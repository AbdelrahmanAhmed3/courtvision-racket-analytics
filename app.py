"""CourtVision's local app: analyze a clip in a few clicks, or use every option."""

# ruff: noqa: E402, I001

from __future__ import annotations

import base64
import csv
import hashlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from imageio_ffmpeg import get_ffmpeg_exe
from streamlit_image_coordinates import streamlit_image_coordinates

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from courtvision.calibration.homography import estimate_template_homography  # noqa: E402
from courtvision.calibration.court_surface import (  # noqa: E402
    CourtSurfaceSettings,
    SurfaceBoundarySettings,
    analyze_court_surface_color,
    draw_court_color_distance,
    draw_court_color_samples,
    draw_surface_boundary_refinement,
    evaluate_surface_threshold_stability,
)
from courtvision.calibration.auto import (  # noqa: E402
    MIN_CONFIDENCE,
    AutoCalibrationProposal,
    AutoCalibrationSettings,
    analyze_auto_calibration,
    draw_auto_proposal,
    draw_ransac_debug,
)
from courtvision.calibration.io import (  # noqa: E402
    CalibrationRecord,
    LandmarkObservation,
    save_calibration,
)
from courtvision.calibration.landmarks import (  # noqa: E402
    default_landmark_names,
    display_landmark_name,
    landmark_by_name,
)
from courtvision.calibration.keypoints import (  # noqa: E402
    KeypointCalibrationProposal,
    default_keypoint_model_id,
    detect_keypoint_calibration,
    draw_landmark_audit,
    landmark_audit_entries,
)
from courtvision.calibration.keypoint_refinement import (  # noqa: E402
    KeypointRefinementSettings,
    draw_keypoint_refinement,
)
from courtvision.calibration.validation import validate_homography  # noqa: E402
from courtvision.visualization.court_overlay import (  # noqa: E402
    draw_calibration_overlay,
)
from courtvision.detectors.rfdetr_detector import (  # noqa: E402
    DEFAULT_SIZE as DEFAULT_RFDETR_SIZE,
)
from courtvision.detectors.rfdetr_detector import RFDETR_SIZES  # noqa: E402

DEFAULT_MODEL_ID = "tennis-v4d0h/2"
LOCAL_DETECTOR = "RF-DETR (local)"
HOSTED_DETECTOR = "Roboflow (hosted)"
DEFAULT_TRACKNET_DIR = REPO_ROOT / "tracknet-model" / "TrackNet"
DEFAULT_TRACKNET_WEIGHTS = REPO_ROOT / "tracknet-model" / "model_best.pt"
AUTO_SEARCH_WINDOW_SECONDS = 6.0
# Invalidates only Streamlit's cached keypoint proposal after a point-order fix.
KEYPOINT_MAPPING_REVISION = "2026-08-19-padel-front-back-v2"


def reset_calibration() -> None:
    # The last click is kept: the image keeps returning it on every rerun, and
    # forgetting it would add that point back.
    st.session_state.calibration_points = []


def save_uploaded_video(uploaded_file) -> Path:
    content = uploaded_file.getvalue()
    digest = hashlib.sha256(content).hexdigest()[:12]
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    path = REPO_ROOT / "outputs" / "ui_uploads" / f"{digest}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(content)
    return path


def video_metadata(path: Path) -> tuple[float, int, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("The uploaded video could not be opened.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if frame_count <= 0:
        raise ValueError("The uploaded video has no readable frames.")
    return fps, frame_count, width, height


def read_frame(path: Path, frame_index: int):
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"Could not read frame {frame_index}.")
    return frame


def search_auto_calibration_window(
    video_path: Path,
    fps: float,
    frame_count: int,
    start_frame_index: int,
    window_seconds: float,
    court_type: str,
    settings: AutoCalibrationSettings,
) -> tuple[int, AutoCalibrationProposal] | None:
    """Return the strongest valid Auto proposal sampled from a short window."""
    step = max(1, round(fps * 0.5))
    last_frame_index = min(
        frame_count - 1,
        start_frame_index + round(window_seconds * fps),
    )
    best: tuple[int, AutoCalibrationProposal] | None = None
    for frame_index in range(start_frame_index, last_frame_index + 1, step):
        debug = analyze_auto_calibration(
            read_frame(video_path, frame_index),
            court_type,
            minimum_confidence=MIN_CONFIDENCE,
            settings=settings,
        )
        proposal = debug.proposal
        if proposal is None:
            continue
        if best is None or proposal.confidence > best[1].confidence:
            best = (frame_index, proposal)
    return best


def draw_selected_points(
    frame,
    landmark_names: list[str],
    points: list[tuple[float, float]],
):
    annotated = frame.copy()
    for index, point in enumerate(points):
        x, y = (int(round(value)) for value in point)
        cv2.circle(annotated, (x, y), 7, (255, 0, 255), -1)
        cv2.putText(
            annotated,
            str(index + 1),
            (x + 9, y - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def clicked_point(click: dict, frame) -> tuple[float, float]:
    """A click on the frame as shown, in the frame's own pixels.

    The frame is drawn to fit its column, so a 1280-pixel frame on a laptop
    screen is shown smaller than it is.
    """
    scale = frame.shape[1] / click["width"] if click.get("width") else 1.0
    return float(click["x"]) * scale, float(click["y"]) * scale


def build_calibration(
    video_path: Path,
    court_type: str,
    frame_index: int,
    frame,
    points: list[tuple[float, float]],
    landmark_names: list[str] | None = None,
    source: str = "manual-ui",
) -> CalibrationRecord:
    names = landmark_names or default_landmark_names(court_type)
    definitions = [landmark_by_name(court_type, name) for name in names]
    observations = {
        definition.name: LandmarkObservation(
            image=point,
            template=definition.template,
            visible=True,
            confidence=1.0,
            source=source,
        )
        for definition, point in zip(definitions, points, strict=True)
    }
    height, width = frame.shape[:2]
    return CalibrationRecord(
        video_id=video_path.name,
        frame_index=frame_index,
        frame_width=width,
        frame_height=height,
        court_type=court_type,
        landmarks=observations,
    )


def run_pipeline(
    video_path: Path,
    output_dir: Path,
    calibration_path: Path,
    detector_args: list[str],
    seconds: float,
    tracknet_dir: str | None,
    tracknet_model_path: str | None,
    tracknet_device: str,
    clone_tracknet: bool,
    track_calibration: bool,
    confidence: float,
    max_missing_seconds: float,
    reassociation_distance_px: float,
    allow_calibration_warning: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "scripts/run_full_pipeline.py",
        "--input",
        str(video_path),
        "--output-dir",
        str(output_dir),
        *detector_args,
        "--confidence",
        str(confidence),
        "--max-missing-seconds",
        str(max_missing_seconds),
        "--reassociation-distance-px",
        str(reassociation_distance_px),
        "--max-seconds",
        str(seconds),
        "--calibration",
        str(calibration_path),
        "--draw-court-map",
        "--draw-calibration-overlay",
    ]
    if tracknet_model_path:
        command.extend(
            [
                "--tracknet-dir",
                tracknet_dir or "",
                "--tracknet-model-path",
                tracknet_model_path,
                "--tracknet-device",
                tracknet_device,
            ]
        )
        if clone_tracknet:
            command.append("--clone-tracknet")
    if track_calibration:
        command.append("--track-calibration")
    if allow_calibration_warning:
        command.append("--allow-calibration-warning")
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def transcode_for_browser(video_path: Path) -> Path:
    """Create an H.264/yuv420p MP4 that browser video elements can decode."""
    output_path = video_path.with_name(f"{video_path.stem}_web.mp4")
    if output_path.exists() and (
        output_path.stat().st_mtime >= video_path.stat().st_mtime
    ):
        return output_path
    result = subprocess.run(
        [
            get_ffmpeg_exe(),
            "-y",
            "-i",
            str(video_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or "Could not prepare video for the browser.")
    return output_path


def video_data_url(video_path: Path) -> str:
    encoded = base64.b64encode(video_path.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


HEATMAP_LINE = "![Heatmaps](heatmaps.png)"


def show_match_report(output_dir: Path) -> None:
    """The match report under the videos, with the heatmaps where they belong."""
    report_path = output_dir / "report.md"
    if not report_path.exists():
        return
    st.subheader("Match report")
    # Its sections sit under the subheader, so they are a level smaller.
    text = report_path.read_text().replace("\n## ", "\n#### ")
    before, _, after = text.partition(HEATMAP_LINE)
    # The report's own title repeats the file name; the subheader replaces it.
    st.markdown(before.split("\n", 1)[1] if before.startswith("# ") else before)
    heatmaps = output_dir / "heatmaps.png"
    if after and heatmaps.exists():
        st.image(
            str(heatmaps), caption="Seconds per square metre; far baseline at the top"
        )
        st.markdown(after)
    download_pdf, download_report, download_json = st.columns(3)
    if (output_dir / "report.pdf").exists():
        download_pdf.download_button(
            "Download report (PDF)",
            data=(output_dir / "report.pdf").read_bytes(),
            file_name="report.pdf",
            mime="application/pdf",
            type="primary",
        )
    download_report.download_button(
        "Download report (Markdown)",
        data=report_path.read_bytes(),
        file_name="report.md",
        mime="text/markdown",
    )
    download_json.download_button(
        "Download report (JSON)",
        data=(output_dir / "report.json").read_bytes(),
        file_name="report.json",
        mime="application/json",
    )


def projection_counts(path: Path) -> dict[str, int]:
    counts = {"player": 0, "ball": 0}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            object_type = row["object_type"]
            if object_type in counts:
                counts[object_type] += 1
    return counts


def shot_count(path: Path) -> int:
    with path.open(newline="") as file:
        return sum(1 for _ in csv.DictReader(file))


def render_synchronized_videos(tracked_video: Path, court_map_video: Path) -> None:
    tracked_url = video_data_url(tracked_video)
    map_url = video_data_url(court_map_video)
    components.html(
        f"""
        <style>
          body {{ margin: 0; background: #101817; color: #eef2ef;
                 font-family: sans-serif; }}
          .results {{ display: grid;
                     grid-template-columns: minmax(0, 3fr) minmax(260px, 2fr);
                     gap: 12px; }}
          .panel {{ background: #182422; padding: 10px; border-radius: 6px; }}
          .title {{ font-size: 14px; margin: 0 0 8px; color: #d8e5dc; }}
          video {{ width: 100%; max-height: 72vh; background: #000; display: block; }}
        </style>
        <div class="results">
          <section class="panel">
            <p class="title">Tracked video</p>
            <video id="tracked" controls preload="metadata" src="{tracked_url}"></video>
          </section>
          <section class="panel">
            <p class="title">Court map</p>
            <video id="court-map" controls preload="metadata" src="{map_url}"></video>
          </section>
        </div>
        <script>
          const tracked = document.getElementById('tracked');
          const courtMap = document.getElementById('court-map');
          let synchronizing = false;
          function syncTime(source, target) {{
            if (synchronizing ||
                Math.abs(source.currentTime - target.currentTime) < 0.08) return;
            synchronizing = true;
            target.currentTime = source.currentTime;
            synchronizing = false;
          }}
          function syncPlay(source, target) {{
            if (!source.paused) target.play().catch(() => {{}});
          }}
          tracked.addEventListener('timeupdate', () => syncTime(tracked, courtMap));
          courtMap.addEventListener('timeupdate', () => syncTime(courtMap, tracked));
          tracked.addEventListener('play', () => syncPlay(tracked, courtMap));
          courtMap.addEventListener('play', () => syncPlay(courtMap, tracked));
          tracked.addEventListener('pause', () => courtMap.pause());
          courtMap.addEventListener('pause', () => tracked.pause());
          tracked.addEventListener('seeking', () => syncTime(tracked, courtMap));
          courtMap.addEventListener('seeking', () => syncTime(courtMap, tracked));
        </script>
        """,
        height=700,
        scrolling=False,
    )


# The "Analyze a clip" page: drop a clip, mark the court, get the report.
QUICK_LANDMARKS = (
    "far_left_service_intersection",
    "far_service_center",
    "far_right_service_intersection",
    "near_right_service_intersection",
    "near_service_center",
    "near_left_service_intersection",
)
QUICK_POINT_HINTS = (
    "far service line, left end",
    "far service line, middle (where the centre line meets it)",
    "far service line, right end",
    "near service line, right end",
    "near service line, middle (where the centre line meets it)",
    "near service line, left end",
)
# The guide's points in metres: across the court, then from the far back wall.
QUICK_GUIDE_POINTS_M = ((0, 3), (5, 3), (10, 3), (10, 17), (5, 17), (0, 17))
VIDEO_TYPES = ("mp4", "mov", "avi", "mkv")
SECONDS_PER_CLIP_SECOND = 8  # about 2 minutes for a 15 s clip on an Apple M4


def reset_quick_points() -> None:
    # The last click is kept, as in reset_calibration().
    st.session_state.quick_points = []


def court_guide(next_index: int) -> np.ndarray:
    """A small top-down padel court with the points to click, the next one ringed."""
    scale, margin = 14, 16
    width, height = 10 * scale + 2 * margin, 20 * scale + 2 * margin
    image = np.full((height, width, 3), (60, 110, 60), np.uint8)

    def at(x_m: float, y_m: float) -> tuple[int, int]:
        return margin + round(x_m * scale), margin + round(y_m * scale)

    white = (235, 240, 235)
    cv2.rectangle(image, at(0, 0), at(10, 20), white, 1)
    cv2.line(image, at(0, 10), at(10, 10), white, 2)
    for y_m in (3, 17):
        cv2.line(image, at(0, y_m), at(10, y_m), white, 1)
    cv2.line(image, at(5, 3), at(5, 17), white, 1)
    for index, (x_m, y_m) in enumerate(QUICK_GUIDE_POINTS_M):
        done = index < next_index
        colour = (255, 0, 255) if done else (0, 220, 255)
        cv2.circle(image, at(x_m, y_m), 5, colour, -1)
        if index == next_index:
            cv2.circle(image, at(x_m, y_m), 10, (0, 220, 255), 2)
        x, y = at(x_m, y_m)
        cv2.putText(
            image,
            str(index + 1),
            (x - 4, y - 9 if y_m < 10 else y + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        image, "far side", at(3.2, -0.4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, white, 1
    )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def choose_clip() -> Path | None:
    uploaded = st.file_uploader("Drop a padel clip", type=list(VIDEO_TYPES))
    if uploaded is not None:
        return save_uploaded_video(uploaded)
    local = sorted(
        path
        for path in (REPO_ROOT / "data" / "raw").glob("*")
        if path.suffix.lower().lstrip(".") in VIDEO_TYPES
    )
    if not local:
        return None
    choice = st.selectbox(
        "...or pick a clip from data/raw/",
        [None, *local],
        format_func=lambda path: "Choose a clip" if path is None else path.name,
    )
    return choice


def quick_pipeline_command(
    video_path: Path, output_dir: Path, calibration_path: Path, ball: bool
) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_full_pipeline.py",
        "--input",
        str(video_path),
        "--output-dir",
        str(output_dir),
        "--calibration",
        str(calibration_path),
        "--draw-court-map",
    ]
    if ball:
        command += [
            "--tracknet-dir",
            str(DEFAULT_TRACKNET_DIR),
            "--tracknet-model-path",
            str(DEFAULT_TRACKNET_WEIGHTS),
        ]
    return command


def run_with_progress(command: list[str]) -> tuple[int, str]:
    """Run the pipeline, showing its latest output line while it works."""
    log: list[str] = []
    with st.status("Analyzing the clip...", expanded=True) as status:
        latest = st.empty()
        # Text mode reads tqdm's carriage-return updates as lines too.
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if line:
                log.append(line)
                latest.caption(line[:160])
        code = process.wait()
        status.update(
            label="Done." if code == 0 else "The analysis stopped with an error.",
            state="complete" if code == 0 else "error",
            expanded=code != 0,
        )
    return code, "\n".join(log[-40:])


def show_results(output_dir: Path) -> None:
    try:
        tracked_video = transcode_for_browser(output_dir / "annotated.mp4")
        court_map_video = transcode_for_browser(output_dir / "court_map.mp4")
    except RuntimeError as error:
        st.error(f"Could not prepare the videos for the browser: {error}")
    else:
        render_synchronized_videos(tracked_video, court_map_video)
    if (output_dir / "report.md").exists():
        show_match_report(output_dir)
    else:
        st.warning("No report: no player was tracked long enough on the court.")
    st.caption(f"Every output is in {output_dir.relative_to(REPO_ROOT)}.")


def analyze_page() -> None:
    st.title("CourtVision")
    st.caption(
        "Drop a padel clip filmed from one high camera, without cuts. CourtVision "
        "tracks the four players and the ball, maps them onto the court, and writes "
        "a match report you can download as a PDF."
    )
    video_path = choose_clip()
    if video_path is None:
        return
    try:
        fps, frame_count, width, height = video_metadata(video_path)
    except ValueError as error:
        st.error(str(error))
        return
    duration = frame_count / fps
    minutes = max(1, round(duration * SECONDS_PER_CLIP_SECOND / 60))
    st.caption(
        f"{video_path.name}: {width} x {height}, {fps:.0f} fps, {duration:.1f} s. "
        f"Analyzing it takes about {minutes} min."
    )

    st.subheader("1. Mark the court")
    frame_seconds = st.slider(
        "Frame to mark",
        min_value=0.0,
        max_value=max(0.0, duration - 1 / fps),
        value=min(3.0, max(0.0, duration - 1 / fps)),
        step=0.5,
        help="Pick a frame where both service lines are clear of players.",
    )
    frame_index = min(int(round(frame_seconds * fps)), frame_count - 1)
    frame_key = (str(video_path), frame_index)
    if st.session_state.get("quick_frame_key") != frame_key:
        reset_quick_points()
        st.session_state.quick_frame_key = frame_key
        st.session_state.quick_result = None
    frame = read_frame(video_path, frame_index)
    points = st.session_state.quick_points
    next_index = len(points)

    calibration = estimate = validation = None
    if next_index == len(QUICK_LANDMARKS):
        calibration = build_calibration(
            video_path,
            "padel",
            frame_index,
            frame,
            points,
            list(QUICK_LANDMARKS),
            source="quick-ui",
        )
        estimate = estimate_template_homography(calibration)
        validation = validate_homography(estimate)

    guide_column, frame_column = st.columns([1, 5])
    with guide_column:
        st.image(court_guide(next_index), caption="Click these points in order.")
    with frame_column:
        if calibration is None:
            st.markdown(
                f"Click point **{next_index + 1} of {len(QUICK_LANDMARKS)}**: the "
                f"{QUICK_POINT_HINTS[next_index]}."
            )
            shown = draw_selected_points(frame, list(QUICK_LANDMARKS), points)
        else:
            st.markdown("The court lines below should sit on the painted lines.")
            shown = draw_calibration_overlay(frame, calibration, estimate, validation)
        click = streamlit_image_coordinates(
            cv2.cvtColor(shown, cv2.COLOR_BGR2RGB),
            key=f"quick-frame-{frame_key}",
            use_column_width="always",
            cursor="crosshair",
        )
    if click and calibration is None:
        if click["unix_time"] != st.session_state.quick_last_click:
            st.session_state.quick_points.append(clicked_point(click, frame))
            st.session_state.quick_last_click = click["unix_time"]
            st.rerun()
    undo_column, reset_column, _ = st.columns([1, 1, 4])
    if undo_column.button("Undo last point", disabled=not points):
        st.session_state.quick_points.pop()
        st.session_state.quick_result = None
        st.rerun()
    if reset_column.button("Start again", disabled=not points):
        reset_quick_points()
        st.session_state.quick_result = None
        st.rerun()
    if calibration is None:
        return
    if validation.status == "bad":
        worst = max(
            validation.per_landmark_error_px, key=validation.per_landmark_error_px.get
        )
        st.error(
            "These points don't fit a court. The one furthest off is the "
            f"{QUICK_POINT_HINTS[QUICK_LANDMARKS.index(worst)]}: undo back to it, "
            "or start again."
        )
        return
    st.success(
        f"Court marked: the points fit within "
        f"{validation.max_reprojection_error_px:.1f} px."
    )

    st.subheader("2. Analyze")
    ball = DEFAULT_TRACKNET_WEIGHTS.exists() and DEFAULT_TRACKNET_DIR.exists()
    if not ball:
        st.info(
            "Ball tracking is off: no TrackNet weights at "
            "tracknet-model/model_best.pt. You'll get the players, teams, movement "
            "and heatmaps, but no shots or rallies. The README explains how to add "
            "them."
        )
    if st.button("Analyze", type="primary"):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / "outputs" / "ui_runs" / f"{video_path.stem}_{run_id}"
        calibration_path = output_dir / "calibration.json"
        save_calibration(calibration_path, calibration)
        code, log = run_with_progress(
            quick_pipeline_command(video_path, output_dir, calibration_path, ball)
        )
        if code != 0:
            st.code(log)
            return
        st.session_state.quick_result = str(output_dir)
    if st.session_state.get("quick_result"):
        st.subheader("3. Results")
        show_results(Path(st.session_state.quick_result))


def initialize_state() -> None:
    st.session_state.setdefault("calibration_points", [])
    st.session_state.setdefault("last_click", None)
    st.session_state.setdefault("frame_key", None)
    st.session_state.setdefault("auto_seed_key", None)
    st.session_state.setdefault("auto_search_result", None)


def advanced_page() -> None:
    st.title("CourtVision: advanced")
    st.caption(
        "Every option: tennis or padel, detector and tracking settings, and the "
        "experimental automatic calibration."
    )

    with st.sidebar:
        st.header("Run setup")
        uploaded_video = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv"])
        court_type = st.selectbox("Court type", ["tennis", "padel"])
        detector_choice = st.radio(
            "Player detector",
            [LOCAL_DETECTOR, HOSTED_DETECTOR],
            help=(
                "RF-DETR runs on this computer and needs no API key. Roboflow "
                "sends frames to its hosted API and needs ROBOFLOW_API_KEY."
            ),
        )
        if detector_choice == HOSTED_DETECTOR:
            model_id = st.text_input("Roboflow model", value=DEFAULT_MODEL_ID)
            detector_args = ["--detector", "roboflow", "--model-id", model_id]
        else:
            rfdetr_size = st.selectbox(
                "RF-DETR size",
                RFDETR_SIZES,
                index=RFDETR_SIZES.index(DEFAULT_RFDETR_SIZE),
                help="Larger models are more accurate and slower.",
            )
            detector_args = ["--detector", "rfdetr", "--rfdetr-size", rfdetr_size]
        confidence = st.slider(
            "Player detection confidence",
            min_value=0.05,
            max_value=0.95,
            value=0.30,
            step=0.05,
        )
        max_missing_seconds = st.slider(
            "Player ID recovery window (seconds)",
            min_value=0.5,
            max_value=10.0,
            value=3.0,
            step=0.5,
        )
        reassociation_distance_px = st.number_input(
            "ID reassociation distance (pixels)",
            min_value=50,
            max_value=1000,
            value=250,
            step=25,
        )
        track_calibration = st.checkbox(
            "Adapt calibration to camera movement",
            value=True,
            help=(
                "Tracks the clicked court landmarks with sparse optical flow and "
                "uses RANSAC to refresh the homography per frame."
            ),
        )
        track_ball = st.checkbox(
            "Include TrackNet ball projection",
            value=DEFAULT_TRACKNET_WEIGHTS.exists(),
        )
        tracknet_dir = None
        tracknet_model_path = None
        tracknet_device = "auto"
        clone_tracknet = False
        if track_ball:
            st.caption(
                "Runs player and ball projection together on the same court map."
            )
            tracknet_dir = st.text_input(
                "TrackNet source directory",
                value=str(DEFAULT_TRACKNET_DIR),
            )
            tracknet_model_path = st.text_input(
                "TrackNet weights",
                value=str(DEFAULT_TRACKNET_WEIGHTS),
            )
            tracknet_device = st.selectbox(
                "TrackNet device",
                ["auto", "mps", "cuda", "cpu"],
            )
            clone_tracknet = st.checkbox(
                "Download TrackNet source if missing",
                value=True,
            )

    if uploaded_video is None:
        st.info("Choose a local video to begin.")
        return

    try:
        video_path = save_uploaded_video(uploaded_video)
        fps, frame_count, width, height = video_metadata(video_path)
    except ValueError as error:
        st.error(str(error))
        return

    duration = frame_count / fps
    st.video(str(video_path))
    setup_column, warning_column = st.columns([3, 2])
    with setup_column:
        calibration_seconds = st.slider(
            "Calibration timestamp (seconds)",
            min_value=0.0,
            max_value=float(duration),
            value=min(1.0, float(duration)),
            step=0.1,
        )
        run_seconds = st.slider(
            "Process from the start (seconds)",
            min_value=0.1,
            max_value=float(duration),
            value=min(15.0, float(duration)),
            step=0.1,
        )
        calibration_input = st.radio(
            "Calibration input",
            ["Manual", "Auto", "Assisted"],
            horizontal=True,
            help=(
                "Manual is the validated path. Auto and Assisted are experimental: "
                "Auto accepts a validated model or line-based proposal; Assisted "
                "starts from a proposal and lets you correct its landmarks."
            ),
        )
        proposal_engine = "Keypoint model"
        if calibration_input != "Manual":
            proposal_engine = st.radio(
                "Automatic proposal engine",
                ["Keypoint model", "RANSAC court lines"],
                horizontal=True,
            )
            st.caption(
                "Experimental: automatic proposals are reliable on the near side "
                "of the court but weak on the far side. Check every landmark."
            )
        calibration_geometry = st.radio(
            "Calibration mode",
            ["Eight landmarks", "Four baseline corners (wide-angle fallback)"],
            horizontal=True,
            disabled=calibration_input != "Manual",
            help=(
                "Use the four-corner fallback for strong barrel distortion. "
                "It maps the court boundary exactly, but is less accurate away "
                "from those anchors."
            ),
        )
    with warning_column:
        st.warning(
            "Temporal calibration starts at the selected calibration timestamp. "
            "After a camera cut or failed validation, that frame is left unmapped."
        )
        st.caption(f"Video: {width} x {height}, {fps:.2f} fps, {duration:.1f}s")

    requested_frame_index = min(int(round(calibration_seconds * fps)), frame_count - 1)
    auto_search_context = (str(video_path), court_type, requested_frame_index)
    auto_search_result = st.session_state.auto_search_result
    frame_index = requested_frame_index
    if (
        calibration_input != "Manual"
        and auto_search_result is not None
        and auto_search_result["context"] == auto_search_context
        and auto_search_result["frame_index"] is not None
    ):
        frame_index = auto_search_result["frame_index"]
    use_four_corner_fallback = (
        calibration_input == "Manual"
        and calibration_geometry.startswith("Four baseline")
    )
    all_landmark_names = default_landmark_names(court_type)
    landmark_names = (
        all_landmark_names[:4] if use_four_corner_fallback else all_landmark_names
    )
    frame_key = (
        str(video_path),
        court_type,
        frame_index,
        calibration_input,
        calibration_geometry,
    )
    if st.session_state.frame_key != frame_key:
        reset_calibration()
        st.session_state.frame_key = frame_key
    frame = read_frame(video_path, frame_index)
    auto_settings = AutoCalibrationSettings()
    keypoint_model_id = default_keypoint_model_id(court_type)
    keypoint_confidence = 0.15
    refine_keypoint_lines = True
    keypoint_refinement_settings = KeypointRefinementSettings()
    show_keypoint_refinement = True
    show_court_color_debug = False
    court_surface_settings = CourtSurfaceSettings()
    refine_padel_surface = True
    show_surface_boundary_debug = False
    surface_boundary_settings = SurfaceBoundarySettings()
    show_auto_debug = False
    if calibration_input != "Manual" and proposal_engine == "RANSAC court lines":
        with st.expander("Auto calibration lab", expanded=True):
            st.caption(
                "Near-white pixels become a point cloud. RANSAC fits full court "
                "lines across gaps caused by players, then court geometry chooses "
                "the outer doubles court. If this frame has no valid proposal, "
                "Auto searches the next six seconds."
            )
            pixel_column, ransac_column = st.columns(2)
            with pixel_column:
                minimum_value = st.slider("Near-white brightness", 80, 245, 170, 5)
                maximum_saturation = st.slider("Near-white saturation", 0, 160, 80, 5)
            with ransac_column:
                min_line_length = st.slider(
                    "Minimum line length (% width)", 2, 35, 8, 1
                )
                line_distance = st.slider(
                    "RANSAC line distance (px)", 1.0, 4.0, 2.5, 0.5
                )
                ransac_trials = st.slider(
                    "RANSAC trials per line", 100, 2_000, 700, 100
                )
                maximum_line_count = st.slider("Maximum fitted lines", 4, 40, 20, 1)
            crop_left, crop_right, crop_top, crop_bottom = 0, 100, 0, 100
            if st.checkbox("Advanced: limit search region manually", value=False):
                st.caption(
                    "Use only for debugging; Auto normally searches the full frame."
                )
                crop_column, crop_preview_column = st.columns(2)
                with crop_column:
                    crop_left = st.slider("Search left", 0, 80, 0, 1)
                    crop_right = st.slider("Search right", 20, 100, 100, 1)
                with crop_preview_column:
                    crop_top = st.slider("Search top", 0, 80, 0, 1)
                    crop_bottom = st.slider("Search bottom", 20, 100, 100, 1)
                if crop_left >= crop_right - 5 or crop_top >= crop_bottom - 5:
                    st.error("Keep the search region at least 5% wide and high.")
                    return
            show_auto_debug = st.checkbox(
                "Show near-white mask and RANSAC lines", value=True
            )
            auto_settings = AutoCalibrationSettings(
                minimum_value=minimum_value,
                maximum_saturation=maximum_saturation,
                min_line_length_ratio=min_line_length / 100,
                line_distance_px=line_distance,
                ransac_trials_per_line=ransac_trials,
                maximum_line_count=maximum_line_count,
                crop_top_ratio=crop_top / 100,
                crop_bottom_ratio=crop_bottom / 100,
                crop_left_ratio=crop_left / 100,
                crop_right_ratio=crop_right / 100,
            )
    elif calibration_input != "Manual":
        with st.expander("Keypoint model", expanded=True):
            st.caption(
                "The selected calibration frame is sent to Roboflow's hosted "
                "keypoint model. Painted-line intersections refine its points; "
                "Assisted lets you correct unresolved landmarks."
            )
            keypoint_model_id = st.text_input(
                "Roboflow keypoint model ID",
                value=keypoint_model_id,
            )
            keypoint_confidence = st.slider(
                "Minimum keypoint confidence",
                min_value=0.05,
                max_value=0.95,
                value=0.15,
                step=0.05,
            )
            refine_keypoint_lines = st.checkbox(
                "Refine model points with painted court-line intersections",
                value=True,
                help=(
                    "The model identifies each landmark. Near-white pixels and "
                    "local RANSAC lines determine its final image coordinate."
                ),
            )
            refinement_left, refinement_right = st.columns(2)
            with refinement_left:
                keypoint_line_value = st.slider(
                    "Refinement brightness",
                    80,
                    245,
                    170,
                    5,
                    disabled=not refine_keypoint_lines,
                )
                keypoint_line_saturation = st.slider(
                    "Refinement saturation",
                    0,
                    160,
                    80,
                    5,
                    disabled=not refine_keypoint_lines,
                )
            with refinement_right:
                keypoint_corridor_percent = st.slider(
                    "Model-guided line corridor (% frame height)",
                    2.0,
                    8.0,
                    4.5,
                    0.5,
                    disabled=not refine_keypoint_lines,
                )
                show_keypoint_refinement = st.checkbox(
                    "Show line-refinement diagnostics",
                    value=True,
                    disabled=not refine_keypoint_lines,
                )
            keypoint_refinement_settings = KeypointRefinementSettings(
                minimum_value=keypoint_line_value,
                maximum_saturation=keypoint_line_saturation,
                corridor_width_ratio=keypoint_corridor_percent / 100,
            )
            if court_type == "padel":
                st.divider()
                refine_padel_surface = st.checkbox(
                    "Refine outer padel boundary from court color",
                    value=True,
                    help=(
                        "The model supplies approximate far and near levels. "
                        "The full-frame color mask supplies the final outer "
                        "ground-level endpoints."
                    ),
                )
                show_court_color_debug = st.checkbox(
                    "Show court-color diagnostics",
                    value=True,
                )
                court_color_tolerance = st.slider(
                    "Court-color distance tolerance",
                    min_value=5.0,
                    max_value=60.0,
                    value=24.0,
                    step=1.0,
                    disabled=not show_court_color_debug,
                )
                court_surface_settings = CourtSurfaceSettings(
                    color_distance_threshold=court_color_tolerance,
                )
                show_surface_boundary_debug = st.checkbox(
                    "Show surface-boundary refinement",
                    value=True,
                    disabled=not show_court_color_debug,
                )
                support_column, gap_control_column = st.columns(2)
                with support_column:
                    minimum_surface_support = st.slider(
                        "Minimum sustained row support",
                        min_value=0.40,
                        max_value=0.95,
                        value=0.65,
                        step=0.05,
                        disabled=not refine_padel_surface,
                    )
                with gap_control_column:
                    horizontal_close_px = st.slider(
                        "Boundary gap closing (px)",
                        min_value=3,
                        max_value=31,
                        value=15,
                        step=2,
                        disabled=not refine_padel_surface,
                    )
                surface_boundary_settings = SurfaceBoundarySettings(
                    minimum_central_support=minimum_surface_support,
                    horizontal_close_px=horizontal_close_px,
                )
    proposal: AutoCalibrationProposal | KeypointCalibrationProposal | None = None
    auto_debug = None
    if calibration_input != "Manual" and proposal_engine == "RANSAC court lines":
        auto_debug = analyze_auto_calibration(
            frame,
            court_type,
            minimum_confidence=(
                0.0 if calibration_input == "Assisted" else MIN_CONFIDENCE
            ),
            settings=auto_settings,
        )
        proposal = auto_debug.proposal
        if proposal is not None and calibration_input == "Auto":
            auto_landmark_names = [
                name for name in all_landmark_names if name in proposal.landmarks
            ]
            landmark_names = auto_landmark_names
            use_four_corner_fallback = len(landmark_names) < 6
        if proposal is not None and st.session_state.auto_seed_key != frame_key:
            st.session_state.calibration_points = [
                proposal.landmarks[name]
                for name in landmark_names
                if name in proposal.landmarks
            ]
            st.session_state.auto_seed_key = frame_key
        elif proposal is None and st.session_state.auto_seed_key != frame_key:
            st.session_state.auto_seed_key = frame_key
    elif calibration_input != "Manual":
        keypoint_context = (
            str(video_path),
            court_type,
            frame_index,
            keypoint_model_id,
            keypoint_confidence,
            refine_keypoint_lines,
            keypoint_refinement_settings,
            refine_padel_surface,
            court_surface_settings,
            surface_boundary_settings,
            KEYPOINT_MAPPING_REVISION,
        )
        cached_keypoint = st.session_state.get("keypoint_proposal")
        if cached_keypoint is None or cached_keypoint["context"] != keypoint_context:
            try:
                proposal = detect_keypoint_calibration(
                    frame=frame,
                    court_type=court_type,
                    model_id=keypoint_model_id,
                    minimum_keypoint_confidence=keypoint_confidence,
                    refine_with_court_lines=refine_keypoint_lines,
                    refinement_settings=keypoint_refinement_settings,
                    refine_padel_surface=refine_padel_surface,
                    court_surface_settings=court_surface_settings,
                    surface_boundary_settings=surface_boundary_settings,
                )
                st.session_state.keypoint_proposal = {
                    "context": keypoint_context,
                    "proposal": proposal,
                }
            except Exception as error:
                st.error(f"Keypoint model inference failed: {error}")
                st.session_state.keypoint_proposal = {
                    "context": keypoint_context,
                    "proposal": None,
                }
        else:
            proposal = cached_keypoint["proposal"]
        keypoint_seed_key = (
            "keypoint",
            keypoint_context,
            None if proposal is None else tuple(proposal.landmarks.items()),
        )
        if proposal is not None and st.session_state.auto_seed_key != keypoint_seed_key:
            st.session_state.calibration_points = list(proposal.landmarks.values())
            st.session_state.auto_seed_key = keypoint_seed_key
    if proposal is not None:
        landmark_names = list(proposal.landmarks)
        use_four_corner_fallback = len(landmark_names) < 6
    search_already_attempted = (
        auto_search_result is not None
        and auto_search_result["context"] == auto_search_context
    )
    if (
        calibration_input == "Auto"
        and proposal_engine == "RANSAC court lines"
        and proposal is None
        and not search_already_attempted
    ):
        with st.spinner("Searching the next six seconds for a valid court..."):
            result = search_auto_calibration_window(
                video_path,
                fps,
                frame_count,
                requested_frame_index,
                AUTO_SEARCH_WINDOW_SECONDS,
                court_type,
                auto_settings,
            )
        st.session_state.auto_search_result = {
            "context": auto_search_context,
            "frame_index": None if result is None else result[0],
            "confidence": None if result is None else result[1].confidence,
        }
        st.rerun()
    if frame_index != requested_frame_index:
        st.info(
            "Auto-search selected "
            f"{frame_index / fps:.1f}s from the window beginning at "
            f"{calibration_seconds:.1f}s."
        )
    points = st.session_state.calibration_points
    next_index = len(points)

    st.subheader("Court calibration")
    if auto_debug is not None and show_auto_debug:
        mask_column, hough_column = st.columns(2)
        with mask_column:
            st.caption("1. Near-white pixel mask")
            st.image(auto_debug.line_mask, clamp=True)
        with hough_column:
            st.caption(f"2. RANSAC fitted lines ({len(auto_debug.lines)} found)")
            st.image(
                cv2.cvtColor(
                    draw_ransac_debug(frame, auto_debug),
                    cv2.COLOR_BGR2RGB,
                )
            )
    if (
        isinstance(proposal, KeypointCalibrationProposal)
        and proposal.refinement is not None
        and show_keypoint_refinement
    ):
        mask_column, refinement_column = st.columns(2)
        with mask_column:
            st.caption("1. Near-white refinement mask")
            st.image(proposal.refinement.line_mask, clamp=True)
        with refinement_column:
            st.caption(
                "2. Model-guided lines and intersections "
                f"({proposal.refined_count}/{proposal.landmark_count} refined)"
            )
            st.image(
                cv2.cvtColor(
                    draw_keypoint_refinement(frame, proposal.refinement),
                    cv2.COLOR_BGR2RGB,
                )
            )
    if (
        isinstance(proposal, KeypointCalibrationProposal)
        and court_type == "padel"
        and show_court_color_debug
    ):
        color_landmarks = (
            proposal.surface_refinement.original_landmarks
            if proposal.surface_refinement is not None
            else proposal.refinement.original_landmarks
            if proposal.refinement is not None
            else proposal.landmarks
        )
        try:
            surface_analysis = analyze_court_surface_color(
                frame,
                color_landmarks,
                court_surface_settings,
            )
        except ValueError as error:
            st.warning(f"Court-color diagnostics unavailable: {error}")
        else:
            st.caption("Court-surface color evidence")
            swatch_column, sample_column, distance_column, mask_column = st.columns(
                [1, 2, 2, 2]
            )
            with swatch_column:
                swatch = np.full(
                    (96, 160, 3),
                    surface_analysis.court_bgr,
                    dtype=np.uint8,
                )
                st.image(cv2.cvtColor(swatch, cv2.COLOR_BGR2RGB))
                st.metric("Sampled color", surface_analysis.court_rgb_hex)
                st.metric(
                    "Full-frame match",
                    f"{surface_analysis.coverage_ratio:.1%}",
                )
            with sample_column:
                st.caption("1. Model-guided samples")
                st.image(
                    cv2.cvtColor(
                        draw_court_color_samples(frame, surface_analysis),
                        cv2.COLOR_BGR2RGB,
                    )
                )
            with distance_column:
                st.caption("2. Color similarity")
                st.image(
                    cv2.cvtColor(
                        draw_court_color_distance(surface_analysis),
                        cv2.COLOR_BGR2RGB,
                    )
                )
            with mask_column:
                st.caption("3. Full-frame raw mask")
                st.image(surface_analysis.surface_mask, clamp=True)
            if show_surface_boundary_debug:
                surface_refinement = proposal.surface_refinement
                if surface_refinement is None:
                    st.warning(
                        "No sustained far/near court-color boundary was found. "
                        "The model corners were kept for this proposal."
                    )
                else:
                    threshold_stability = evaluate_surface_threshold_stability(
                        frame,
                        color_landmarks,
                        court_surface_settings,
                        surface_boundary_settings,
                    )
                    confidence_column, shift_column, row_column = st.columns(3)
                    confidence_column.metric(
                        "Boundary confidence",
                        f"{surface_refinement.confidence:.0%}",
                    )
                    shift_column.metric(
                        "Mean model-point move",
                        f"{surface_refinement.mean_shift_px:.1f}px",
                    )
                    row_column.metric(
                        "Selected ground rows",
                        f"{surface_refinement.far_row} / {surface_refinement.near_row}",
                    )
                    stability_column, change_column = st.columns(2)
                    stability_column.metric(
                        "Nearby tolerance check",
                        "Stable" if threshold_stability.stable else "Sensitive",
                        help=(
                            "Runs the boundary at the current color tolerance "
                            "and at plus/minus 6."
                        ),
                    )
                    maximum_change = threshold_stability.maximum_corner_change_px
                    change_column.metric(
                        "Maximum corner change",
                        f"{maximum_change:.1f}px"
                        if np.isfinite(maximum_change)
                        else "Unavailable",
                    )
                    st.caption(
                        "Tolerance trials: "
                        + ", ".join(
                            f"{threshold:g}"
                            + (
                                " passed"
                                if threshold
                                in threshold_stability.successful_thresholds
                                else " failed"
                            )
                            for threshold in threshold_stability.tested_thresholds
                        )
                    )
                    st.caption(
                        "4. Outer surface boundary: magenta is the model, "
                        "green is the color-derived corner"
                    )
                    st.image(
                        cv2.cvtColor(
                            draw_surface_boundary_refinement(
                                frame,
                                surface_refinement,
                            ),
                            cv2.COLOR_BGR2RGB,
                        )
                    )
    if calibration_input == "Auto" and proposal is None:
        if proposal_engine == "Keypoint model":
            st.error(
                "The keypoint model did not return four reliable court corners "
                "for this frame. Choose Assisted or a different calibration frame."
            )
        else:
            st.error(
                "Auto calibration could not find a valid court at the selected "
                "frame or in the next six seconds. Choose Assisted or Manual."
            )
        return
    if proposal is not None:
        auto_metric, line_metric, support_metric = st.columns(3)
        auto_metric.metric("Auto confidence", f"{proposal.confidence:.0%}")
        if isinstance(proposal, AutoCalibrationProposal):
            line_metric.metric("RANSAC candidate lines", proposal.line_count)
            support_metric.metric("Line support", f"{proposal.line_support:.0%}")
        else:
            line_metric.metric(
                "Refined intersections",
                f"{proposal.refined_count}/{proposal.landmark_count}",
            )
            support_metric.metric(
                "Line support",
                f"{proposal.mean_line_support:.0%}",
            )
            st.caption(f"Keypoint model: {proposal.model_id}")
        st.caption(
            f"Temporal landmarks: {len(proposal.landmarks)} available for "
            "optical-flow tracking."
        )
        if isinstance(proposal, KeypointCalibrationProposal):
            audit_entries = landmark_audit_entries(proposal)
            with st.expander("Final landmark audit", expanded=True):
                st.caption(
                    "Green: court-color boundary | Yellow: near-white service "
                    "line | Cyan: RANSAC intersection | Magenta: model only"
                )
                st.image(
                    cv2.cvtColor(
                        draw_landmark_audit(frame, proposal),
                        cv2.COLOR_BGR2RGB,
                    )
                )
                st.dataframe(
                    {
                        "#": list(range(1, len(audit_entries) + 1)),
                        "landmark": [
                            display_landmark_name(entry.name) for entry in audit_entries
                        ],
                        "source": [entry.source for entry in audit_entries],
                        "model x,y": [
                            f"{entry.original[0]:.1f}, {entry.original[1]:.1f}"
                            for entry in audit_entries
                        ],
                        "final x,y": [
                            f"{entry.final[0]:.1f}, {entry.final[1]:.1f}"
                            for entry in audit_entries
                        ],
                        "move (px)": [
                            round(entry.shift_px, 1) for entry in audit_entries
                        ],
                    },
                    hide_index=True,
                    use_container_width=True,
                )
        if isinstance(proposal, AutoCalibrationProposal):
            boundary_anchor = getattr(proposal, "boundary_anchor", "doubles")
            st.caption(
                "Calibration anchors: "
                f"{boundary_anchor}. Output corners are always "
                "doubles-court corners."
            )
        if calibration_input == "Assisted" and proposal.confidence < MIN_CONFIDENCE:
            st.warning(
                "Low-confidence auto proposal. Review and correct every corner before "
                "running the pipeline."
            )
        if (
            calibration_input == "Assisted"
            and isinstance(proposal, KeypointCalibrationProposal)
            and proposal.refined_count < proposal.landmark_count
        ):
            unresolved = proposal.landmark_count - proposal.refined_count
            st.warning(
                f"{unresolved} model landmark(s) could not be confirmed by "
                "painted-line intersections. Review them before processing."
            )
    if (
        calibration_input == "Auto"
        and isinstance(proposal, KeypointCalibrationProposal)
        and not proposal.auto_ready
    ):
        st.error(
            "The keypoint model returned a court, but line refinement did not "
            "confirm all four boundary corners and at least two interior "
            "landmarks. Choose Assisted or another calibration frame."
        )
        return
    if calibration_input == "Assisted" and len(points) == len(landmark_names):
        st.write("Select a landmark, then click its corrected position.")
    elif calibration_input == "Assisted":
        st.warning("Auto proposal unavailable. Add the four boundary corners manually.")
        st.write(
            f"Click {next_index + 1}/{len(landmark_names)}: "
            f"**{display_landmark_name(landmark_names[next_index])}**"
        )
    elif calibration_input == "Manual" and next_index < len(landmark_names):
        st.write(
            f"Click {next_index + 1}/{len(landmark_names)}: "
            f"**{display_landmark_name(landmark_names[next_index])}**"
        )
    else:
        st.success("All landmarks selected. Review them, then run the pipeline.")

    preview = (
        draw_auto_proposal(frame, proposal)
        if isinstance(proposal, AutoCalibrationProposal)
        else frame
    )
    clickable_frame = draw_selected_points(preview, landmark_names, points)
    edit_index = None
    if calibration_input == "Assisted" and len(points) == len(landmark_names):
        edit_name = st.selectbox(
            "Landmark to correct",
            landmark_names,
            format_func=display_landmark_name,
        )
        edit_index = landmark_names.index(edit_name)
    if calibration_input == "Auto":
        st.image(cv2.cvtColor(clickable_frame, cv2.COLOR_BGR2RGB))
        click = None
    else:
        click = streamlit_image_coordinates(
            cv2.cvtColor(clickable_frame, cv2.COLOR_BGR2RGB),
            key=f"court-frame-{frame_key}",
            use_column_width="always",
            cursor="crosshair",
        )
    if click and click["unix_time"] != st.session_state.last_click:
        point = clicked_point(click, frame)
        if calibration_input == "Assisted" and edit_index is not None:
            st.session_state.calibration_points[edit_index] = point
        elif next_index < len(landmark_names):
            st.session_state.calibration_points.append(point)
        st.session_state.last_click = click["unix_time"]
        st.rerun()

    if points:
        labels = [display_landmark_name(name) for name in landmark_names[: len(points)]]
        st.dataframe(
            {
                "landmark": labels,
                "x": [round(point[0], 1) for point in points],
                "y": [round(point[1], 1) for point in points],
            },
            hide_index=True,
            use_container_width=True,
        )
    undo_column, reset_column = st.columns(2)
    with undo_column:
        if st.button(
            "Undo last point",
            disabled=not points or calibration_input != "Manual",
        ):
            st.session_state.calibration_points.pop()
            st.rerun()
    with reset_column:
        reset_label = (
            "Restore auto proposal"
            if calibration_input == "Assisted"
            else "Reset points"
        )
        if st.button(reset_label, disabled=calibration_input == "Auto"):
            if calibration_input == "Assisted" and proposal is not None:
                st.session_state.calibration_points = list(proposal.landmarks.values())
            else:
                reset_calibration()
            st.rerun()

    if len(points) != len(landmark_names):
        return

    calibration = build_calibration(
        video_path,
        court_type,
        frame_index,
        frame,
        points,
        landmark_names,
        source=(
            "auto-keypoint-ui"
            if calibration_input == "Auto"
            and isinstance(proposal, KeypointCalibrationProposal)
            else "auto-ransac-ui"
            if calibration_input == "Auto"
            else "assisted-keypoint-ui"
            if isinstance(proposal, KeypointCalibrationProposal)
            else "assisted-ransac-ui"
            if calibration_input == "Assisted"
            else "manual-ui"
        ),
    )
    validation = validate_homography(estimate_template_homography(calibration))
    if validation.status == "bad":
        st.error("Calibration failed validation. Reset the points and try again.")
        return
    calibration_message = (
        f"Calibration {validation.status}: {validation.inlier_count}/"
        f"{validation.landmark_count} inliers, mean reprojection error "
        f"{validation.mean_reprojection_error_px:.2f}px."
    )
    if use_four_corner_fallback:
        st.warning(
            f"{calibration_message} Four-corner mapping is enabled; temporal "
            "calibration will be disabled for this run."
        )
    else:
        st.success(calibration_message)

    run_label = (
        "Run player + ball projection" if track_ball else "Run player projection"
    )
    if st.button(run_label, type="primary"):
        if track_ball and (not tracknet_dir or not tracknet_model_path):
            st.error(
                "Ball tracking requires both the TrackNet directory and weights path."
            )
            return
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / "outputs" / "ui_runs" / f"{video_path.stem}_{run_id}"
        calibration_path = output_dir / "calibration.json"
        save_calibration(calibration_path, calibration)
        with st.spinner("Running player tracking and court mapping..."):
            result = run_pipeline(
                video_path,
                output_dir,
                calibration_path,
                detector_args,
                run_seconds,
                tracknet_dir,
                tracknet_model_path,
                tracknet_device,
                clone_tracknet,
                track_calibration and not use_four_corner_fallback,
                confidence,
                max_missing_seconds,
                float(reassociation_distance_px),
                allow_calibration_warning=use_four_corner_fallback,
            )
        if result.returncode != 0:
            st.error("The pipeline did not finish.")
            st.code(result.stderr or result.stdout)
            return
        st.success(f"Finished. Results saved to {output_dir.relative_to(REPO_ROOT)}")
        counts = projection_counts(output_dir / "tracks_with_court_coords.csv")
        player_metric, ball_metric, shot_metric = st.columns(3)
        player_metric.metric("Player projections", counts["player"])
        ball_metric.metric("Ball projections", counts["ball"])
        if track_ball and (output_dir / "shots.csv").exists():
            shot_metric.metric(
                "Estimated shots (experimental)",
                shot_count(output_dir / "shots.csv"),
            )
        if track_ball and counts["ball"] == 0:
            st.warning(
                "TrackNet found no visible ball in this time range. Increase the run "
                "duration or choose a segment where the ball is visible."
            )
        try:
            tracked_video = transcode_for_browser(output_dir / "annotated.mp4")
            court_map_video = transcode_for_browser(output_dir / "court_map.mp4")
        except RuntimeError as error:
            st.error(f"Could not prepare browser videos: {error}")
            return
        render_synchronized_videos(tracked_video, court_map_video)
        show_match_report(output_dir)
        st.download_button(
            "Download court coordinates CSV",
            data=(output_dir / "tracks_with_court_coords.csv").read_bytes(),
            file_name="tracks_with_court_coords.csv",
            mime="text/csv",
        )
        if track_ball and (output_dir / "shots.csv").exists():
            st.download_button(
                "Download estimated shots CSV",
                data=(output_dir / "shots.csv").read_bytes(),
                file_name="shots.csv",
                mime="text/csv",
            )


def main() -> None:
    st.set_page_config(page_title="CourtVision", layout="wide")
    initialize_state()
    reset = {"quick_points": [], "quick_last_click": None, "quick_result": None}
    for key, value in reset.items():
        st.session_state.setdefault(key, value)
    st.navigation(
        [
            st.Page(analyze_page, title="Analyze a clip", default=True),
            st.Page(advanced_page, title="Advanced"),
        ]
    ).run()


if __name__ == "__main__":
    main()
