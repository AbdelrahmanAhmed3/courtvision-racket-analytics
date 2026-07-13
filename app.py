"""Local Streamlit interface for CourtVision calibration and mapping."""

# ruff: noqa: E402, I001

from __future__ import annotations

import base64
import hashlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import streamlit as st
import streamlit.components.v1 as components
from imageio_ffmpeg import get_ffmpeg_exe
from streamlit_image_coordinates import streamlit_image_coordinates

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from courtvision.calibration.homography import estimate_template_homography  # noqa: E402
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
from courtvision.calibration.validation import validate_homography  # noqa: E402

DEFAULT_MODEL_ID = "tennis-v4d0h/2"


def reset_calibration() -> None:
    st.session_state.calibration_points = []
    st.session_state.last_click = None


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


def build_calibration(
    video_path: Path,
    court_type: str,
    frame_index: int,
    frame,
    points: list[tuple[float, float]],
) -> CalibrationRecord:
    names = default_landmark_names(court_type)
    definitions = [landmark_by_name(court_type, name) for name in names]
    observations = {
        definition.name: LandmarkObservation(
            image=point,
            template=definition.template,
            visible=True,
            confidence=1.0,
            source="manual-ui",
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
    model_id: str,
    seconds: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/run_full_pipeline.py",
            "--input",
            str(video_path),
            "--output-dir",
            str(output_dir),
            "--model-id",
            model_id,
            "--max-seconds",
            str(seconds),
            "--calibration",
            str(calibration_path),
            "--draw-court-map",
            "--draw-calibration-overlay",
        ],
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


def initialize_state() -> None:
    st.session_state.setdefault("calibration_points", [])
    st.session_state.setdefault("last_click", None)
    st.session_state.setdefault("frame_key", None)


def main() -> None:
    st.set_page_config(page_title="CourtVision", layout="wide")
    initialize_state()
    st.title("CourtVision")
    st.caption("Calibrate a court, track players, and map movement onto the court.")

    with st.sidebar:
        st.header("Run setup")
        uploaded_video = st.file_uploader(
            "Video", type=["mp4", "mov", "avi", "mkv"]
        )
        court_type = st.selectbox("Court type", ["tennis", "padel"])
        model_id = st.text_input("Roboflow model", value=DEFAULT_MODEL_ID)

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
    with warning_column:
        st.warning(
            "V1 assumes a stable camera. Recalibrate after a cut or significant "
            "camera move; per-frame automatic recalibration is the next phase."
        )
        st.caption(f"Video: {width} x {height}, {fps:.2f} fps, {duration:.1f}s")

    frame_index = min(int(round(calibration_seconds * fps)), frame_count - 1)
    frame_key = (str(video_path), court_type, frame_index)
    if st.session_state.frame_key != frame_key:
        reset_calibration()
        st.session_state.frame_key = frame_key
    frame = read_frame(video_path, frame_index)
    landmark_names = default_landmark_names(court_type)
    next_index = len(st.session_state.calibration_points)

    st.subheader("Court calibration")
    if next_index < len(landmark_names):
        st.write(
            f"Click {next_index + 1}/{len(landmark_names)}: "
            f"**{display_landmark_name(landmark_names[next_index])}**"
        )
    else:
        st.success("All landmarks selected. Review them, then run the pipeline.")

    clickable_frame = draw_selected_points(
        frame,
        landmark_names,
        st.session_state.calibration_points,
    )
    click = streamlit_image_coordinates(
        cv2.cvtColor(clickable_frame, cv2.COLOR_BGR2RGB),
        key=f"court-frame-{frame_key}",
    )
    if click and next_index < len(landmark_names):
        point = (float(click["x"]), float(click["y"]))
        if point != st.session_state.last_click:
            st.session_state.calibration_points.append(point)
            st.session_state.last_click = point
            st.rerun()

    points = st.session_state.calibration_points
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
        if st.button("Undo last point", disabled=not points):
            st.session_state.calibration_points.pop()
            st.session_state.last_click = None
            st.rerun()
    with reset_column:
        if st.button("Reset points"):
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
    )
    validation = validate_homography(estimate_template_homography(calibration))
    if validation.status == "bad":
        st.error("Calibration failed validation. Reset the points and try again.")
        return
    st.success(
        f"Calibration {validation.status}: {validation.inlier_count}/"
        f"{validation.landmark_count} inliers, mean reprojection error "
        f"{validation.mean_reprojection_error_px:.2f}px."
    )

    if st.button("Run tracking and court mapping", type="primary"):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / "outputs" / "ui_runs" / f"{video_path.stem}_{run_id}"
        calibration_path = output_dir / "calibration.json"
        save_calibration(calibration_path, calibration)
        with st.spinner("Running player tracking and court mapping..."):
            result = run_pipeline(
                video_path,
                output_dir,
                calibration_path,
                model_id,
                run_seconds,
            )
        if result.returncode != 0:
            st.error("The pipeline did not finish.")
            st.code(result.stderr or result.stdout)
            return
        st.success(f"Finished. Results saved to {output_dir.relative_to(REPO_ROOT)}")
        try:
            tracked_video = transcode_for_browser(output_dir / "annotated.mp4")
            court_map_video = transcode_for_browser(output_dir / "court_map.mp4")
        except RuntimeError as error:
            st.error(f"Could not prepare browser videos: {error}")
            return
        render_synchronized_videos(tracked_video, court_map_video)
        st.download_button(
            "Download court coordinates CSV",
            data=(output_dir / "tracks_with_court_coords.csv").read_bytes(),
            file_name="tracks_with_court_coords.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
