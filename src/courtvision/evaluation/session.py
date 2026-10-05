"""Labelling logic behind scripts/label_clip.py, kept free of any UI code.

The window passes key presses, clicks and drags to ``LabelSession``; everything
that changes the labels happens here, so it can be tested without a display.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from courtvision.evaluation.checks import describe, find_all_problems, find_problems
from courtvision.evaluation.labels import (
    BallEvent,
    PlayerBox,
    Segment,
    VideoLabels,
)

MAX_UNDO = 50
MIN_BOX_SIZE_PX = 6
# A copied player takes over a proposed box overlapping its old box this much.
COPY_MIN_IOU = 0.3

KEY_HELP = (
    ("a / d", "previous / next frame"),
    ("s / f", "back / forward 10 frames"),
    (", / .", "previous / next segment"),
    ("g", "next frame that needs boxes"),
    ("space", "play / pause"),
    ("l / o", "label / skip this segment"),
    ("c / m", "cut here / merge with previous segment"),
    ("r / e", "rally starts / ends here"),
    ("1-4", "impact by player (or assign box)"),
    ("b / w", "bounce / wall rebound, then click ball"),
    ("click box", "select; drag: draw or resize selected"),
    ("x", "delete selected box or this frame's events"),
    ("p", "copy players from previous box frame"),
    ("y", "boxes done; unassigned ones removed"),
    ("n", "next problem (labels breaking a rule)"),
    ("u / esc", "undo / cancel"),
    ("q", "save and quit"),
)


@dataclass(frozen=True)
class ChecklistItem:
    text: str
    done: bool | None  # None: information only


class LabelSession:
    def __init__(self, labels: VideoLabels, frame: int = 0) -> None:
        self.labels = labels
        self.frame = 0
        self.pending: str | None = None  # "bounce" or "wall_rebound" awaiting a click
        self.selected: int | None = None  # index into this frame's boxes
        self.message = ""
        self.dirty = False
        self._undo: list[VideoLabels] = []
        self._armed: tuple[str, int] | None = None  # a mark waiting to be moved
        self.go_to(frame)

    # Navigation -----------------------------------------------------------
    @property
    def segment(self) -> Segment:
        return self.labels.segment_at(self.frame)

    def go_to(self, frame: int) -> None:
        self.frame = max(0, min(frame, self.labels.frame_count - 1))
        self.selected = None
        self.pending = None

    def next_box_frame(self) -> int | None:
        """The next labelled-segment frame whose boxes are not done, wrapping."""
        due = [
            frame
            for segment in self.labels.segments
            if segment.status == "label"
            for frame in self.labels.box_frames(segment)
            if frame not in segment.reviewed_box_frames
        ]
        later = [frame for frame in due if frame > self.frame]
        return (later or due or [None])[0]

    def next_problem_frame(self) -> int | None:
        """The next frame with a problem (see checks.py), wrapping."""
        frames = sorted({problem.frame for problem in find_all_problems(self.labels)})
        later = [frame for frame in frames if frame > self.frame]
        return (later or frames or [None])[0]

    # Boxes ----------------------------------------------------------------
    def is_box_frame(self) -> bool:
        segment = self.segment
        return segment.status == "label" and self.frame in self.labels.box_frames(
            segment
        )

    def boxes(self) -> list[PlayerBox]:
        return self.segment.boxes.get(self.frame, [])

    def needs_prefill(self) -> bool:
        return self.is_box_frame() and self.frame not in self.segment.boxes

    def prefill(self, boxes: list[tuple[float, float, float, float]]) -> None:
        """Store model-proposed boxes for this frame, without identities."""
        self.segment.boxes[self.frame] = [PlayerBox(*box) for box in boxes]
        self.dirty = True

    # Input ----------------------------------------------------------------
    def key(self, key: str) -> None:
        self.message = ""
        if key not in {"r", "e"}:
            self._armed = None
        actions = {
            "a": lambda: self.go_to(self.frame - 1),
            "d": lambda: self.go_to(self.frame + 1),
            "s": lambda: self.go_to(self.frame - 10),
            "f": lambda: self.go_to(self.frame + 10),
            ",": self._previous_segment,
            ".": self._next_segment,
            "g": self._go_to_next_box_frame,
            "l": lambda: self._set_status("label"),
            "o": lambda: self._set_status("skip"),
            "c": self._cut,
            "m": self._merge_with_previous,
            "r": self._rally_start,
            "e": self._rally_end,
            "b": lambda: self._await_click("bounce"),
            "w": lambda: self._await_click("wall_rebound"),
            "x": self._delete,
            "p": self._copy_previous_players,
            "y": self._boxes_done,
            "n": self._go_to_next_problem,
            "u": self._undo_last,
            "esc": self._cancel,
        }
        if key in {"1", "2", "3", "4"}:
            self._number(int(key))
        elif key in actions:
            actions[key]()

    def click(self, x: float, y: float) -> None:
        self.message = ""
        if self.pending is not None:
            self._edit()
            self.segment.events.append(BallEvent(self.pending, self.frame, x=x, y=y))
            self.message = (
                f"{self.pending.replace('_', ' ')} at frame {self.frame}"
                + self._outside_rally()
            )
            self.pending = None
            return
        inside = [index for index, box in enumerate(self.boxes()) if box.contains(x, y)]
        # Overlapping boxes: pick the smallest, usually the player in front.
        self.selected = (
            min(inside, key=lambda index: _area(self.boxes()[index]))
            if inside
            else None
        )

    def drag(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.message = ""
        if not self.is_box_frame():
            self.message = "Boxes are only drawn on box frames (press g)"
            return
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        if right - left < MIN_BOX_SIZE_PX or bottom - top < MIN_BOX_SIZE_PX:
            return
        self._edit()
        if self.selected is not None:
            box = self.boxes()[self.selected]
            box.x1, box.y1, box.x2, box.y2 = left, top, right, bottom
            self.message = "Box resized"
            return
        self.segment.boxes.setdefault(self.frame, []).append(
            PlayerBox(left, top, right, bottom)
        )
        self.selected = len(self.boxes()) - 1
        self.message = "New box: press 1-4 to assign the player"
        self._reopen_if_unassigned()

    # Side panel -----------------------------------------------------------
    def checklist(self) -> list[ChecklistItem]:
        segment = self.segment
        index = self.labels.segments.index(segment) + 1
        items = [
            ChecklistItem(
                f"Segment {index}/{len(self.labels.segments)}: {segment.status}",
                segment.status != "unreviewed",
            )
        ]
        if segment.status != "label":
            return items
        events_here = [event for event in segment.events if event.frame == self.frame]
        impacts = [event for event in segment.events if event.kind == "impact"]
        items += [
            ChecklistItem("Rally start marked (r)", segment.rally_start is not None),
            ChecklistItem("Rally end marked (e)", segment.rally_end is not None),
            ChecklistItem(
                f"Impacts {len(impacts)}, bounces "
                f"{sum(event.kind == 'bounce' for event in segment.events)}, "
                f"wall rebounds "
                f"{sum(event.kind == 'wall_rebound' for event in segment.events)}",
                None,
            ),
            ChecklistItem(
                "This frame: "
                + (
                    ", ".join(describe(event) for event in events_here)
                    if events_here
                    else "no events"
                ),
                None,
            ),
        ]
        due = [
            frame
            for frame in self.labels.box_frames(segment)
            if frame not in segment.reviewed_box_frames
        ]
        items.append(
            ChecklistItem(
                f"Box frames done {len(self.labels.box_frames(segment)) - len(due)}"
                f"/{len(self.labels.box_frames(segment))}",
                not due,
            )
        )
        if self.is_box_frame():
            boxes = self.boxes()
            assigned = sum(box.player is not None for box in boxes)
            items.append(
                ChecklistItem(
                    f"Box frame: {assigned}/{len(boxes)} boxes assigned, "
                    + (
                        "done"
                        if self.frame in segment.reviewed_box_frames
                        else "press y when done"
                    ),
                    self.frame in segment.reviewed_box_frames,
                )
            )
        problems = find_problems(self.labels, segment)
        items.append(
            ChecklistItem(
                f"Problems in this segment: {len(problems)} (n)", not problems
            )
        )
        items += [
            ChecklistItem(f"Here: {problem.text}", False)
            for problem in problems
            if problem.frame == self.frame
        ]
        return items

    # Actions --------------------------------------------------------------
    def _edit(self) -> None:
        self._undo.append(copy.deepcopy(self.labels))
        del self._undo[:-MAX_UNDO]
        self.dirty = True

    def _undo_last(self) -> None:
        if not self._undo:
            self.message = "Nothing to undo"
            return
        self.labels = self._undo.pop()
        self.selected = None
        self.dirty = True
        self.message = "Undone"

    def _cancel(self) -> None:
        self.pending = None
        self.selected = None

    def _previous_segment(self) -> None:
        index = self.labels.segments.index(self.segment)
        self.go_to(self.labels.segments[max(0, index - 1)].start)

    def _next_segment(self) -> None:
        index = self.labels.segments.index(self.segment)
        if index + 1 < len(self.labels.segments):
            self.go_to(self.labels.segments[index + 1].start)

    def _go_to_next_problem(self) -> None:
        frame = self.next_problem_frame()
        if frame is None:
            self.message = "No problems found"
        else:
            self.go_to(frame)

    def _go_to_next_box_frame(self) -> None:
        frame = self.next_box_frame()
        if frame is None:
            self.message = "No box frames left in labelled segments"
        else:
            self.go_to(frame)

    def _set_status(self, status: str) -> None:
        self._edit()
        self.segment.status = status

    def _cut(self) -> None:
        segment = self.segment
        if self.frame == segment.start:
            self.message = "A segment already starts here"
            return
        self._edit()
        tail = segment.split_at(self.frame)
        self.labels.segments.insert(self.labels.segments.index(segment) + 1, tail)
        self.message = f"New segment starts at frame {self.frame}"

    def _merge_with_previous(self) -> None:
        index = self.labels.segments.index(self.segment)
        if index == 0:
            self.message = "This is the first segment"
            return
        self._edit()
        current = self.labels.segments.pop(index)
        self.labels.segments[index - 1].absorb(current)
        self.message = "Merged with the previous segment"

    def _rally_start(self) -> None:
        segment = self.segment
        if not self._labelling():
            return
        if segment.rally_end is not None and self.frame > segment.rally_end:
            self.message = "The rally cannot start after it ends"
            return
        if not self._may_move("start", segment.rally_start):
            return
        self._edit()
        segment.rally_start = self.frame

    def _rally_end(self) -> None:
        segment = self.segment
        if not self._labelling():
            return
        if segment.rally_start is not None and self.frame < segment.rally_start:
            self.message = "The rally cannot end before it starts"
            return
        if not self._may_move("end", segment.rally_end):
            return
        self._edit()
        segment.rally_end = self.frame

    def _may_move(self, mark: str, current: int | None) -> bool:
        """A rally mark that is set moves only on a second press, so r and e,
        neighbours on the keyboard, cannot move one by a slip."""
        if current in (None, self.frame) or self._armed == (mark, self.frame):
            self._armed = None
            return True
        self._armed = (mark, self.frame)
        key = "r" if mark == "start" else "e"
        self.message = (
            f"The rally {mark}s at frame {current}: press {key} again to move"
        )
        return False

    def _labelling(self) -> bool:
        if self.segment.status != "label":
            self.message = "Mark the segment for labelling first (l)"
            return False
        return True

    def _outside_rally(self) -> str:
        segment = self.segment
        before = segment.rally_start is not None and self.frame < segment.rally_start
        after = segment.rally_end is not None and self.frame > segment.rally_end
        return " (outside the rally!)" if before or after else ""

    def _await_click(self, kind: str) -> None:
        if not self._labelling():
            return
        self.pending = kind
        self.message = f"Click the ball for the {kind.replace('_', ' ')}"

    def _number(self, player: int) -> None:
        if self.selected is not None:
            self._edit()
            for box in self.boxes():
                if box.player == player:
                    box.player = None
            self.boxes()[self.selected].player = player
            self.message = f"Box assigned to player {player}"
            self.selected = None
            self._reopen_if_unassigned()
            return
        if not self._labelling():
            return
        self._edit()
        segment = self.segment
        segment.events = [
            event
            for event in segment.events
            if not (event.kind == "impact" and event.frame == self.frame)
        ]
        segment.events.append(BallEvent("impact", self.frame, player=player))
        self.message = (
            f"Impact by player {player} at frame {self.frame}" + self._outside_rally()
        )

    def _delete(self) -> None:
        if self.selected is not None:
            self._edit()
            del self.boxes()[self.selected]
            self.selected = None
            self.message = "Box deleted"
            return
        segment = self.segment
        if not any(event.frame == self.frame for event in segment.events):
            return
        self._edit()
        segment.events = [
            event for event in segment.events if event.frame != self.frame
        ]
        self.message = "Events on this frame deleted"

    def _copy_previous_players(self) -> None:
        """Give this frame's boxes the players from the last finished box frame.

        Each previous player takes the proposed box that overlaps its old box
        most; a player with no overlapping proposal gets its old box copied.
        """
        if not self.is_box_frame():
            self.message = "This is not a box frame"
            return
        segment = self.segment
        earlier = [frame for frame in segment.reviewed_box_frames if frame < self.frame]
        if not earlier:
            self.message = "No earlier finished box frame in this segment"
            return
        source = max(earlier)
        self._edit()
        boxes = segment.boxes.setdefault(self.frame, [])
        for box in boxes:
            box.player = None
        matched = copied = 0
        for old in (box for box in segment.boxes[source] if box.player is not None):
            free = [box for box in boxes if box.player is None]
            best = max(free, key=lambda box: _iou(box, old), default=None)
            if best is not None and _iou(best, old) >= COPY_MIN_IOU:
                best.player = old.player
                matched += 1
            else:
                boxes.append(PlayerBox(old.x1, old.y1, old.x2, old.y2, old.player))
                copied += 1
        self.selected = None
        self.message = (
            f"Players from frame {source}: {matched} matched, {copied} copied as-is."
            " Check, then y"
        )
        self._reopen_if_unassigned()

    def _reopen_if_unassigned(self) -> None:
        """A finished box frame has a player on every box; reopen it if not."""
        reviewed = self.segment.reviewed_box_frames
        if self.frame in reviewed and any(box.player is None for box in self.boxes()):
            reviewed.remove(self.frame)
            self.message += " (box frame reopened: press y when done)"

    def _boxes_done(self) -> None:
        """Finish this box frame; boxes left without a player are removed."""
        if not self.is_box_frame():
            self.message = "This is not a box frame"
            return
        if self.frame in self.segment.reviewed_box_frames:
            self.message = "Boxes on this frame are already done"
            return
        self._edit()
        boxes = self.boxes()
        removed = sum(box.player is None for box in boxes)
        boxes[:] = [box for box in boxes if box.player is not None]
        self.segment.reviewed_box_frames.append(self.frame)
        self.selected = None
        self.message = f"Boxes done ({len(boxes)} players, {removed} removed)"


def _area(box: PlayerBox) -> float:
    return (box.x2 - box.x1) * (box.y2 - box.y1)


def _iou(first: PlayerBox, second: PlayerBox) -> float:
    width = min(first.x2, second.x2) - max(first.x1, second.x1)
    height = min(first.y2, second.y2) - max(first.y1, second.y1)
    if width <= 0 or height <= 0:
        return 0.0
    overlap = width * height
    return overlap / (_area(first) + _area(second) - overlap)
