from __future__ import annotations

import os
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st


FPS = 15
PPE_CLASSES = ["helmet", "vest", "gloves", "goggles"]

STATUS_ICON = {
    "compliant": "✅",
    "violation": "❌",
    "uncertain": "⚠️",
    "offline": "⏸️",
}

SHORT_STATUS = {
    "compliant": "OK",
    "violation": "MISS",
    "uncertain": "?",
    "offline": "--",
}

ROLES = [
    "operator",
    "contractor",
    "supervisor",
    "forklift",
    "technician",
    "inspector",
]

_FACE_CASCADE: Any = None
_SYNTH_BG: Optional[np.ndarray] = None
_FRAGMENT_SUPPORT = {"auto": hasattr(st, "fragment")}


@dataclass
class TrackedWorker:
    track_id: int
    label: str
    role: str
    zone: str
    bbox: Tuple[int, int, int, int]
    kind: str = "real"
    prev_cx: float = 0.0
    prev_cy: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    lost_frames: int = 0
    present: bool = True

    current_true: Dict[str, int] = field(default_factory=dict)
    violation_remaining: Dict[str, int] = field(default_factory=dict)
    history: Dict[str, Deque[int]] = field(default_factory=dict)
    last_status: Dict[str, str] = field(default_factory=dict)
    last_confidence: Dict[str, float] = field(default_factory=dict)
    alert_cooldown: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        x, y, w, h = self.bbox
        self.prev_cx = float(x + w / 2.0)
        self.prev_cy = float(y + h / 2.0)

        for ppe in PPE_CLASSES:
            self.current_true[ppe] = 1
            self.violation_remaining[ppe] = 0
            self.history[ppe] = deque(maxlen=12)
            self.last_status[ppe] = "uncertain"
            self.last_confidence[ppe] = 0.0
            self.alert_cooldown[ppe] = -10.0


def remove_temp_file(path: Optional[str]) -> None:
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x, y, w, h = bbox
    return float(x + w / 2.0), float(y + h / 2.0)


def choose_zone(bbox: Tuple[int, int, int, int], frame_width: int) -> str:
    cx, _ = bbox_center(bbox)
    ratio = cx / float(max(1, frame_width))

    if ratio < 0.25:
        return "Gate A"
    if ratio < 0.50:
        return "Assembly"
    if ratio < 0.75:
        return "Welding"
    return "Loading dock"


def create_worker(
    track_id: int,
    bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, ...],
    kind: str = "real",
) -> TrackedWorker:
    role = ROLES[track_id % len(ROLES)]
    zone = choose_zone(bbox, frame_shape[1])

    return TrackedWorker(
        track_id=track_id,
        label=f"W-{track_id}",
        role=role,
        zone=zone,
        bbox=tuple(int(v) for v in bbox),
        kind=kind,
    )


def nms_boxes(boxes: List[Tuple[int, int, int, int]], thresh: float = 0.35) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []

    arr = np.array([[x, y, x + w, y + h] for x, y, w, h in boxes], dtype=float)
    x1 = arr[:, 0]
    y1 = arr[:, 1]
    x2 = arr[:, 2]
    y2 = arr[:, 3]

    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    order = np.argsort(areas)
    keep: List[int] = []

    while order.size > 0:
        i = int(order[-1])
        keep.append(i)

        if order.size == 1:
            break

        rest = order[:-1]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        w = np.maximum(0.0, xx2 - xx1 + 1.0)
        h = np.maximum(0.0, yy2 - yy1 + 1.0)
        inter = w * h
        iou = inter / areas[rest]

        order = rest[iou <= thresh]

    return [boxes[i] for i in keep]


def get_face_cascade() -> Optional[Any]:
    global _FACE_CASCADE

    if _FACE_CASCADE is None:
        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        classifier = cv2.CascadeClassifier(path)

        if classifier.empty():
            _FACE_CASCADE = False
        else:
            _FACE_CASCADE = classifier

    if _FACE_CASCADE is False:
        return None

    return _FACE_CASCADE


def resize_frame(frame: Optional[np.ndarray], max_width: int = 960) -> Optional[np.ndarray]:
    if frame is None:
        return None

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    h, w = frame.shape[:2]

    if w <= max_width:
        return frame

    scale = float(max_width) / float(w)
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_AREA)


def detect_faces(frame: np.ndarray, min_neighbors: int) -> List[Tuple[int, int, int, int]]:
    cascade = get_face_cascade()

    if cascade is None or frame is None:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=int(min_neighbors),
        minSize=(36, 36),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if faces is None or len(faces) == 0:
        return []

    boxes = [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
    return nms_boxes(boxes, thresh=0.35)


def iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter = float(inter_w * inter_h)

    area_a = float(aw * ah)
    area_b = float(bw * bh)
    union = area_a + area_b - inter

    if union <= 0.0:
        return 0.0

    return inter / union


def sync_tracks(
    workers: List[TrackedWorker],
    detections: List[Tuple[int, int, int, int]],
    frame_idx: int,
    next_track_id: int,
    frame_shape: Tuple[int, ...],
) -> Tuple[List[TrackedWorker], int]:
    active = [w for w in workers if w.kind == "real" and w.lost_frames < 30]

    used_workers = set()
    used_detections = set()
    scores: List[Tuple[float, int, int]] = []

    for wi, worker in enumerate(active):
        for di, det in enumerate(detections):
            score = iou(worker.bbox, det)
            if score > 0.20:
                scores.append((score, wi, di))

    scores.sort(key=lambda item: item[0], reverse=True)

    for _, wi, di in scores:
        if wi in used_workers or di in used_detections:
            continue

        used_workers.add(wi)
        used_detections.add(di)

        worker = active[wi]
        worker.bbox = tuple(int(v) for v in detections[di])
        worker.zone = choose_zone(worker.bbox, frame_shape[1])
        worker.lost_frames = 0
        worker.present = True

    for di, det in enumerate(detections):
        if di not in used_detections:
            new_worker = create_worker(next_track_id, det, frame_shape, kind="real")
            next_track_id += 1
            workers.append(new_worker)

    for wi, worker in enumerate(active):
        if wi not in used_workers:
            worker.lost_frames += 1
            worker.present = worker.lost_frames < 15

    workers = [w for w in workers if w.present or w.lost_frames < 90]
    return workers, next_track_id


def create_synthetic_workers(
    frame_shape: Tuple[int, ...],
    next_track_id: int,
    count: int = 5,
) -> Tuple[List[TrackedWorker], int]:
    h, w = frame_shape[:2]
    rng = np.random.default_rng(777 + next_track_id)

    workers: List[TrackedWorker] = []
    margin_x = 70
    usable_w = max(80, w - 2 * margin_x)

    for i in range(count):
        scale = 0.78 + 0.34 * (float(i) / float(max(1, count - 1)))
        bw = int(62 * scale)
        bh = int(135 * scale)

        x = int(margin_x + (i + 0.5) * usable_w / float(count) - bw / 2.0)
        y = int(h * 0.52 + (i % 2) * 45)

        x = max(10, min(w - bw - 10, x))
        y = max(int(h * 0.35), min(h - bh - 40, y))

        worker = create_worker(next_track_id + i, (x, y, bw, bh), frame_shape, kind="synthetic")
        worker.vx = float(rng.normal(0.0, 0.45))
        worker.vy = float(rng.normal(0.0, 0.18))

        workers.append(worker)

    return workers, next_track_id + count


def update_synthetic_tracks(
    workers: List[TrackedWorker],
    frame_idx: int,
    next_track_id: int,
    frame_shape: Tuple[int, ...],
    base_seed: int,
) -> Tuple[List[TrackedWorker], int]:
    present_synth = [w for w in workers if w.kind == "synthetic" and w.present]

    if not present_synth:
        new_workers, next_track_id = create_synthetic_workers(frame_shape, next_track_id)
        workers.extend(new_workers)
        present_synth = new_workers

    h, w = frame_shape[:2]
    rng = np.random.default_rng(base_seed + frame_idx)

    for worker in present_synth:
        worker.vx += float(rng.normal(0.0, 0.08))
        worker.vy += float(rng.normal(0.0, 0.04))

        speed = float(np.hypot(worker.vx, worker.vy))
        max_speed = 1.8

        if speed > max_speed and speed > 0.0:
            worker.vx *= max_speed / speed
            worker.vy *= max_speed / speed

        x, y, bw, bh = worker.bbox
        nx = float(x) + worker.vx
        ny = float(y) + worker.vy

        min_y = int(h * 0.35)
        max_y = h - bh - 40

        if nx < 20:
            nx = 20.0
            worker.vx = abs(worker.vx)
        if nx + bw > w - 20:
            nx = float(w - 20 - bw)
            worker.vx = -abs(worker.vx)

        if ny < min_y:
            ny = float(min_y)
            worker.vy = abs(worker.vy)
        if ny > max_y:
            ny = float(max_y)
            worker.vy = -abs(worker.vy)

        worker.bbox = (int(nx), int(ny), bw, bh)
        worker.lost_frames = 0
        worker.present = True

    workers = [w for w in workers if w.present or w.lost_frames < 90]
    return workers, next_track_id


def fade_synthetic_workers(workers: List[TrackedWorker]) -> List[TrackedWorker]:
    for worker in workers:
        if worker.kind == "synthetic":
            worker.lost_frames += 1
            worker.present = worker.lost_frames < 15

    return [w for w in workers if w.present or w.lost_frames < 90]


def simulate_ppe(worker: TrackedWorker, frame_idx: int, base_seed: int) -> None:
    if not worker.present:
        return

    rng = np.random.default_rng(base_seed + worker.track_id * 1000003 + frame_idx)

    cx, cy = bbox_center(worker.bbox)
    motion = float(np.hypot(cx - worker.prev_cx, cy - worker.prev_cy))

    worker.prev_cx = cx
    worker.prev_cy = cy

    for ppe in PPE_CLASSES:
        if worker.violation_remaining.get(ppe, 0) > 0:
            true_value = 0
            worker.violation_remaining[ppe] -= 1
        else:
            true_value = 1

            p_start = 0.00040

            if worker.role == "contractor":
                p_start *= 1.7
            if worker.role == "forklift" and ppe == "vest":
                p_start *= 1.3
            if worker.zone == "Welding" and ppe == "goggles":
                p_start *= 2.2
            if worker.zone == "Loading dock" and ppe == "helmet":
                p_start *= 1.4
            if motion > 2.0:
                p_start *= 1.5
            if motion > 5.0 and ppe in ("gloves", "goggles"):
                p_start *= 1.4

            if float(rng.random()) < p_start:
                worker.violation_remaining[ppe] = int(rng.integers(18, 50))
                true_value = 0

        worker.current_true[ppe] = true_value

        if true_value == 1:
            observed = 0 if float(rng.random()) < 0.010 else 1
        else:
            observed = 1 if float(rng.random()) < 0.050 else 0

        worker.history[ppe].append(int(observed))


def temporal_status(
    history: Deque[int],
    window: int = 5,
    min_observations: int = 3,
) -> Tuple[str, float]:
    if len(history) < min_observations:
        return "uncertain", 0.0

    values = list(history)[-window:]
    score = float(sum(values)) / float(len(values))

    if score >= 0.68:
        return "compliant", score
    if score <= 0.32:
        return "violation", 1.0 - score

    return "uncertain", score


def refresh_statuses(workers: List[TrackedWorker], frame_idx: int, record_alerts: bool = True) -> None:
    current_time = frame_idx / float(FPS)

    try:
        window = int(st.session_state.get("voting_window", 5) or 5)
    except Exception:
        window = 5

    window = max(1, min(12, window))

    for worker in workers:
        if not worker.present:
            for ppe in PPE_CLASSES:
                worker.last_status[ppe] = "offline"
                worker.last_confidence[ppe] = 0.0
            continue

        for ppe in PPE_CLASSES:
            hist = worker.history[ppe]
            status, confidence = temporal_status(hist, window=window)
            previous_status = worker.last_status.get(ppe, "uncertain")

            if record_alerts and status == "compliant" and len(hist) > 0 and hist[-1] == 0:
                st.session_state["suppressed_flickers"] = int(st.session_state.get("suppressed_flickers", 0)) + 1

            if (
                record_alerts
                and status == "violation"
                and previous_status != "violation"
                and current_time >= worker.alert_cooldown.get(ppe, -10.0)
            ):
                worker.alert_cooldown[ppe] = current_time + 8.0

                is_false_alarm = worker.current_true.get(ppe, 1) == 1
                severity = "high" if ppe in ("helmet", "goggles") else "medium"

                event = {
                    "Time (s)": round(current_time, 2),
                    "Worker": worker.label,
                    "Role": worker.role,
                    "Zone": worker.zone,
                    "Source": worker.kind,
                    "PPE": ppe,
                    "Confidence": round(confidence, 3),
                    "Severity": severity,
                    "Type": "false-alarm" if is_false_alarm else "violation",
                    "Message": f"{worker.label} missing {ppe} in {worker.zone}",
                }

                st.session_state["events"].appendleft(event)
                st.session_state["alert_count"] = int(st.session_state.get("alert_count", 0)) + 1

                if is_false_alarm:
                    st.session_state["false_alert_count"] = int(st.session_state.get("false_alert_count", 0)) + 1
                else:
                    st.session_state["true_alert_count"] = int(st.session_state.get("true_alert_count", 0)) + 1

            worker.last_status[ppe] = status
            worker.last_confidence[ppe] = confidence


def clip_box(box: Tuple[int, int, int, int], frame_shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x, y, bw, bh = box

    x = int(max(0, min(w - 1, x)))
    y = int(max(0, min(h - 1, y)))
    bw = int(max(1, min(w - x, bw)))
    bh = int(max(1, min(h - y, bh)))

    return x, y, bw, bh


def add_transparent_rect(
    overlay: np.ndarray,
    box: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    alpha_unused: float = 0.22,
) -> None:
    x, y, w, h = clip_box(box, overlay.shape)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)


def ppe_region_boxes(bbox: Tuple[int, int, int, int]) -> Dict[str, List[Tuple[int, int, int, int]]]:
    x, y, w, h = bbox

    return {
        "helmet": [
            (x - int(0.05 * w), y - int(0.45 * h), int(1.10 * w), int(0.45 * h))
        ],
        "goggles": [
            (x + int(0.08 * w), y + int(0.22 * h), int(0.84 * w), int(0.18 * h))
        ],
        "vest": [
            (x - int(0.35 * w), y + int(0.90 * h), int(1.70 * w), int(1.20 * h))
        ],
        "gloves": [
            (x - int(0.55 * w), y + int(1.15 * h), int(0.28 * w), int(0.35 * h)),
            (x + int(1.27 * w), y + int(1.15 * h), int(0.28 * w), int(0.35 * h)),
        ],
    }


def ppe_color(status: str) -> Tuple[int, int, int]:
    if status == "compliant":
        return (0, 190, 0)
    if status == "violation":
        return (0, 0, 225)
    if status == "offline":
        return (110, 110, 110)
    return (0, 210, 255)


def overall_color(statuses: List[str]) -> Tuple[int, int, int]:
    if any(status == "violation" for status in statuses):
        return (0, 0, 225)
    if any(status == "uncertain" for status in statuses):
        return (0, 210, 255)
    if all(status == "compliant" for status in statuses):
        return (0, 190, 0)
    return (120, 120, 120)


def make_synthetic_frame() -> np.ndarray:
    global _SYNTH_BG

    if _SYNTH_BG is not None:
        return _SYNTH_BG.copy()

    width = 960
    height = 540
    bg = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        t = y / float(height)
        bg[y, :] = (
            int(42 + 22 * t),
            int(48 + 26 * t),
            int(58 + 32 * t),
        )

    floor_y = 300

    for y in range(floor_y, height):
        t = (y - floor_y) / float(height - floor_y)
        top = np.array([72, 78, 86], dtype=float)
        bottom = np.array([38, 41, 46], dtype=float)
        bg[y, :] = (top * (1.0 - t) + bottom * t).astype(np.uint8)

    for x in range(0, width, 120):
        x2 = int(width / 2 + (x - width / 2) * 2.4)
        cv2.line(bg, (x, floor_y), (x2, height), (92, 98, 108), 1, cv2.LINE_AA)

    for y in range(floor_y, height, 36):
        cv2.line(bg, (0, y), (width, y), (88, 94, 104), 1, cv2.LINE_AA)

    cv2.rectangle(bg, (0, 0), (width, 38), (24, 27, 34), -1)

    zone_labels = [
        ("Gate A", 80),
        ("Assembly", 300),
        ("Welding", 520),
        ("Loading dock", 720),
    ]

    for label, x in zone_labels:
        cv2.putText(
            bg,
            label,
            (x, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (185, 190, 198),
            1,
            cv2.LINE_AA,
        )

    cv2.line(bg, (0, floor_y), (width, floor_y), (0, 210, 255), 2, cv2.LINE_AA)

    _SYNTH_BG = bg
    return bg.copy()


def draw_scene(
    frame: Optional[np.ndarray],
    workers: List[TrackedWorker],
    frame_idx: int,
    events: List[Dict[str, Any]],
    source_label: str,
    fps: float,
    show_overlay: bool = True,
) -> np.ndarray:
    if frame is None:
        frame = make_synthetic_frame()

    frame = frame.copy()
    h, w = frame.shape[:2]

    present_workers = [worker for worker in workers if worker.present]
    present_workers.sort(key=lambda item: item.bbox[1] + item.bbox[3])

    overlay = frame.copy()

    if show_overlay:
        for worker in present_workers:
            box = clip_box(worker.bbox, frame.shape)
            regions = ppe_region_boxes(box)

            for ppe, boxes in regions.items():
                color = ppe_color(worker.last_status.get(ppe, "uncertain"))
                for region_box in boxes:
                    add_transparent_rect(overlay, region_box, color)

        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for worker in present_workers:
        x, y, bw, bh = clip_box(worker.bbox, frame.shape)
        statuses = [worker.last_status.get(ppe, "uncertain") for ppe in PPE_CLASSES]
        color = overall_color(statuses)

        if show_overlay:
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2, cv2.LINE_AA)

        font_scale = max(0.34, min(0.55, w / 1800.0))

        label = f"{worker.label} {worker.role} {worker.zone}"
        status_text = " ".join(
            f"{ppe[:1].upper()}:{SHORT_STATUS.get(worker.last_status.get(ppe, 'uncertain'), '?')}"
            for ppe in PPE_CLASSES
        )

        text_y = y - 18 if y > 48 else y + bh + 14

        cv2.putText(
            frame,
            label,
            (x, text_y),
            font,
            font_scale,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            status_text,
            (x, text_y + 14),
            font,
            font_scale * 0.9,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.rectangle(frame, (0, 0), (w, 34), (18, 20, 26), -1)

    t = frame_idx / float(FPS)
    mm = int(t // 60)
    ss = int(t % 60)

    active = len(present_workers)
    violations = sum(
        1
        for worker in present_workers
        if any(worker.last_status.get(ppe) == "violation" for ppe in PPE_CLASSES)
    )

    left_hud = f"PPE REAL DEMO {mm:02d}:{ss:02d} {fps:4.1f} FPS"
    right_hud = f"{source_label} | Workers {active} | Violations {violations}"

    cv2.putText(
        frame,
        left_hud,
        (8, 22),
        font,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    (tw, _), _ = cv2.getTextSize(right_hud, font, 0.50, 1)
    cv2.putText(
        frame,
        right_hud,
        (max(8, w - tw - 8), 22),
        font,
        0.50,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    cv2.rectangle(frame, (0, h - 30), (w, h), (16, 18, 24), -1)

    if events:
        event = events[0]
        msg = (
            f"{float(event.get('Time (s)', 0.0)):.1f}s | "
            f"{str(event.get('Message', ''))} | "
            f"conf {float(event.get('Confidence', 0.0)):.2f}"
        )
    else:
        msg = "Real-image demo: show a face for tracking. If no face is detected, simulated workers are overlaid after a short wait."

    cv2.putText(
        frame,
        msg[:125],
        (8, h - 10),
        font,
        0.45,
        (0, 210, 255),
        1,
        cv2.LINE_AA,
    )

    return frame


def record_annotated_frame(frame: np.ndarray) -> None:
    recorder = st.session_state.get("recorder")

    if recorder is None or recorder.get("status") != "recording":
        return

    if frame is None:
        return

    if recorder.get("writer") is None:
        h, w = frame.shape[:2]

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        path = tmp.name
        tmp.close()

        writer = None

        for codec in ("avc1", "H264", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            candidate = cv2.VideoWriter(path, fourcc, FPS, (w, h))

            if candidate.isOpened():
                writer = candidate
                break

            candidate.release()

        if writer is None:
            try:
                os.unlink(path)
            except OSError:
                pass

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
            path = tmp.name
            tmp.close()

            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            candidate = cv2.VideoWriter(path, fourcc, FPS, (w, h))

            if candidate.isOpened():
                writer = candidate
            else:
                candidate.release()
                try:
                    os.unlink(path)
                except OSError:
                    pass
                recorder["status"] = "failed"
                return

        recorder["writer"] = writer
        recorder["path"] = path

    writer = recorder.get("writer")

    if writer is None:
        return

    writer.write(frame)
    recorder["frames_written"] = int(recorder.get("frames_written", 0)) + 1

    if recorder["frames_written"] >= int(recorder.get("target_frames", 75)):
        writer.release()
        recorder["writer"] = None
        recorder["status"] = "done"
        st.session_state["demo_video_path"] = recorder.get("path")


def _camera_backends() -> List[Optional[int]]:
    backends: List[Optional[int]] = [None]

    for name in ("CAP_DSHOW", "CAP_MSMF", "CAP_V4L2", "CAP_GSTREAMER"):
        value = getattr(cv2, name, None)
        if isinstance(value, int):
            backends.append(value)

    return backends


def try_open_camera() -> Tuple[Optional[cv2.VideoCapture], int, str]:
    if not bool(st.session_state.get("use_webcam", True)):
        return None, -1, "Local webcam disabled."

    for idx in (0, 1, 2):
        for backend in _camera_backends():
            try:
                if backend is None:
                    cap = cv2.VideoCapture(idx)
                else:
                    cap = cv2.VideoCapture(idx, backend)
            except Exception:
                continue

            if not cap.isOpened():
                cap.release()
                continue

            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
                cap.set(cv2.CAP_PROP_FPS, FPS)
            except Exception:
                pass

            ok, test = cap.read()

            if ok and test is not None and test.size > 0:
                return cap, idx, f"Local webcam {idx} active."

            cap.release()

    return None, -1, "No local webcam found. Upload a real video or use synthetic fallback."


def close_source() -> None:
    cap = st.session_state.get("cap")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    st.session_state["cap"] = None

    source_cap = st.session_state.get("source_cap")
    if source_cap is not None:
        try:
            source_cap.release()
        except Exception:
            pass
    st.session_state["source_cap"] = None

    recorder = st.session_state.get("recorder")
    if recorder is not None:
        writer = recorder.get("writer")
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
            recorder["writer"] = None


def read_latest_frame(cap: cv2.VideoCapture, max_skip: int = 3) -> Optional[np.ndarray]:
    latest = None

    for _ in range(max(1, max_skip)):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        latest = frame

    return latest


def save_uploaded_file(uploaded_file: Any) -> str:
    suffix = os.path.splitext(str(getattr(uploaded_file, "name", "video.mp4")))[1]
    if not suffix:
        suffix = ".mp4"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    return tmp.name


def ensure_widget_defaults() -> None:
    defaults = {
        "playing": True,
        "show_overlay": True,
        "use_webcam": True,
        "voting_window": 5,
        "min_neighbors": 5,
        "step_once": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sanitize_widget_state() -> None:
    ensure_widget_defaults()

    try:
        st.session_state["voting_window"] = int(np.clip(int(st.session_state.get("voting_window", 5)), 3, 10))
    except Exception:
        st.session_state["voting_window"] = 5

    try:
        st.session_state["min_neighbors"] = int(np.clip(int(st.session_state.get("min_neighbors", 5)), 3, 10))
    except Exception:
        st.session_state["min_neighbors"] = 5

    for key in ("playing", "show_overlay", "use_webcam"):
        st.session_state[key] = bool(st.session_state.get(key, True))

    st.session_state["step_once"] = bool(st.session_state.get("step_once", False))


def initialize_state() -> None:
    ensure_widget_defaults()
    sanitize_widget_state()

    st.session_state["initialized"] = True
    st.session_state["base_seed"] = 20261028
    st.session_state["frame_count"] = 0
    st.session_state["next_track_id"] = 1001
    st.session_state["no_face_frames"] = 0
    st.session_state["camera_fail_count"] = 0

    st.session_state["workers"] = []
    st.session_state["events"] = deque(maxlen=250)

    st.session_state["alert_count"] = 0
    st.session_state["true_alert_count"] = 0
    st.session_state["false_alert_count"] = 0
    st.session_state["suppressed_flickers"] = 0

    st.session_state["last_frame"] = None
    st.session_state["last_process_time"] = None
    st.session_state["fps_ema"] = 0.0

    st.session_state["cap"] = None
    st.session_state["source_cap"] = None
    st.session_state["uploaded_video_path"] = None
    st.session_state["uploaded_video_name"] = None

    st.session_state["demo_video_path"] = None
    st.session_state["recorder"] = {
        "status": "recording",
        "frames_written": 0,
        "target_frames": 75,
        "path": None,
        "writer": None,
    }

    cap, idx, msg = try_open_camera()
    st.session_state["cap"] = cap
    st.session_state["camera_message"] = msg
    st.session_state["source_type"] = "webcam" if cap is not None else "synthetic"


def get_source_frame() -> np.ndarray:
    source_type = st.session_state.get("source_type", "synthetic")

    if source_type == "webcam":
        cap = st.session_state.get("cap")

        if cap is None or not cap.isOpened():
            cap, idx, msg = try_open_camera()
            st.session_state["cap"] = cap
            st.session_state["camera_message"] = msg

            if cap is None:
                st.session_state["source_type"] = "synthetic"
                return make_synthetic_frame()

        frame = read_latest_frame(cap, max_skip=3)

        if frame is None:
            st.session_state["camera_fail_count"] = int(st.session_state.get("camera_fail_count", 0)) + 1

            if st.session_state["camera_fail_count"] > 10:
                try:
                    cap.release()
                except Exception:
                    pass

                st.session_state["cap"] = None
                st.session_state["source_type"] = "synthetic"
                st.session_state["camera_message"] = "Webcam read failed. Using fallback."

            return make_synthetic_frame()

        st.session_state["camera_fail_count"] = 0
        return frame

    if source_type == "uploaded_video":
        cap = st.session_state.get("source_cap")
        path = st.session_state.get("uploaded_video_path")

        if (cap is None or not cap.isOpened()) and path:
            cap = cv2.VideoCapture(path)
            st.session_state["source_cap"] = cap

        if cap is None or not cap.isOpened():
            st.session_state["source_type"] = "synthetic"
            return make_synthetic_frame()

        ok, frame = cap.read()

        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()

        if not ok or frame is None:
            st.session_state["source_type"] = "synthetic"
            return make_synthetic_frame()

        return frame

    return make_synthetic_frame()


def process_frame(raw_frame: Optional[np.ndarray]) -> np.ndarray:
    frame = resize_frame(raw_frame, max_width=960)

    if frame is None:
        frame = make_synthetic_frame()

    frame_idx = int(st.session_state.get("frame_count", 0))
    source_type = st.session_state.get("source_type", "synthetic")
    base_seed = int(st.session_state.get("base_seed", 1))

    workers = list(st.session_state.get("workers", []))
    next_track_id = int(st.session_state.get("next_track_id", 1001))

    present_real = False

    if source_type in ("webcam", "uploaded_video"):
        min_neighbors = int(st.session_state.get("min_neighbors", 5))
        detections = detect_faces(frame, min_neighbors)

        workers, next_track_id = sync_tracks(
            workers=workers,
            detections=detections,
            frame_idx=frame_idx,
            next_track_id=next_track_id,
            frame_shape=frame.shape,
        )

        present_real = any(worker.kind == "real" and worker.present for worker in workers)

        if present_real:
            st.session_state["no_face_frames"] = 0
            workers = fade_synthetic_workers(workers)
        else:
            st.session_state["no_face_frames"] = int(st.session_state.get("no_face_frames", 0)) + 1

            if st.session_state["no_face_frames"] > 45:
                workers, next_track_id = update_synthetic_tracks(
                    workers=workers,
                    frame_idx=frame_idx,
                    next_track_id=next_track_id,
                    frame_shape=frame.shape,
                    base_seed=base_seed,
                )
    else:
        workers, next_track_id = update_synthetic_tracks(
            workers=workers,
            frame_idx=frame_idx,
            next_track_id=next_track_id,
            frame_shape=frame.shape,
            base_seed=base_seed,
        )

    st.session_state["workers"] = workers
    st.session_state["next_track_id"] = next_track_id

    for worker in workers:
        simulate_ppe(worker, frame_idx, base_seed)

    refresh_statuses(workers, frame_idx, record_alerts=True)

    now = time.time()
    last_time = st.session_state.get("last_process_time")

    if last_time:
        dt = max(0.001, now - float(last_time))
        instant_fps = 1.0 / dt
        old_fps = float(st.session_state.get("fps_ema", 0.0))

        if old_fps <= 0.0:
            st.session_state["fps_ema"] = instant_fps
        else:
            st.session_state["fps_ema"] = 0.75 * old_fps + 0.25 * instant_fps

    st.session_state["last_process_time"] = now

    source_label = {
        "webcam": "REAL WEBCAM",
        "uploaded_video": "UPLOADED VIDEO",
        "synthetic": "SYNTHETIC FALLBACK",
    }.get(source_type, "LIVE")

    if source_type in ("webcam", "uploaded_video") and not present_real and int(st.session_state.get("no_face_frames", 0)) > 45:
        source_label += " + SIM WORKERS"

    annotated = draw_scene(
        frame=frame,
        workers=workers,
        frame_idx=frame_idx,
        events=list(st.session_state.get("events", []))[:1],
        source_label=source_label,
        fps=float(st.session_state.get("fps_ema", 0.0)),
        show_overlay=bool(st.session_state.get("show_overlay", True)),
    )

    record_annotated_frame(annotated)

    st.session_state["frame_count"] = frame_idx + 1
    st.session_state["last_frame"] = annotated

    return annotated


def inject_random_violation() -> None:
    workers = [worker for worker in st.session_state.get("workers", []) if worker.present]

    if not workers:
        return

    frame_idx = int(st.session_state.get("frame_count", 0))
    base_seed = int(st.session_state.get("base_seed", 1))
    rng = np.random.default_rng(base_seed + frame_idx + 99991)

    worker = workers[int(rng.integers(0, len(workers)))]
    ppe = PPE_CLASSES[int(rng.integers(0, len(PPE_CLASSES)))]

    worker.violation_remaining[ppe] = int(rng.integers(35, 70))
    worker.current_true[ppe] = 0

    for _ in range(10):
        worker.history[ppe].append(0)

    refresh_statuses(st.session_state.workers, frame_idx, record_alerts=True)


def toggle_playing_callback() -> None:
    st.session_state["playing"] = not bool(st.session_state.get("playing", True))


def step_frame_callback() -> None:
    st.session_state["step_once"] = True
    st.session_state["playing"] = False


def inject_violation_callback() -> None:
    inject_random_violation()


def retry_camera_callback() -> None:
    close_source()
    st.session_state["use_webcam"] = True
    st.session_state["source_type"] = "webcam"
    st.session_state["cap"] = None
    st.session_state["no_face_frames"] = 0
    st.session_state["camera_fail_count"] = 0
    st.session_state["camera_message"] = "Retrying local webcam..."


def use_webcam_change_callback() -> None:
    close_source()

    if bool(st.session_state.get("use_webcam", True)):
        st.session_state["source_type"] = "webcam"
        st.session_state["cap"] = None
        st.session_state["no_face_frames"] = 0
        st.session_state["camera_fail_count"] = 0
        st.session_state["camera_message"] = "Local webcam selected."
    else:
        if st.session_state.get("uploaded_video_path"):
            st.session_state["source_type"] = "uploaded_video"
            st.session_state["camera_message"] = "Using uploaded video."
        else:
            st.session_state["source_type"] = "synthetic"
            st.session_state["camera_message"] = "Webcam disabled. Using synthetic fallback."


def uploaded_video_change_callback() -> None:
    uploaded_file = st.session_state.get("uploaded_video")

    if uploaded_file is None:
        return

    try:
        path = save_uploaded_file(uploaded_file)
        close_source()

        cap = cv2.VideoCapture(path)

        recorder = st.session_state.get("recorder")
        if recorder is not None:
            recorder["status"] = "recording"
            recorder["frames_written"] = 0
            recorder["path"] = None
            recorder["writer"] = None

        st.session_state["demo_video_path"] = None

        if cap.isOpened():
            st.session_state["source_cap"] = cap
            st.session_state["uploaded_video_path"] = path
            st.session_state["uploaded_video_name"] = getattr(uploaded_file, "name", "uploaded_video")
            st.session_state["source_type"] = "uploaded_video"
            st.session_state["camera_message"] = f"Playing uploaded video: {st.session_state['uploaded_video_name']}"
        else:
            st.session_state["source_type"] = "synthetic"
            st.session_state["camera_message"] = "Uploaded video could not be opened. Using synthetic fallback."

    except Exception as exc:
        st.session_state["source_type"] = "synthetic"
        st.session_state["camera_message"] = f"Upload failed: {exc}"


def record_new_clip_callback() -> None:
    recorder = st.session_state.get("recorder")

    if recorder is None:
        recorder = {
            "status": "recording",
            "frames_written": 0,
            "target_frames": 75,
            "path": None,
            "writer": None,
        }
        st.session_state["recorder"] = recorder

    writer = recorder.get("writer")
    if writer is not None:
        try:
            writer.release()
        except Exception:
            pass

    remove_temp_file(st.session_state.get("demo_video_path"))

    recorder["status"] = "recording"
    recorder["frames_written"] = 0
    recorder["target_frames"] = 75
    recorder["path"] = None
    recorder["writer"] = None

    st.session_state["demo_video_path"] = None


def reset_callback() -> None:
    close_source()
    remove_temp_file(st.session_state.get("demo_video_path"))
    remove_temp_file(st.session_state.get("uploaded_video_path"))
    st.session_state.clear()
    initialize_state()


def worker_overview() -> pd.DataFrame:
    columns = [
        "Worker",
        "Role",
        "Zone",
        "Helmet",
        "Vest",
        "Gloves",
        "Goggles",
        "Overall",
        "Confidence",
        "Source",
        "Present",
    ]

    rows: List[Dict[str, Any]] = []

    for worker in st.session_state.get("workers", []):
        if not worker.present and worker.lost_frames >= 60:
            continue

        statuses = [worker.last_status.get(ppe, "uncertain") for ppe in PPE_CLASSES]

        if not worker.present:
            overall = "Offline"
        elif any(status == "violation" for status in statuses):
            overall = "Non-compliant"
        elif any(status == "uncertain" for status in statuses):
            overall = "Checking"
        else:
            overall = "Compliant"

        avg_confidence = float(
            np.mean([worker.last_confidence.get(ppe, 0.0) for ppe in PPE_CLASSES])
        )

        rows.append(
            {
                "Worker": worker.label,
                "Role": worker.role,
                "Zone": worker.zone,
                "Helmet": STATUS_ICON.get(worker.last_status.get("helmet", "uncertain"), "⚠️"),
                "Vest": STATUS_ICON.get(worker.last_status.get("vest", "uncertain"), "⚠️"),
                "Gloves": STATUS_ICON.get(worker.last_status.get("gloves", "uncertain"), "⚠️"),
                "Goggles": STATUS_ICON.get(worker.last_status.get("goggles", "uncertain"), "⚠️"),
                "Overall": overall,
                "Confidence": round(avg_confidence, 2),
                "Source": worker.kind,
                "Present": worker.present,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def render_metrics() -> None:
    overview = worker_overview()

    if overview.empty:
        active = 0
        compliant = 0
        non_compliant = 0
        checking = 0
    else:
        active = int(overview["Present"].sum())
        compliant = int((overview["Overall"] == "Compliant").sum())
        non_compliant = int((overview["Overall"] == "Non-compliant").sum())
        checking = int((overview["Overall"] == "Checking").sum())

    frame_count = int(st.session_state.get("frame_count", 0))
    camera_hours = max(0.001, frame_count / float(FPS) / 3600.0)

    false_alerts = int(st.session_state.get("false_alert_count", 0))
    true_alerts = int(st.session_state.get("true_alert_count", 0))
    alert_count = int(st.session_state.get("alert_count", 0))
    suppressed_flickers = int(st.session_state.get("suppressed_flickers", 0))

    false_alarm_rate = float(false_alerts) / camera_hours

    cols = st.columns(5)

    cols[0].metric(
        "Active workers",
        active,
        help="Workers currently visible or tracked in the real/synthetic feed.",
    )

    cols[1].metric(
        "Compliant",
        f"{compliant}/{active}" if active else "0/0",
        help="All four PPE classes compliant after temporal voting.",
    )

    cols[2].metric(
        "Violations",
        non_compliant,
        help="Workers with at least one confirmed missing PPE item.",
    )

    cols[3].metric(
        "Alerts",
        alert_count,
        help=f"True alerts: {true_alerts} | False alarms: {false_alerts}",
    )

    cols[4].metric(
        "False alarms / hr",
        f"{false_alarm_rate:.2f}",
        help="Demo acceptance gate: < 1.00 false alarm per camera-hour.",
    )

    st.caption(
        f"Temporal voting suppressed {suppressed_flickers} single-frame flickers. "
        f"Checking state: {checking} workers."
    )


def render_worker_table() -> None:
    st.subheader("Per-worker PPE verification")

    df = worker_overview()

    if df.empty:
        st.caption("Waiting for workers. Show a face to the webcam or upload a real video.")
        return

    st.dataframe(
        df.drop(columns=["Present"]),
        hide_index=True,
        use_container_width=True,
    )


def render_event_table() -> None:
    st.subheader("Violation and false-alarm log")

    events = list(st.session_state.get("events", []))

    if not events:
        st.caption("No violations logged yet. Use Inject violation or wait for simulated events.")
        return

    df = pd.DataFrame(events)
    st.dataframe(df, hide_index=True, use_container_width=True)


def render_generated_video() -> None:
    path = st.session_state.get("demo_video_path")

    if path and os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            try:
                st.video(path, loop=True, muted=True)
            except TypeError:
                st.video(path)
        except Exception:
            try:
                with open(path, "rb") as video_file:
                    st.video(video_file.read())
            except Exception:
                st.warning("Video file was generated, but preview failed in this runtime. Use download below.")

        suffix = os.path.splitext(path)[1] or ".mp4"
        file_name = f"ppe_real_demo{suffix}"
        mime = "video/mp4" if suffix.lower() == ".mp4" else "video/x-msvideo"

        try:
            with open(path, "rb") as video_file:
                st.download_button(
                    "Download generated live clip",
                    data=video_file.read(),
                    file_name=file_name,
                    mime=mime,
                    use_container_width=True,
                )
        except Exception:
            st.error("Could not read the generated video file for download.")
    else:
        recorder = st.session_state.get("recorder", {})
        status = recorder.get("status", "idle")
        written = int(recorder.get("frames_written", 0))
        target = int(recorder.get("target_frames", 75))

        if status == "recording":
            st.info(f"Recording live demo clip: {written}/{target} frames.")
        elif status == "failed":
            st.warning("Automatic MP4/AVI recording failed in this runtime. The live feed above still works.")
        else:
            st.info("No generated clip yet.")

        st.button(
            "Record new clip",
            on_click=record_new_clip_callback,
            use_container_width=True,
        )


def sidebar_toggle(label: str, key: str, on_change: Optional[Callable[[], None]] = None) -> None:
    if hasattr(st.sidebar, "toggle"):
        st.sidebar.toggle(label, key=key, on_change=on_change)
    else:
        st.sidebar.checkbox(label, key=key, on_change=on_change)


def make_fragment(func: Callable[[], None], interval: float) -> Callable[[], None]:
    if _FRAGMENT_SUPPORT["auto"]:
        try:
            return st.fragment(func, run_every=interval)
        except Exception:
            _FRAGMENT_SUPPORT["auto"] = False
            try:
                return st.fragment(func)
            except Exception:
                return func

    return func


def live_fragment_impl() -> None:
    playing = bool(st.session_state.get("playing", True))
    step = bool(st.session_state.get("step_once", False))

    if playing or step or st.session_state.get("last_frame") is None:
        raw = get_source_frame()
        annotated = process_frame(raw)

        if step:
            st.session_state["step_once"] = False
    else:
        annotated = st.session_state.get("last_frame")

    if annotated is not None:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    status = "LIVE" if playing else "PAUSED"
    source = str(st.session_state.get("source_type", "synthetic")).upper()
    frame_count = int(st.session_state.get("frame_count", 0))

    st.caption(
        f"{status} • {source} • frame {frame_count} • real-image pipeline with per-worker PPE status"
    )


def metrics_fragment_impl() -> None:
    render_metrics()


def worker_fragment_impl() -> None:
    render_worker_table()


def event_fragment_impl() -> None:
    render_event_table()


st.set_page_config(
    page_title="Real-Time PPE Compliance Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_widget_defaults()
sanitize_widget_state()

if not st.session_state.get("initialized", False):
    with st.spinner("Starting real-image demo, loading webcam/upload source and recording clip..."):
        initialize_state()

st.sidebar.title("Control room")
st.sidebar.caption("No API key required. Uses local webcam, uploaded video, or synthetic fallback.")

sidebar_toggle(
    "Use local webcam",
    key="use_webcam",
    on_change=use_webcam_change_callback,
)

st.sidebar.slider(
    "Temporal voting window",
    min_value=3,
    max_value=10,
    step=1,
    key="voting_window",
)

st.sidebar.slider(
    "Face detection strictness",
    min_value=3,
    max_value=10,
    step=1,
    key="min_neighbors",
    help="Lower value detects more faces but may create more false tracks.",
)

sidebar_toggle(
    "Show PPE overlays",
    key="show_overlay",
)

st.sidebar.button(
    "Pause / Resume",
    on_click=toggle_playing_callback,
    use_container_width=True,
)

st.sidebar.button(
    "Step one frame",
    on_click=step_frame_callback,
    use_container_width=True,
)

st.sidebar.button(
    "Inject violation",
    on_click=inject_violation_callback,
    use_container_width=True,
)

st.sidebar.button(
    "Retry webcam",
    on_click=retry_camera_callback,
    use_container_width=True,
)

st.sidebar.button(
    "Record new clip",
    on_click=record_new_clip_callback,
    use_container_width=True,
)

st.sidebar.button(
    "Reset simulation",
    on_click=reset_callback,
    use_container_width=True,
)

st.sidebar.file_uploader(
    "Upload real video, optional",
    type=["mp4", "avi", "mov", "mkv"],
    key="uploaded_video",
    on_change=uploaded_video_change_callback,
)

st.sidebar.caption(st.session_state.get("camera_message", ""))

sidebar_video_path = st.session_state.get("demo_video_path")

if (
    sidebar_video_path
    and os.path.exists(sidebar_video_path)
    and os.path.getsize(sidebar_video_path) > 0
):
    suffix = os.path.splitext(sidebar_video_path)[1] or ".mp4"
    file_name = f"ppe_real_demo{suffix}"
    mime = "video/mp4" if suffix.lower() == ".mp4" else "video/x-msvideo"

    try:
        with open(sidebar_video_path, "rb") as sidebar_video_file:
            st.sidebar.download_button(
                "Download live clip",
                data=sidebar_video_file.read(),
                file_name=file_name,
                mime=mime,
                use_container_width=True,
            )
    except Exception:
        pass

st.title("🪖 Real-Time PPE Compliance Monitor")

st.caption(
    "Real-image live demo: uses webcam or uploaded video frames, tracks workers, and verifies "
    "**helmet, vest, gloves and goggles per worker** with temporal voting."
)

live_fragment = make_fragment(live_fragment_impl, 0.10)
metrics_fragment = make_fragment(metrics_fragment_impl, 0.50)
worker_fragment = make_fragment(worker_fragment_impl, 0.50)
event_fragment = make_fragment(event_fragment_impl, 1.00)

st.subheader("Live real-image feed")

live_fragment()
metrics_fragment()
worker_fragment()
event_fragment()

st.subheader("Generated live-video file")
render_generated_video()

with st.expander("How this demo works and what is simulated"):
    st.markdown(
        """
        **Real-image pipeline**
        - The app tries to open your local webcam.
        - It reads real video frames and resizes them for fast processing.
        - It detects faces using OpenCV Haar cascades.
        - Detected faces are tracked as workers using simple IoU association.
        - Each worker gets helmet, vest, gloves and goggles status.
        - Temporal voting suppresses single-frame flicker before raising an alert.
        - The app records a short live demo clip automatically and provides download.

        **Important honesty note**
        - This demo uses real camera/video images.
        - The PPE compliance states are simulated for demonstration because no trained PPE detector or API key is used.
        - To make it production-grade, replace `simulate_ppe()` with outputs from a real person detector, PPE detector, tracker and keypoint/ROI association model.

        **If webcam does not work**
        - Run the app locally, not on a remote server.
        - Allow camera access for Python/terminal/browser as required by your OS.
        - Or upload a real MP4/AVI/MOV/MKV video in the sidebar.
        - If neither is available, the app uses a synthetic fallback so the dashboard is never blank.
        """
    )

if not _FRAGMENT_SUPPORT["auto"] and (
    bool(st.session_state.get("playing", True))
    or bool(st.session_state.get("step_once", False))
):
    time.sleep(0.12)
    st.rerun()