"""Pipeline complet : YOLO11 -> tracker (ByteTrack / BoT-SORT) -> analytics -> rendu."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .analytics import Heatmap, LineCounter, SpeedEstimator, anchor_point

# Classes véhicules de COCO (car, motorcycle, bus, truck) : utilisées quand on
# lance le modèle d'origine non fine-tuné, pour la comparaison avec la baseline.
COCO_VEHICLES = [2, 3, 5, 7]

PALETTE = {
    "car": (0, 200, 255),
    "bus": (255, 120, 0),
    "van": (0, 220, 0),
    "others": (200, 0, 200),
    "truck": (255, 120, 0),
    "motorcycle": (0, 220, 0),
}


@dataclass
class TrackStats:
    cls: str
    first_frame: int
    last_frame: int
    speeds: list[float] = field(default_factory=list)


def load_scene(path: str | Path | None) -> dict:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


class TrafficPipeline:
    def __init__(
        self,
        weights: str = "yolo11n.pt",
        tracker: str = "bytetrack.yaml",
        scene: dict | None = None,
        conf: float = 0.25,
        imgsz: int = 640,
        device: str | None = None,
    ):
        from ultralytics import YOLO  # import tardif : les tests n'en ont pas besoin

        self.model = YOLO(weights)
        self.tracker = tracker
        self.scene = scene or {}
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        names = self.model.names
        self.classes = COCO_VEHICLES if len(names) == 80 else None

    def process_video(
        self,
        video_in: str | Path,
        out_dir: str | Path,
        mot_out: str | Path | None = None,
        progress: Callable[[int, int], None] | None = None,
        max_frames: int | None = None,
    ) -> dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_in))
        if not cap.isOpened():
            raise FileNotFoundError(f"Impossible d'ouvrir la vidéo : {video_in}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if max_frames:
            n_total = min(n_total, max_frames) if n_total else max_frames

        video_out = out_dir / "annotated.mp4"
        # H.264 (avc1) pour que la vidéo soit lisible dans le navigateur (démo Gradio) ; mp4v en secours.
        writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
        if not writer.isOpened():
            writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        line = self.scene.get("count_line")
        counter = LineCounter(*line) if line else None
        hom = self.scene.get("homography")
        speed = SpeedEstimator(hom["image_points"], hom["world_points"], fps) if hom else None
        heat = Heatmap(w, h)
        tracks: dict[int, TrackStats] = {}
        mot_rows: list[str] = []
        last_frame = None

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and frame_idx >= max_frames):
                break
            frame_idx += 1
            last_frame = frame
            res = self.model.track(
                frame, persist=True, tracker=self.tracker, conf=self.conf, imgsz=self.imgsz,
                classes=self.classes, device=self.device, verbose=False,
            )[0]
            boxes = res.boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.int().cpu().numpy()
                cls_ids = boxes.cls.int().cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), tid, c, s in zip(xyxy, ids, cls_ids, confs):
                    tid, cname = int(tid), self.model.names[int(c)]
                    pt = anchor_point(x1, y1, x2, y2)
                    heat.update(pt)
                    st = tracks.setdefault(tid, TrackStats(cname, frame_idx, frame_idx))
                    st.last_frame = frame_idx
                    kmh = speed.update(tid, pt, frame_idx) if speed else None
                    if kmh is not None:
                        st.speeds.append(kmh)
                    if counter:
                        counter.update(tid, cname, pt, frame_idx)
                    mot_rows.append(
                        f"{frame_idx},{tid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{s:.3f},-1,-1,-1"
                    )
                    self._draw_box(frame, (x1, y1, x2, y2), tid, cname, kmh)
            self._draw_overlay(frame, counter)
            writer.write(frame)
            if progress:
                progress(frame_idx, n_total)

        cap.release()
        writer.release()

        if last_frame is not None:
            cv2.imwrite(str(out_dir / "heatmap.jpg"), heat.render(last_frame))
        self._write_csv(out_dir, tracks, counter, fps)
        if mot_out:
            Path(mot_out).parent.mkdir(parents=True, exist_ok=True)
            Path(mot_out).write_text("\n".join(mot_rows) + "\n")

        return {
            "video": str(video_out),
            "heatmap": str(out_dir / "heatmap.jpg"),
            "frames": frame_idx,
            "tracks": len(tracks),
            "counts": {k: dict(v) for k, v in counter.counts.items()} if counter else {},
        }

    # ------------------------------------------------------------------ rendu
    @staticmethod
    def _draw_box(img, box, tid, cname, kmh):
        x1, y1, x2, y2 = map(int, box)
        color = PALETTE.get(cname, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"#{tid} {cname}" + (f" {kmh:.0f} km/h" if kmh is not None else "")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_overlay(img, counter: LineCounter | None):
        if counter is None:
            return
        a, b = tuple(map(int, counter.a)), tuple(map(int, counter.b))
        cv2.line(img, a, b, (0, 0, 255), 2)
        lines = [f"Total : {counter.total}"] + [
            f"{c} : in {v['in']} / out {v['out']}" for c, v in sorted(counter.counts.items())
        ]
        pad = 8
        hbox = pad * 2 + 22 * len(lines)
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (260, 10 + hbox), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, dst=img)
        for i, t in enumerate(lines):
            cv2.putText(img, t, (20, 10 + pad + 16 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _write_csv(out_dir: Path, tracks: dict[int, TrackStats], counter, fps: float):
        with open(out_dir / "tracks.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["track_id", "class", "first_frame", "last_frame", "duration_s",
                         "mean_speed_kmh", "max_speed_kmh"])
            for tid, st in sorted(tracks.items()):
                sp = np.array(st.speeds) if st.speeds else None
                wr.writerow([tid, st.cls, st.first_frame, st.last_frame,
                             round((st.last_frame - st.first_frame) / fps, 2),
                             round(float(sp.mean()), 1) if sp is not None else "",
                             round(float(sp.max()), 1) if sp is not None else ""])
        if counter:
            with open(out_dir / "crossings.csv", "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["frame", "time_s", "track_id", "class", "direction"])
                for e in counter.events:
                    wr.writerow([e.frame, round(e.frame / fps, 2), e.track_id, e.cls, e.direction])
