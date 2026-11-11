"""Modules d'analyse ajoutés au-dessus du suivi (nos apports).

Chaque véhicule suivi est représenté par son point d'ancrage = milieu du bas de
la boîte : c'est le point qui touche la chaussée, donc le seul qui soit
cohérent avec une homographie du plan de la route.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import cv2
import numpy as np

Point = tuple[float, float]


def anchor_point(x1: float, y1: float, x2: float, y2: float) -> Point:
    """Milieu du bord inférieur de la boîte (contact avec le sol)."""
    return ((x1 + x2) / 2.0, y2)


def _side(a: Point, b: Point, p: Point) -> float:
    """Signe du produit vectoriel (b - a) x (p - a) : côté de p par rapport à la droite ab."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """Vrai si le segment p1p2 coupe strictement le segment q1q2."""
    d1, d2 = _side(q1, q2, p1), _side(q1, q2, p2)
    d3, d4 = _side(p1, p2, q1), _side(p1, p2, q2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


@dataclass
class Crossing:
    track_id: int
    cls: str
    direction: str  # "in" ou "out"
    frame: int


class LineCounter:
    """Compte les véhicules qui franchissent une ligne, par classe et par sens.

    Un véhicule n'est compté qu'une fois (par identifiant), ce qui rend le
    comptage robuste aux oscillations de la boîte autour de la ligne.
    """

    def __init__(self, a: Point, b: Point):
        self.a, self.b = tuple(map(float, a)), tuple(map(float, b))
        self._last: dict[int, Point] = {}
        self._counted: set[int] = set()
        self.counts: dict[str, dict[str, int]] = defaultdict(lambda: {"in": 0, "out": 0})
        self.events: list[Crossing] = []

    def update(self, track_id: int, cls: str, point: Point, frame: int) -> Crossing | None:
        prev = self._last.get(track_id)
        self._last[track_id] = point
        if prev is None or track_id in self._counted:
            return None
        if not segments_intersect(prev, point, self.a, self.b):
            return None
        # côté positif -> côté négatif = "in" (convention ; dépend de l'orientation a->b)
        direction = "in" if _side(self.a, self.b, prev) > 0 else "out"
        self._counted.add(track_id)
        self.counts[cls][direction] += 1
        ev = Crossing(track_id, cls, direction, frame)
        self.events.append(ev)
        return ev

    @property
    def total(self) -> int:
        return sum(c["in"] + c["out"] for c in self.counts.values())


class SpeedEstimator:
    """Vitesse (km/h) à partir de la trajectoire projetée sur le sol par homographie.

    H : image (pixels) -> sol (mètres), calculée à partir de 4 points connus.
    La vitesse est la distance parcourue sur une fenêtre glissante divisée par
    la durée, puis lissée par moyenne exponentielle (les boîtes « tremblent »).
    """

    def __init__(self, image_points, world_points, fps: float, window: int = 10, ema: float = 0.3):
        src = np.asarray(image_points, dtype=np.float32)
        dst = np.asarray(world_points, dtype=np.float32)
        if src.shape != (4, 2) or dst.shape != (4, 2):
            raise ValueError("Il faut exactement 4 points image et 4 points monde")
        self.H = cv2.getPerspectiveTransform(src, dst)
        self.fps = float(fps)
        self.window = window
        self.ema = ema
        self._hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))
        self.speed: dict[int, float] = {}

    def to_world(self, point: Point) -> np.ndarray:
        p = np.array([[point]], dtype=np.float32)
        return cv2.perspectiveTransform(p, self.H)[0, 0]

    def update(self, track_id: int, point: Point, frame: int) -> float | None:
        hist = self._hist[track_id]
        hist.append((frame, self.to_world(point)))
        if len(hist) < max(3, self.window // 2):
            return None
        (f0, w0), (f1, w1) = hist[0], hist[-1]
        dt = (f1 - f0) / self.fps
        if dt <= 0:
            return self.speed.get(track_id)
        kmh = float(np.linalg.norm(w1 - w0) / dt * 3.6)
        prev = self.speed.get(track_id)
        self.speed[track_id] = kmh if prev is None else self.ema * kmh + (1 - self.ema) * prev
        return self.speed[track_id]


class Heatmap:
    """Accumule les positions des véhicules pour visualiser les zones de passage."""

    def __init__(self, width: int, height: int, radius: int = 12):
        self.acc = np.zeros((height, width), dtype=np.float32)
        self.radius = radius

    def update(self, point: Point) -> None:
        cv2.circle(self.acc, (int(point[0]), int(point[1])), self.radius, 1.0, -1)

    def render(self, background: np.ndarray | None = None, alpha: float = 0.55) -> np.ndarray:
        norm = self.acc / self.acc.max() if self.acc.max() > 0 else self.acc
        norm = cv2.GaussianBlur(norm, (0, 0), self.radius)
        norm = norm / norm.max() if norm.max() > 0 else norm
        color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        if background is None:
            return color
        mask = (norm > 0.02)[..., None]
        blend = cv2.addWeighted(background, 1 - alpha, color, alpha, 0)
        return np.where(mask, blend, background)
