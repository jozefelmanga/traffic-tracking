"""Retire les boîtes situées dans les « ignored regions » d'UA-DETRAC avant TrackEval.

Pourquoi : UA-DETRAC n'annote pas les véhicules des zones ignorées (souvent la route
au loin). Un détecteur qui les voit quand même est compté comme faux positif, ce qui
fait chuter le MOTA sans que ce soit une vraie erreur. Le protocole officiel
d'UA-DETRAC écarte donc ces boîtes ; on fait la même chose ici, pour la vérité
terrain ET pour les sorties des trackers.

Règle : une boîte est écartée si plus de --thr (50 %) de sa surface est couverte par
l'union des zones ignorées de la séquence.

Exemple :
    python scripts/filter_ignored.py --gt data/detrac_yolo/mot_gt --trackers trackeval/trackers \\
        --out-gt data/detrac_yolo/mot_gt_ign --out-trackers trackeval/trackers_ign
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

IMG_W, IMG_H = 960, 540


def load_mask(ignored_txt: Path, width: int = IMG_W, height: int = IMG_H) -> np.ndarray:
    """Masque booléen (H, W) de l'union des zones ignorées."""
    mask = np.zeros((height, width), bool)
    if ignored_txt.exists():
        for line in ignored_txt.read_text().split():
            x, y, w, h = map(float, line.split(","))
            mask[max(0, int(y)):max(0, int(y + h)), max(0, int(x)):max(0, int(x + w))] = True
    return mask


def ignored_fraction(mask: np.ndarray, x: float, y: float, w: float, h: float) -> float:
    """Part de la boîte (x, y, w, h) couverte par le masque (0 si la boîte est hors image)."""
    H, W = mask.shape
    x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
    x1, y1 = min(W, int(round(x + w))), min(H, int(round(y + h)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(mask[y0:y1, x0:x1].mean())


def filter_mot_file(src: Path, dst: Path, mask: np.ndarray, thr: float) -> tuple[int, int]:
    """Copie un fichier MOT (frame,id,x,y,w,h,...) sans les boîtes ignorées. Retourne (gardées, retirées)."""
    kept, removed = [], 0
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        x, y, w, h = map(float, line.split(",")[2:6])
        if ignored_fraction(mask, x, y, w, h) > thr:
            removed += 1
        else:
            kept.append(line)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(kept) + ("\n" if kept else ""))
    return len(kept), removed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, required=True, help="dossier mot_gt (MVI_xxxxx/gt/gt.txt + ignored.txt)")
    ap.add_argument("--trackers", type=Path, required=True, help="dossier des trackers (<nom>/data/MVI_xxxxx.txt)")
    ap.add_argument("--out-gt", type=Path, required=True)
    ap.add_argument("--out-trackers", type=Path, required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    seqs = sorted(d.name for d in args.gt.iterdir() if (d / "gt" / "gt.txt").exists())
    if not any((args.gt / s / "ignored.txt").exists() for s in seqs):
        raise SystemExit("Aucun ignored.txt : relancer convert_detrac.py --mot-gt (version à jour)")
    masks = {s: load_mask(args.gt / s / "ignored.txt") for s in seqs}

    for s in seqs:
        k, r = filter_mot_file(args.gt / s / "gt" / "gt.txt", args.out_gt / s / "gt" / "gt.txt", masks[s], args.thr)
        shutil.copy(args.gt / s / "seqinfo.ini", args.out_gt / s / "seqinfo.ini")
        print(f"GT {s}: {k} boîtes gardées, {r} retirées ({masks[s].mean():.0%} de l'image ignorée)")

    for trk in sorted(d for d in args.trackers.iterdir() if (d / "data").is_dir()):
        tot_k = tot_r = 0
        for s in seqs:
            src = trk / "data" / f"{s}.txt"
            if src.exists():
                k, r = filter_mot_file(src, args.out_trackers / trk.name / "data" / f"{s}.txt", masks[s], args.thr)
                tot_k, tot_r = tot_k + k, tot_r + r
        print(f"{trk.name}: {tot_k} boîtes gardées, {tot_r} retirées ({tot_r / max(1, tot_k + tot_r):.0%})")


if __name__ == "__main__":
    main()
