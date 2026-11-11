"""Convertit UA-DETRAC (annotations XML par séquence) au format YOLO.

Apports par rapport à une conversion naïve :
  * découpage train/val PAR SÉQUENCE (des images consécutives d'une même vidéo
    dans train et val gonfleraient artificiellement le mAP) ;
  * sous-échantillonnage temporel (--stride) : 25 images/s quasi identiques
    n'apportent rien et multiplient le temps d'entraînement ;
  * masquage des « ignored regions » en gris : ces zones contiennent des
    véhicules NON annotés, qui seraient sinon appris comme du fond ;
  * export optionnel de la vérité terrain au format MOTChallenge pour
    évaluer le suivi avec TrackEval (MOTA / IDF1 / HOTA).

Exemple :
    python scripts/convert_detrac.py --images DETRAC-train-data/Insight-MVT_Annotation_Train \
        --xml DETRAC-Train-Annotations-XML --out data/detrac_yolo --stride 5 --val-ratio 0.15
"""
from __future__ import annotations

import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
from tqdm import tqdm

CLASSES = {"car": 0, "bus": 1, "van": 2, "others": 3}
IMG_W, IMG_H = 960, 540
MASK_COLOR = (114, 114, 114)  # même gris que le padding « letterbox » de YOLO


def parse_sequence(xml_path: Path):
    """Retourne (ignored_boxes, frames) ; frames = {num: [(id, cls, x, y, w, h), ...]}."""
    root = ET.parse(xml_path).getroot()
    ignored = [
        tuple(float(b.get(k)) for k in ("left", "top", "width", "height"))
        for b in root.iterfind("ignored_region/box")
    ]
    frames: dict[int, list] = {}
    for fr in root.iterfind("frame"):
        num = int(fr.get("num"))
        objs = []
        for t in fr.iterfind("target_list/target"):
            box = t.find("box")
            attr = t.find("attribute")
            vtype = attr.get("vehicle_type") if attr is not None else "others"
            x, y, w, h = (float(box.get(k)) for k in ("left", "top", "width", "height"))
            objs.append((int(t.get("id")), CLASSES.get(vtype, 3), x, y, w, h))
        frames[num] = objs
    return ignored, frames


def to_yolo_line(cls: int, x: float, y: float, w: float, h: float, iw: int, ih: int) -> str | None:
    # On rogne la boîte dans l'image (certaines annotations débordent).
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(iw), x + w), min(float(ih), y + h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    cx, cy = (x0 + x1) / 2 / iw, (y0 + y1) / 2 / ih
    return f"{cls} {cx:.6f} {cy:.6f} {(x1 - x0) / iw:.6f} {(y1 - y0) / ih:.6f}"


def write_mot_gt(seq: str, frames: dict, n_frames: int, out_root: Path) -> None:
    """GT MOTChallenge : frame,id,x,y,w,h,conf,class,visibility + seqinfo.ini."""
    seq_dir = out_root / seq
    (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
    with open(seq_dir / "gt" / "gt.txt", "w") as f:
        for num in sorted(frames):
            for tid, cls, x, y, w, h in frames[num]:
                f.write(f"{num},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1\n")
    (seq_dir / "seqinfo.ini").write_text(
        f"[Sequence]\nname={seq}\nimDir=img1\nframeRate=25\nseqLength={n_frames}\n"
        f"imWidth={IMG_W}\nimHeight={IMG_H}\nimExt=.jpg\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True, help="dossier contenant MVI_xxxxx/img00001.jpg")
    ap.add_argument("--xml", type=Path, required=True, help="dossier des MVI_xxxxx.xml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=5, help="garder 1 image sur N")
    ap.add_argument("--val-ratio", type=float, default=0.15, help="part des séquences mises en validation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-mask", action="store_true", help="ne pas masquer les zones ignorées")
    ap.add_argument("--mot-gt", action="store_true", help="exporter la GT MOT des séquences de validation")
    args = ap.parse_args()

    xml_files = sorted(args.xml.glob("MVI_*.xml"))
    if not xml_files:
        raise SystemExit(f"Aucun fichier MVI_*.xml dans {args.xml}")

    seqs = [p.stem for p in xml_files]
    random.Random(args.seed).shuffle(seqs)
    n_val = max(1, round(len(seqs) * args.val_ratio))
    val_seqs = set(seqs[:n_val])
    print(f"{len(seqs)} séquences -> {len(seqs) - n_val} train / {n_val} val")

    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
    (args.out / "val_sequences.txt").write_text("\n".join(sorted(val_seqs)) + "\n")

    stats = {"images": 0, "boxes": 0, "missing": 0}
    for xml_path in tqdm(xml_files, desc="séquences"):
        seq = xml_path.stem
        split = "val" if seq in val_seqs else "train"
        ignored, frames = parse_sequence(xml_path)
        img_dir = args.images / seq

        if args.mot_gt and split == "val":
            n_frames = len(list(img_dir.glob("img*.jpg"))) or max(frames, default=0)
            write_mot_gt(seq, frames, n_frames, args.out / "mot_gt")

        for num in sorted(frames)[:: args.stride]:
            src = img_dir / f"img{num:05d}.jpg"
            img = cv2.imread(str(src))
            if img is None:
                stats["missing"] += 1
                continue
            ih, iw = img.shape[:2]
            if not args.no_mask:
                for x, y, w, h in ignored:
                    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), MASK_COLOR, -1)

            name = f"{seq}_img{num:05d}"
            cv2.imwrite(str(args.out / "images" / split / f"{name}.jpg"), img)
            lines = [l for o in frames[num] if (l := to_yolo_line(o[1], *o[2:], iw, ih))]
            (args.out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
            stats["images"] += 1
            stats["boxes"] += len(lines)

    print(f"Terminé : {stats['images']} images, {stats['boxes']} boîtes, {stats['missing']} images introuvables")


if __name__ == "__main__":
    main()
