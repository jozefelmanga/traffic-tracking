"""Lance le pipeline détection + suivi + analytics sur une vidéo.

Exemples :
    # modèle fine-tuné + ByteTrack réglé
    python scripts/track_video.py --video demo.mp4 --weights weights/best.pt \
        --tracker configs/trackers/bytetrack_tuned.yaml --scene configs/scene_example.json

    # baseline : modèle COCO d'origine + tracker par défaut, export MOT pour TrackEval
    python scripts/track_video.py --video MVI_40701.mp4 --weights yolo11n.pt \
        --tracker bytetrack.yaml --mot-out outputs/mot/baseline/MVI_40701.txt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm  # noqa: E402

from traffic.pipeline import TrafficPipeline, load_scene  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="vidéo ou motif d'images (ex. MVI_40701/img%%05d.jpg)")
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--tracker", default="configs/trackers/bytetrack_tuned.yaml")
    ap.add_argument("--scene", default=None, help="JSON ligne de comptage + homographie")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--mot-out", default=None, help="fichier de sortie au format MOTChallenge")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="'cpu', '0', ...")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    pipe = TrafficPipeline(args.weights, args.tracker, load_scene(args.scene),
                           conf=args.conf, imgsz=args.imgsz, device=args.device)
    bar = tqdm(unit="frame")

    def progress(i: int, n: int) -> None:
        if n and bar.total != n:
            bar.total = n
        bar.update(1)

    summary = pipe.process_video(args.video, args.out, mot_out=args.mot_out,
                                 progress=progress, max_frames=args.max_frames)
    bar.close()
    for k, v in summary.items():
        print(f"{k:>8} : {v}")


if __name__ == "__main__":
    main()
