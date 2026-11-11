"""Démo Gradio : vidéo routière -> vidéo annotée + comptage + vitesse + heatmap."""
import sys
import tempfile
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from traffic.pipeline import TrafficPipeline, load_scene  # noqa: E402

FINETUNED = ROOT / "weights" / "best.pt"
WEIGHTS = {"YOLO11n fine-tuné UA-DETRAC": str(FINETUNED), "YOLO11n COCO (baseline)": "yolo11n.pt"}
TRACKERS = {  # le premier est la valeur par défaut : la meilleure configuration (HOTA 76,0)
    "BoT-SORT (réglé)": str(ROOT / "configs/trackers/botsort_tuned.yaml"),
    "ByteTrack (réglé)": str(ROOT / "configs/trackers/bytetrack_tuned.yaml"),
    "ByteTrack (défaut)": "bytetrack.yaml",
}
EXAMPLE_SCENE = ROOT / "configs" / "scene_example.json"


def run(video, weights_name, tracker_name, line_y, use_calibration, max_frames, progress=gr.Progress()):
    if video is None:
        raise gr.Error("Chargez une vidéo.")
    weights = WEIGHTS[weights_name]
    if weights == str(FINETUNED) and not FINETUNED.exists():
        raise gr.Error("weights/best.pt introuvable : entraînez le modèle ou choisissez la baseline.")

    import cv2
    cap = cv2.VideoCapture(video)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    scene = load_scene(EXAMPLE_SCENE) if use_calibration else {}
    y = int(h * line_y / 100)
    scene["count_line"] = [[int(0.05 * w), y], [int(0.95 * w), y]]

    pipe = TrafficPipeline(weights, TRACKERS[tracker_name], scene)
    out = Path(tempfile.mkdtemp(prefix="traffic_"))
    summary = pipe.process_video(
        video, out, max_frames=int(max_frames) or None,
        progress=lambda i, n: progress(i / n if n else 0, desc=f"image {i}/{n}"),
    )
    return summary["video"], summary["heatmap"], summary["counts"], str(out / "tracks.csv")


with gr.Blocks(title="Analyse de trafic") as demo:
    gr.Markdown("# Analyse de trafic — détection et suivi de véhicules\n"
                "YOLO11 + ByteTrack / BoT-SORT, comptage au franchissement de ligne, vitesse estimée.")
    with gr.Row():
        with gr.Column():
            vid = gr.Video(label="Vidéo d'entrée")
            wsel = gr.Dropdown(list(WEIGHTS), value=list(WEIGHTS)[0], label="Modèle")
            tsel = gr.Dropdown(list(TRACKERS), value=list(TRACKERS)[0], label="Tracker")
            line = gr.Slider(10, 95, value=70, step=1, label="Position de la ligne de comptage (% hauteur)")
            calib = gr.Checkbox(True, label="Vitesse : utiliser la calibration configs/scene_example.json")
            maxf = gr.Number(300, label="Nombre max. d'images (0 = toute la vidéo)", precision=0)
            btn = gr.Button("Lancer", variant="primary")
        with gr.Column():
            vout = gr.Video(label="Vidéo annotée")
            hout = gr.Image(label="Carte de chaleur")
            cout = gr.JSON(label="Comptage")
            fout = gr.File(label="tracks.csv")
    btn.click(run, [vid, wsel, tsel, line, calib, maxf], [vout, hout, cout, fout])

if __name__ == "__main__":
    demo.launch()
