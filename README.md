# Analyse de trafic par détection et suivi de véhicules

Projet de Vision par Ordinateur Avancée — 3ᵉ année cycle ingénieur en informatique (ISSAT Sousse).

**Entrée :** une vidéo de caméra routière (MP4).
**Sortie :** la vidéo annotée (boîte + classe + identifiant persistant par véhicule), le comptage par classe au franchissement d'une ligne virtuelle, la vitesse estimée, une carte de chaleur du trafic et un export CSV.

```
frame ──► YOLO11 (fine-tuné UA-DETRAC) ──► ByteTrack / BoT-SORT ──► analytics (comptage, vitesse, heatmap) ──► vidéo + CSV
           boîtes, classes, scores          ID par véhicule (Kalman + IoU)
```

## Nos apports par rapport à l'implémentation d'origine (Ultralytics)

| # | Apport | Fichier |
|---|--------|---------|
| 1 | Conversion UA-DETRAC (XML) → YOLO, masquage des zones ignorées, sous-échantillonnage temporel, découpage par séquence (pas de fuite train/val) | `scripts/convert_detrac.py` |
| 2 | Fine-tuning de YOLO11 sur 4 classes de véhicules + comparaison avec le modèle COCO d'origine | `notebooks/01_train_kaggle.ipynb` |
| 3 | Comptage directionnel par classe au franchissement d'une ligne | `src/traffic/analytics.py` (`LineCounter`) |
| 4 | Estimation de vitesse par homographie image → sol (mètres) | `src/traffic/analytics.py` (`SpeedEstimator`) |
| 5 | Carte de chaleur des trajectoires + export CSV | `src/traffic/analytics.py` (`Heatmap`) |
| 6 | Comparaison ByteTrack vs BoT-SORT et réglage de leurs paramètres (MOTA, IDF1, HOTA) | `configs/trackers/`, `scripts/track_video.py --mot-out` |
| 7 | Démo Gradio | `app.py` |

## Structure

```
traffic-tracking/
├── app.py                      # démo Gradio
├── configs/
│   ├── detrac.yaml             # dataset YOLO (chemins + classes)
│   ├── scene_example.json      # ligne de comptage + points d'homographie
│   └── trackers/               # ByteTrack / BoT-SORT réglés
├── scripts/
│   ├── convert_detrac.py       # UA-DETRAC XML → YOLO (+ GT format MOTChallenge)
│   └── track_video.py          # pipeline complet en ligne de commande
├── src/traffic/
│   ├── analytics.py            # comptage, vitesse, heatmap (numpy pur, testé)
│   └── pipeline.py             # YOLO + tracker + analytics + rendu
├── notebooks/01_train_kaggle.ipynb
├── tests/test_analytics.py
└── docs/errors_log.md          # journal des erreurs réelles (Partie 6)
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac : source .venv/bin/activate)
pip install -r requirements.txt
```

L'entraînement se fait sur **Kaggle Notebooks (GPU T4 x2, Internet activé)** via `notebooks/01_train_kaggle.ipynb`. L'inférence et la démo tournent sur CPU.

## Utilisation

```bash
# 1) Préparer le dataset (après téléchargement de UA-DETRAC)
python scripts/convert_detrac.py --images DETRAC-train-data/Insight-MVT_Annotation_Train \
    --xml DETRAC-Train-Annotations-XML --out data/detrac_yolo --stride 5 --val-ratio 0.15

# 2) Lancer le suivi sur une vidéo
python scripts/track_video.py --video demo.mp4 --weights weights/best.pt \
    --tracker configs/trackers/bytetrack_tuned.yaml --scene configs/scene_example.json \
    --out outputs/

# 3) Démo
python app.py
```

## Dataset

**UA-DETRAC** — 100 séquences (≈140 000 images, 960×540) filmées à Pékin et Tianjin, classes `car`, `bus`, `van`, `others`, avec identifiants de suivi.
Licence : **CC BY-NC-SA 3.0** (usage académique non commercial) — à vérifier au téléchargement.

## Métriques

- Détection : mAP50, mAP50-95 (Ultralytics `val`).
- Suivi : MOTA, IDF1, HOTA, ID switches ([TrackEval](https://github.com/JonathonLuiten/TrackEval)).

## Références

- Wen et al., *UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking*, CVIU 2020.
- Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, ECCV 2022.
- Aharon et al., *BoT-SORT: Robust Associations Multi-Pedestrian Tracking*, 2022.
- Jocher et al., *Ultralytics YOLO11*, 2024 — https://github.com/ultralytics/ultralytics
- Luiten et al., *HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking*, IJCV 2021.

## Licence

AGPL-3.0 (hérité d'Ultralytics).
