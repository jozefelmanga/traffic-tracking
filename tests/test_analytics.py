"""Tests des modules d'analyse et du convertisseur (sans GPU ni ultralytics)."""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic.analytics import Heatmap, LineCounter, SpeedEstimator, anchor_point, segments_intersect  # noqa: E402


def test_anchor_point_is_bottom_center():
    assert anchor_point(10, 20, 30, 60) == (20, 60)


def test_segments_intersect():
    assert segments_intersect((0, 0), (10, 10), (0, 10), (10, 0))
    assert not segments_intersect((0, 0), (1, 1), (5, 0), (5, 10))


def test_line_counter_counts_once_with_direction():
    c = LineCounter((0, 100), (200, 100))
    # véhicule 1 descend et traverse la ligne, puis oscille autour : compté une seule fois
    for f, y in enumerate([80, 95, 105, 98, 110]):
        c.update(1, "car", (50, y), f)
    # véhicule 2 monte
    for f, y in enumerate([130, 110, 90]):
        c.update(2, "bus", (120, y), f)
    # véhicule 3 ne traverse pas
    for f, y in enumerate([10, 20, 30]):
        c.update(3, "car", (60, y), f)
    assert c.total == 2
    assert c.counts["car"]["in"] + c.counts["car"]["out"] == 1
    assert c.counts["car"] != c.counts["bus"]
    dirs = {e.track_id: e.direction for e in c.events}
    assert dirs[1] != dirs[2]


def test_speed_estimator_recovers_known_speed():
    # Homographie identité à l'échelle 1 px = 0.1 m
    img = [[0, 0], [100, 0], [100, 100], [0, 100]]
    world = [[0, 0], [10, 0], [10, 10], [0, 10]]
    fps = 25.0
    est = SpeedEstimator(img, world, fps, window=10)
    v_ms = 15.0  # 54 km/h -> 150 px/s -> 6 px/frame
    speed = None
    for f in range(30):
        speed = est.update(7, (50, f * v_ms / 0.1 / fps), f)
    assert speed == pytest.approx(54.0, rel=1e-3)


def test_speed_estimator_rejects_bad_points():
    with pytest.raises(ValueError):
        SpeedEstimator([[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 0], [1, 1]], 25)


def test_heatmap_render_shape():
    hm = Heatmap(64, 48, radius=4)
    hm.update((32, 24))
    bg = np.zeros((48, 64, 3), np.uint8)
    out = hm.render(bg)
    assert out.shape == bg.shape and out[24, 32].any()


XML = """<?xml version="1.0" encoding="utf-8"?>
<sequence name="MVI_00001">
  <sequence_attribute camera_state="stable" sence_weather="sunny"/>
  <ignored_region><box left="0" top="0" width="20" height="20"/></ignored_region>
  <frame density="2" num="1"><target_list>
    <target id="1"><box left="100" top="100" width="50" height="40"/>
      <attribute orientation="0" speed="1" trajectory_length="1" truncation_ratio="0" vehicle_type="car"/></target>
    <target id="2"><box left="940" top="500" width="60" height="60"/>
      <attribute orientation="0" speed="1" trajectory_length="1" truncation_ratio="0" vehicle_type="bus"/></target>
  </target_list></frame>
  <frame density="1" num="2"><target_list>
    <target id="1"><box left="105" top="100" width="50" height="40"/>
      <attribute orientation="0" speed="1" trajectory_length="2" truncation_ratio="0" vehicle_type="car"/></target>
  </target_list></frame>
</sequence>
"""


def test_convert_detrac_end_to_end(tmp_path):
    xml_dir, img_root, out = tmp_path / "xml", tmp_path / "imgs", tmp_path / "out"
    xml_dir.mkdir()
    (xml_dir / "MVI_00001.xml").write_text(XML)
    (img_root / "MVI_00001").mkdir(parents=True)
    for n in (1, 2):
        cv2.imwrite(str(img_root / "MVI_00001" / f"img{n:05d}.jpg"), np.full((540, 960, 3), 255, np.uint8))

    subprocess.run([sys.executable, str(ROOT / "scripts" / "convert_detrac.py"), "--images", str(img_root),
                    "--xml", str(xml_dir), "--out", str(out), "--stride", "1", "--mot-gt"], check=True)

    # une seule séquence -> elle va en validation
    lab = (out / "labels" / "val" / "MVI_00001_img00001.txt").read_text().splitlines()
    assert len(lab) == 2
    cls, cx, cy, w, h = map(float, lab[0].split())
    assert cls == 0 and cx == pytest.approx(125 / 960, abs=1e-6) and w == pytest.approx(50 / 960, abs=1e-6)
    # la boîte du bus déborde de l'image : elle doit être rognée dans [0, 1]
    cls, cx, cy, w, h = map(float, lab[1].split())
    assert cls == 1 and cx + w / 2 <= 1 + 1e-6 and cy + h / 2 <= 1 + 1e-6
    # zone ignorée masquée en gris
    img = cv2.imread(str(out / "images" / "val" / "MVI_00001_img00001.jpg"))
    assert abs(int(img[5, 5, 0]) - 114) < 10 and img[300, 500, 0] > 240
    # vérité terrain MOT
    gt = (out / "mot_gt" / "MVI_00001" / "gt" / "gt.txt").read_text().splitlines()
    assert gt[0].startswith("1,1,100.00,100.00,50.00,40.00") and len(gt) == 3
    assert (out / "mot_gt" / "MVI_00001" / "ignored.txt").read_text() == "0.00,0.00,20.00,20.00\n"


def test_filter_ignored_end_to_end(tmp_path):
    gt, trk = tmp_path / "gt", tmp_path / "trk"
    (gt / "MVI_1" / "gt").mkdir(parents=True)
    (gt / "MVI_1" / "seqinfo.ini").write_text("[Sequence]\n")
    (gt / "MVI_1" / "ignored.txt").write_text("0.00,0.00,100.00,100.00\n")
    (gt / "MVI_1" / "gt" / "gt.txt").write_text("1,1,500,300,40,40,1,1,1\n")
    (trk / "t" / "data").mkdir(parents=True)
    # boîte 1 entièrement ignorée, boîte 2 à 25 % dans la zone (gardée), boîte 3 hors zone
    (trk / "t" / "data" / "MVI_1.txt").write_text(
        "1,1,10,10,20,20,0.9,-1,-1,-1\n1,2,90,90,20,20,0.9,-1,-1,-1\n1,3,500,300,40,40,0.9,-1,-1,-1\n")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "filter_ignored.py"), "--gt", str(gt), "--trackers",
                    str(trk), "--out-gt", str(tmp_path / "gt2"), "--out-trackers", str(tmp_path / "trk2")], check=True)
    ids = [l.split(",")[1] for l in (tmp_path / "trk2" / "t" / "data" / "MVI_1.txt").read_text().splitlines()]
    assert ids == ["2", "3"]
    assert (tmp_path / "gt2" / "MVI_1" / "gt" / "gt.txt").read_text().startswith("1,1,500")
    assert (tmp_path / "gt2" / "MVI_1" / "seqinfo.ini").exists()
