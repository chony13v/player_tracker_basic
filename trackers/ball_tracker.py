from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import sys
sys.path.append('../')

from utils import read_stub, save_stub


class BallTracker:
    """
    Detecta y sigue la pelota. Rellena huecos por interpolación
    y elimina bboxes fuera de rango o con NaN.
    """

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    # -------------------------------------------------------------- #
    # 1) detección por lotes
    # -------------------------------------------------------------- #
    def detect_frames(self, frames, conf=0.5, batch_size=20):
        detections = []
        for i in range(0, len(frames), batch_size):
            detections += self.model.predict(frames[i:i + batch_size], conf=conf)
        return detections

    # -------------------------------------------------------------- #
    # 2) genera tracks (1 dict por frame)
    # -------------------------------------------------------------- #
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None and len(tracks) == len(frames):
            return tracks

        detections = self.detect_frames(frames)
        tracks = []

        for det in detections:
            cls_inv = {v: k for k, v in det.names.items()}
            det_sv = sv.Detections.from_ultralytics(det)

            frame_dict = {}
            best_bbox, best_conf = None, 0.0
            for d in det_sv:
                bbox, conf, cls_id = d[0].tolist(), d[2], d[3]
                if cls_id == cls_inv.get("Ball") and conf > best_conf:
                    best_bbox, best_conf = bbox, conf

            if best_bbox is not None:
                frame_dict[1] = {"bbox": best_bbox}

            tracks.append(frame_dict)

        save_stub(stub_path, tracks)
        return tracks

    # -------------------------------------------------------------- #
    # 3) descarta saltos imposibles
    # -------------------------------------------------------------- #
    def remove_wrong_detections(self, ball_tracks, max_px=25):
        last_good = -1
        for i, frame in enumerate(ball_tracks):
            box = frame.get(1, {}).get("bbox", [])
            if not box:
                continue
            if last_good == -1:
                last_good = i
                continue

            last_box = ball_tracks[last_good][1]["bbox"]
            gap = i - last_good
            if np.linalg.norm(np.array(last_box[:2]) - np.array(box[:2])) > max_px * gap:
                ball_tracks[i] = {}
            else:
                last_good = i
        return ball_tracks

    # -------------------------------------------------------------- #
    # 4) interpolación con limpieza de NaN
    # -------------------------------------------------------------- #
    def interpolate_ball_positions(self, ball_tracks):
        series = []
        for f in ball_tracks:
            box = f.get(1, {}).get("bbox", [])
            series.append(box if len(box) == 4 else [np.nan] * 4)

        df = pd.DataFrame(series, columns=["x1", "y1", "x2", "y2"]).interpolate().bfill()

        interpolated = []
        for row in df.itertuples(index=False, name=None):
            if any(np.isnan(row)):
                interpolated.append({})           # frame sin info usable
            else:
                interpolated.append({1: {"bbox": list(row)}})

        return interpolated
