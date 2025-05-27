import sys
import math
sys.path.append('../')

from utils.bbox_utils import measure_distance, get_center_of_bbox


class BallAquisitionDetector:
    """
    Determina qué jugador está en posesión del balón, frame a frame.
    Ignora bboxes nulos o con NaN para evitar fallos aguas abajo.
    """

    def __init__(self):
        # Umbrales heuristicos
        self.possession_threshold = 50      # px
        self.min_frames = 11               # frames consecutivos
        self.containment_threshold = 0.8   # % balón dentro del bbox jugador

    # ------------------------------------------------------------------ #
    # 1) puntos clave alrededor del jugador
    # ------------------------------------------------------------------ #
    def get_key_basketball_player_assignment_points(self, player_bbox, ball_center):
        x1, y1, x2, y2 = player_bbox
        cx, cy = ball_center
        w, h = x2 - x1, y2 - y1

        pts = []
        if y1 < cy < y2:
            pts.extend([(x1, cy), (x2, cy)])
        if x1 < cx < x2:
            pts.extend([(cx, y1), (cx, y2)])

        # esquinas + centro + algunos puntos intermedios
        pts.extend([
            (x1 + w // 2, y1), (x2, y1), (x1, y1),
            (x2, y1 + h // 2), (x1, y1 + h // 2),
            (x1 + w // 2, y1 + h // 2),
            (x2, y2), (x1, y2), (x1 + w // 2, y2),
            (x1 + w // 2, y1 + h // 3)
        ])
        return pts

    # ------------------------------------------------------------------ #
    # 2) % de balón dentro de bbox jugador
    # ------------------------------------------------------------------ #
    @staticmethod
    def calculate_ball_containment_ratio(player_bbox, ball_bbox):
        px1, py1, px2, py2 = player_bbox
        bx1, by1, bx2, by2 = ball_bbox

        ix1, iy1 = max(px1, bx1), max(py1, by1)
        ix2, iy2 = min(px2, bx2), min(py2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        inter = (ix2 - ix1) * (iy2 - iy1)
        ball_area = (bx2 - bx1) * (by2 - by1)
        return inter / ball_area if ball_area else 0.0

    # ------------------------------------------------------------------ #
    # 3) distancia mínima balón-jugador
    # ------------------------------------------------------------------ #
    def find_minimum_distance_to_ball(self, ball_center, player_bbox):
        pts = self.get_key_basketball_player_assignment_points(player_bbox, ball_center)
        return min(measure_distance(ball_center, p) for p in pts)

    # ------------------------------------------------------------------ #
    # 4) mejor candidato en un frame
    # ------------------------------------------------------------------ #
    def find_best_candidate_for_possession(self, ball_center, players_frame, ball_bbox):
        high, regular = [], []

        for pid, info in players_frame.items():
            pb = info.get("bbox", [])
            if len(pb) != 4:
                continue

            cont = self.calculate_ball_containment_ratio(pb, ball_bbox)
            dist = self.find_minimum_distance_to_ball(ball_center, pb)

            if cont > self.containment_threshold:
                high.append((pid, dist))
            else:
                regular.append((pid, dist))

        if high:
            return max(high, key=lambda x: x[1])[0]       # mayor contención
        if regular:
            pid, d = min(regular, key=lambda x: x[1])
            if d < self.possession_threshold:
                return pid
        return -1

    # ------------------------------------------------------------------ #
    # 5) bucle principal sobre todos los frames
    # ------------------------------------------------------------------ #
    def detect_ball_possession(self, player_tracks, ball_tracks):
        n = len(ball_tracks)
        possession_list = [-1] * n
        consecutive = {}          # {player_id: nº frames}

        for f in range(n):
            bb = ball_tracks[f].get(1, {}).get("bbox", [])
            if len(bb) != 4 or any(math.isnan(c) for c in bb):
                consecutive.clear()
                continue

            center = get_center_of_bbox(bb)
            best_pid = self.find_best_candidate_for_possession(center,
                                                                player_tracks[f],
                                                                bb)
            if best_pid != -1:
                consecutive = {best_pid: consecutive.get(best_pid, 0) + 1}
                if consecutive[best_pid] >= self.min_frames:
                    possession_list[f] = int(best_pid)
            else:
                consecutive.clear()

        return possession_list
