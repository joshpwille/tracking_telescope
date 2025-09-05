"""
tracking_table.py
Maintains the table of active tracks per camera.

Each row is a dict with:
    id      (int)   – unique track ID
    bbox    (tuple) – (x, y, w, h)
    center  (tuple) – (cx, cy)
    scale   (float) – estimated scale (e.g., height or area)
    conf    (float) – detector confidence
"""

from typing import Dict, List, Tuple


class TrackingTable:
    def __init__(self, cam_id: str):
        self.cam_id = cam_id
        self.tracks: Dict[int, Dict] = {}

    def update_track(self, tid: int,
                     bbox: Tuple[int, int, int, int],
                     conf: float,
                     scale: float = None) -> None:
        """Insert or update a track in the table."""
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        if scale is None:
            scale = h  # default scale from bbox height

        self.tracks[tid] = {
            "id": tid,
            "bbox": (x, y, w, h),
            "center": (cx, cy),
            "scale": float(scale),
            "conf": float(conf),
        }

    def remove_track(self, tid: int) -> None:
        """Remove a track if it no longer exists."""
        if tid in self.tracks:
            del self.tracks[tid]

    def get_active_tracks(self) -> List[Dict]:
        """Return a list of all active tracks."""
        return list(self.tracks.values())

    def get_track(self, tid: int) -> Dict:
        """Return a single track by ID, or None."""
        return self.tracks.get(tid, None)

    def clear(self) -> None:
        """Clear all tracks (e.g., on reset)."""
        self.tracks.clear()

    def __repr__(self) -> str:
        return f"<TrackingTable cam={self.cam_id} n={len(self.tracks)}>"

