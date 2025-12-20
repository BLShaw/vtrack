from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class VehicleCounter:
    """Handles vehicle counting using line intersection logic."""
    
    def __init__(self, line_up: List[int], line_down: List[int]) -> None:
        """
        Initialize the counter.
        
        Args:
            line_up: [x1, y1, x2, y2] coordinates for the 'up' counting line.
            line_down: [x1, y1, x2, y2] coordinates for the 'down' counting line.
        """
        self.line_up = line_up
        self.line_down = line_down
        
        self.previous_positions: Dict[int, Tuple[int, int]] = {}
        
        self.count_up: List[int] = []
        self.count_down: List[int] = []
        self.total_count: List[int] = []

    def _ccw(self, A: Tuple[int, int], B: Tuple[int, int], C: Tuple[int, int]) -> bool:
        """Check if three points are listed in counter-clockwise order."""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def _intersect(self, A: Tuple[int, int], B: Tuple[int, int], C: Tuple[int, int], D: Tuple[int, int]) -> bool:
        """
        Return True if line segment AB intersects line segment CD.
        """
        return (self._ccw(A, C, D) != self._ccw(B, C, D)) and (self._ccw(A, B, C) != self._ccw(A, B, D))

    def update(self, cx: int, cy: int, id: int) -> None:
        """
        Update the position of a vehicle and check if it crossed a line.
        """
        if id not in self.previous_positions:
            self.previous_positions[id] = (cx, cy)
            return

        prev_cx, prev_cy = self.previous_positions[id]
        
        p1 = (prev_cx, prev_cy)
        p2 = (cx, cy)
        
        l1_start = (self.line_up[0], self.line_up[1])
        l1_end = (self.line_up[2], self.line_up[3])
        
        if self._intersect(p1, p2, l1_start, l1_end):
            if id not in self.total_count:
                self.total_count.append(id)
                if id not in self.count_up:
                    self.count_up.append(id)
                    logger.debug(f"Vehicle {id} crossed UP line")

        l2_start = (self.line_down[0], self.line_down[1])
        l2_end = (self.line_down[2], self.line_down[3])

        if self._intersect(p1, p2, l2_start, l2_end):
             if id not in self.total_count:
                self.total_count.append(id)
                if id not in self.count_down:
                    self.count_down.append(id)
                    logger.debug(f"Vehicle {id} crossed DOWN line")
        
        self.previous_positions[id] = (cx, cy)

    def get_counts(self) -> Tuple[List[int], List[int], List[int]]:
        """Return the current counts."""
        return self.count_up, self.count_down, self.total_count