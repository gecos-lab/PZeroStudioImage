"""Direction- and curvature-aware corridor A* fracture tracing.

The search state includes the incoming direction.  This is essential: curvature
is a transition property and cannot be represented correctly by one scalar cost
per pixel.  Search is restricted to a click-defined corridor and all dense state
is stored in NumPy arrays; Python dictionaries are intentionally avoided.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple

import numpy as np

from .models import EvidenceMap, PixelXY, RasterDocument


class TraceStatus(str, Enum):
    """Terminal state of a trace request."""

    SUCCESS = "success"
    NO_PATH = "no_path"
    CANCELLED = "cancelled"
    LIMIT_REACHED = "limit_reached"


@dataclass(frozen=True)
class TraceConfig:
    """Non-negative terms in the A* path functional.

    The integrated traversal cost is

    ``base + probability_weight * -log(P) + uncertainty + gap + orientation``

    per unit pixel length.  A squared turn-angle term is added at each state
    transition. Keeping ``base_cost`` strictly positive gives every valid edge
    a finite positive traversal cost and supports the geometric fallback bound.
    """

    base_cost: float = 0.05
    probability_weight: float = 1.0
    orientation_weight: float = 0.35
    uncertainty_weight: float = 0.25
    curvature_weight: float = 0.30
    gap_weight: float = 0.75
    gap_threshold: float = 0.20
    probability_epsilon: float = 1.0e-4
    cancellation_interval: int = 256
    progress_interval: int = 2048

    def __post_init__(self) -> None:
        weights = (
            self.probability_weight,
            self.orientation_weight,
            self.uncertainty_weight,
            self.curvature_weight,
            self.gap_weight,
        )
        if self.base_cost <= 0.0:
            raise ValueError("base_cost must be strictly positive")
        if any(weight < 0.0 for weight in weights):
            raise ValueError("trace cost weights must be non-negative")
        if not 0.0 <= self.gap_threshold <= 1.0:
            raise ValueError("gap_threshold must be in [0, 1]")
        if not 0.0 < self.probability_epsilon < 1.0:
            raise ValueError("probability_epsilon must be in (0, 1)")
        if self.cancellation_interval <= 0 or self.progress_interval <= 0:
            raise ValueError("callback intervals must be positive integers")


@dataclass(frozen=True)
class TraceRequest:
    """Endpoints and spatial limits for one interactive trace.

    Coordinates use image ``(x, y)`` order and are rounded to the nearest pixel.
    ``corridor_radius`` is the maximum Euclidean distance, in pixels, from the
    straight segment joining the endpoints.
    """

    start_xy: Tuple[float, float]
    end_xy: Tuple[float, float]
    corridor_radius: float = 64.0
    max_expansions: Optional[int] = None

    def __post_init__(self) -> None:
        coordinates = (*self.start_xy, *self.end_xy)
        if len(self.start_xy) != 2 or len(self.end_xy) != 2:
            raise ValueError("start_xy and end_xy must each contain x and y")
        if not all(np.isfinite(value) for value in coordinates):
            raise ValueError("trace coordinates must be finite")
        if self.corridor_radius <= 0.0:
            raise ValueError("corridor_radius must be positive")
        if self.max_expansions is not None and self.max_expansions <= 0:
            raise ValueError("max_expansions must be positive when provided")

    @property
    def start_pixel(self) -> PixelXY:
        return int(round(self.start_xy[0])), int(round(self.start_xy[1]))

    @property
    def end_pixel(self) -> PixelXY:
        return int(round(self.end_xy[0])), int(round(self.end_xy[1]))


@dataclass(frozen=True)
class TraceProgress:
    """Lightweight progress snapshot suitable for a UI worker signal."""

    expanded_states: int
    frontier_states: int
    searchable_states: int
    best_cost: float

    @property
    def fraction(self) -> float:
        if self.searchable_states <= 0:
            return 0.0
        return min(1.0, self.expanded_states / self.searchable_states)


@dataclass
class TraceResult:
    """A full-resolution pixel polyline and A* diagnostics."""

    status: TraceStatus
    path_xy: np.ndarray
    cost: float
    expanded_states: int
    searchable_pixels: int
    message: str = ""

    def __post_init__(self) -> None:
        path = np.asarray(self.path_xy, dtype=np.int32)
        if path.size == 0:
            path = np.empty((0, 2), dtype=np.int32)
        if path.ndim != 2 or path.shape[1] != 2:
            raise ValueError("path_xy must have shape (n, 2)")
        self.path_xy = np.ascontiguousarray(path)

    @property
    def succeeded(self) -> bool:
        return self.status is TraceStatus.SUCCESS

    def simplified_xy(self, tolerance: float = 1.0) -> np.ndarray:
        """Return a Ramer-Douglas-Peucker simplification in pixel coordinates."""

        return simplify_polyline(self.path_xy, tolerance)

    def world_xy(
        self,
        document: RasterDocument,
        tolerance: Optional[float] = None,
        center: bool = True,
    ) -> np.ndarray:
        """Transform the full or simplified trace through ``document``'s affine."""

        path = self.path_xy if tolerance is None else self.simplified_xy(tolerance)
        if len(path) == 0:
            return np.empty((0, 2), dtype=np.float64)
        return document.pixel_to_world(path, center=center)


CancelCallback = Callable[[], bool]
ProgressCallback = Callable[[TraceProgress], None]


# Directions are clockwise in image coordinates (positive y points down).
_DY = np.asarray((-1, -1, 0, 1, 1, 1, 0, -1), dtype=np.int8)
_DX = np.asarray((0, 1, 1, 1, 0, -1, -1, -1), dtype=np.int8)
_STEP_LENGTH = np.asarray((1.0, math.sqrt(2.0), 1.0, math.sqrt(2.0),
                           1.0, math.sqrt(2.0), 1.0, math.sqrt(2.0)), dtype=np.float32)
_DIRECTION_ANGLE = np.arctan2(_DY.astype(np.float32), _DX.astype(np.float32))


class CorridorAStarTracer:
    """Trace a fracture through dense evidence using an admissible A* search."""

    direction_count = 8
    _heuristic_lookahead_steps = 1

    def __init__(self, config: Optional[TraceConfig] = None) -> None:
        self.config = config or TraceConfig()
        differences = np.abs(
            _DIRECTION_ANGLE[:, None].astype(np.float64)
            - _DIRECTION_ANGLE[None, :].astype(np.float64)
        )
        turn_angles = np.minimum(differences, 2.0 * np.pi - differences)
        self._turn_penalty = np.asarray(turn_angles**2, dtype=np.float32)

    def trace(
        self,
        evidence: EvidenceMap,
        request: TraceRequest,
        cancel: Optional[CancelCallback] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> TraceResult:
        """Find the minimum-cost path inside the endpoint corridor.

        Cancellation and progress callbacks are optional and are invoked only at
        configured intervals, making the method suitable for a background UI
        worker without coupling the core to a particular concurrency framework.
        """

        start_x, start_y = request.start_pixel
        end_x, end_y = request.end_pixel
        height, width = evidence.shape
        for name, x, y in (("start", start_x, start_y), ("end", end_x, end_y)):
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"{name} pixel {(x, y)} lies outside evidence bounds")
            if not evidence.valid_mask[y, x]:
                raise ValueError(f"{name} pixel {(x, y)} is invalid or nodata")
        if start_x == end_x and start_y == end_y:
            return TraceResult(
                TraceStatus.SUCCESS,
                np.asarray(((start_x, start_y),), dtype=np.int32),
                0.0,
                0,
                1,
                "start and end coincide",
            )

        crop = self._make_corridor(evidence, request)
        x_min, y_min, allowed = crop
        crop_height, crop_width = allowed.shape
        local_start_x, local_start_y = start_x - x_min, start_y - y_min
        local_end_x, local_end_y = end_x - x_min, end_y - y_min
        allowed[local_start_y, local_start_x] = True
        allowed[local_end_y, local_end_x] = True
        corridor_y, corridor_x = np.nonzero(allowed)
        searchable_pixels = int(len(corridor_y))
        searchable_states = searchable_pixels * self.direction_count
        if searchable_pixels > np.iinfo(np.int32).max:
            raise MemoryError("the requested corridor contains too many pixels")
        # A dense crop-to-corridor lookup makes neighbour access O(1), while all
        # costly direction states are allocated only for pixels inside the
        # corridor.  This matters for long diagonal traces whose bounding box is
        # much larger than the narrow search region.
        corridor_lookup = np.full(allowed.shape, -1, dtype=np.int32)
        corridor_lookup[corridor_y, corridor_x] = np.arange(
            searchable_pixels, dtype=np.int32
        )
        raster_slice = np.s_[y_min : y_min + crop_height, x_min : x_min + crop_width]
        probability = evidence.probability[raster_slice][allowed]
        orientation = evidence.orientation[raster_slice][allowed]
        uncertainty = evidence.uncertainty[raster_slice][allowed]
        traversal_cost, orientation_cost = self._cost_surfaces(
            probability, orientation, uncertainty
        )
        neighbour_pixels, scalar_edges = self._packed_edges(
            corridor_x,
            corridor_y,
            corridor_lookup,
            traversal_cost,
            orientation_cost,
        )
        start_pixel_index = int(corridor_lookup[local_start_y, local_start_x])
        end_pixel_index = int(corridor_lookup[local_end_y, local_end_x])
        scalar_heuristic, scalar_cancelled = self._reverse_scalar_heuristic(
            start_pixel_index,
            end_pixel_index,
            neighbour_pixels,
            scalar_edges,
            cancel,
        )
        if scalar_cancelled:
            return TraceResult(
                TraceStatus.CANCELLED,
                np.empty((0, 2), dtype=np.int32),
                math.inf,
                0,
                searchable_pixels,
                "trace cancelled while preparing the search heuristic",
            )
        if scalar_heuristic is None:
            return TraceResult(
                TraceStatus.NO_PATH,
                np.empty((0, 2), dtype=np.int32),
                math.inf,
                0,
                searchable_pixels,
                "no valid path exists inside the corridor",
            )
        incumbent_path, incumbent_cost = self._scalar_path_upper_bound(
            start_pixel_index,
            end_pixel_index,
            neighbour_pixels,
            scalar_edges,
            scalar_heuristic,
        )
        per_unit_edges = scalar_edges / _STEP_LENGTH[:, None]
        minimum_step_cost = float(np.min(per_unit_edges[np.isfinite(per_unit_edges)]))
        minimum_step_cost = float(np.nextafter(minimum_step_cost, -math.inf))
        geometric_heuristic = self._geometric_heuristic_vector(
            corridor_x,
            corridor_y,
            local_end_x,
            local_end_y,
            minimum_step_cost,
        )
        scalar_heuristic = np.maximum(scalar_heuristic, geometric_heuristic)
        heuristic = self._directional_lookahead_heuristic(
            scalar_heuristic,
            neighbour_pixels,
            scalar_edges,
            end_pixel_index,
        )

        state_count = searchable_pixels * self.direction_count
        # Float64 keeps the heap key and stored g-score bit-identical.  With a
        # float32 array, a rounded-down score can incorrectly make its own heap
        # entry look stale and break A* optimality on long paths.
        scores = np.full(state_count, np.inf, dtype=np.float64)
        parent_dtype = np.int32 if state_count <= np.iinfo(np.int32).max else np.int64
        parents = np.full(state_count, -2, dtype=parent_dtype)
        closed = np.zeros(state_count, dtype=bool)
        frontier = []

        for direction in range(self.direction_count):
            state = start_pixel_index * self.direction_count + direction
            scores[state] = 0.0
            parents[state] = -1
            heapq.heappush(
                frontier,
                (float(heuristic[direction, start_pixel_index]), state),
            )

        expanded = 0
        goal_state = -1
        terminal_status = TraceStatus.NO_PATH
        terminal_message = "no valid path exists inside the corridor"
        config = self.config

        use_incumbent = False
        while frontier:
            estimated_cost, state = heapq.heappop(frontier)
            pixel_index, incoming_direction = divmod(state, self.direction_count)
            queued_cost = float(scores[state])
            current_estimate = queued_cost + float(
                heuristic[incoming_direction, pixel_index]
            )
            if closed[state] or estimated_cost > current_estimate:
                continue
            if estimated_cost >= incumbent_cost:
                terminal_status = TraceStatus.SUCCESS
                terminal_message = "minimum-cost path found"
                use_incumbent = True
                break
            closed[state] = True
            expanded += 1

            if pixel_index == end_pixel_index:
                goal_state = state
                terminal_status = TraceStatus.SUCCESS
                terminal_message = "minimum-cost path found"
                break
            if request.max_expansions is not None and expanded >= request.max_expansions:
                terminal_status = TraceStatus.LIMIT_REACHED
                terminal_message = "maximum state expansion limit reached"
                break
            if cancel is not None and expanded % config.cancellation_interval == 0 and cancel():
                terminal_status = TraceStatus.CANCELLED
                terminal_message = "trace cancelled"
                break
            if progress is not None and expanded % config.progress_interval == 0:
                progress(
                    TraceProgress(
                        expanded,
                        len(frontier),
                        searchable_states,
                        float(queued_cost),
                    )
                )

            for next_direction in range(self.direction_count):
                next_pixel_index = int(neighbour_pixels[next_direction, pixel_index])
                if next_pixel_index < 0:
                    continue
                next_state = next_pixel_index * self.direction_count + next_direction
                if closed[next_state]:
                    continue
                edge_cost = float(scalar_edges[next_direction, pixel_index])
                edge_cost += config.curvature_weight * float(
                    self._turn_penalty[incoming_direction, next_direction]
                )
                tentative = float(scores[state]) + edge_cost
                estimate = tentative + float(
                    heuristic[next_direction, next_pixel_index]
                )
                if estimate >= incumbent_cost or tentative >= float(scores[next_state]):
                    continue
                scores[next_state] = tentative
                parents[next_state] = state
                heapq.heappush(frontier, (estimate, next_state))

        if (
            goal_state < 0
            and not use_incumbent
            and terminal_status is TraceStatus.NO_PATH
            and np.isfinite(incumbent_cost)
        ):
            terminal_status = TraceStatus.SUCCESS
            terminal_message = "minimum-cost path found"
            use_incumbent = True
        if progress is not None:
            if goal_state >= 0:
                best_cost = float(scores[goal_state])
            elif use_incumbent:
                best_cost = incumbent_cost
            else:
                best_cost = math.inf
            progress(TraceProgress(expanded, len(frontier), searchable_states, best_cost))
        if goal_state < 0 and not use_incumbent:
            return TraceResult(
                terminal_status,
                np.empty((0, 2), dtype=np.int32),
                math.inf,
                expanded,
                searchable_pixels,
                terminal_message,
            )
        if use_incumbent:
            path = np.column_stack(
                (
                    corridor_x[incumbent_path] + x_min,
                    corridor_y[incumbent_path] + y_min,
                )
            ).astype(np.int32)
            return TraceResult(
                terminal_status,
                path,
                incumbent_cost,
                expanded,
                searchable_pixels,
                terminal_message,
            )
        path = self._reconstruct_path(
            parents,
            goal_state,
            corridor_x,
            corridor_y,
            x_min,
            y_min,
        )
        return TraceResult(
            terminal_status,
            path,
            float(scores[goal_state]),
            expanded,
            searchable_pixels,
            terminal_message,
        )

    def _make_corridor(
        self,
        evidence: EvidenceMap,
        request: TraceRequest,
    ) -> Tuple[int, int, np.ndarray]:
        start_x, start_y = request.start_pixel
        end_x, end_y = request.end_pixel
        radius = float(request.corridor_radius)
        height, width = evidence.shape
        x_min = max(0, int(math.floor(min(start_x, end_x) - radius)))
        x_max = min(width - 1, int(math.ceil(max(start_x, end_x) + radius)))
        y_min = max(0, int(math.floor(min(start_y, end_y) - radius)))
        y_max = min(height - 1, int(math.ceil(max(start_y, end_y) + radius)))
        yy, xx = np.ogrid[y_min : y_max + 1, x_min : x_max + 1]
        segment_x = float(end_x - start_x)
        segment_y = float(end_y - start_y)
        squared_length = segment_x * segment_x + segment_y * segment_y
        projection = (
            (xx.astype(np.float32) - start_x) * segment_x
            + (yy.astype(np.float32) - start_y) * segment_y
        ) / squared_length
        projection = np.clip(projection, 0.0, 1.0)
        closest_x = start_x + projection * segment_x
        closest_y = start_y + projection * segment_y
        squared_distance = (xx - closest_x) ** 2 + (yy - closest_y) ** 2
        allowed = squared_distance <= radius * radius
        allowed &= evidence.valid_mask[y_min : y_max + 1, x_min : x_max + 1]
        return x_min, y_min, np.ascontiguousarray(allowed)

    def _cost_surfaces(
        self,
        probability: np.ndarray,
        orientation: np.ndarray,
        uncertainty: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        config = self.config
        clipped_probability = np.clip(
            probability, config.probability_epsilon, 1.0
        )
        evidence_penalty = -np.log(clipped_probability)
        if config.gap_threshold > 0.0:
            gap = np.maximum(
                0.0, (config.gap_threshold - probability) / config.gap_threshold
            )
            gap_penalty = gap**2
        else:
            gap_penalty = np.zeros(probability.shape, dtype=np.float32)
        traversal = (
            config.base_cost
            + config.probability_weight * evidence_penalty
            + config.uncertainty_weight * uncertainty
            + config.gap_weight * gap_penalty
        ).astype(np.float32)
        # Axial orientations repeat after four movement directions.  A model's
        # angle is not meaningful where fracture presence is unlikely or the
        # prediction is uncertain, so mismatch is gated by evidence reliability.
        reliability = probability * (1.0 - uncertainty)
        directional = np.empty((4,) + probability.shape, dtype=np.float32)
        for direction, angle in enumerate(_DIRECTION_ANGLE[:4]):
            # sin^2 is axial: theta and theta + pi receive exactly the same cost.
            mismatch = np.sin(float(angle) - orientation) ** 2
            directional[direction] = config.orientation_weight * reliability * mismatch
        return traversal, directional

    def _packed_edges(
        self,
        corridor_x: np.ndarray,
        corridor_y: np.ndarray,
        corridor_lookup: np.ndarray,
        traversal_cost: np.ndarray,
        orientation_cost: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Precompute packed neighbours and scalar edge integrals."""

        pixel_count = len(corridor_x)
        crop_height, crop_width = corridor_lookup.shape
        neighbours = np.full(
            (self.direction_count, pixel_count), -1, dtype=np.int32
        )
        edges = np.full(
            (self.direction_count, pixel_count), np.inf, dtype=np.float32
        )
        for direction in range(self.direction_count // 2):
            next_y = corridor_y + int(_DY[direction])
            next_x = corridor_x + int(_DX[direction])
            inside = (
                (next_y >= 0)
                & (next_y < crop_height)
                & (next_x >= 0)
                & (next_x < crop_width)
            )
            source = np.flatnonzero(inside)
            targets = corridor_lookup[next_y[inside], next_x[inside]]
            connected = targets >= 0
            source = source[connected]
            targets = targets[connected]
            neighbours[direction, source] = targets
            integrated = 0.5 * (
                traversal_cost[source]
                + traversal_cost[targets]
                + orientation_cost[direction % 4, source]
                + orientation_cost[direction % 4, targets]
            )
            edge = float(_STEP_LENGTH[direction]) * integrated
            edges[direction, source] = edge
            opposite = direction + self.direction_count // 2
            neighbours[opposite, targets] = source
            edges[opposite, targets] = edge
        return neighbours, edges

    def _reverse_scalar_heuristic(
        self,
        start_pixel_index: int,
        end_pixel_index: int,
        neighbour_pixels: np.ndarray,
        scalar_edges: np.ndarray,
        cancel: Optional[CancelCallback],
    ) -> Tuple[Optional[np.ndarray], bool]:
        """Build a consistent cost-to-go bound by reverse scalar Dijkstra.

        The scalar graph uses every path term except curvature.  Its exact
        distance is therefore a lower bound on the direction-state objective.
        Search stops once the start is finalized.  Finalized pixels retain
        their exact scalar distance; every remaining pixel receives the start
        distance, which is also a valid and consistent lower bound because
        Dijkstra has proved that no unfinalized distance is smaller.
        """

        pixel_count = neighbour_pixels.shape[1]
        distance = np.full(pixel_count, np.inf, dtype=np.float64)
        settled = np.zeros(pixel_count, dtype=bool)
        distance[end_pixel_index] = 0.0
        frontier = [(0.0, end_pixel_index)]
        expanded = 0
        cancellation_interval = self.config.cancellation_interval

        while frontier:
            queued_cost, pixel_index = heapq.heappop(frontier)
            if settled[pixel_index] or queued_cost > float(distance[pixel_index]):
                continue
            settled[pixel_index] = True
            expanded += 1
            if pixel_index == start_pixel_index:
                cutoff = queued_cost
                heuristic = np.full(pixel_count, cutoff, dtype=np.float64)
                heuristic[settled] = distance[settled]
                return heuristic, False
            if cancel is not None and expanded % cancellation_interval == 0 and cancel():
                return None, True

            for direction in range(self.direction_count):
                next_pixel_index = int(neighbour_pixels[direction, pixel_index])
                if next_pixel_index < 0 or settled[next_pixel_index]:
                    continue
                tentative = queued_cost + float(scalar_edges[direction, pixel_index])
                if tentative >= float(distance[next_pixel_index]):
                    continue
                distance[next_pixel_index] = tentative
                heapq.heappush(frontier, (tentative, next_pixel_index))
        return None, False

    def _scalar_path_upper_bound(
        self,
        start_pixel_index: int,
        end_pixel_index: int,
        neighbour_pixels: np.ndarray,
        scalar_edges: np.ndarray,
        scalar_heuristic: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Recover a feasible scalar-optimal path and evaluate full curvature cost."""

        path = [start_pixel_index]
        current = start_pixel_index
        incoming_direction: Optional[int] = None
        full_cost = 0.0
        while current != end_pixel_index:
            best_value = math.inf
            best_edge = math.inf
            best_pixel = -1
            best_direction = -1
            for direction in range(self.direction_count):
                next_pixel = int(neighbour_pixels[direction, current])
                if next_pixel < 0:
                    continue
                edge = float(scalar_edges[direction, current])
                value = edge + float(scalar_heuristic[next_pixel])
                if value < best_value:
                    best_value = value
                    best_edge = edge
                    best_pixel = next_pixel
                    best_direction = direction
            if (
                best_pixel < 0
                or best_value > float(scalar_heuristic[current]) + 1.0e-6
                or len(path) > neighbour_pixels.shape[1]
            ):
                # The reverse Dijkstra result guarantees this should be
                # unreachable, but an infinite bound keeps failure safe.
                return np.asarray(path, dtype=np.int32), math.inf
            full_cost += best_edge
            if incoming_direction is not None:
                full_cost += self.config.curvature_weight * float(
                    self._turn_penalty[incoming_direction, best_direction]
                )
            incoming_direction = best_direction
            current = best_pixel
            path.append(current)
        return np.asarray(path, dtype=np.int32), full_cost

    def _directional_lookahead_heuristic(
        self,
        scalar_heuristic: np.ndarray,
        neighbour_pixels: np.ndarray,
        scalar_edges: np.ndarray,
        end_pixel_index: int,
    ) -> np.ndarray:
        """Add exact curvature-aware Bellman backups to the lower bound.

        Starting from a consistent scalar cost-to-go, each backup includes one
        additional transition's curvature.  The sequence is monotone and every
        member remains consistent and admissible for direction-state A*.
        """

        pixel_count = neighbour_pixels.shape[1]
        previous = np.broadcast_to(
            scalar_heuristic[None, :], (self.direction_count, pixel_count)
        )
        curvature = self.config.curvature_weight * self._turn_penalty
        for _ in range(self._heuristic_lookahead_steps):
            current = np.full(
                (self.direction_count, pixel_count), np.inf, dtype=np.float64
            )
            for next_direction in range(self.direction_count):
                targets = neighbour_pixels[next_direction]
                valid = targets >= 0
                source = np.flatnonzero(valid)
                downstream = previous[next_direction, targets[valid]]
                base = scalar_edges[next_direction, source].astype(np.float64) + downstream
                for incoming_direction in range(self.direction_count):
                    candidate = base + float(
                        curvature[incoming_direction, next_direction]
                    )
                    current[incoming_direction, source] = np.minimum(
                        current[incoming_direction, source], candidate
                    )
            current[:, end_pixel_index] = 0.0
            previous = current
        return np.asarray(previous)

    @staticmethod
    def _geometric_heuristic_vector(
        x: np.ndarray,
        y: np.ndarray,
        end_x: int,
        end_y: int,
        minimum_step_cost: float,
    ) -> np.ndarray:
        """Return the admissible octile bound for packed corridor pixels."""

        delta_x = np.abs(x.astype(np.int64) - end_x)
        delta_y = np.abs(y.astype(np.int64) - end_y)
        diagonal = np.minimum(delta_x, delta_y)
        axial = np.maximum(delta_x, delta_y) - diagonal
        return minimum_step_cost * (axial + float(_STEP_LENGTH[1]) * diagonal)

    def _heuristic(
        self,
        x: int,
        y: int,
        end_x: int,
        end_y: int,
        minimum_step_cost: Optional[float] = None,
    ) -> float:
        delta_x = abs(end_x - x)
        delta_y = abs(end_y - y)
        diagonal = min(delta_x, delta_y)
        axial = max(delta_x, delta_y) - diagonal
        octile_distance = axial + float(_STEP_LENGTH[1]) * diagonal
        # Every transition costs at least the global minimum surface cost.  The
        # fallback retains the same proof using the strictly positive base term.
        lower_bound = (
            self.config.base_cost if minimum_step_cost is None else minimum_step_cost
        )
        return lower_bound * octile_distance

    def _reconstruct_path(
        self,
        parents: np.ndarray,
        goal_state: int,
        corridor_x: np.ndarray,
        corridor_y: np.ndarray,
        x_offset: int,
        y_offset: int,
    ) -> np.ndarray:
        reversed_path = []
        state = int(goal_state)
        while state >= 0:
            pixel_index = (state // self.direction_count)
            x = int(corridor_x[pixel_index])
            y = int(corridor_y[pixel_index])
            reversed_path.append((x + x_offset, y + y_offset))
            state = int(parents[state])
        reversed_path.reverse()
        return np.asarray(reversed_path, dtype=np.int32)


def simplify_polyline(points_xy: np.ndarray, tolerance: float = 1.0) -> np.ndarray:
    """Simplify an ``(n, 2)`` polyline while preserving its endpoints.

    This iterative Ramer-Douglas-Peucker implementation avoids recursion depth
    failures on long, full-resolution traces.
    """

    points = np.asarray(points_xy)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape (n, 2)")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    if len(points) <= 2 or tolerance == 0.0:
        return np.array(points, copy=True)
    work = points.astype(np.float64, copy=False)
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = True
    keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        segment = work[end] - work[start]
        squared_length = float(np.dot(segment, segment))
        candidates = work[start + 1 : end]
        if squared_length == 0.0:
            distances = np.linalg.norm(candidates - work[start], axis=1)
        else:
            offsets = candidates - work[start]
            projection = np.clip(offsets @ segment / squared_length, 0.0, 1.0)
            nearest = work[start] + projection[:, None] * segment
            distances = np.linalg.norm(candidates - nearest, axis=1)
        relative_index = int(np.argmax(distances))
        maximum_distance = float(distances[relative_index])
        if maximum_distance > tolerance:
            split = start + 1 + relative_index
            keep[split] = True
            stack.append((start, split))
            stack.append((split, end))
    return np.array(points[keep], copy=True)
