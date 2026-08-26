"""Deterministic topology and analysis for accepted fracture traces.

Each accepted polyline is one graph edge. Its first and last points are
clustered into graph nodes using the configured snap tolerance. Interior
polyline intersections are deliberately not inferred here: they must already
have been split into branch traces by the vectorization or review workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from domstudio.evaluation import polyline_length


def _trace_array(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or len(value) < 2:
        raise ValueError("Each trace must have shape (n, 2) with n >= 2.")
    if not np.all(np.isfinite(value)):
        raise ValueError("Trace coordinates must be finite.")
    return value


def _clean_coordinate(value: float) -> float:
    result = float(value)
    return 0.0 if result == 0.0 else result


def _axial_orientation_degrees(trace: np.ndarray) -> float | None:
    """Return the length-weighted axial mean orientation in ``[0, 180)``."""

    segments = np.diff(trace, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    nonzero = lengths > np.finfo(np.float64).eps
    if not np.any(nonzero):
        return None
    segments = segments[nonzero]
    lengths = lengths[nonzero]
    angles = np.arctan2(segments[:, 1], segments[:, 0])
    axis_x = float(np.sum(lengths * np.cos(2.0 * angles)))
    axis_y = float(np.sum(lengths * np.sin(2.0 * angles)))
    resultant = float(np.hypot(axis_x, axis_y))
    scale = max(float(lengths.sum()), 1.0)
    if resultant <= np.finfo(np.float64).eps * scale * 16.0:
        return None
    orientation = float(np.degrees(0.5 * np.arctan2(axis_y, axis_x)) % 180.0)
    return 0.0 if np.isclose(orientation, 180.0) else orientation


@dataclass(frozen=True, slots=True)
class NetworkNode:
    """A spatially clustered set of trace endpoints."""

    id: str
    coordinate: tuple[float, float]
    component_id: str
    degree: int
    clustered_endpoint_count: int
    incident_edge_ids: tuple[str, ...]
    neighbor_node_ids: tuple[str, ...]

    @property
    def x(self) -> float:
        return self.coordinate[0]

    @property
    def y(self) -> float:
        return self.coordinate[1]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "coordinate": [self.x, self.y],
            "x": self.x,
            "y": self.y,
            "component_id": self.component_id,
            "degree": self.degree,
            "clustered_endpoint_count": self.clustered_endpoint_count,
            "incident_edge_ids": list(self.incident_edge_ids),
            "neighbor_node_ids": list(self.neighbor_node_ids),
        }


@dataclass(frozen=True, slots=True)
class NetworkEdge:
    """One accepted fracture trace and its graph-topology attributes."""

    id: str
    trace_index: int
    from_node_id: str
    to_node_id: str
    component_id: str
    coordinates: tuple[tuple[float, float], ...]
    length: float
    orientation_degrees: float | None
    closed: bool

    @property
    def axial_orientation_degrees(self) -> float | None:
        return self.orientation_degrees

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "trace_index": self.trace_index,
            "from_node_id": self.from_node_id,
            "to_node_id": self.to_node_id,
            "component_id": self.component_id,
            "coordinates": [[x, y] for x, y in self.coordinates],
            "length": self.length,
            "orientation_degrees": self.orientation_degrees,
            "orientation_convention": "axial_0_180_degrees",
            "closed": self.closed,
        }


@dataclass(frozen=True, slots=True)
class NetworkComponent:
    """A connected component and its independent-cycle topology."""

    id: str
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    closed_cycle_count: int
    closed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "node_ids": list(self.node_ids),
            "edge_ids": list(self.edge_ids),
            "closed_cycle_count": self.closed_cycle_count,
            "closed": self.closed,
        }


@dataclass(frozen=True, slots=True)
class NetworkSummary:
    """Compact metrics retained for the existing UI plus topology metrics."""

    trace_count: int
    total_length: float
    endpoint_count: int
    junction_count: int
    units: str = "px"
    node_count: int = 0
    connected_component_count: int = 0
    closed_cycle_count: int = 0

    @property
    def edge_count(self) -> int:
        return self.trace_count

    @property
    def mean_length(self) -> float:
        return self.total_length / self.edge_count if self.edge_count else 0.0

    @property
    def component_count(self) -> int:
        return self.connected_component_count

    @property
    def cycle_count(self) -> int:
        return self.closed_cycle_count

    def to_dict(self) -> dict[str, object]:
        return {
            "trace_count": self.trace_count,
            "edge_count": self.edge_count,
            "total_length": self.total_length,
            "mean_length": self.mean_length,
            "node_count": self.node_count,
            "endpoint_count": self.endpoint_count,
            "junction_count": self.junction_count,
            "connected_component_count": self.connected_component_count,
            "closed_cycle_count": self.closed_cycle_count,
            "units": self.units,
        }


@dataclass(frozen=True, slots=True)
class NetworkGraph:
    """Deterministic multigraph derived from fracture-trace endpoints."""

    nodes: tuple[NetworkNode, ...]
    edges: tuple[NetworkEdge, ...]
    adjacency: Mapping[str, tuple[str, ...]]
    components: tuple[NetworkComponent, ...]
    units: str = "px"

    @property
    def degrees(self) -> dict[str, int]:
        return {node.id: node.degree for node in self.nodes}

    @property
    def closed_cycle_count(self) -> int:
        return sum(component.closed_cycle_count for component in self.components)

    def summary(self) -> NetworkSummary:
        total_length = float(sum(edge.length for edge in self.edges))
        return NetworkSummary(
            trace_count=len(self.edges),
            total_length=total_length,
            endpoint_count=sum(node.degree == 1 for node in self.nodes),
            junction_count=sum(node.degree >= 3 for node in self.nodes),
            units=self.units,
            node_count=len(self.nodes),
            connected_component_count=len(self.components),
            closed_cycle_count=self.closed_cycle_count,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "summary": self.summary().to_dict(),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "adjacency": {
                node_id: list(neighbor_ids)
                for node_id, neighbor_ids in self.adjacency.items()
            },
            "degrees": self.degrees,
            "components": [component.to_dict() for component in self.components],
        }


@dataclass(frozen=True, slots=True)
class _EdgeSeed:
    id: str
    trace_index: int
    from_node_id: str
    to_node_id: str
    coordinates: tuple[tuple[float, float], ...]
    length: float
    orientation_degrees: float | None
    closed: bool


@dataclass(slots=True)
class FractureNetwork:
    traces: list[np.ndarray]
    snap_tolerance: float = 3.0

    def __post_init__(self) -> None:
        self.traces = [_trace_array(trace) for trace in self.traces]
        self.snap_tolerance = float(self.snap_tolerance)
        if not np.isfinite(self.snap_tolerance) or self.snap_tolerance < 0:
            raise ValueError("snap_tolerance must be finite and non-negative")

    @classmethod
    def from_traces(
        cls,
        traces: Iterable[Sequence[Sequence[float]] | np.ndarray],
        *,
        snap_tolerance: float = 3.0,
    ) -> "FractureNetwork":
        return cls(list(traces), snap_tolerance=snap_tolerance)

    def _cluster_endpoints(self) -> tuple[list[tuple[float, float]], list[int], list[int]]:
        """Return sorted centroids, endpoint-to-node indices, and cluster sizes."""

        if not self.traces:
            return [], [], []
        endpoints = np.vstack([trace[[0, -1]] for trace in self.traces])
        parent = list(range(len(endpoints)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root == right_root:
                return
            lower, upper = sorted((left_root, right_root))
            parent[upper] = lower

        if len(endpoints) > 1:
            pairs = cKDTree(endpoints).query_pairs(self.snap_tolerance)
            for left, right in sorted(pairs):
                union(int(left), int(right))

        clusters: dict[int, list[int]] = {}
        for endpoint_index in range(len(endpoints)):
            clusters.setdefault(find(endpoint_index), []).append(endpoint_index)

        records: list[tuple[tuple[float, float], tuple[tuple[float, float], ...], list[int]]] = []
        for members in clusters.values():
            member_points = endpoints[members]
            centroid = tuple(_clean_coordinate(value) for value in member_points.mean(axis=0))
            signature = tuple(
                sorted(
                    (_clean_coordinate(point[0]), _clean_coordinate(point[1]))
                    for point in member_points
                )
            )
            records.append((centroid, signature, members))
        records.sort(key=lambda record: (record[0], record[1]))

        endpoint_nodes = [0] * len(endpoints)
        coordinates: list[tuple[float, float]] = []
        cluster_sizes: list[int] = []
        for node_index, (centroid, _, members) in enumerate(records):
            coordinates.append(centroid)
            cluster_sizes.append(len(members))
            for endpoint_index in members:
                endpoint_nodes[endpoint_index] = node_index
        return coordinates, endpoint_nodes, cluster_sizes

    def build_graph(self) -> NetworkGraph:
        """Build a deterministic undirected multigraph without external graph libraries."""

        coordinates, endpoint_nodes, cluster_sizes = self._cluster_endpoints()
        if not coordinates:
            return NetworkGraph(nodes=(), edges=(), adjacency={}, components=())

        node_ids = [f"node-{index:06d}" for index in range(len(coordinates))]
        degrees = {node_id: 0 for node_id in node_ids}
        incident_edges: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        neighbors: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
        edge_seeds: list[_EdgeSeed] = []

        for trace_index, trace in enumerate(self.traces):
            edge_id = f"edge-{trace_index:06d}"
            from_node_id = node_ids[endpoint_nodes[2 * trace_index]]
            to_node_id = node_ids[endpoint_nodes[2 * trace_index + 1]]
            closed = from_node_id == to_node_id
            coordinates_value = tuple(
                (_clean_coordinate(point[0]), _clean_coordinate(point[1]))
                for point in trace
            )
            edge_seeds.append(
                _EdgeSeed(
                    id=edge_id,
                    trace_index=trace_index,
                    from_node_id=from_node_id,
                    to_node_id=to_node_id,
                    coordinates=coordinates_value,
                    length=float(polyline_length(trace)),
                    orientation_degrees=_axial_orientation_degrees(trace),
                    closed=closed,
                )
            )
            incident_edges[from_node_id].append(edge_id)
            if closed:
                degrees[from_node_id] += 2
                neighbors[from_node_id].add(from_node_id)
            else:
                incident_edges[to_node_id].append(edge_id)
                degrees[from_node_id] += 1
                degrees[to_node_id] += 1
                neighbors[from_node_id].add(to_node_id)
                neighbors[to_node_id].add(from_node_id)

        component_nodes: list[tuple[str, ...]] = []
        visited: set[str] = set()
        for start in node_ids:
            if start in visited:
                continue
            pending = [start]
            members: list[str] = []
            visited.add(start)
            while pending:
                current = pending.pop()
                members.append(current)
                for neighbor in sorted(neighbors[current], reverse=True):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        pending.append(neighbor)
            component_nodes.append(tuple(sorted(members)))

        node_component: dict[str, str] = {}
        components: list[NetworkComponent] = []
        for component_index, member_node_ids in enumerate(component_nodes):
            component_id = f"component-{component_index:06d}"
            member_set = set(member_node_ids)
            edge_ids = tuple(
                seed.id for seed in edge_seeds if seed.from_node_id in member_set
            )
            cycle_count = max(0, len(edge_ids) - len(member_node_ids) + 1)
            closed = cycle_count > 0 and all(
                degrees[node_id] == 2 for node_id in member_node_ids
            )
            components.append(
                NetworkComponent(
                    id=component_id,
                    node_ids=member_node_ids,
                    edge_ids=edge_ids,
                    closed_cycle_count=cycle_count,
                    closed=closed,
                )
            )
            for node_id in member_node_ids:
                node_component[node_id] = component_id

        nodes = tuple(
            NetworkNode(
                id=node_id,
                coordinate=coordinates[node_index],
                component_id=node_component[node_id],
                degree=degrees[node_id],
                clustered_endpoint_count=cluster_sizes[node_index],
                incident_edge_ids=tuple(sorted(incident_edges[node_id])),
                neighbor_node_ids=tuple(sorted(neighbors[node_id])),
            )
            for node_index, node_id in enumerate(node_ids)
        )
        edges = tuple(
            NetworkEdge(
                id=seed.id,
                trace_index=seed.trace_index,
                from_node_id=seed.from_node_id,
                to_node_id=seed.to_node_id,
                component_id=node_component[seed.from_node_id],
                coordinates=seed.coordinates,
                length=seed.length,
                orientation_degrees=seed.orientation_degrees,
                closed=seed.closed,
            )
            for seed in edge_seeds
        )
        adjacency = {node.id: node.neighbor_node_ids for node in nodes}
        return NetworkGraph(
            nodes=nodes,
            edges=edges,
            adjacency=adjacency,
            components=tuple(components),
        )

    def graph(self) -> NetworkGraph:
        """Compatibility-friendly shorthand for :meth:`build_graph`."""

        return self.build_graph()

    def _endpoint_degrees(self) -> list[int]:
        return [node.degree for node in self.build_graph().nodes]

    def summary(self) -> NetworkSummary:
        return self.build_graph().summary()

    def analysis(self) -> dict[str, object]:
        """Return complete JSON-serializable topology and measurement data."""

        return self.build_graph().to_dict()

    def to_dict(self) -> dict[str, object]:
        return self.analysis()
