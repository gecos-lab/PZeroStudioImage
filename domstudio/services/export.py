"""Geospatially correct vector export for accepted fracture traces."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

from shapely.geometry import LineString

from domstudio.core import RasterDocument
from domstudio.network import FractureNetwork


class ExportError(RuntimeError):
    """Raised when traces cannot be exported."""


def _pixel_to_world(document: RasterDocument, point: Sequence[float]) -> tuple[float, float]:
    x, y = float(point[0]), float(point[1])
    if document.transform is None:
        return x, y
    try:
        world_x, world_y = document.pixel_to_world((x, y), center=True)
    except (TypeError, ValueError) as exc:
        raise ExportError("Unsupported affine transform representation.") from exc
    return float(world_x), float(world_y)


def export_traces(
    path: str | Path,
    traces: Iterable[Sequence[Sequence[float]]],
    document: RasterDocument,
) -> Path:
    """Export traces to GeoJSON, GPKG, ESRI Shapefile, or vertex CSV.

    Pixel centers are transformed through the original raster affine. When an
    ordinary image has no spatial reference, coordinates remain in pixel space
    and no CRS is assigned.
    """

    destination = Path(path).expanduser().resolve()
    pixel_coordinate_sets: list[list[tuple[float, float]]] = []
    coordinate_sets: list[list[tuple[float, float]]] = []
    pixel_lengths: list[float] = []
    coordinate_lengths: list[float] = []
    lines: list[LineString] = []
    for trace in traces:
        pixel_coordinates = [
            (float(point[0]), float(point[1])) for point in trace
        ]
        coordinates = [
            _pixel_to_world(document, point) for point in pixel_coordinates
        ]
        if len(coordinates) >= 2:
            pixel_line = LineString(pixel_coordinates)
            line = LineString(coordinates)
            if not line.is_empty and line.length > 0 and pixel_line.length > 0:
                pixel_coordinate_sets.append(pixel_coordinates)
                coordinate_sets.append(coordinates)
                pixel_lengths.append(float(pixel_line.length))
                coordinate_lengths.append(float(line.length))
                lines.append(line)
    if not lines:
        raise ExportError("There are no valid traces to export.")

    # Build topology in pixel space so the endpoint snap tolerance keeps its
    # documented pixel meaning even when the raster uses projected coordinates.
    graph = FractureNetwork.from_traces(pixel_coordinate_sets).build_graph()
    edge_attributes = {edge.trace_index: edge for edge in graph.edges}
    node_degrees = graph.degrees

    suffix = destination.suffix.lower()
    if suffix == ".csv":
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    (
                        "trace_id",
                        "vertex_id",
                        "x",
                        "y",
                        "length_px",
                        "length_coordinate_units",
                        "edge_id",
                        "from_node_id",
                        "to_node_id",
                        "component_id",
                        "from_degree",
                        "to_degree",
                        "orientation_degrees",
                        "closed",
                        "crs",
                    )
                )
                for trace_id, coordinates in enumerate(coordinate_sets, start=1):
                    edge = edge_attributes[trace_id - 1]
                    for vertex_id, (x, y) in enumerate(coordinates, start=1):
                        writer.writerow(
                            (
                                trace_id,
                                vertex_id,
                                x,
                                y,
                                pixel_lengths[trace_id - 1],
                                coordinate_lengths[trace_id - 1],
                                edge.id,
                                edge.from_node_id,
                                edge.to_node_id,
                                edge.component_id,
                                node_degrees[edge.from_node_id],
                                node_degrees[edge.to_node_id],
                                edge.orientation_degrees,
                                edge.closed,
                                str(document.crs) if document.crs else "pixel",
                            )
                        )
        except OSError as exc:
            raise ExportError(f"Failed to export {destination}: {exc}") from exc
        return destination

    if suffix not in {".geojson", ".json", ".gpkg", ".shp"}:
        destination = destination.with_suffix(".geojson")
        suffix = ".geojson"

    # GeoJSON has a small, stable schema and does not need the heavyweight GIS
    # writer stack. Keeping this path native also makes the default export
    # available in minimal/headless installations.
    if suffix in {".geojson", ".json"}:
        payload: dict[str, object] = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "trace_id": index,
                        "length_px": pixel_lengths[index - 1],
                        "length_crs": coordinate_lengths[index - 1],
                        "edge_id": edge_attributes[index - 1].id,
                        "from_node": edge_attributes[index - 1].from_node_id,
                        "to_node": edge_attributes[index - 1].to_node_id,
                        "component": edge_attributes[index - 1].component_id,
                        "from_degree": node_degrees[
                            edge_attributes[index - 1].from_node_id
                        ],
                        "to_degree": node_degrees[
                            edge_attributes[index - 1].to_node_id
                        ],
                        "orientation_deg": edge_attributes[
                            index - 1
                        ].orientation_degrees,
                        "closed": edge_attributes[index - 1].closed,
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [list(point) for point in coordinates],
                    },
                }
                for index, coordinates in enumerate(coordinate_sets, start=1)
            ],
        }
        if document.crs:
            payload["crs"] = {
                "type": "name",
                "properties": {"name": str(document.crs)},
            }
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as exc:
            raise ExportError(f"Failed to export {destination}: {exc}") from exc
        return destination

    try:
        import geopandas as gpd

        frame = gpd.GeoDataFrame(
            {
                "trace_id": list(range(1, len(lines) + 1)),
                "length_px": pixel_lengths,
                "length_crs": coordinate_lengths,
                "edge_id": [edge.id for edge in graph.edges],
                "from_node": [edge.from_node_id for edge in graph.edges],
                "to_node": [edge.to_node_id for edge in graph.edges],
                "component": [edge.component_id for edge in graph.edges],
                "from_deg": [
                    node_degrees[edge.from_node_id] for edge in graph.edges
                ],
                "to_deg": [node_degrees[edge.to_node_id] for edge in graph.edges],
                "orient_deg": [edge.orientation_degrees for edge in graph.edges],
                "closed": [edge.closed for edge in graph.edges],
            },
            geometry=lines,
            crs=document.crs,
        )
        if suffix == ".gpkg":
            driver = "GPKG"
        else:
            driver = "ESRI Shapefile"
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_file(destination, driver=driver)
    except Exception as exc:
        raise ExportError(f"Failed to export {destination}: {exc}") from exc
    return destination
