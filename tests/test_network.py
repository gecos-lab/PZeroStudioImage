"""Topology and serialization tests for the fracture network model."""

from __future__ import annotations

import json
import unittest

import numpy as np

from domstudio.evaluation import polyline_length
from domstudio.network import FractureNetwork


class FractureNetworkTopologyTests(unittest.TestCase):
    def test_near_branch_endpoints_form_one_degree_three_junction(self) -> None:
        traces = [
            np.asarray(((0.0, 10.0), (10.0, 10.0))),
            np.asarray(((10.0, 0.0), (10.2, 10.1))),
            np.asarray(((10.1, 10.2), (20.0, 20.0))),
        ]
        network = FractureNetwork.from_traces(traces, snap_tolerance=0.5)

        graph = network.build_graph()
        summary = graph.summary()
        junction = next(node for node in graph.nodes if node.degree == 3)

        self.assertEqual(len(graph.nodes), 4)
        self.assertEqual(len(graph.edges), 3)
        self.assertEqual(junction.clustered_endpoint_count, 3)
        self.assertEqual(len(junction.incident_edge_ids), 3)
        self.assertTrue(np.allclose(junction.coordinate, (10.1, 10.1)))
        self.assertEqual(summary.trace_count, 3)
        self.assertEqual(summary.edge_count, 3)
        self.assertEqual(summary.node_count, 4)
        self.assertEqual(summary.endpoint_count, 3)
        self.assertEqual(summary.junction_count, 1)
        self.assertEqual(summary.connected_component_count, 1)
        self.assertEqual(summary.closed_cycle_count, 0)
        expected_length = sum(polyline_length(trace) for trace in traces)
        self.assertAlmostEqual(summary.total_length, expected_length)
        self.assertAlmostEqual(summary.mean_length, expected_length / 3.0)

    def test_zero_tolerance_still_clusters_exact_endpoints(self) -> None:
        network = FractureNetwork.from_traces(
            [
                ((0.0, 0.0), (1.0, 0.0)),
                ((1.0, 0.0), (2.0, 0.0)),
                ((10.0, 0.0), (11.0, 0.0)),
            ],
            snap_tolerance=0.0,
        )

        graph = network.build_graph()
        shared = next(node for node in graph.nodes if node.coordinate == (1.0, 0.0))

        self.assertEqual(shared.degree, 2)
        self.assertEqual(shared.clustered_endpoint_count, 2)
        self.assertEqual(len(graph.nodes), 5)
        self.assertEqual(len(graph.components), 2)
        self.assertEqual(graph.summary().endpoint_count, 4)
        self.assertEqual(graph.summary().connected_component_count, 2)

    def test_self_loop_has_degree_two_and_adds_one_cycle(self) -> None:
        network = FractureNetwork.from_traces(
            [
                ((0.0, 0.0), (1.0, 0.0), (0.0, 0.0)),
                ((0.0, 0.0), (-1.0, 0.0)),
            ],
            snap_tolerance=0.0,
        )

        graph = network.build_graph()
        loop = graph.edges[0]
        loop_node = next(node for node in graph.nodes if node.coordinate == (0.0, 0.0))
        summary = graph.summary()

        self.assertTrue(loop.closed)
        self.assertEqual(loop.from_node_id, loop.to_node_id)
        self.assertEqual(loop_node.degree, 3)
        self.assertEqual(len(loop_node.incident_edge_ids), 2)
        self.assertEqual(summary.endpoint_count, 1)
        self.assertEqual(summary.junction_count, 1)
        self.assertEqual(summary.closed_cycle_count, 1)
        self.assertEqual(graph.components[0].closed_cycle_count, 1)
        self.assertFalse(graph.components[0].closed)

    def test_parallel_edges_add_one_independent_cycle(self) -> None:
        network = FractureNetwork.from_traces(
            [
                ((0.0, 0.0), (2.0, 0.0)),
                ((2.0, 0.0), (0.0, 0.0)),
            ],
            snap_tolerance=0.0,
        )

        graph = network.build_graph()

        self.assertEqual([node.degree for node in graph.nodes], [2, 2])
        self.assertEqual(graph.closed_cycle_count, 1)
        self.assertTrue(graph.components[0].closed)
        self.assertEqual(graph.edges[0].orientation_degrees, 0.0)
        self.assertEqual(graph.edges[1].orientation_degrees, 0.0)

    def test_analysis_is_deterministic_and_json_serializable(self) -> None:
        network = FractureNetwork.from_traces(
            [
                ((0.0, 0.0), (1.0, 1.0)),
                ((1.0, 1.0), (2.0, 1.0)),
            ],
            snap_tolerance=0.01,
        )

        first = network.analysis()
        second = network.to_dict()
        encoded = json.dumps(first, allow_nan=False, sort_keys=True)

        self.assertEqual(first, second)
        self.assertEqual(encoded, json.dumps(second, allow_nan=False, sort_keys=True))
        self.assertEqual([edge["id"] for edge in first["edges"]], [
            "edge-000000",
            "edge-000001",
        ])
        first_edge = first["edges"][0]
        self.assertEqual(first_edge["from_node_id"], "node-000000")
        self.assertEqual(first_edge["to_node_id"], "node-000001")
        self.assertEqual(first_edge["component_id"], "component-000000")
        self.assertAlmostEqual(first_edge["orientation_degrees"], 45.0)
        self.assertFalse(first_edge["closed"])
        self.assertEqual(first["summary"]["edge_count"], 2)
        self.assertEqual(first["summary"]["mean_length"], (2.0**0.5 + 1.0) / 2.0)

    def test_non_finite_geometry_and_tolerance_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            FractureNetwork.from_traces([((0.0, 0.0), (np.nan, 1.0))])
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            FractureNetwork.from_traces([((0.0, 0.0), (1.0, 1.0))], snap_tolerance=np.nan)


if __name__ == "__main__":
    unittest.main()
