from unittest.mock import patch
import unittest
import src.cli
import os
import sys
# Ensure project root is on PYTHONPATH so 'src' can be imported
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class TestCLIIntegration(unittest.TestCase):
    @patch('src.cli.traci')
    @patch('src.cli.generate_grid_network')
    @patch('src.cli.extract_zones_from_junctions')
    @patch('src.cli.set_lane_counts')
    @patch('src.cli.assign_edge_attractiveness')
    @patch('src.cli.generate_vehicle_routes')
    @patch('src.cli.generate_sumo_conf_file', return_value='dummy.sumocfg')
    @patch('src.cli.load_tree', return_value=('tree_data', 'run_config'))
    @patch('src.cli.compute_phases', return_value={'TLS_1': 'GGrr'})
    def test_main_runs_through(
        self,
        mock_compute_phases,
        mock_load_tree,
        mock_conf,
        mock_routes,
        mock_attr,
        mock_lanes,
        mock_zones,
        mock_generate,
        mock_traci
    ):
        # Prepare test arguments for a short run
        test_args = [
            'cli.py',
            '--grid_dimension', '2',
            '--block_size_m', '100',
            '--blocks_to_remove', '0',
            '--num_vehicles', '1',
            '--step-length', '1.0',
            '--end-time', '2'
        ]
        with patch.object(sys, 'argv', test_args):
            from src.cli import main
            main()

        # Verify each major step was invoked once
        mock_generate.assert_called_once()
        mock_zones.assert_called_once()
        mock_lanes.assert_called_once()
        mock_attr.assert_called_once()
        mock_routes.assert_called_once()
        mock_conf.assert_called_once()
        mock_load_tree.assert_called_once()

        # And that TraCI was started, stepped, and closed
        mock_traci.start.assert_called_once()
        self.assertTrue(mock_traci.simulationStep.called)
        mock_traci.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
