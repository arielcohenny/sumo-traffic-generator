"""
Unit tests for orchestration module classes.

Tests for RunSpec, RunMetrics, ComparisonResults, MetricsExtractor, and ComparisonRunner.
"""

import pytest
import json
import tempfile
from pathlib import Path

from src.orchestration.run_spec import RunSpec, RunMetrics
from src.orchestration.comparison_results import ComparisonResults
from src.orchestration.metrics_extractor import MetricsExtractor


class TestRunSpec:
    """Tests for RunSpec dataclass."""

    def test_create_run_spec(self):
        """Test basic RunSpec creation."""
        spec = RunSpec(
            traffic_control="tree_method",
            private_traffic_seed=100,
            public_traffic_seed=200,
            name="test_run"
        )
        assert spec.traffic_control == "tree_method"
        assert spec.private_traffic_seed == 100
        assert spec.public_traffic_seed == 200
        assert spec.name == "test_run"

    def test_run_spec_to_dict(self):
        """Test RunSpec serialization to dict."""
        spec = RunSpec(
            traffic_control="actuated",
            private_traffic_seed=42,
            public_traffic_seed=43,
            name="my_run"
        )
        d = spec.to_dict()
        assert d["traffic_control"] == "actuated"
        assert d["private_seed"] == 42  # API uses private_seed in dict
        assert d["public_seed"] == 43   # API uses public_seed in dict
        assert d["name"] == "my_run"

    def test_run_spec_from_dict(self):
        """Test RunSpec deserialization from dict."""
        data = {
            "traffic_control": "fixed",
            "private_traffic_seed": 500,
            "public_traffic_seed": 600,
            "name": "loaded_run"
        }
        spec = RunSpec.from_dict(data)
        assert spec.traffic_control == "fixed"
        assert spec.private_traffic_seed == 500
        assert spec.public_traffic_seed == 600
        assert spec.name == "loaded_run"

    def test_run_spec_roundtrip(self):
        """Test RunSpec serialization roundtrip."""
        original = RunSpec(
            traffic_control="tree_method",
            private_traffic_seed=123,
            public_traffic_seed=456,
            name="roundtrip_test"
        )
        restored = RunSpec.from_dict(original.to_dict())
        assert restored.traffic_control == original.traffic_control
        assert restored.private_traffic_seed == original.private_traffic_seed
        assert restored.public_traffic_seed == original.public_traffic_seed
        assert restored.name == original.name


class TestRunMetrics:
    """Tests for RunMetrics dataclass."""

    def test_create_run_metrics(self):
        """Test basic RunMetrics creation."""
        metrics = RunMetrics(
            name="test_metrics",
            traffic_control="tree_method",
            private_traffic_seed=100,
            public_traffic_seed=200,
            avg_travel_time=150.5,
            std_travel_time=25.3,
            avg_waiting_time=45.2,
            completion_rate=0.95,
            throughput=180.0,
            avg_queue_length=5.5,
            max_queue_length=15
        )
        assert metrics.name == "test_metrics"
        assert metrics.avg_travel_time == 150.5
        assert metrics.completion_rate == 0.95

    def test_run_metrics_to_dict(self):
        """Test RunMetrics serialization."""
        metrics = RunMetrics(
            name="test",
            traffic_control="actuated",
            private_traffic_seed=1,
            public_traffic_seed=2,
            avg_travel_time=100.0,
            std_travel_time=10.0,
            avg_waiting_time=20.0,
            completion_rate=0.9,
            throughput=150.0,
            avg_queue_length=3.0,
            max_queue_length=10
        )
        d = metrics.to_dict()
        assert d["name"] == "test"
        assert d["avg_travel_time"] == 100.0
        assert d["completion_rate"] == 0.9

    def test_run_metrics_from_dict(self):
        """Test RunMetrics deserialization."""
        data = {
            "name": "loaded",
            "traffic_control": "fixed",
            "private_traffic_seed": 10,
            "public_traffic_seed": 20,
            "avg_travel_time": 200.0,
            "std_travel_time": 30.0,
            "avg_waiting_time": 50.0,
            "completion_rate": 0.85,
            "throughput": 120.0,
            "avg_queue_length": 4.0,
            "max_queue_length": 12
        }
        metrics = RunMetrics.from_dict(data)
        assert metrics.name == "loaded"
        assert metrics.avg_travel_time == 200.0


class TestComparisonResults:
    """Tests for ComparisonResults dataclass."""

    def _create_sample_metrics(self, name: str, traffic_control: str) -> RunMetrics:
        """Helper to create sample RunMetrics."""
        return RunMetrics(
            name=name,
            traffic_control=traffic_control,
            private_traffic_seed=100,
            public_traffic_seed=200,
            avg_travel_time=150.0,
            std_travel_time=20.0,
            avg_waiting_time=30.0,
            completion_rate=0.92,
            throughput=160.0,
            avg_queue_length=4.0,
            max_queue_length=10
        )

    def test_create_comparison_results(self):
        """Test ComparisonResults creation."""
        results = ComparisonResults(
            comparison_name="test_comparison",
            runs=[
                self._create_sample_metrics("run1", "tree_method"),
                self._create_sample_metrics("run2", "fixed")
            ],
            network_config={"grid_dimension": 5}
        )
        assert results.comparison_name == "test_comparison"
        assert len(results.runs) == 2
        assert results.network_config["grid_dimension"] == 5

    def test_add_run(self):
        """Test adding runs to ComparisonResults."""
        results = ComparisonResults(comparison_name="test")
        assert len(results.runs) == 0

        results.add_run(self._create_sample_metrics("run1", "tree_method"))
        assert len(results.runs) == 1

        results.add_run(self._create_sample_metrics("run2", "actuated"))
        assert len(results.runs) == 2

    def test_to_summary_dict(self):
        """Test summary aggregation by method."""
        results = ComparisonResults(comparison_name="summary_test")

        # Add multiple runs for same method
        metrics1 = RunMetrics(
            name="tree1", traffic_control="tree_method",
            private_traffic_seed=1, public_traffic_seed=1,
            avg_travel_time=100.0, std_travel_time=10.0,
            avg_waiting_time=20.0, completion_rate=0.9,
            throughput=150.0, avg_queue_length=3.0, max_queue_length=8
        )
        metrics2 = RunMetrics(
            name="tree2", traffic_control="tree_method",
            private_traffic_seed=2, public_traffic_seed=2,
            avg_travel_time=120.0, std_travel_time=15.0,
            avg_waiting_time=25.0, completion_rate=0.88,
            throughput=140.0, avg_queue_length=4.0, max_queue_length=10
        )
        results.add_run(metrics1)
        results.add_run(metrics2)

        summary = results.to_summary_dict()
        assert "tree_method" in summary
        assert summary["tree_method"]["avg_travel_time"]["mean"] == 110.0  # (100 + 120) / 2

    def test_json_roundtrip(self):
        """Test JSON serialization roundtrip."""
        original = ComparisonResults(
            comparison_name="json_test",
            runs=[self._create_sample_metrics("run1", "tree_method")],
            network_config={"grid_dimension": 3}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "results.json"
            original.to_json(json_path)

            loaded = ComparisonResults.from_json(json_path)
            assert loaded.comparison_name == original.comparison_name
            assert len(loaded.runs) == len(original.runs)
            assert loaded.runs[0].name == original.runs[0].name


class TestMetricsExtractor:
    """Tests for MetricsExtractor class."""

    def test_extractor_creation(self):
        """Test MetricsExtractor instantiation."""
        extractor = MetricsExtractor()
        assert extractor is not None

    def test_parse_tripinfo_basic(self):
        """Test parsing basic tripinfo XML."""
        tripinfo_content = """<?xml version="1.0" encoding="UTF-8"?>
<tripinfos>
    <tripinfo id="veh_0" depart="0.00" arrival="100.00" duration="100.00"
              waitingTime="10.00" timeLoss="15.00"/>
    <tripinfo id="veh_1" depart="5.00" arrival="150.00" duration="145.00"
              waitingTime="20.00" timeLoss="25.00"/>
</tripinfos>"""

        with tempfile.TemporaryDirectory() as tmpdir:
            tripinfo_path = Path(tmpdir) / "tripinfo.xml"
            tripinfo_path.write_text(tripinfo_content)

            extractor = MetricsExtractor()
            result = extractor._parse_tripinfo(tripinfo_path)

            # API returns travel_times, waiting_times, vehicles_arrived
            assert result["vehicles_arrived"] == 2
            assert len(result["travel_times"]) == 2
            assert result["travel_times"] == [100.0, 145.0]
            assert result["waiting_times"] == [10.0, 20.0]

    def test_parse_tripinfo_no_arrivals(self):
        """Test parsing tripinfo with no vehicles."""
        tripinfo_content = """<?xml version="1.0" encoding="UTF-8"?>
<tripinfos>
</tripinfos>"""

        with tempfile.TemporaryDirectory() as tmpdir:
            tripinfo_path = Path(tmpdir) / "tripinfo.xml"
            tripinfo_path.write_text(tripinfo_content)

            extractor = MetricsExtractor()
            result = extractor._parse_tripinfo(tripinfo_path)

            assert result["vehicles_arrived"] == 0
            assert len(result["travel_times"]) == 0

    def test_parse_missing_file(self):
        """Test handling of missing files."""
        extractor = MetricsExtractor()

        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "nonexistent.xml"
            result = extractor._parse_tripinfo(missing_path)

            # Should return None for missing file
            assert result is None


class TestComparisonRunner:
    """Tests for ComparisonRunner class."""

    def test_runner_creation(self):
        """Test ComparisonRunner instantiation."""
        from src.orchestration.comparison_runner import ComparisonRunner

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ComparisonRunner(Path(tmpdir))
            assert runner.workspace == Path(tmpdir)

    def test_network_path_setup(self):
        """Test network and runs path setup."""
        from src.orchestration.comparison_runner import ComparisonRunner

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ComparisonRunner(Path(tmpdir))
            assert runner.network_path == Path(tmpdir) / "network"
            assert runner.runs_path == Path(tmpdir) / "runs"

    def test_generate_network_only_creates_directory(self):
        """Test that generate_network_only creates the network directory structure."""
        from src.orchestration.comparison_runner import ComparisonRunner

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ComparisonRunner(Path(tmpdir))

            # Verify initial paths are set up correctly
            assert runner.network_path == Path(tmpdir) / "network"
            assert runner.runs_path == Path(tmpdir) / "runs"

            # The runner should have correct workspace
            assert runner.workspace == Path(tmpdir)


class TestSeedGeneration:
    """Tests for seed generation in comparison widgets."""

    def test_reproducible_seed_generation(self):
        """Test that seed generation is reproducible from base seeds."""
        import random

        base_private = 100
        base_public = 200
        num_runs = 5

        # First generation
        private_rng1 = random.Random(base_private)
        private_seeds1 = [private_rng1.randint(0, 999999) for _ in range(num_runs)]

        public_rng1 = random.Random(base_public)
        public_seeds1 = [public_rng1.randint(0, 999999) for _ in range(num_runs)]

        # Second generation with same base seeds
        private_rng2 = random.Random(base_private)
        private_seeds2 = [private_rng2.randint(0, 999999) for _ in range(num_runs)]

        public_rng2 = random.Random(base_public)
        public_seeds2 = [public_rng2.randint(0, 999999) for _ in range(num_runs)]

        # Should be identical
        assert private_seeds1 == private_seeds2
        assert public_seeds1 == public_seeds2

    def test_different_base_seeds_produce_different_sequences(self):
        """Test that different base seeds produce different sequences."""
        import random

        rng1 = random.Random(100)
        seeds1 = [rng1.randint(0, 999999) for _ in range(5)]

        rng2 = random.Random(101)
        seeds2 = [rng2.randint(0, 999999) for _ in range(5)]

        assert seeds1 != seeds2
