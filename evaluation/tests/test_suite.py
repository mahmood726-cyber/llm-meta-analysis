"""
Comprehensive Test Suite for LLM Meta-Analysis

This module provides unit and integration tests for the meta-analysis system.
"""

import pytest
import os
import json
import tempfile
import shutil
from typing import Dict, List
from unittest.mock import Mock, MagicMock, patch

# Test configuration
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output")


class TestModelInterface:
    """Tests for the Model abstract class and implementations"""

    @pytest.fixture
    def sample_model_output(self):
        return "The answer is A"

    def test_model_abstract_class(self):
        """Test that Model cannot be instantiated directly"""
        from models.model import Model
        with pytest.raises(TypeError):
            Model()

    def test_gpt35_model_interface(self):
        """Test GPT-3.5 model interface"""
        from models.gpt35 import GPT35

        model = GPT35()
        assert model.get_context_length() == 16385
        assert model.encode_text("test text") is not None

    def test_claude_model_interface(self):
        """Test Claude model interface"""
        from models.claude import Claude

        model = Claude()
        assert model.get_context_length() == 200000
        assert model.encode_text("test text") is not None

    def test_llama3_model_interface(self):
        """Test Llama 3 model interface"""
        # Note: This test would require actual model loading
        # For unit testing, we might mock the model loading
        from models.llama3 import Llama38B

        # Test that the class can be instantiated
        # (actual model loading would require transformers and torch)
        try:
            model = Llama38B()
            assert model.get_context_length() == 8192
        except Exception as e:
            # Model loading might fail without GPU or model files
            pytest.skip(f"Model loading skipped: {e}")


class TestUtils:
    """Tests for utility functions"""

    def test_load_json_file(self):
        """Test JSON file loading"""
        from utils import load_json_file, save_json_file

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = [{"test": "data"}]
            json.dump(test_data, f)
            temp_path = f.name

        try:
            result = load_json_file(temp_path)
            assert result == test_data
        finally:
            os.unlink(temp_path)

    def test_save_json_file(self):
        """Test JSON file saving"""
        from utils import save_json_file, load_json_file

        test_data = {"test": "data", "number": 123}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            save_json_file(temp_path, test_data)
            result = load_json_file(temp_path)
            assert result == test_data
        finally:
            os.unlink(temp_path)

    def test_convert_character_to_string_outcome_type(self):
        """Test outcome type conversion"""
        from utils import convert_character_to_string_outcome_type

        assert convert_character_to_string_outcome_type("A") == "binary"
        assert convert_character_to_string_outcome_type("B") == "continuous"
        assert convert_character_to_string_outcome_type("C") == "x"
        assert convert_character_to_string_outcome_type("invalid") == "x"

    def test_convert_string_to_character_outcome_type(self):
        """Test outcome type conversion (reverse)"""
        from utils import convert_string_to_character_outcome_type

        assert convert_string_to_character_outcome_type("binary") == "A"
        assert convert_string_to_character_outcome_type("continuous") == "B"
        assert convert_string_to_character_outcome_type("x") == "C"

    def test_calculate_log_odds_ratio(self):
        """Test log odds ratio calculation"""
        from utils import calculate_log_odds_ratio

        # Test case: 10/50 vs 20/50
        lor, var = calculate_log_odds_ratio(10, 20, 50, 50)

        assert lor is not None
        assert var is not None
        assert isinstance(lor, float)
        # LOR should be negative (intervention worse than comparator)
        assert lor < 0

    def test_calculate_standardized_mean_difference(self):
        """Test SMD calculation"""
        from utils import calculate_standardized_mean_difference

        # Test case
        smd, var = calculate_standardized_mean_difference(10.0, 8.0, 2.0, 2.5, 50, 50)

        assert smd is not None
        assert var is not None
        assert isinstance(smd, float)
        # SMD should be positive (intervention better than comparator)
        assert smd > 0

    def test_clean_yaml_output(self):
        """Test YAML output cleaning"""
        from utils import clean_yaml_output

        raw_output = """
        ```yaml
        intervention:
          events: 10
          group_size: 50
        comparator:
          events: 20
          group_size: 50
        ```
        """

        cleaned = clean_yaml_output(raw_output)
        assert "```" not in cleaned
        assert "yaml" not in cleaned
        assert "intervention:" in cleaned or "comparator:" in cleaned


class TestMetricsCalculator:
    """Tests for the MetricsCalculator class"""

    @pytest.fixture
    def sample_outcome_type_data(self):
        return [
            {"outcome_type": "binary", "outcome_type_output": "binary"},
            {"outcome_type": "binary", "outcome_type_output": "continuous"},
            {"outcome_type": "continuous", "outcome_type_output": "continuous"},
            {"outcome_type": "continuous", "outcome_type_output": "x"},
        ]

    @pytest.fixture
    def sample_binary_outcomes_data(self):
        return [
            {
                "intervention_events": "10", "intervention_group_size": "50",
                "comparator_events": "20", "comparator_group_size": "50",
                "intervention_events_output": "10", "intervention_group_size_output": "50",
                "comparator_events_output": "20", "comparator_group_size_output": "50",
            },
            {
                "intervention_events": "15", "intervention_group_size": "60",
                "comparator_events": "25", "comparator_group_size": "60",
                "intervention_events_output": "15", "intervention_group_size_output": "60",
                "comparator_events_output": "x", "comparator_group_size_output": "60",
            },
        ]

    def test_metrics_calculator_initialization(self):
        """Test MetricsCalculator initialization"""
        from calculate_metrics import MetricsCalculator

        calc = MetricsCalculator("outcome_type")
        assert calc.task == "outcome_type"

    def test_calculate_accuracy(self):
        """Test accuracy calculation"""
        from calculate_metrics import MetricsCalculator

        calc = MetricsCalculator("outcome_type")
        actual = ["binary", "continuous", "binary"]
        predicted = ["binary", "binary", "binary"]

        accuracy = calc._MetricsCalculator__calculate_accuracy(actual, predicted)
        assert accuracy == 2/3

    def test_outcome_type_metrics(self):
        """Test outcome type metrics calculation"""
        from calculate_metrics import MetricsCalculator

        data = [
            {"outcome_type": "binary", "outcome_type_output": "binary"},
            {"outcome_type": "binary", "outcome_type_output": "continuous"},
            {"outcome_type": "continuous", "outcome_type_output": "continuous"},
            {"outcome_type": "x", "outcome_type_output": "x"},
        ]

        calc = MetricsCalculator("outcome_type")
        metrics = calc.calculate_metrics(data)

        assert "exact_match_accuracy" in metrics
        assert "outcome_type_f_score" in metrics
        assert metrics["exact_match_accuracy"]["total"] == 0.75


class TestErrorAnalyzer:
    """Tests for the ErrorAnalyzer class"""

    @pytest.fixture
    def sample_output_data(self):
        return [
            {
                "pmcid": "1234567",
                "intervention": "Drug A",
                "comparator": "Placebo",
                "outcome": "Mortality",
                "outcome_type": "binary",
                "intervention_events": "10",
                "intervention_group_size": "50",
                "comparator_events": "20",
                "comparator_group_size": "50",
                "intervention_events_output": "10",
                "intervention_group_size_output": "50",
                "comparator_events_output": "15",  # Wrong!
                "comparator_group_size_output": "50",
            },
            {
                "pmcid": "7654321",
                "intervention": "Drug B",
                "comparator": "Placebo",
                "outcome": "Recovery",
                "outcome_type": "binary",
                "intervention_events": "30",
                "intervention_group_size": "100",
                "comparator_events": "20",
                "comparator_group_size": "100",
                "intervention_events_output": "x",
                "intervention_group_size_output": "100",
                "comparator_events_output": "20",
                "comparator_group_size_output": "100",
            },
        ]

    def test_error_analyzer_initialization(self, tmp_path):
        """Test ErrorAnalyzer initialization"""
        from error_analyzer import ErrorAnalyzer

        # Create temporary output file
        output_file = tmp_path / "test_output.json"
        test_data = [{"pmcid": "123", "outcome_type": "binary", "outcome_type_output": "continuous"}]
        with open(output_file, 'w') as f:
            json.dump(test_data, f)

        analyzer = ErrorAnalyzer("outcome_type", str(output_file))
        assert analyzer.task == "outcome_type"

    def test_classify_error(self):
        """Test error classification"""
        from error_analyzer import ErrorAnalyzer

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = [{"pmcid": "123", "outcome_type": "binary", "outcome_type_output": "continuous"}]
            json.dump(test_data, f)
            temp_path = f.name

        try:
            analyzer = ErrorAnalyzer("outcome_type", temp_path)

            # Test various error classifications
            assert analyzer._classify_error("binary", "x", "test") == "unknown_prediction"
            assert analyzer._classify_error(10, 15, "events") == "extraction_error"
        finally:
            os.unlink(temp_path)


class TestMetaAnalysisMethods:
    """Tests for the meta-analysis statistical methods"""

    def test_meta_analyzer_initialization(self):
        """Test MetaAnalyzer initialization"""
        from meta_analysis_methods import MetaAnalyzer

        analyzer = MetaAnalyzer()
        assert len(analyzer.studies) == 0
        assert analyzer.results is None

    def test_add_study(self):
        """Test adding a study"""
        from meta_analysis_methods import MetaAnalyzer, StudyResult

        analyzer = MetaAnalyzer()
        study = StudyResult(
            study_id="test1",
            pmcid="123",
            intervention="Drug A",
            comparator="Placebo",
            outcome="Mortality",
            intervention_events=10,
            intervention_total=50,
            comparator_events=20,
            comparator_total=50
        )

        analyzer.add_study(study)
        assert len(analyzer.studies) == 1

    def test_binary_effects_calculation(self):
        """Test binary effect size calculation"""
        from meta_analysis_methods import MetaAnalyzer, StudyResult

        analyzer = MetaAnalyzer()

        # Add two studies
        analyzer.add_study(StudyResult(
            study_id="study1", pmcid="123", intervention="A", comparator="B", outcome="Mortality",
            intervention_events=10, intervention_total=50,
            comparator_events=20, comparator_total=50
        ))

        analyzer.add_study(StudyResult(
            study_id="study2", pmcid="456", intervention="A", comparator="B", outcome="Mortality",
            intervention_events=15, intervention_total=60,
            comparator_events=25, comparator_total=60
        ))

        effects, variances = analyzer.calculate_binary_effects()

        assert len(effects) == 2
        assert len(variances) == 2
        assert all(v > 0 for v in variances)

    def test_meta_analysis_execution(self):
        """Test full meta-analysis"""
        from meta_analysis_methods import MetaAnalyzer, StudyResult

        analyzer = MetaAnalyzer()

        # Add multiple studies
        for i in range(5):
            analyzer.add_study(StudyResult(
                study_id=f"study{i}", pmcid=str(i), intervention="A", comparator="B", outcome="Mortality",
                intervention_events=10+i*5, intervention_total=50,
                comparator_events=20+i*5, comparator_total=50
            ))

        result = analyzer.analyze(method='dl', outcome_type='binary')

        assert result is not None
        assert result.pooled_effect is not None
        assert result.n_studies == 5
        assert result.heterogeneity_i2 >= 0


class TestDTAProIntegration:
    """Tests for DTA PRO integration"""

    def test_dta_integrator_initialization(self):
        """Test DTAProIntegrator initialization"""
        from dta_integration import DTAProIntegrator

        integrator = DTAProIntegrator()
        assert len(integrator.studies) == 0

    def test_convert_from_binary_outcomes(self):
        """Test conversion from binary outcomes to DTA format"""
        from dta_integration import DTAProIntegrator

        test_data = [
            {
                "pmcid": "123",
                "intervention": "Drug A",
                "comparator": "Placebo",
                "outcome": "Mortality",
                "intervention_events_output": "10",
                "intervention_group_size_output": "50",
                "comparator_events_output": "20",
                "comparator_group_size_output": "50",
            }
        ]

        integrator = DTAProIntegrator()
        integrator.convert_from_binary_outcomes(test_data)

        assert len(integrator.studies) == 1
        assert integrator.studies[0].true_positives == 10
        assert integrator.studies[0].false_positives == 20

    def test_diagnostic_accuracy_calculation(self):
        """Test diagnostic accuracy statistics"""
        from dta_integration import DTAProIntegrator, DTAProStudy

        integrator = DTAProIntegrator()
        integrator.studies.append(DTAProStudy(
            study_id="test", author="Test", year=2023,
            condition="Test", index_test="Test", reference_standard="Test", target_condition="Test",
            true_positives=10, false_positives=5,
            false_negatives=3, true_negatives=32,
            total_n=50
        ))

        df = integrator.calculate_diagnostic_accuracy()

        assert len(df) == 1
        assert 'sensitivity' in df.columns
        assert 'specificity' in df.columns
        assert df.iloc[0]['sensitivity'] == 10/13  # TP / (TP + FN)
        assert df.iloc[0]['specificity'] == 32/37  # TN / (TN + FP)


class TestRealTimeProcessor:
    """Tests for real-time processing"""

    def test_processor_initialization(self):
        """Test RealTimeProcessor initialization"""
        from realtime_processor import RealTimeProcessor
        from models.model import Model

        # Create a mock model
        mock_model = Mock(spec=Model)
        mock_model.get_context_length.return_value = 1000
        mock_model.encode_text.return_value = [1, 2, 3]
        mock_model.generate_output.return_value = "Test output"

        processor = RealTimeProcessor(mock_model, "test_model", "outcome_type")

        assert processor.model_name == "test_model"
        assert processor.task == "outcome_type"
        assert not processor.is_running

    def test_submit_document(self):
        """Test document submission"""
        from realtime_processor import RealTimeProcessor
        from models.model import Model

        mock_model = Mock(spec=Model)
        mock_model.get_context_length.return_value = 1000
        mock_model.encode_text.return_value = [1, 2, 3]

        processor = RealTimeProcessor(mock_model, "test_model", "outcome_type")

        doc = {
            "pmcid": "123",
            "intervention": "A",
            "comparator": "B",
            "outcome": "Mortality"
        }

        doc_id = processor.submit_document(doc, "Test content")

        assert doc_id == "123"
        assert processor.stats.total_submitted == 1


class TestIntegration:
    """Integration tests for the full pipeline"""

    def test_full_pipeline_simulation(self, tmp_path):
        """Test simulated full pipeline from extraction to meta-analysis"""

        # This is a simulation test - doesn't actually run models
        # but tests the data flow

        # 1. Simulate extracted data
        extracted_data = [
            {
                "pmcid": str(i),
                "intervention": "Drug A",
                "comparator": "Placebo",
                "outcome": "Mortality",
                "outcome_type": "binary",
                "intervention_events_output": str(10 + i),
                "intervention_group_size_output": "50",
                "comparator_events_output": str(20 + i),
                "comparator_group_size_output": "50",
            }
            for i in range(1, 6)
        ]

        # 2. Test meta-analysis
        from meta_analysis_methods import MetaAnalyzer, StudyResult

        analyzer = MetaAnalyzer()

        for item in extracted_data:
            analyzer.add_study(StudyResult(
                study_id=item['pmcid'],
                pmcid=item['pmcid'],
                intervention=item['intervention'],
                comparator=item['comparator'],
                outcome=item['outcome'],
                intervention_events=int(item['intervention_events_output']),
                intervention_total=50,
                comparator_events=int(item['comparator_events_output']),
                comparator_total=50
            ))

        result = analyzer.analyze(method='dl', outcome_type='binary')

        assert result.n_studies == 5
        assert result.pooled_effect is not None

    def test_error_analysis_pipeline(self, tmp_path):
        """Test error analysis pipeline"""

        # Create test data with errors
        test_data = [
            {
                "pmcid": "123",
                "intervention": "Drug A",
                "comparator": "Placebo",
                "outcome": "Mortality",
                "outcome_type": "binary",
                "intervention_events": "10",
                "intervention_group_size": "50",
                "comparator_events": "20",
                "comparator_group_size": "50",
                "intervention_events_output": "10",
                "intervention_group_size_output": "50",
                "comparator_events_output": "x",  # Error
                "comparator_group_size_output": "50",
            }
        ]

        output_file = tmp_path / "test_output.json"
        with open(output_file, 'w') as f:
            json.dump(test_data, f)

        from error_analyzer import ErrorAnalyzer

        analyzer = ErrorAnalyzer("binary_outcomes", str(output_file))
        report = analyzer.run_analysis("test_model")

        assert report.total_instances == 1
        assert report.error_instances > 0


# Test fixtures
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment"""
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    yield
    # Cleanup
    if os.path.exists(TEST_OUTPUT_DIR):
        shutil.rmtree(TEST_OUTPUT_DIR)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
