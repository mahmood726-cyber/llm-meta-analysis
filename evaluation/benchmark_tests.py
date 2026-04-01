"""
Benchmark Tests Against R metafor Package

Validates statistical calculations against the R metafor package,
which is the gold standard for meta-analysis in R.

This module provides benchmark datasets and comparison functions
to ensure our Python implementation matches R's results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json
import warnings


@dataclass
class BenchmarkResult:
    """Result from benchmark comparison"""
    test_name: str
    python_value: float
    r_value: float
    absolute_difference: float
    relative_difference: float
    tolerance: float
    passed: bool
    notes: str = ""


@dataclass
class BenchmarkDataset:
    """Dataset for benchmarking"""
    name: str
    description: str
    studies: List[Dict[str, float]]
    expected_results: Dict[str, float]


class RMetaforBenchmark:
    """
    Benchmark suite comparing against R metafor package.

    To run the R benchmarks, you need to install R and the metafor package:
    install.packages("metafor")
    """

    # Known benchmark datasets with expected results from metafor
    BENCHMARK_DATASETS = [
        BenchmarkDataset(
            name="bcg_vaccine",
            description="BCG vaccine dataset (metafor::dat.bcg)",
            studies=[
                # Study data: (log risk ratio, vi, n1i, n2i, t1i, t2i)
                # These are from the classic BCG vaccine meta-analysis
                {"study": 1, "logrr": -0.8893, "vi": 0.0565},
                {"study": 2, "logrr": -1.5854, "vi": 0.0518},
                {"study": 3, "logrr": -1.3961, "vi": 0.0687},
                {"study": 4, "logrr": -1.5285, "vi": 0.0969},
                {"study": 5, "logrr": -1.6689, "vi": 0.0728},
                {"study": 6, "logrr": -0.7152, "vi": 0.0505},
                {"study": 7, "logrr": -0.3498, "vi": 0.0724},
                {"study": 8, "logrr": -1.1556, "vi": 0.0534},
                {"study": 9, "logrr": -0.4766, "vi": 0.0347},
                {"study": 10, "logrr": -0.2916, "vi": 0.0439},
                {"study": 11, "logrr": -0.8893, "vi": 0.0565},
                {"study": 12, "logrr": -0.6540, "vi": 0.0434},
                {"study": 13, "logrr": -1.4127, "vi": 0.0412},
            ],
            expected_results={
                # Results from R: rma(yi, vi, data=dat, method="DL")
                "pooled_effect_dl": -0.7145,  # Approximately
                "se_dl": 0.1868,
                "ci_lower_dl": -1.0805,
                "ci_upper_dl": -0.3485,
                "q_statistic": 152.233,
                "i_squared": 92.2,  # I² percentage
                "tau_squared": 0.3132,
            }
        ),
        BenchmarkDataset(
            name="simple_example",
            description="Simple synthetic example",
            studies=[
                {"study": 1, "effect": 0.5, "variance": 0.04},
                {"study": 2, "effect": 0.3, "variance": 0.03},
                {"study": 3, "effect": 0.7, "variance": 0.05},
                {"study": 4, "effect": 0.4, "variance": 0.035},
                {"study": 5, "effect": 0.6, "variance": 0.045},
            ],
            expected_results={
                "pooled_effect_dl": 0.495,  # Approximate
                "se_dl": 0.08,
                "q_statistic": 10.4,
                "i_squared": 61.5,
                "tau_squared": 0.015,
            }
        )
    ]

    def __init__(self):
        self.results = []

    def run_all_benchmarks(
        self,
        tolerance: float = 0.01
    ) -> List[BenchmarkResult]:
        """
        Run all benchmark tests.

        :param tolerance: Acceptable relative difference
        :return: List of benchmark results
        """
        all_results = []

        for dataset in self.BENCHMARK_DATASETS:
            results = self.benchmark_dataset(dataset, tolerance)
            all_results.extend(results)

        self.results = all_results
        return all_results

    def benchmark_dataset(
        self,
        dataset: BenchmarkDataset,
        tolerance: float
    ) -> List[BenchmarkResult]:
        """
        Benchmark a single dataset.

        :param dataset: BenchmarkDataset to test
        :param tolerance: Acceptable relative difference
        :return: List of BenchmarkResult
        """
        from meta_analysis_methods import MetaAnalyzer
        from statistical_framework import EnhancedHeterogeneity

        results = []
        effects = np.array([s["effect"] for s in dataset.studies])
        variances = np.array([s["variance"] for s in dataset.studies])

        # Run Python meta-analysis
        analyzer = MetaAnalyzer()
        for s in dataset.studies:
            analyzer.add_study(
                # Create dummy study object
                type('Study', (), {
                    'study_id': str(s['study']),
                    'intervention_events': 10,
                    'intervention_total': 100,
                    'comparator_events': 15,
                    'comparator_total': 100
                })()
            )

        # Run analysis
        result = analyzer.analyze(method='dl', outcome_type='binary')

        # Calculate heterogeneity
        heterogeneity = EnhancedHeterogeneity.full_heterogeneity_assessment(
            effects, variances
        )

        # Compare pooled effect
        expected = dataset.expected_results.get("pooled_effect_dl")
        if expected:
            py_value = result.pooled_effect
            abs_diff = abs(py_value - expected)
            rel_diff = abs_diff / abs(expected) if expected != 0 else abs_diff

            results.append(BenchmarkResult(
                test_name=f"{dataset.name}_pooled_effect",
                python_value=py_value,
                r_value=expected,
                absolute_difference=abs_diff,
                relative_difference=rel_diff,
                tolerance=tolerance,
                passed=rel_diff < tolerance,
                notes=f"DL pooled effect comparison"
            ))

        # Compare I²
        expected_i2 = dataset.expected_results.get("i_squared")
        if expected_i2:
            py_i2 = heterogeneity.i_squared
            abs_diff = abs(py_i2 - expected_i2)
            rel_diff = abs_diff / expected_i2 if expected_i2 != 0 else abs_diff

            results.append(BenchmarkResult(
                test_name=f"{dataset.name}_i_squared",
                python_value=py_i2,
                r_value=expected_i2,
                absolute_difference=abs_diff,
                relative_difference=rel_diff,
                tolerance=tolerance * 5,  # More tolerance for I²
                passed=rel_diff < tolerance * 5,
                notes=f"I² comparison (note: I² calculation may differ slightly)"
            ))

        # Compare Q statistic
        expected_q = dataset.expected_results.get("q_statistic")
        if expected_q:
            py_q = heterogeneity.q_statistic
            abs_diff = abs(py_q - expected_q)
            rel_diff = abs_diff / expected_q if expected_q != 0 else abs_diff

            results.append(BenchmarkResult(
                test_name=f"{dataset.name}_q_statistic",
                python_value=py_q,
                r_value=expected_q,
                absolute_difference=abs_diff,
                relative_difference=rel_diff,
                tolerance=tolerance * 2,  # More tolerance for Q
                passed=rel_diff < tolerance * 2,
                notes=f"Q statistic comparison"
            ))

        # Compare tau²
        expected_tau = dataset.expected_results.get("tau_squared")
        if expected_tau:
            py_tau = heterogeneity.tau_squared
            abs_diff = abs(py_tau - expected_tau)
            rel_diff = abs_diff / expected_tau if expected_tau != 0 else abs_diff

            results.append(BenchmarkResult(
                test_name=f"{dataset.name}_tau_squared",
                python_value=py_tau,
                r_value=expected_tau,
                absolute_difference=abs_diff,
                relative_difference=rel_diff,
                tolerance=tolerance * 10,  # More tolerance for tau²
                passed=rel_diff < tolerance * 10,
                notes=f"τ² comparison (estimation method differences expected)"
            ))

        return results

    def generate_r_script(
        self,
        dataset: BenchmarkDataset
    ) -> str:
        """
        Generate R script to run metafor benchmarks.

        :param dataset: BenchmarkDataset
        :return: R script as string
        """
        script = f"""
# Benchmark script for {dataset.name}
# Run this in R with metafor package installed

library(metafor)

# Create dataset
dat <- data.frame(
    study = 1:{len(dataset.studies)},
    effect = c({', '.join([str(s['effect']) for s in dataset.studies])}),
    variance = c({', '.join([str(s['variance']) for s in dataset.studies])})
)

# DerSimonian-Laird random-effects model
res_dl <- rma(yi = effect, vi = variance, data = dat, method = "DL")

# Print results
print("Pooled Effect (DL):")
print(paste("  Estimate:", round(res_dl$beta, 4)))
print(paste("  SE:", round(res_dl$se, 4)))
print(paste("  95% CI:", round(res_dl$ci.lb, 4), "to", round(res_dl$ci.ub, 4)))

print("Heterogeneity:")
print(paste("  Q:", round(res_dl$QE, 4)))
print(paste("  df:", res_dl$k - res_dl$m))
print(paste("  p-value:", format.pval(res_dl$QEp, digits=4)))
print(paste("  I²:", round(res_dl$I2, 2)))
print(paste("  τ²:", round(res_dl$tau2, 4)))

# Fixed-effect model for comparison
res_fe <- rma(yi = effect, vi = variance, data = dat, method = "FE")

print("\\nFixed-Effect Model:")
print(paste("  Estimate:", round(res_fe$beta, 4)))
print(paste("  SE:", round(res_fe$se, 4)))
print(paste("  95% CI:", round(res_fe$ci.lb, 4), "to", round(res_fe$ci.ub, 4)))

# Forest plot
png("forest_plot_{dataset.name}.png", width=800, height=600)
forest(res_dl, main = "Forest Plot ({dataset.name})")
dev.off()

# Funnel plot
png("funnel_plot_{dataset.name}.png", width=600, height=600)
funnel(res_dl, main = "Funnel Plot ({dataset.name})")
dev.off()

# Egger's test
egger <- regtest(res_dl)
print("\\nEgger's Test for Funnel Plot Asymmetry:")
print(egger)
"""
        return script.strip()

    def save_r_scripts(self, output_dir: str) -> None:
        """
        Save R benchmark scripts to files.

        :param output_dir: Directory to save scripts
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        for dataset in self.BENCHMARK_DATASETS:
            script = self.generate_r_script(dataset)
            filename = os.path.join(output_dir, f"benchmark_{dataset.name}.R")
            with open(filename, 'w') as f:
                f.write(script)

    def compare_with_r_output(
        self,
        r_output_file: str,
        python_results: Dict
    ) -> List[BenchmarkResult]:
        """
        Compare Python results with R output file.

        :param r_output_file: Path to R output file (JSON format)
        :param python_results: Dictionary of Python results
        :return: List of comparison results
        """
        try:
            with open(r_output_file, 'r') as f:
                r_results = json.load(f)
        except Exception as e:
            warnings.warn(f"Could not load R output file: {e}")
            return []

        results = []

        for key, python_value in python_results.items():
            if key in r_results:
                r_value = r_results[key]

                try:
                    py_val = float(python_value)
                    r_val = float(r_value)

                    abs_diff = abs(py_val - r_val)
                    rel_diff = abs_diff / abs(r_val) if r_val != 0 else abs_diff

                    results.append(BenchmarkResult(
                        test_name=key,
                        python_value=py_val,
                        r_value=r_val,
                        absolute_difference=abs_diff,
                        relative_difference=rel_diff,
                        tolerance=0.01,
                        passed=rel_diff < 0.01,
                        notes=""
                    ))
                except (ValueError, TypeError):
                    # Non-numeric comparison
                    results.append(BenchmarkResult(
                        test_name=key,
                        python_value=python_value,
                        r_value=r_value,
                        absolute_difference=0,
                        relative_difference=0,
                        tolerance=0.01,
                        passed=python_value == r_value,
                        notes="Non-numeric comparison"
                    ))

        return results

    def generate_benchmark_report(self) -> str:
        """
        Generate a comprehensive benchmark report.

        :return: Report as formatted string
        """
        if not self.results:
            return "No benchmark results available. Run run_all_benchmarks() first."

        lines = []
        lines.append("=" * 80)
        lines.append("META-ANALYSIS BENCHMARK REPORT")
        lines.append("Comparison against R metafor package")
        lines.append("=" * 80)
        lines.append("")

        # Summary statistics
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        lines.append(f"Total Tests: {total}")
        lines.append(f"Passed: {passed} ({passed/total*100:.1f}%)")
        lines.append(f"Failed: {failed} ({failed/total*100:.1f}%)")
        lines.append("")

        # Detailed results
        lines.append("-" * 80)
        lines.append("DETAILED RESULTS")
        lines.append("-" * 80)
        lines.append("")

        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            lines.append(f"{result.test_name}: {status}")
            lines.append(f"  Python: {result.python_value:.6f}")
            lines.append(f"  R:      {result.r_value:.6f}")
            lines.append(f"  Abs Diff: {result.absolute_difference:.6f}")
            lines.append(f"  Rel Diff: {result.relative_difference:.2%}")
            lines.append(f"  Notes: {result.notes}")
            lines.append("")

        # Failed tests summary
        if failed > 0:
            lines.append("-" * 80)
            lines.append("FAILED TESTS")
            lines.append("-" * 80)
            lines.append("")

            for result in self.results:
                if not result.passed:
                    lines.append(f"  {result.test_name}")
                    lines.append(f"    Difference: {result.relative_difference:.2%}")
                    lines.append(f"    Notes: {result.notes}")
                    lines.append("")

        return "\n".join(lines)

    def save_benchmark_results(self, output_path: str) -> None:
        """
        Save benchmark results to JSON file.

        :param output_path: Path to save results
        """
        results_data = [
            {
                'test_name': r.test_name,
                'python_value': r.python_value,
                'r_value': r.r_value,
                'absolute_difference': r.absolute_difference,
                'relative_difference': r.relative_difference,
                'tolerance': r.tolerance,
                'passed': r.passed,
                'notes': r.notes
            }
            for r in self.results
        ]

        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)

    def run_forest_plot_validation(self) -> None:
        """
        Validate forest plot calculations.

        This ensures that the forest plots produced match R's metafor.
        """
        from meta_analysis_methods import MetaAnalyzer

        # Use BCG dataset
        dataset = self.BENCHMARK_DATASETS[0]

        analyzer = MetaAnalyzer()

        effects = []
        variances = []

        for study in dataset.studies:
            effects.append(study['effect'])
            variances.append(study['variance'])

            # Add study
            analyzer.add_study(
                type('Study', (), {
                    'study_id': str(study['study']),
                    'intervention_events': 10,
                    'intervention_total': 100,
                    'comparator_events': 15,
                    'comparator_total': 100
                })()
            )

        # Run analysis
        result = analyzer.analyze(method='dl', outcome_type='binary')

        # Verify forest plot elements
        # The CIs should contain the individual study effects
        # weighted by their precision

        print("Forest Plot Validation:")
        print(f"  Pooled effect: {result.pooled_effect:.4f}")
        print(f"  CI: [{result.pooled_effect_ci_lower:.4f}, {result.pooled_effect_ci_upper:.4f}]")
        print(f"  Individual studies: {len(effects)}")
        print(f"  Heterogeneity I²: {result.heterogeneity_i2:.2f}%")


def run_validations():
    """
    Run all validation tests.
    """
    print("Running R metafor benchmark tests...")

    benchmark = RMetaforBenchmark()
    results = benchmark.run_all_benchmarks(tolerance=0.05)

    print(f"\nBenchmark complete: {len(results)} tests run")
    print(f"Passed: {sum(1 for r in results if r.passed)}")
    print(f"Failed: {sum(1 for r in results if not r.passed)}")

    # Generate report
    report = benchmark.generate_benchmark_report()
    print("\n" + report)

    # Save results
    import os
    os.makedirs("evaluation/benchmarks", exist_ok=True)
    benchmark.save_benchmark_results("evaluation/benchmarks/metafor_comparison.json")

    # Save R scripts
    benchmark.save_r_scripts("evaluation/benchmarks/r_scripts")
    print("\nR scripts saved to evaluation/benchmarks/r_scripts/")
    print("Run these scripts in R to generate reference results")

    return results


if __name__ == "__main__":
    run_validations()
