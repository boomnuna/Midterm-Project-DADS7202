"""
src/evaluation/statistics.py

Implements the assignment's two explicit statistics requirements:
  1. "report every number as mean±SD" across the N repeated runs
  2. "compute p-values to check if the best architecture's win is
      statistically significant vs the others"
"""

import numpy as np
from scipy import stats


class StatisticalComparator:
    def __init__(self, results: dict):
        """
        results: {backbone_name: [run1_metrics_dict, run2_metrics_dict, ...]}
                 as produced by ExperimentRunner.run_all()
        """
        self.results = results

    def mean_std_table(self, metric_name: str = "accuracy") -> dict:
        """Returns {backbone_name: (mean, std)} for the given scalar metric."""
        table = {}
        for backbone_name, runs in self.results.items():
            values = [run[metric_name] for run in runs]
            table[backbone_name] = (float(np.mean(values)), float(np.std(values)))
        return table

    def print_mean_std_table(self, metric_names: list = None):
        metric_names = metric_names or ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
        print("=" * 70)
        print("MEAN ± SD ACROSS REPEATED RUNS")
        print("=" * 70)
        for metric_name in metric_names:
            print(f"\n{metric_name}:")
            table = self.mean_std_table(metric_name)
            for backbone_name, (mean, std) in table.items():
                n = len(self.results[backbone_name])
                print(f"  {backbone_name:20s}: {mean:.4f} ± {std:.4f}  (n={n} runs)")

    def pairwise_significance(self, metric_name: str = "accuracy") -> dict:
        """
        Independent two-sample t-test between every pair of architectures'
        per-run metric values. Returns {(backbone_a, backbone_b): p_value}.

        NOTE: with small n (3-10 runs per the assignment's minimum), the
        t-test has limited power — treat p-values as indicative, and say
        so explicitly in your Discussion section rather than overclaiming
        significance from a handful of runs.
        """
        backbone_names = list(self.results.keys())
        p_values = {}

        for i in range(len(backbone_names)):
            for j in range(i + 1, len(backbone_names)):
                name_a, name_b = backbone_names[i], backbone_names[j]
                values_a = [run[metric_name] for run in self.results[name_a]]
                values_b = [run[metric_name] for run in self.results[name_b]]

                _, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)  # Welch's t-test
                p_values[(name_a, name_b)] = p_value

        return p_values

    def print_pairwise_significance(self, metric_name: str = "accuracy", alpha: float = 0.05):
        print("=" * 70)
        print(f"PAIRWISE SIGNIFICANCE TEST (Welch's t-test) on '{metric_name}'")
        print(f"(alpha = {alpha})")
        print("=" * 70)
        p_values = self.pairwise_significance(metric_name)
        for (name_a, name_b), p in p_values.items():
            verdict = "SIGNIFICANT" if p < alpha else "not significant"
            print(f"  {name_a} vs {name_b}: p={p:.4f}  -> {verdict}")

    def best_architecture(self, metric_name: str = "accuracy") -> str:
        table = self.mean_std_table(metric_name)
        return max(table.items(), key=lambda kv: kv[1][0])[0]
