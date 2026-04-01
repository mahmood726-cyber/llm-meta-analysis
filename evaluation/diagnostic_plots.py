"""
Statistical Diagnostic Plots for Meta-Analysis

Provides publication-quality diagnostic plots for meta-analysis including:
- Forest plots (with subgroups)
- Funnel plots (for publication bias)
- Radial plots
- Baujat plots (influential studies)
- Influence plots
- Network plots (for NMA)
- Labbe plots
- Galbraith plots

References:
- Borenstein et al. (2009). Introduction to Meta-Analysis.
- Viechtbauer (2010). Conducting Meta-Analyses in R.
- Cochrane Handbook for Systematic Reviews.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import warnings


# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.3


@dataclass
class PlotConfig:
    """Configuration for diagnostic plots."""
    figsize: Tuple[float, float] = (8, 6)
    dpi: int = 300
    colors: List[str] = None
    effect_measure: str = "MD"  # MD, SMD, OR, RR, HR, etc.
    confidence_level: float = 0.95
    publication_style: str = "bmj"  # bmj, lancet, nature, etc.

    def __post_init__(self):
        if self.colors is None:
            self.colors = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
            ]


class MetaAnalysisPlots:
    """
    Publication-quality diagnostic plots for meta-analysis.

    All plots follow standards from leading medical journals.
    """

    def __init__(self, config: Optional[PlotConfig] = None):
        """
        Initialize plot generator.

        :param config: Plot configuration
        """
        self.config = config or PlotConfig()

    def forest_plot(
        self,
        effects: np.ndarray,
        variances: np.ndarray,
        study_names: Optional[List[str]] = None,
        pooled_effect: Optional[float] = None,
        pooled_ci: Optional[Tuple[float, float]] = None,
        prediction_interval: Optional[Tuple[float, float]] = None,
        subgroups: Optional[np.ndarray] = None,
        title: str = "Forest Plot",
        xlabel: Optional[str] = None,
        annotate_stats: bool = True,
        sort_by: str = "input"
    ) -> plt.Figure:
        """
        Create publication-quality forest plot.

        Features:
        - Individual study effects with CIs
        - Pooled effect diamond
        - Prediction interval (if provided)
        - Subgroup panels (if provided)
        - Weight display
        - Proper statistical annotations

        :param effects: Study effects
        :param variances: Study variances
        :param study_names: Optional study names
        :param pooled_effect: Pooled effect estimate
        :param pooled_ci: Pooled effect CI (lower, upper)
        :param prediction_interval: Prediction interval (lower, upper)
        :param subgroups: Optional subgroup labels
        :param title: Plot title
        :param xlabel: X-axis label
        :param annotate_stats: Show statistical annotations
        :param sort_by: How to order studies ('input', 'effect', 'precision', 'size')
        :return: Matplotlib figure
        """
        n = len(effects)

        if study_names is None:
            study_names = [f"Study {i+1}" for i in range(n)]

        # Sort if requested
        if sort_by != "input":
            if sort_by == "effect":
                order = np.argsort(effects)
            elif sort_by == "precision":
                order = np.argsort(variances)
            elif sort_by == "size":
                order = np.argsort(np.sqrt(1/variances))
            else:
                order = np.arange(n)

            effects = effects[order]
            variances = variances[order]
            study_names = [study_names[i] for i in order]
            if subgroups is not None:
                subgroups = subgroups[order]

        # Calculate CIs
        ses = np.sqrt(variances)
        ci_width = stats.norm.ppf(1 - self.config.confidence_level/2)
        ci_lower = effects - ci_width * ses
        ci_upper = effects + ci_width * ses

        # Calculate weights (for display)
        weights = 1 / variances
        weights_pct = weights / np.sum(weights) * 100

        # Determine layout
        if subgroups is not None:
            unique_subgroups = np.unique(subgroups)
            n_subgroups = len(unique_subgroups)
            fig = plt.figure(figsize=(10, 4 + n * 0.4 + n_subgroups * 0.5))
            gs = gridspec.GridSpec(n + n_subgroups + 1, 1, height_ratios=[1] * (n + n_subgroups) + [0.3])
        else:
            fig = plt.figure(figsize=(10, 4 + n * 0.4))
            gs = gridspec.GridSpec(n + 1, 1, height_ratios=[1] * n + [0.3])

        ax = fig.add_subplot(gs[0, 0])

        # Set up x-axis
        all_effects = list(effects)
        if pooled_effect is not None:
            all_effects.append(pooled_effect)
        xlim = self._calculate_forest_xlim(all_effects, ci_lower, ci_upper)

        y_pos = n
        row_idx = 0

        # Subgroup handling
        if subgroups is not None:
            current_subgroup = None
            subgroup_results = {}

            for i, (effect, ci_l, ci_u, name, subgroup) in enumerate(
                zip(effects, ci_lower, ci_upper, study_names, subgroups)
            ):
                # New subgroup panel
                if subgroup != current_subgroup:
                    if current_subgroup is not None:
                        # Add subgroup pooled effect
                        self._add_forest_diamond(ax, subgroup_results[current_subgroup]['effect'],
                                                subgroup_results[current_subgroup]['ci'],
                                                y_pos, self.config.colors[1], label=f"{current_subgroup} Pooled")
                        y_pos -= 1
                        row_idx += 1
                        ax = fig.add_subplot(gs[row_idx, 0])
                        ax.set_xlim(xlim)
                        ax.axvline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)

                    current_subgroup = subgroup
                    subgroup_results[subgroup] = {'effects': [], 'vars': []}

                subgroup_results[subgroup]['effects'].append(effect)
                subgroup_results[subgroup]['vars'].append(variances[i])

                # Add study
                self._add_forest_line(ax, effect, ci_l, ci_u, y_pos, self.config.colors[0])
                ax.text(-0.05, y_pos, name, ha='right', va='center', fontsize=9)
                ax.text(1.05, y_pos, f"{weights_pct[i]:.1f}%", ha='left', va='center', fontsize=8)
                y_pos -= 1

            # Add final subgroup
            if current_subgroup in subgroup_results:
                self._add_forest_diamond(ax, subgroup_results[current_subgroup]['effect'],
                                        subgroup_results[current_subgroup]['ci'],
                                        y_pos, self.config.colors[1], label=f"{current_subgroup} Pooled")

        else:
            # No subgroups
            for i, (effect, ci_l, ci_u, name, w_pct) in enumerate(
                zip(effects, ci_lower, ci_upper, study_names, weights_pct)
            ):
                self._add_forest_line(ax, effect, ci_l, ci_u, y_pos, self.config.colors[0])
                ax.text(-0.05, y_pos, name, ha='right', va='center', fontsize=9)
                ax.text(1.05, y_pos, f"{w_pct:.1f}%", ha='left', va='center', fontsize=8)
                y_pos -= 1

        # Add overall pooled effect
        if pooled_effect is not None and pooled_ci is not None:
            self._add_forest_diamond(ax, pooled_effect, pooled_ci, y_pos, self.config.colors[2],
                                   label="Overall Pooled")
            y_pos -= 1

        # Add prediction interval
        if prediction_interval is not None:
            ax = fig.add_subplot(gs[row_idx, 0])
            ax.axvspan(prediction_interval[0], prediction_interval[1], alpha=0.15, color='gray',
                      label=f"{int(self.config.confidence_level*100)}% PI")

        # Styling
        ax.set_xlim(xlim)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        ax.set_xlabel(xlabel or f"{self.config.effect_measure} ({int(self.config.confidence_level*100)}% CI)")
        ax.set_title(title, fontweight='bold', pad=10)

        # Add reference line text
        ax.text(0, 1.02, "No effect", ha='center', transform=ax.get_xaxis_transform(), fontsize=8)

        # Legend
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

        plt.tight_layout()
        return fig

    def _add_forest_line(self, ax, effect, ci_lower, ci_upper, y, color):
        """Add a single study line to forest plot."""
        ax.plot([ci_lower, ci_upper], [y, y], color=color, linewidth=1.5)
        ax.plot([effect], [y], marker='s', markersize=6, color=color)
        ax.plot([ci_lower], [y], marker='|', markersize=8, color=color)
        ax.plot([ci_upper], [y], marker='|', markersize=8, color=color)

    def _add_forest_diamond(self, ax, effect, ci, y, color, label="Pooled"):
        """Add pooled effect diamond to forest plot."""
        diamond_width = ci[1] - ci[0]
        diamond = mpatches.Polygon(
            [[effect, y + diamond_width/2],
             [ci[1], y],
             [effect, y - diamond_width/2],
             [ci[0], y]],
            closed=True, facecolor=color, edgecolor='black', linewidth=0.8
        )
        ax.add_patch(diamond)
        ax.text(effect, y + 0.3, label, ha='center', va='bottom', fontsize=8, fontweight='bold')

    def _calculate_forest_xlim(self, effects, ci_lower, ci_upper, padding=1.5):
        """Calculate appropriate x-axis limits for forest plot."""
        min_val = min(np.min(ci_lower), np.min(effects))
        max_val = max(np.max(ci_upper), np.max(effects))
        range_val = max_val - min_val
        return (min_val - padding * range_val/10, max_val + padding * range_val/10)

    def funnel_plot(
        self,
        effects: np.ndarray,
        variances: np.ndarray,
        study_names: Optional[List[str]] = None,
        pooled_effect: Optional[float] = None,
        pseudo_ci: bool = True,
        contour_levels: Optional[List[float]] = None,
        title: str = "Funnel Plot"
    ) -> plt.Figure:
        """
        Create funnel plot for publication bias assessment.

        Features:
        - Standard error on y-axis (recommended)
        - Pseudo confidence intervals
        - Contour-enhanced funnel (optional)
        - Study labels (hover in interactive version)

        :param effects: Study effects
        :param variances: Study variances
        :param study_names: Optional study names
        :param pooled_effect: Pooled effect (center of funnel)
        :param pseudo_ci: Show pseudo confidence intervals
        :param contour_levels: Statistical significance levels for contours
        :param title: Plot title
        :return: Matplotlib figure
        """
        n = len(effects)
        ses = np.sqrt(variances)

        if pooled_effect is None:
            # Inverse variance weighted average
            weights = 1 / variances
            pooled_effect = np.sum(weights * effects) / np.sum(weights)

        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Scatter plot
        ax.scatter(effects, ses, s=50, alpha=0.7, color=self.config.colors[0], edgecolors='black', linewidth=0.5)

        # Add study labels
        if study_names is not None and n <= 30:
            for i, name in enumerate(study_names):
                ax.annotate(name, (effects[i], ses[i]), fontsize=7, alpha=0.7)

        # Pseudo confidence intervals
        if pseudo_ci:
            se_max = np.max(ses) * 1.1
            se_vals = np.linspace(0, se_max, 100)
            z_alpha = stats.norm.ppf(1 - self.config.confidence_level/2)

            # 95% CI
            ax.plot(pooled_effect + z_alpha * se_vals, se_vals, 'k--', linewidth=0.8, alpha=0.5, label='95% CI')
            ax.plot(pooled_effect - z_alpha * se_vals, se_vals, 'k--', linewidth=0.8, alpha=0.5)

            # 99.7% CI (approx 3 sigma)
            z_997 = stats.norm.ppf(1 - 0.003/2)
            ax.plot(pooled_effect + z_997 * se_vals, se_vals, 'k:', linewidth=0.8, alpha=0.3)
            ax.plot(pooled_effect - z_997 * se_vals, se_vals, 'k:', linewidth=0.8, alpha=0.3)

        # Contour-enhanced funnel
        if contour_levels:
            self._add_funnel_contours(ax, effects, ses, pooled_effect, contour_levels)

        # Vertical line at pooled effect
        ax.axvline(pooled_effect, color='red', linewidth=1.5, linestyle='-', label='Pooled effect')

        # Styling
        ax.set_xlabel(f"{self.config.effect_measure}")
        ax.set_ylabel("Standard Error")
        ax.set_title(title, fontweight='bold')
        ax.invert_yaxis()  # Larger SE at top
        ax.set_ylim(top=np.max(ses) * 1.15, bottom=-0.01)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', frameon=True, shadow=True)

        plt.tight_layout()
        return fig

    def _add_funnel_contours(self, ax, effects, ses, pooled_effect, levels):
        """Add statistical significance contours to funnel plot."""
        se_max = np.max(ses) * 1.1
        x_range = np.linspace(np.min(effects) - 1, np.max(effects) + 1, 100)
        y_range = np.linspace(0, se_max, 100)
        X, Y = np.meshgrid(x_range, y_range)

        # Calculate z-scores
        Z = np.abs((X - pooled_effect) / Y)

        # Create contour
        if levels is None:
            levels = [0.9, 0.95, 0.99]

        CS = ax.contourf(X, Y, Z, levels=levels, colors=['white', 'lightgray', 'gray'], alpha=0.3)
        # Add legend for contours
        ax.text(0.02, 0.98, "White: p > 0.1\nGray: 0.1 > p > 0.05\nDark: p < 0.05",
               transform=ax.transAxes, fontsize=7, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def radial_plot(
        self,
        effects: np.ndarray,
        variances: np.ndarray,
        study_names: Optional[List[str]] = None,
        pooled_effect: Optional[float] = None,
        title: str = "Radial Plot"
    ) -> plt.Figure:
        """
        Create radial plot (alternative to funnel plot).

        Plots standardized residual against precision.
        Useful for identifying outliers and publication bias.

        :param effects: Study effects
        :param variances: Study variances
        :param study_names: Optional study names
        :param pooled_effect: Pooled effect
        :param title: Plot title
        :return: Matplotlib figure
        """
        if pooled_effect is None:
            weights = 1 / variances
            pooled_effect = np.sum(weights * effects) / np.sum(weights)

        # Calculate precision and standardized residual
        precision = 1 / np.sqrt(variances)
        std_residual = (effects - pooled_effect) * precision

        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Scatter plot
        ax.scatter(precision, std_residual, s=50, alpha=0.7, color=self.config.colors[0],
                  edgecolors='black', linewidth=0.5)

        # Add study labels
        if study_names is not None and len(study_names) <= 25:
            for i, name in enumerate(study_names):
                ax.annotate(name, (precision[i], std_residual[i]), fontsize=7)

        # Reference lines
        z_alpha = stats.norm.ppf(1 - self.config.confidence_level/2)
        ax.axhline(z_alpha, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'{int(self.config.confidence_level*100)}% CI')
        ax.axhline(-z_alpha, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

        # Styling
        ax.set_xlabel("Precision (1/SE)")
        ax.set_ylabel("Standardized Residual")
        ax.set_title(title, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        plt.tight_layout()
        return fig

    def baujat_plot(
        self,
        effects: np.ndarray,
        variances: np.ndarray,
        study_names: Optional[List[str]] = None,
        tau2: float = 0,
        title: str = "Baujat Plot"
    ) -> plt.Figure:
        """
        Create Baujat plot for identifying influential studies.

        Plots contribution to heterogeneity vs contribution to pooled effect.
        Studies in upper-right corner are most influential.

        :param effects: Study effects
        :param variances: Study variances
        :param study_names: Optional study names
        :param tau2: Between-study variance
        :param title: Plot title
        :return: Matplotlib figure
        """
        n = len(effects)
        if study_names is None:
            study_names = [f"S{i+1}" for i in range(n)]

        # Calculate contributions
        weights = 1 / (variances + tau2)
        pooled = np.sum(weights * effects) / np.sum(weights)

        # Contribution to Q statistic (heterogeneity)
        q_contributions = weights * (effects - pooled)**2
        q_pct = q_contributions / np.sum(q_contributions) * 100

        # Contribution to pooled effect
        effect_contributions = weights * (effects - pooled)
        effect_pct = np.abs(effect_contributions) / np.sum(weights) * 100

        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Scatter plot
        scatter = ax.scatter(q_pct, effect_pct, s=100, alpha=0.7, c=range(n),
                           cmap='viridis', edgecolors='black', linewidth=0.5)

        # Add study labels
        for i, name in enumerate(study_names):
            ax.annotate(name, (q_pct[i], effect_pct[i]), fontsize=7, alpha=0.8)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Study Index', rotation=270, labelpad=15)

        # Styling
        ax.set_xlabel("Contribution to Heterogeneity (Q) %")
        ax.set_ylabel("Contribution to Pooled Effect %")
        ax.set_title(title, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)

        # Add diagonal reference line
        max_val = max(np.max(q_pct), np.max(effect_pct))
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=0.8, alpha=0.3)

        plt.tight_layout()
        return fig

    def influence_plot(
        self,
        leave_one_out_results: pd.DataFrame,
        influence_metric: str = "influence_pct",
        title: str = "Influence Plot"
    ) -> plt.Figure:
        """
        Create influence plot from leave-one-out analysis.

        Shows how pooled effect changes when each study is omitted.

        :param leave_one_out_results: DataFrame from leave_one_out()
        :param influence_metric: Metric to plot ('influence', 'influence_pct', 'dfbetas')
        :param title: Plot title
        :return: Matplotlib figure
        """
        df = leave_one_out_results[leave_one_out_results['omitted_study'] != 'Overall (all studies)'].copy()

        n = len(df)
        study_names = df['omitted_study'].values
        influences = df[influence_metric].values

        fig, ax = plt.subplots(figsize=(10, max(6, n * 0.3)))

        y_pos = np.arange(n)
        colors = ['red' if abs(x) > 10 else 'gray' for x in influences]

        ax.barh(y_pos, influences, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

        # Add reference line
        ax.axvline(0, color='black', linewidth=0.8)

        # Styling
        ax.set_yticks(y_pos)
        ax.set_yticklabels(study_names, fontsize=8)
        ax.set_xlabel(f"{influence_metric.replace('_', ' ').title()}")
        ax.set_title(title, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()
        return fig

    def labbe_plot(
        self,
        events_experimental: np.ndarray,
        total_experimental: np.ndarray,
        events_control: np.ndarray,
        total_control: np.ndarray,
        study_names: Optional[List[str]] = None,
        title: str = "L'Abbé Plot"
    ) -> plt.Figure:
        """
        Create L'Abbé plot for binary outcome meta-analysis.

        Plots event rates in experimental vs control group.
        Diagonal line represents no effect.

        :param events_experimental: Events in experimental group
        :param total_experimental: Total in experimental group
        :param events_control: Events in control group
        :param total_control: Total in control group
        :param study_names: Optional study names
        :param title: Plot title
        :return: Matplotlib figure
        """
        # Calculate event rates
        rate_exp = events_experimental / total_experimental
        rate_ctrl = events_control / total_control

        # Study sizes (for bubble size)
        sizes = total_experimental + total_control
        sizes_norm = sizes / np.max(sizes) * 200

        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Scatter plot
        ax.scatter(rate_ctrl, rate_exp, s=sizes_norm, alpha=0.6, c=rate_exp/rate_ctrl,
                  cmap='RdYlGn', edgecolors='black', linewidth=0.5, vmin=0, vmax=2)

        # Add diagonal line (no effect)
        max_rate = max(np.max(rate_ctrl), np.max(rate_exp)) * 1.05
        ax.plot([0, max_rate], [0, max_rate], 'k--', linewidth=1.5, label='No effect line')

        # Add study labels
        if study_names is not None:
            for i, name in enumerate(study_names):
                ax.annotate(name, (rate_ctrl[i], rate_exp[i]), fontsize=7, alpha=0.7)

        # Add 45-degree reference zones
        ax.fill_between([0, max_rate], [0, max_rate*0.8], [0, max_rate*1.2],
                       alpha=0.1, color='green', label='20% equivalence zone')

        # Styling
        ax.set_xlabel("Control Group Event Rate")
        ax.set_ylabel("Experimental Group Event Rate")
        ax.set_title(title, fontweight='bold')
        ax.set_xlim(-0.02, max_rate)
        ax.set_ylim(-0.02, max_rate)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', frameon=True)

        # Add colorbar
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Risk Ratio', rotation=270, labelpad=15)

        plt.tight_layout()
        return fig

    def galbraith_plot(
        self,
        effects: np.ndarray,
        variances: np.ndarray,
        study_names: Optional[List[str]] = None,
        pooled_effect: Optional[float] = None,
        title: str = "Galbraith Plot"
    ) -> plt.Figure:
        """
        Create Galbraith plot for assessing heterogeneity.

        Similar to radial plot but with different scaling.
        Helps identify studies contributing to heterogeneity.

        :param effects: Study effects
        :param variances: Study variances
        :param study_names: Optional study names
        :param pooled_effect: Pooled effect
        :param title: Plot title
        :return: Matplotlib figure
        """
        if pooled_effect is None:
            weights = 1 / variances
            pooled_effect = np.sum(weights * effects) / np.sum(weights)

        # Calculate z-scores and precision
        z_scores = (effects - pooled_effect) / np.sqrt(variances)
        precision = np.sqrt(weights)

        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Scatter plot
        ax.scatter(precision, z_scores, s=50, alpha=0.7, color=self.config.colors[0],
                  edgecolors='black', linewidth=0.5)

        # Add study labels
        if study_names is not None and len(study_names) <= 25:
            for i, name in enumerate(study_names):
                ax.annotate(name, (precision[i], z_scores[i]), fontsize=7)

        # Reference lines (95% CI)
        z_alpha = stats.norm.ppf(1 - self.config.confidence_level/2)
        ax.axhline(z_alpha, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'{int(self.config.confidence_level*100)}% CI')
        ax.axhline(-z_alpha, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

        # Styling
        ax.set_xlabel("Precision (sqrt(weight))")
        ax.set_ylabel("Z-score")
        ax.set_title(title, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        plt.tight_layout()
        return fig

    def network_plot(
        self,
        treatments: List[str],
        comparisons: List[Tuple[str, str]],
        n_studies: Optional[List[int]] = None,
        title: str = "Network Plot"
    ) -> plt.Figure:
        """
        Create network plot for network meta-analysis.

        Shows treatment network with edges representing comparisons.

        :param treatments: List of treatment names
        :param comparisons: List of (treatment1, treatment2) tuples
        :param n_studies: Number of studies for each comparison
        :param title: Plot title
        :return: Matplotlib figure
        """
        import networkx as nx

        # Create graph
        G = nx.Graph()

        # Add nodes
        for treatment in treatments:
            G.add_node(treatment)

        # Add edges
        for i, (t1, t2) in enumerate(comparisons):
            weight = n_studies[i] if n_studies else 1
            G.add_edge(t1, t2, weight=weight)

        # Calculate layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Draw edges with thickness proportional to number of studies
        if n_studies:
            max_studies = max(n_studies)
            edge_widths = [3 * w / max_studies for w in n_studies]
            nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray', ax=ax)

            # Add edge labels (number of studies)
            edge_labels = {(t1, t2): str(n) for (t1, t2), n in zip(comparisons, n_studies)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)
        else:
            nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray', ax=ax)

        # Draw nodes with size proportional to degree
        node_sizes = [300 * (G.degree(n) + 1) for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                              node_color=self.config.colors[:len(treatments)],
                              alpha=0.7, edgecolors='black', linewidth=2, ax=ax)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

        # Styling
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.axis('off')

        # Add legend
        if n_studies:
            legend_elements = [
                plt.Line2D([0], [0], color='gray', linewidth=3 * max_studies/max_studies,
                          label=f'Max comparisons ({max_studies} studies)'),
                plt.Line2D([0], [0], color='gray', linewidth=3 * 1/max_studies,
                          label=f'Min comparisons (1 study)')
            ]
            ax.legend(handles=legend_elements, loc='upper left', frameon=True)

        plt.tight_layout()
        return fig

    def subgroup_forest_plot(
        self,
        effects: np.ndarray,
        variances: np.ndarray,
        subgroups: np.ndarray,
        study_names: Optional[List[str]] = None,
        title: str = "Subgroup Analysis Forest Plot"
    ) -> plt.Figure:
        """
        Create forest plot with separate panels for each subgroup.

        Includes formal between-subgroup test results.

        :param effects: Study effects
        :param variances: Study variances
        :param subgroups: Subgroup labels
        :param study_names: Optional study names
        :param title: Plot title
        :return: Matplotlib figure
        """
        from evaluation.sensitivity_analysis import SubgroupAnalysis

        # Perform subgroup analysis
        result = SubgroupAnalysis.analyze(effects, variances, subgroups)

        unique_subgroups = result['subgroups'].keys()
        n_subgroups = len(unique_subgroups)

        # Calculate subplot dimensions
        n_cols = min(2, n_subgroups)
        n_rows = (n_subgroups + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
        if n_subgroups == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.flatten()

        # Plot each subgroup
        for i, subgroup in enumerate(unique_subgroups):
            ax = axes[i]
            sg_result = result['subgroups'][subgroup]

            # Get studies in this subgroup
            mask = subgroups == subgroup
            sg_effects = effects[mask]
            sg_variances = variances[mask]
            sg_names = [study_names[j] for j in np.where(mask)[0]] if study_names else None

            # Mini forest plot for this subgroup
            n_sg = len(sg_effects)
            y_pos = np.arange(n_sg)[::-1]

            ses = np.sqrt(sg_variances)
            ci_width = stats.norm.ppf(1 - self.config.confidence_level/2)

            for j, (effect, se, y) in enumerate(zip(sg_effects, ses, y_pos)):
                ci_l = effect - ci_width * se
                ci_u = effect + ci_width * se
                self._add_forest_line(ax, effect, ci_l, ci_u, y, self.config.colors[i % len(self.config.colors)])
                if sg_names and n_sg <= 15:
                    ax.text(-0.05, y, sg_names[j], ha='right', va='center', fontsize=7)

            # Add subgroup pooled effect diamond
            pooled = sg_result.pooled_effect
            ci = [sg_result.ci_lower, sg_result.ci_upper]
            self._add_forest_diamond(ax, pooled, ci, -1, self.config.colors[2],
                                   label=f"Pooled (n={n_sg})")

            # Styling
            all_effects = list(sg_effects) + [pooled]
            xlim = self._calculate_forest_xlim(all_effects,
                                              [e - ci_width * np.sqrt(v) for e, v in zip(sg_effects, sg_variances)],
                                              [e + ci_width * np.sqrt(v) for e, v in zip(sg_effects, sg_variances)])
            ax.set_xlim(xlim)
            ax.axvline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)
            ax.set_yticks([])
            ax.set_xlabel(f"{self.config.effect_measure}")
            ax.set_title(f"{subgroup}\nI²={sg_result.i_squared:.1f}%, τ²={sg_result.tau_squared:.4f}",
                        fontweight='bold')

        # Hide extra subplots
        for i in range(n_subgroups, len(axes)):
            axes[i].set_visible(False)

        # Add overall title with test result
        fig.suptitle(f"{title}\nBetween-subgroup test: Q={result['between_subgroups']['q_statistic']:.2f}, "
                    f"df={result['between_subgroups']['df']}, "
                    f"p={result['between_subgroups']['p_value']:.4f}",
                    fontweight='bold', fontsize=12)

        plt.tight_layout()
        return fig

    def cumulative_forest_plot(
        self,
        cumulative_results: pd.DataFrame,
        title: str = "Cumulative Meta-Analysis"
    ) -> plt.Figure:
        """
        Create forest plot showing cumulative evidence accumulation.

        :param cumulative_results: DataFrame from cumulative_meta_analysis()
        :param title: Plot title
        :return: Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left panel: Cumulative forest plot
        n = len(cumulative_results)
        y_pos = np.arange(n)[::-1]

        for i, row in cumulative_results.iterrows():
            y = y_pos[i]
            effect = row['pooled_effect']
            ci_l = row['ci_lower']
            ci_u = row['ci_upper']

            self._add_forest_line(ax1, effect, ci_l, ci_u, y, self.config.colors[0])

            # Add significance indicator
            if row['significant']:
                ax1.text(effect, y + 0.3, '*', ha='center', fontsize=12, fontweight='bold')

        # Reference line
        ax1.axvline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)

        # Styling
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f"{int(row['n_studies'])}" for _, row in cumulative_results.iterrows()])
        ax1.set_xlabel(f"{self.config.effect_measure} ({int(self.config.confidence_level*100)}% CI)")
        ax1.set_ylabel("Number of Studies")
        ax1.set_title(title, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Right panel: Evidence accumulation
        ax2.plot(cumulative_results['n_studies'], cumulative_results['pooled_effect'],
                marker='o', linewidth=2, markersize=6, color=self.config.colors[0], label='Pooled effect')
        ax2.fill_between(cumulative_results['n_studies'],
                        cumulative_results['ci_lower'],
                        cumulative_results['ci_upper'],
                        alpha=0.3, color=self.config.colors[0], label='95% CI')

        ax2.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)
        ax2.set_xlabel("Number of Studies")
        ax2.set_ylabel(f"{self.config.effect_measure}")
        ax2.set_title("Evidence Accumulation Over Time", fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


class DiagnosticReport:
    """
    Generate comprehensive diagnostic report with all plots.
    """

    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        self.plotter = MetaAnalysisPlots(config)

    def generate_full_report(
        self,
        effects: np.ndarray,
        variances: np.ndarray,
        study_names: Optional[List[str]] = None,
        pooled_effect: Optional[float] = None,
        pooled_ci: Optional[Tuple[float, float]] = None,
        tau2: float = 0,
        subgroups: Optional[np.ndarray] = None,
        output_path: str = "diagnostic_report.pdf"
    ) -> None:
        """
        Generate comprehensive diagnostic report with all plots.

        :param effects: Study effects
        :param variances: Study variances
        :param study_names: Optional study names
        :param pooled_effect: Pooled effect estimate
        :param pooled_ci: Pooled effect CI
        :param tau2: Between-study variance
        :param subgroups: Optional subgroup labels
        :param output_path: Path to save report (PDF or HTML)
        """
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(output_path) as pdf:
            # Forest plot
            fig = self.plotter.forest_plot(
                effects, variances, study_names, pooled_effect, pooled_ci
            )
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Funnel plot
            fig = self.plotter.funnel_plot(
                effects, variances, study_names, pooled_effect
            )
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Radial plot
            fig = self.plotter.radial_plot(
                effects, variances, study_names, pooled_effect
            )
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Baujat plot
            fig = self.plotter.baujat_plot(
                effects, variances, study_names, tau2
            )
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Galbraith plot
            fig = self.plotter.galbraith_plot(
                effects, variances, study_names, pooled_effect
            )
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Subgroup forest plot (if subgroups provided)
            if subgroups is not None:
                fig = self.plotter.subgroup_forest_plot(
                    effects, variances, subgroups, study_names
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)


if __name__ == "__main__":
    print("Statistical Diagnostic Plots Module loaded")
    print("Features:")
    print("  - Forest plots (with subgroups, sorting)")
    print("  - Funnel plots (contour-enhanced)")
    print("  - Radial plots")
    print("  - Baujat plots (influential studies)")
    print("  - Influence plots")
    print("  - L'Abbé plots (binary outcomes)")
    print("  - Galbraith plots")
    print("  - Network plots (NMA)")
    print("  - Subgroup forest plots")
    print("  - Cumulative forest plots")
    print("  - Comprehensive PDF reports")
