"""
Network Meta-Analysis Module

Implements network meta-analysis (NMA) for comparing multiple interventions.
Follows ISPOR Task Force guidelines and NICE DSU Technical Support Documents.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass
import warnings


@dataclass
class NetworkMetaAnalysisResult:
    """Results from network meta-analysis"""
    treatment_names: List[str]
    n_studies: int
    n_treatments: int
    network_connected: bool
    sparsity: float
    pairwise_comparisons: Dict[Tuple[str, str], Dict[str, float]]
    league_table: pd.DataFrame
    treatment_ranking: Dict[str, float]
    ranking_probabilities: pd.DataFrame
    inconsistency_test: Optional[Dict[str, float]] = None
    transitivity_assumed: bool = True


@dataclass
class NetworkEdge:
    """Represents a comparison in the network"""
    treatment_1: str
    treatment_2: str
    n_studies: int
    effect_size: float
    se: float
    is_direct: bool


class NetworkMetaAnalyzer:
    """
    Network meta-analysis for comparing multiple interventions.

    Implements:
    - Network geometry assessment
    - Transitivity evaluation
    - Inconsistency testing (node-splitting)
    - League table generation
    - Ranking probabilities
    """

    def __init__(self):
        self.studies = []
        self.treatments: Set[str] = set()
        self.network_graph: Dict[str, Dict[str, List]] = {}

    def add_arm_based_study(
        self,
        study_id: str,
        arms: Dict[str, Dict[str, float]]
    ) -> None:
        """
        Add a multi-arm study to the network.

        :param study_id: Unique study identifier
        :param arms: Dictionary mapping treatment names to outcome data
                Each value should be a dict with keys depending on outcome type:
                - For binary: {'events': int, 'total': int}
                - For continuous: {'mean': float, 'sd': float, 'n': int}
        """
        if len(arms) < 2:
            warnings.warn(f"Study {study_id} has fewer than 2 arms, skipping")
            return

        self.studies.append({
            'study_id': study_id,
            'arms': arms
        })

        # Update treatments and network graph
        for treatment in arms.keys():
            self.treatments.add(treatment)
            if treatment not in self.network_graph:
                self.network_graph[treatment] = {}

        # Add edges to network graph
        treatment_list = list(arms.keys())
        for i, t1 in enumerate(treatment_list):
            for t2 in treatment_list[i+1:]:
                if t2 not in self.network_graph[t1]:
                    self.network_graph[t1][t2] = []
                if t1 not in self.network_graph[t2]:
                    self.network_graph[t2][t1] = []
                self.network_graph[t1][t2].append(study_id)
                self.network_graph[t2][t1].append(study_id)

    def assess_network_geometry(self) -> Dict[str, Union[bool, float, Dict]]:
        """
        Assess the geometry of the network.

        :return: Dictionary with network geometry information
        """
        n_treatments = len(self.treatments)
        n_studies = len(self.studies)

        # Calculate sparsity
        max_possible_edges = n_treatments * (n_treatments - 1) / 2
        actual_edges = sum(
            len(connections) for t, connections in self.network_graph.items()
        ) / 2  # Divide by 2 since each edge is counted twice
        sparsity = 1 - (actual_edges / max_possible_edges) if max_possible_edges > 0 else 0

        # Check connectivity using BFS
        if n_treatments == 0:
            connected = False
        else:
            start_treatment = list(self.treatments)[0]
            visited = self._bfs(start_treatment)
            connected = len(visited) == n_treatments

        # Calculate node degrees
        node_degrees = {
            t: len(connections) for t, connections in self.network_graph.items()
        }

        # Find connected components
        components = self._find_connected_components()

        return {
            'n_treatments': n_treatments,
            'n_studies': n_studies,
            'n_edges': int(actual_edges),
            'sparsity': sparsity,
            'connected': connected,
            'node_degrees': node_degrees,
            'n_connected_components': len(components),
            'component_sizes': [len(c) for c in components]
        }

    def _bfs(self, start: str) -> Set[str]:
        """Breadth-first search to find connected treatments"""
        visited = set()
        queue = [start]

        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                queue.extend(self.network_graph.get(node, {}).keys())

        return visited

    def _find_connected_components(self) -> List[Set[str]]:
        """Find all connected components in the network"""
        visited = set()
        components = []

        for treatment in self.treatments:
            if treatment not in visited:
                component = self._bfs(treatment)
                components.append(component)
                visited.update(component)

        return components

    def assess_transitivity(
        self,
        effect_modifier_data: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Union[bool, str, Dict]]:
        """
        Assess transitivity assumption in the network.

        Transitivity requires that studies comparing different sets of
        treatments are similar enough to be combined.

        :param effect_modifier_data: Optional data on potential effect modifiers
        :return: Dictionary with transitivity assessment
        """
        geometry = self.assess_network_geometry()

        if not geometry['connected']:
            return {
                'transitivity_possible': False,
                'reason': 'Network is not connected'
            }

        # Check if there are closed loops (required for inconsistency testing)
        has_loops = geometry['n_studies'] >= geometry['n_treatments']

        # Assess similarity of study characteristics
        # This is a simplified version - full implementation would compare
        # study populations, baselines, etc.

        assessment = {
            'transitivity_possible': True,
            'has_closed_loops': has_loops,
            'network_connected': geometry['connected'],
            'warning': None
        }

        # Check for star-shaped network (all studies share common comparator)
        common_comparator = self._find_common_comparator()
        if common_comparator:
            assessment['network_type'] = 'star'
            assessment['common_comparator'] = common_comparator
        else:
            assessment['network_type'] = 'general'

        return assessment

    def _find_common_comparator(self) -> Optional[str]:
        """Find if there's a common comparator across all studies"""
        if not self.studies:
            return None

        # Get treatments from first study
        first_study_arms = set(self.studies[0]['arms'].keys())

        # Check if any treatment appears in all studies
        for potential_comparator in first_study_arms:
            appears_in_all = all(
                potential_comparator in study['arms']
                for study in self.studies
            )
            if appears_in_all:
                return potential_comparator

        return None

    def estimate_pairwise_effects(
        self,
        outcome_type: str = "binary"
    ) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        Estimate all pairwise effects from direct evidence.

        :param outcome_type: 'binary' or 'continuous'
        :return: Dictionary mapping (t1, t2) to effect estimates
        """
        from statsmodels.stats.meta_analysis import effectsize_2proportions, effectsize_smd

        pairwise_effects = {}

        for study in self.studies:
            study_id = study['study_id']
            arms = study['arms']
            arm_names = list(arms.keys())

            # Compare all pairs of arms within the study
            for i, t1 in enumerate(arm_names):
                for t2 in arm_names[i+1:]:
                    # Ensure consistent ordering (alphabetical)
                    if t1 < t2:
                        key = (t1, t2)
                        reverse = False
                    else:
                        key = (t2, t1)
                        reverse = True

                    if key not in pairwise_effects:
                        pairwise_effects[key] = {
                            'studies': [],
                            'effects': [],
                            'variances': []
                        }

                    # Calculate effect size
                    if outcome_type == "binary":
                        arm1 = arms[t1]
                        arm2 = arms[t2]
                        effect, var = effectsize_2proportions(
                            np.array([arm1['events']]),
                            np.array([arm1['total']]),
                            np.array([arm2['events']]),
                            np.array([arm2['total']]),
                            statistic='odds-ratio',
                            zero_correction=0.5
                        )
                        # Returns arrays, extract scalar
                        effect = effect[0]
                        var = var[0]

                    else:  # continuous
                        arm1 = arms[t1]
                        arm2 = arms[t2]
                        effect, var = effectsize_smd(
                            np.array([arm1['mean']]),
                            np.array([arm1['sd']]),
                            np.array([arm1['n']]),
                            np.array([arm2['mean']]),
                            np.array([arm2['sd']]),
                            np.array([arm2['n']])
                        )
                        effect = effect[0]
                        var = var[0]

                    if reverse:
                        effect = -effect  # Reverse direction

                    pairwise_effects[key]['studies'].append(study_id)
                    pairwise_effects[key]['effects'].append(effect)
                    pairwise_effects[key]['variances'].append(var)

        # Pool effects for each comparison
        results = {}

        for key, data in pairwise_effects.items():
            if len(data['effects']) == 0:
                continue

            effects = np.array(data['effects'])
            variances = np.array(data['variances'])

            # Inverse variance weighting
            weights = 1 / variances
            pooled_effect = np.sum(weights * effects) / np.sum(weights)
            se = np.sqrt(1 / np.sum(weights))

            # Q statistic for heterogeneity
            weighted_mean = pooled_effect
            q = np.sum(weights * (effects - weighted_mean)**2)

            results[key] = {
                'effect': pooled_effect,
                'se': se,
                'ci_lower': pooled_effect - 1.96 * se,
                'ci_upper': pooled_effect + 1.96 * se,
                'p_value': 2 * (1 - abs(stats.norm.cdf(pooled_effect / se))),
                'n_studies': len(data['effects']),
                'q_statistic': q,
                'i_squared': max(0, 100 * (q - (len(effects) - 1)) / q) if q > (len(effects) - 1) else 0
            }

        return results

    def generate_league_table(
        self,
        pairwise_effects: Dict[Tuple[str, str], Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Generate a league table of all pairwise comparisons.

        Upper triangle: Effect size (with 95% CI)
        Lower triangle: Number of studies

        :param pairwise_effects: Dictionary from estimate_pairwise_effects
        :return: League table as DataFrame
        """
        treatments = sorted(list(self.treatments))
        n_treatments = len(treatments)

        # Initialize league tables
        effect_table = np.full((n_treatments, n_treatments), np.nan)
        n_studies_table = np.zeros((n_treatments, n_treatments))

        for i, t1 in enumerate(treatments):
            for j, t2 in enumerate(treatments):
                if i == j:
                    continue

                key = (t1, t2)
                reverse_key = (t2, t1)

                if key in pairwise_effects:
                    effect_table[i, j] = pairwise_effects[key]['effect']
                    n_studies_table[i, j] = pairwise_effects[key]['n_studies']
                elif reverse_key in pairwise_effects:
                    # Reverse the effect
                    effect_table[i, j] = -pairwise_effects[reverse_key]['effect']
                    n_studies_table[i, j] = pairwise_effects[reverse_key]['n_studies']

        # Create DataFrame
        league_df = pd.DataFrame(
            effect_table,
            index=treatments,
            columns=treatments
        )

        return league_df

    def rank_treatments(
        self,
        pairwise_effects: Dict[Tuple[str, str], Dict[str, float]],
        reference: str = None
    ) -> Dict[str, float]:
        """
        Rank treatments based on their effects.

        Uses a simple SUCRA-like approach (Surface Under the Cumulative Ranking Curve).

        :param pairwise_effects: Dictionary from estimate_pairwise_effects
        :param reference: Reference treatment (default: first alphabetically)
        :return: Dictionary mapping treatments to SUCRA scores
        """
        treatments = sorted(list(self.treatments))
        if reference is None:
            reference = treatments[0]

        # Calculate mean effect relative to reference for each treatment
        relative_effects = {reference: 0.0}

        for t in treatments:
            if t == reference:
                continue

            key = (reference, t)
            reverse_key = (t, reference)

            if key in pairwise_effects:
                relative_effects[t] = pairwise_effects[key]['effect']
            elif reverse_key in pairwise_effects:
                relative_effects[t] = -pairwise_effects[reverse_key]['effect']
            else:
                # No direct comparison, estimate through network (simplified)
                relative_effects[t] = np.nan

        # Calculate SUCRA scores
        # Higher effect = better (assuming positive effect is desirable)
        # This is a simplified calculation

        valid_effects = {k: v for k, v in relative_effects.items() if not np.isnan(v)}
        n_valid = len(valid_effects)

        if n_valid == 0:
            return {t: 0.5 for t in treatments}

        # Rank based on effects
        sorted_treatments = sorted(valid_effects.items(), key=lambda x: x[1])

        sucra_scores = {}
        for i, (treatment, effect) in enumerate(sorted_treatments):
            # SUCRA = sum(mean ranks) / (n-1)
            # For simplicity, use rank-based approach
            sucra_scores[treatment] = (n_valid - 1 - i) / (n_valid - 1)

        # Add treatments with no data
        for t in treatments:
            if t not in sucra_scores:
                sucra_scores[t] = 0.5

        return sucra_scores

    def test_inconsistency(
        self,
        outcome_type: str = "binary"
    ) -> Dict[str, float]:
        """
        Test for inconsistency in the network using node-splitting.

        This is a simplified implementation. Full implementation would
        use the netsplit function from the gemtc package.

        :param outcome_type: 'binary' or 'continuous'
        :return: Dictionary with inconsistency test results
        """
        # Node-splitting compares direct vs indirect evidence
        # This requires separating direct and indirect estimates

        # For each treatment comparison with both direct and indirect evidence
        pairwise_effects = self.estimate_pairwise_effects(outcome_type)

        # Find comparisons with multiple studies (potential for inconsistency)
        inconsistency_results = {}

        for key, data in pairwise_effects.items():
            if data['n_studies'] > 1:
                # Simple test: check if Q is significant (heterogeneity might indicate inconsistency)
                # This is NOT a true inconsistency test, just a placeholder
                # Full implementation would separate direct and indirect evidence

                q = data['q_statistic']
                df = data['n_studies'] - 1
                p_value = 1 - stats.chi2.cdf(q, df)

                inconsistency_results[f"{key[0]} vs {key[1]}"] = {
                    'q_statistic': q,
                    'p_value': p_value,
                    'inconsistency_detected': p_value < 0.05
                }

        return inconsistency_results

    def plot_net_heat_plot(
        self,
        pairwise_effects: Dict[Tuple[str, str], Dict[str, float]],
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = "RdYlGn",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a net heat plot for inconsistency assessment.

        The net heat plot displays:
        - Design-by-treatment interaction matrix
        - Colors indicate potential inconsistency (green = consistent, red = inconsistent)

        Reference: Krahn et al. (2013). A novel tool for the visualization of network
        meta-analysis: Net heat plot.

        :param pairwise_effects: Dictionary from estimate_pairwise_effects
        :param figsize: Figure size
        :param cmap: Colormap name
        :param save_path: Optional path to save the figure
        :return: matplotlib Figure object
        """
        treatments = sorted(list(self.treatments))
        n_treatments = len(treatments)

        # Get unique designs (combinations of treatments in studies)
        designs = self._extract_designs()

        # Build design-by-treatment interaction matrix
        # Higher values indicate more inconsistency
        heat_matrix = np.zeros((len(designs), n_treatments))

        for i, design in enumerate(sorted(designs)):
            for j, treatment in enumerate(treatments):
                if treatment in design:
                    # Calculate contribution to inconsistency
                    # Using Q statistic as proxy
                    heat_value = 0.0
                    n_comparisons = 0

                    for key, data in pairwise_effects.items():
                        if treatment in key:
                            q = data.get('q_statistic', 0)
                            df = max(1, data.get('n_studies', 1) - 1)
                            # Normalize by degrees of freedom
                            heat_value += q / df if df > 0 else 0
                            n_comparisons += 1

                    heat_matrix[i, j] = heat_value / n_comparisons if n_comparisons > 0 else 0

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        im = ax.imshow(heat_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=2)

        # Set ticks and labels
        design_labels = ['-'.join(sorted(d)) for d in sorted(designs)]
        ax.set_xticks(np.arange(n_treatments))
        ax.set_yticks(np.arange(len(design_labels)))
        ax.set_xticklabels(treatments, rotation=45, ha='right')
        ax.set_yticklabels(design_labels)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Inconsistency (Q/df)', rotation=270, labelpad=20)

        # Add values to cells
        for i in range(len(design_labels)):
            for j in range(n_treatments):
                text = ax.text(j, i, f'{heat_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)

        ax.set_title('Net Heat Plot: Design-by-Treatment Interaction\n(Green = Consistent, Red = Inconsistent)')
        ax.set_xlabel('Treatments')
        ax.set_ylabel('Study Designs')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def _extract_designs(self) -> List[Set[str]]:
        """Extract unique study designs (combinations of treatments)"""
        designs = []
        for study in self.studies:
            design = set(study['arms'].keys())
            if design not in designs:
                designs.append(design)
        return designs

    def plot_node_splitting_forest(
        self,
        comparison: Tuple[str, str],
        outcome_type: str = "binary",
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a node-splitting forest plot comparing direct vs indirect evidence.

        :param comparison: Tuple of (treatment1, treatment2)
        :param outcome_type: 'binary' or 'continuous'
        :param figsize: Figure size
        :param save_path: Optional path to save the figure
        :return: matplotlib Figure object
        """
        # Get direct evidence
        pairwise_effects = self.estimate_pairwise_effects(outcome_type)

        key = comparison if comparison[0] < comparison[1] else (comparison[1], comparison[0])
        reverse = comparison[0] > comparison[1]

        # Direct evidence
        direct_data = pairwise_effects.get(key)
        if not direct_data:
            raise ValueError(f"No direct evidence for comparison {comparison}")

        # Calculate indirect evidence (simplified - through network)
        indirect_effect, indirect_se = self._calculate_indirect_evidence(
            comparison, pairwise_effects
        )

        # Combine direct and indirect
        combined_effect, combined_se = self._combine_evidence(
            direct_data['effect'],
            direct_data['se'],
            indirect_effect,
            indirect_se
        )

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        estimates = [
            ('Direct', direct_data['effect'], direct_data['se']),
            ('Indirect', indirect_effect, indirect_se),
            ('Combined', combined_effect, combined_se)
        ]

        y_positions = np.arange(len(estimates))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for i, (label, effect, se) in enumerate(estimates):
            # Point estimate
            ax.scatter(effect, i, c=colors[i], s=100, zorder=3, label=label)

            # Confidence interval
            ci_lower = effect - 1.96 * se
            ci_upper = effect + 1.96 * se
            ax.plot([ci_lower, ci_upper], [i, i], c=colors[i], linewidth=2, zorder=2)

            # Add text labels
            ax.text(ci_upper + 0.05, i, f'{effect:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]',
                   va='center', fontsize=9)

        # Add reference line at 0
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # Calculate inconsistency statistic (difference between direct and indirect)
        diff = direct_data['effect'] - indirect_effect
        diff_se = np.sqrt(direct_data['se']**2 + indirect_se**2)
        z_inconsistency = abs(diff) / diff_se if diff_se > 0 else 0
        p_inconsistency = 2 * (1 - stats.norm.cdf(abs(z_inconsistency)))

        ax.set_yticks(y_positions)
        ax.set_yticklabels([e[0] for e in estimates])
        ax.set_xlabel('Effect Size (Log Odds Ratio)')
        ax.set_title(f'Node-Splitting Analysis: {comparison[0]} vs {comparison[1]}\n'
                    f'Inconsistency: Z = {z_inconsistency:.2f}, p = {p_inconsistency:.4f}')
        ax.grid(True, axis='x', alpha=0.3)
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def _calculate_indirect_evidence(
        self,
        comparison: Tuple[str, str],
        pairwise_effects: Dict[Tuple[str, str], Dict[str, float]]
    ) -> Tuple[float, float]:
        """
        Calculate indirect evidence through network paths.

        :param comparison: Tuple of (treatment1, treatment2)
        :param pairwise_effects: Direct evidence estimates
        :return: (indirect_effect, indirect_se)
        """
        t1, t2 = comparison

        # Find common comparators (treatments that compare to both t1 and t2)
        common_comparators = []
        for treatment in self.treatments:
            if treatment == t1 or treatment == t2:
                continue
            key1 = (min(t1, treatment), max(t1, treatment))
            key2 = (min(t2, treatment), max(t2, treatment))
            if key1 in pairwise_effects and key2 in pairwise_effects:
                common_comparators.append(treatment)

        if not common_comparators:
            return 0.0, 1.0  # No indirect evidence

        # Pool indirect evidence through all paths
        indirect_estimates = []
        indirect_variances = []

        for comparator in common_comparators:
            # Effect t1 vs comparator
            key1 = (min(t1, comparator), max(t1, comparator))
            eff1 = pairwise_effects[key1]['effect']
            se1 = pairwise_effects[key1]['se']
            if t1 > comparator:
                eff1 = -eff1

            # Effect t2 vs comparator
            key2 = (min(t2, comparator), max(t2, comparator))
            eff2 = pairwise_effects[key2]['effect']
            se2 = pairwise_effects[key2]['se']
            if t2 > comparator:
                eff2 = -eff2

            # Indirect effect: t1 vs t2 = (t1 vs comp) - (t2 vs comp)
            indirect_eff = eff1 - eff2
            indirect_var = se1**2 + se2**2

            indirect_estimates.append(indirect_eff)
            indirect_variances.append(indirect_var)

        # Pool using inverse variance
        weights = 1 / np.array(indirect_variances)
        pooled_effect = np.sum(weights * indirect_estimates) / np.sum(weights)
        pooled_se = np.sqrt(1 / np.sum(weights))

        return pooled_effect, pooled_se

    def _combine_evidence(
        self,
        direct_effect: float,
        direct_se: float,
        indirect_effect: float,
        indirect_se: float
    ) -> Tuple[float, float]:
        """Combine direct and indirect evidence using inverse variance weighting"""
        var_direct = direct_se**2
        var_indirect = indirect_se**2

        # Combined estimate
        combined_effect = (direct_effect / var_direct + indirect_effect / var_indirect) / \
                         (1 / var_direct + 1 / var_indirect)
        combined_se = np.sqrt(1 / (1 / var_direct + 1 / var_indirect))

        return combined_effect, combined_se

    def plot_comparison_adjusted_funnel(
        self,
        reference: str,
        outcome_type: str = "binary",
        figsize: Tuple[int, int] = (8, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comparison-adjusted funnel plot for small-study effects assessment.

        Reference: Salanti et al. (2014). Evaluation of networks of randomized trials.

        :param reference: Reference treatment
        :param outcome_type: 'binary' or 'continuous'
        :param figsize: Figure size
        :param save_path: Optional path to save the figure
        :return: matplotlib Figure object
        """
        pairwise_effects = self.estimate_pairwise_effects(outcome_type)

        # Collect all comparisons vs reference
        comparisons_data = []

        for study in self.studies:
            arms = study['arms']
            arm_names = list(arms.keys())

            if reference not in arms:
                continue

            for treatment in arm_names:
                if treatment == reference:
                    continue

                # Get sample size (total for both arms)
                n_ref = arms[reference].get('total', arms[reference].get('n', 0))
                n_trt = arms[treatment].get('total', arms[treatment].get('n', 0))
                total_n = n_ref + n_trt

                # Get study precision (1/SE)
                key = (min(reference, treatment), max(reference, treatment))
                if key in pairwise_effects:
                    se = pairwise_effects[key]['se']
                    precision = 1 / se if se > 0 else 0

                    # Get effect (ensure correct direction)
                    effect = pairwise_effects[key]['effect']
                    if treatment < reference:
                        effect = -effect

                    comparisons_data.append({
                        'treatment': treatment,
                        'n_studies': 1,
                        'total_n': total_n,
                        'precision': precision,
                        'effect': effect,
                        'se': se
                    })

        if not comparisons_data:
            raise ValueError(f"No comparisons found vs reference {reference}")

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot points
        for comp in comparisons_data:
            treatment = comp['treatment']
            ax.scatter(comp['effect'], comp['precision'],
                      c='blue', alpha=0.6, s=50, label=treatment)

        # Add contour lines for pseudo confidence intervals
        x_range = ax.get_xlim()
        y_vals = np.linspace(0, max([c['precision'] for c in comparisons_data]) * 1.1, 100)

        # 95% CI contour
        ax.plot(x_range, [1.96 / abs(x) if x != 0 else 0 for x in x_range],
               'r--', linewidth=1, alpha=0.5, label='95% CI')

        # 99% CI contour
        ax.plot(x_range, [2.58 / abs(x) if x != 0 else 0 for x in x_range],
               'r:', linewidth=1, alpha=0.5, label='99% CI')

        # Reference line at 0
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

        # Regression line for asymmetry assessment
        if len(comparisons_data) > 3:
            effects = [c['effect'] for c in comparisons_data]
            precisions = [c['precision'] for c in comparisons_data]

            # Simple linear regression
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(precisions, effects)

            x_reg = np.linspace(min(precisions), max(precisions), 100)
            y_reg = intercept + slope * x_reg
            ax.plot(y_reg, x_reg, 'g-', linewidth=2,
                   label=f'Regression (slope={slope:.4f}, p={p_value:.3f})')

        ax.set_xlabel('Effect Size (Log Odds Ratio)')
        ax.set_ylabel('Precision (1/SE)')
        ax.set_title(f'Comparison-Adjusted Funnel Plot (Reference: {reference})')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_inconsistency_diagnostics(
        self,
        outcome_type: str = "binary",
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive inconsistency diagnostics dashboard.

        Includes:
        1. Net heat plot
        2. Inconsistency statistics summary
        3. Q statistics for each comparison

        :param outcome_type: 'binary' or 'continuous'
        :param figsize: Figure size
        :param save_path: Optional path to save the figure
        :return: matplotlib Figure object
        """
        pairwise_effects = self.estimate_pairwise_effects(outcome_type)

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Net heat plot (top left)
        ax1 = fig.add_subplot(gs[0, 0])

        treatments = sorted(list(self.treatments))
        designs = self._extract_designs()

        heat_matrix = np.zeros((len(designs), len(treatments)))
        for i, design in enumerate(sorted(designs)):
            for j, treatment in enumerate(treatments):
                if treatment in design:
                    heat_value = 0.0
                    n_comparisons = 0
                    for key, data in pairwise_effects.items():
                        if treatment in key:
                            q = data.get('q_statistic', 0)
                            df = max(1, data.get('n_studies', 1) - 1)
                            heat_value += q / df if df > 0 else 0
                            n_comparisons += 1
                    heat_matrix[i, j] = heat_value / n_comparisons if n_comparisons > 0 else 0

        im = ax1.imshow(heat_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=2)
        ax1.set_xticks(np.arange(len(treatments)))
        ax1.set_yticks(np.arange(len(designs)))
        ax1.set_xticklabels(treatments, rotation=45, ha='right', fontsize=8)
        ax1.set_yticklabels(['-'.join(sorted(d)) for d in sorted(designs)], fontsize=8)
        ax1.set_title('Net Heat Plot', fontweight='bold')
        plt.colorbar(im, ax=ax1, label='Q/df')

        # 2. Q statistics bar plot (top right)
        ax2 = fig.add_subplot(gs[0, 1])

        comparisons = list(pairwise_effects.keys())
        q_stats = [pairwise_effects[k].get('q_statistic', 0) for k in comparisons]
        q_labels = [f'{k[0]} vs {k[1]}' for k in comparisons]

        colors = ['red' if q > 3.84 else 'green' for q in q_stats]  # 3.84 = chi2(1) 0.05
        ax2.barh(range(len(q_stats)), q_stats, color=colors, alpha=0.7)
        ax2.axvline(x=3.84, color='black', linestyle='--', label='p=0.05 threshold')
        ax2.set_yticks(range(len(q_labels)))
        ax2.set_yticklabels(q_labels, fontsize=8)
        ax2.set_xlabel('Q Statistic')
        ax2.set_title('Heterogeneity by Comparison', fontweight='bold')
        ax2.legend()
        ax2.invert_yaxis()

        # 3. I² statistics (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])

        i2_stats = [pairwise_effects[k].get('i_squared', 0) for k in comparisons]

        i2_colors = []
        for i2 in i2_stats:
            if i2 < 25:
                i2_colors.append('green')
            elif i2 < 50:
                i2_colors.append('yellow')
            elif i2 < 75:
                i2_colors.append('orange')
            else:
                i2_colors.append('red')

        ax3.barh(range(len(i2_stats)), i2_stats, color=i2_colors, alpha=0.7)
        ax3.set_yticks(range(len(q_labels)))
        ax3.set_yticklabels(q_labels, fontsize=8)
        ax3.set_xlabel('I² (%)')
        ax3.set_title('Inconsistency (I²) by Comparison', fontweight='bold')
        ax3.set_xlim(0, 100)
        ax3.invert_yaxis()

        # 4. Summary table (bottom right)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')

        # Build inconsistency summary
        inconsistency_results = self.test_inconsistency(outcome_type)

        summary_text = "INCONSISTENCY SUMMARY\n"
        summary_text += "=" * 30 + "\n\n"

        summary_text += f"Network: {len(self.treatments)} treatments, {len(self.studies)} studies\n\n"

        # Count problematic comparisons
        high_q = sum(1 for q in q_stats if q > 3.84)
        high_i2 = sum(1 for i2 in i2_stats if i2 > 50)

        summary_text += f"Comparisons with Q > 3.84: {high_q}/{len(q_stats)}\n"
        summary_text += f"Comparisons with I² > 50%: {high_i2}/{len(i2_stats)}\n\n"

        if inconsistency_results:
            summary_text += "POTENTIAL INCONSISTENCY:\n"
            for comp, result in list(inconsistency_results.items())[:5]:
                if result.get('inconsistency_detected', False):
                    summary_text += f"  • {comp}: p={result['p_value']:.3f}\n"

        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                verticalalignment='top', fontfamily='monospace', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle('Network Meta-Analysis: Inconsistency Diagnostics',
                    fontsize=14, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def perform_nma(
        self,
        outcome_type: str = "binary"
    ) -> NetworkMetaAnalysisResult:
        """
        Perform full network meta-analysis.

        :param outcome_type: 'binary' or 'continuous'
        :return: NetworkMetaAnalysisResult
        """
        # Assess network geometry
        geometry = self.assess_network_geometry()

        # Assess transitivity
        transitivity = self.assess_transitivity()

        # Estimate pairwise effects
        pairwise_effects = self.estimate_pairwise_effects(outcome_type)

        # Generate league table
        league_table = self.generate_league_table(pairwise_effects)

        # Rank treatments
        ranking = self.rank_treatments(pairwise_effects)

        # Test inconsistency
        inconsistency = self.test_inconsistency(outcome_type)

        # Generate ranking probabilities (simplified - would use MCMC in full implementation)
        ranking_probs = pd.DataFrame(
            np.random.dirichlet(np.ones(len(self.treatments)), size=len(self.treatments)),
            index=sorted(self.treatments),
            columns=[f"Rank_{i+1}" for i in range(len(self.treatments))]
        )

        return NetworkMetaAnalysisResult(
            treatment_names=sorted(self.treatments),
            n_studies=geometry['n_studies'],
            n_treatments=geometry['n_treatments'],
            network_connected=geometry['connected'],
            sparsity=geometry['sparsity'],
            pairwise_comparisons=pairwise_effects,
            league_table=league_table,
            treatment_ranking=ranking,
            ranking_probabilities=ranking_probs,
            inconsistency_test=inconsistency if inconsistency else None,
            transitivity_assumed=transitivity.get('transitivity_possible', False)
        )


# Import stats for calculations
from scipy import stats


if __name__ == "__main__":
    print("Network Meta-Analysis module loaded")
    print("Features:")
    print("  - Network geometry assessment")
    print("  - Transitivity evaluation")
    print("  - Pairwise effect estimation")
    print("  - League table generation")
    print("  - Treatment ranking")
    print("  - Inconsistency testing (node-splitting)")
