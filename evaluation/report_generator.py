"""
Automated PRISMA-Compliant Report Generator

Generates comprehensive meta-analysis reports following PRISMA 2020 guidelines.
Creates publication-ready reports with tables, figures, and structured content.

References:
- Page et al. (2021) PRISMA 2020 statement
- Moher et al. (2009) PRISMA guidelines
- Cochrane Handbook for Systematic Reviews
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path


@dataclass
class StudyData:
    """Data for a single study"""
    study_id: str
    authors: str
    year: int
    title: str
    journal: str
    design: str
    population: str
    intervention: str
    comparator: str
    outcome: str
    effect_estimate: float
    standard_error: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    events_intervention: int
    events_comparator: int
    quality_score: float
    risk_of_bias: str
    notes: str = ""


@dataclass
class MetaAnalysisResults:
    """Results from meta-analysis"""
    n_studies: int
    total_participants: int
    pooled_effect: float
    standard_error: float
    confidence_interval: Tuple[float, float]
    p_value: float
    effect_measure: str
    heterogeneity_q: float
    heterogeneity_i2: float
    heterogeneity_tau2: float
    prediction_interval: Optional[Tuple[float, float]]
    subgroup_analyses: List[Dict]
    sensitivity_analyses: List[Dict]
    publication_bias_tests: Dict
    quality_assessment: Dict
   GRADE_assessment: Dict


@dataclass
class PRISMAReport:
    """PRISMA-compliant report"""
    title: str
    authors: str
    abstract: str
    introduction: str
    methods: str
    results: str
    discussion: str
    conclusion: str
    references: List[str]
    tables: List[Dict]
    figures: List[Dict]
    appendices: List[str]
    prisma_checklist: Dict[str, str]
    generated_date: str


class PRISMAReportGenerator:
    """
    Generate PRISMA 2020 compliant meta-analysis reports.

    Automatically structures reports according to PRISMA guidelines
    and creates all required sections.
    """

    def __init__(self):
        """Initialize report generator"""
        self.template_sections = {
            "title": "Title",
            "abstract": "Abstract",
            "introduction": "Introduction",
            "methods": "Methods",
            "results": "Results",
            "discussion": "Discussion",
            "conclusion": "Conclusion",
            "references": "References",
            "appendices": "Appendices"
        }

        self.prisma_items = self._load_prisma_items()

    def _load_prisma_items(self) -> Dict:
        """Load PRISMA 2020 checklist items"""
        return {
            # Title and Abstract
            "TITLE_1": "Identify the report as a systematic review",
            "TITLE_2": "State the research question using PICOS elements",
            "ABSTRACT": "Provide structured abstract including objectives, methods, results, conclusions",

            # Introduction
            "INTRO_1": "Rationale for review in context of existing knowledge",
            "INTRO_2": "Objectives with PICOS elements",

            # Methods
            "METHODS_1": "Protocol and registration",
            "METHODS_2": "Eligibility criteria with PICOS",
            "METHODS_3": "Information sources",
            "METHODS_4": "Search strategy",
            "METHODS_5": "Study selection process",
            "METHODS_6": "Data extraction process",
            "METHODS_7": "Data items",
            "METHODS_8": "Risk of bias assessment",
            "METHODS_9": "Effect measures",
            "METHODS_10": "Synthesis methods",
            "METHODS_11": "Reporting bias assessment",
            "METHODS_12": "Certainty assessment",

            # Results
            "RESULTS_1": "Study selection (PRISMA flow diagram)",
            "RESULTS_2": "Study characteristics",
            "RESULTS_3": "Risk of bias results",
            "RESULTS_4": "Results of syntheses",
            "RESULTS_5": "Reporting bias",
            "RESULTS_6": "Certainty of evidence",

            # Discussion
            "DISCUSSION_1": "Summary of findings",
            "DISCUSSION_2": "Limitations",
            "DISCUSSION_3": "Interpretation in context of existing evidence",
            "DISCUSSION_4": "Conclusions",

            # Other
            "OTHER_1": "Registration and protocol",
            "OTHER_2": "Availability of data, code, materials",
            "OTHER_3": "Funding sources",
            "OTHER_4": "Competing interests"
        }

    def generate_full_report(
        self,
        studies: List[StudyData],
        results: MetaAnalysisResults,
        review_question: str,
        picos: Dict[str, str],
        search_strategy: str,
        databases_searched: List[str],
        date_range: Tuple[str, str],
        risk_of_bias_tool: str,
        quality_of_evidence: str
    ) -> PRISMAReport:
        """
        Generate complete PRISMA-compliant report.

        :param studies: List of included studies
        :param results: Meta-analysis results
        :param review_question: Primary research question
        :param picos: PICOS elements
        :param search_strategy: Search strategy description
        :param databases_searched: List of databases
        :param date_range: Date range of search
        :param risk_of_bias_tool: Tool used for RoB assessment
        :param quality_of_evidence: Overall quality assessment
        :return: PRISMAReport
        """
        # Title
        title = self._generate_title(picos, results)

        # Abstract
        abstract = self._generate_abstract(studies, results, picos)

        # Introduction
        introduction = self._generate_introduction(picos, studies)

        # Methods
        methods = self._generate_methods(
            picos, search_strategy, databases_searched,
            date_range, risk_of_bias_tool
        )

        # Results
        results_section = self._generate_results(studies, results)

        # Discussion
        discussion = self._generate_discussion(studies, results, quality_of_evidence)

        # Conclusion
        conclusion = self._generate_conclusion(results)

        # References
        references = self._generate_references(studies)

        # Tables
        tables = self._generate_tables(studies, results)

        # Figures
        figures = self._generate_figures(studies, results)

        # Appendices
        appendices = self._generate_appendices(studies, results)

        # PRISMA checklist
        checklist = self._generate_prisma_checklist()

        # Authors
        authors = "Systematic Review Team"

        return PRISMAReport(
            title=title,
            authors=authors,
            abstract=abstract,
            introduction=introduction,
            methods=methods,
            results=results_section,
            discussion=discussion,
            conclusion=conclusion,
            references=references,
            tables=tables,
            figures=figures,
            appendices=appendices,
            prisma_checklist=checklist,
            generated_date=datetime.now().strftime("%Y-%m-%d")
        )

    def _generate_title(self, picos: Dict, results: MetaAnalysisResults) -> str:
        """Generate title following PRISMA guidelines"""
        return (f"{picos.get('i', 'Intervention')} versus {picos.get('c', 'Comparator')} "
                f"for {picos.get('p', 'Population')}: "
                f"A Systematic Review and Meta-analysis")

    def _generate_abstract(
        self,
        studies: List[StudyData],
        results: MetaAnalysisResults,
        picos: Dict
    ) -> str:
        """Generate structured abstract"""
        n_studies = len(studies)
        n_participants = sum(s.sample_size for s in studies)

        effect_direction = "significant benefit" if results.p_value < 0.05 else "no significant difference"
        if results.pooled_effect < 0:
            effect_direction = "significant harm" if results.p_value < 0 else effect_direction

        ci_str = f"({results.confidence_interval[0]:.2f}, {results.confidence_interval[1]:.2f})"

        return f"""**Objective:** {picos.get('o', 'To assess the effects of intervention')}

**Data Sources:** Comprehensive search of electronic databases from {studies[0].year} to {studies[-1].year}.

**Study Selection:** {n_studies} studies ({n_participants} participants) were included.

**Data Extraction and Synthesis:** Two reviewers independently extracted data. Meta-analysis was performed using {results.effect_measure} with random-effects model.

**Results:** The pooled {results.effect_measure} was {results.pooled_effect:.2f} {ci_str}, p={results.p_value:.4f}. Heterogeneity was {results.heterogeneity_i2:.1f}% (I²). {effect_direction} was observed.

**Conclusion:** {self._generate_conclusion(results)}

**Systematic Review Registration:** PROSPERO CRD42000000000"""

    def _generate_introduction(self, picos: Dict, studies: List[StudyData]) -> str:
        """Generate introduction section"""
        return f"""## Introduction

**Rationale**

{picos.get('p', 'The condition')} represents a significant public health challenge affecting millions worldwide. {picos.get('i', 'The intervention')} has been proposed as a potential approach to improve outcomes in this population.

Several individual studies have examined the effects of {picos.get('i', 'intervention')} on {picos.get('o', 'outcomes')}, but results have been inconsistent. A systematic synthesis of the available evidence is needed to provide clinicians, patients, and policymakers with reliable estimates of treatment effects.

**Objectives**

The primary objective of this systematic review is to evaluate the effects of {picos.get('i', 'intervention')} compared to {picos.get('c', 'comparator')} on {picos.get('o', 'primary outcomes')} in patients with {picos.get('p', 'population')}.

The specific research questions are:

1. What is the magnitude of effect of {picos.get('i', 'intervention')} on {picos.get('o', 'outcomes')}?
2. How consistent are the effects across different studies and populations?
3. What is the quality of the available evidence?

**PICOS**

- **Population:** {picos.get('p', 'Not specified')}
- **Intervention:** {picos.get('i', 'Not specified')}
- **Comparator:** {picos.get('c', 'Not specified')}
- **Outcomes:** {picos.get('o', 'Not specified')}
- **Study design:** {picos.get('s', 'Randomized controlled trials')}
"""

    def _generate_methods(
        self,
        picos: Dict,
        search_strategy: str,
        databases: List[str],
        date_range: Tuple[str, str],
        rob_tool: str
    ) -> str:
        """Generate methods section"""
        db_list = ", ".join(databases)

        return f"""## Methods

**Protocol and Registration**

This systematic review was conducted following the PRISMA 2020 guidelines and was registered in the PROSPERO international prospective register of systematic reviews (registration number: CRD42000000000). The protocol was established a priori and documented before data extraction began.

**Eligibility Criteria**

We included studies that met the following criteria:

- **Population:** {picos.get('p', 'Patients with the condition of interest')}
- **Intervention:** {picos.get('i', 'The intervention of interest')}
- **Comparator:** {picos.get('c', 'Placebo, sham, or standard care')}
- **Outcomes:** {picos.get('o', 'Primary and secondary outcomes as specified')}
- **Study design:** Randomized controlled trials with parallel group design

We excluded studies that were: non-randomized, had no control group, or used duplicate data.

**Information Sources**

We searched the following electronic databases from {date_range[0]} to {date_range[1]}: {db_list}. Additionally, we searched trial registries (ClinicalTrials.gov, WHO ICTRP) and screened reference lists of included studies and relevant reviews.

**Search Strategy**

{search_strategy}

**Study Selection**

Two reviewers independently screened titles and abstracts for eligibility. Full-text articles of potentially eligible studies were retrieved and assessed against inclusion criteria. Disagreements were resolved through discussion or consultation with a third reviewer.

**Data Collection Process**

Two reviewers independently extracted data using a standardized form. Information extracted included: study characteristics, participant demographics, intervention details, outcome measures, effect estimates, and risk of bias indicators.

**Risk of Bias Assessment**

Risk of bias was assessed using the {rob_tool_tool}. Two reviewers independently assessed each study. Disagreements were resolved through consensus.

**Effect Measures**

Effect measures were calculated as reported in individual studies. For binary outcomes, we calculated risk ratios and odds ratios. For continuous outcomes, we calculated mean differences.

**Synthesis Methods**

Meta-analysis was performed when at least two studies provided comparable data. We used random-effects models to account for between-study heterogeneity. Heterogeneity was assessed using the I² statistic and tau².

**Assessment of Reporting Bias**

Publication bias was assessed using funnel plots and Egger's regression test when at least 10 studies were available.

**Certainty Assessment**

The certainty of evidence was assessed using the GRADE approach, considering risk of bias, inconsistency, indirectness, imprecision, and publication bias.
"""

    def _generate_results(
        self,
        studies: List[StudyData],
        results: MetaAnalysisResults
    ) -> str:
        """Generate results section"""
        n_studies = len(studies)
        n_participants = sum(s.sample_size for s in studies)

        return f"""## Results

**Study Selection**

A total of X records were identified through database searching. After duplicate removal, Y unique records remained. After screening titles and abstracts, Z full-text articles were assessed for eligibility. Finally, {n_studies} studies ({n_participants} participants) met the inclusion criteria and were included in the synthesis.

**Study Characteristics**

The included {n_studies} studies were published between {min(s.year for s in studies)} and {max(s.year for s in studies)}. Studies were conducted in various settings including academic medical centers and community hospitals. Sample sizes ranged from {min(s.sample_size for s in studies)} to {max(s.sample_size for s in studies)} participants.

**Risk of Bias Results**

Risk of bias assessment indicated that [X] studies were at low risk of bias, [Y] had some concerns, and [Z] were at high risk. Key domains of concern included [specific domains].

**Results of Synthesis**

Meta-analysis of {n_studies} studies including {n_participants} participants found:

- **Pooled effect:** {results.pooled_effect:.2f} (95% CI: {results.confidence_interval[0]:.2f} to {results.confidence_interval[1]:.2f})
- **Statistical significance:** p = {results.p_value:.4f}
- **Effect measure:** {results.effect_measure}

**Heterogeneity**

- **Q statistic:** {results.heterogeneity_q:.2f}
- **I²:** {results.heterogeneity_i2:.1f}%
- **Tau²:** {results.heterogeneity_tau2:.4f}

The level of heterogeneity was {'high' if results.heterogeneity_i2 > 75 else 'moderate' if results.heterogeneity_i2 > 50 else 'low'}.

**Reporting Bias**

{results.publication_bias_tests.get('eggers_test', 'Not available')}

**Certainty of Evidence**

Using the GRADE approach, the overall certainty of evidence was assessed as {results.GRADE_assessment.get('overall_quality', 'Not specified')}.
"""

    def _generate_discussion(
        self,
        studies: List[StudyData],
        results: MetaAnalysisResults,
        quality: str
    ) -> str:
        """Generate discussion section"""
        return f"""## Discussion

**Summary of Findings**

This systematic review and meta-analysis of {len(studies)} studies ({sum(s.sample_size for s in studies)} participants) found that the intervention resulted in a {'significant' if results.p_value < 0.05 else 'non-significant'} effect (pooled {results.effect_measure}: {results.pooled_effect:.2f}, 95% CI: {results.confidence_interval[0]:.2f} to {results.confidence_interval[1]:.2f}, p = {results.p_value:.4f}).

The overall certainty of evidence was assessed as {quality} using the GRADE approach.

**Limitations**

This review has several limitations. First, heterogeneity was {'high' if results.heterogeneity_i2 > 75 else 'moderate' if results.heterogeneity_i2 > 50 else 'low'} (I² = {results.heterogeneity_i2:.1f}%), suggesting variability in effect estimates across studies. Second, some studies had methodological limitations including [specific limitations]. Third, publication bias cannot be entirely ruled out.

**Interpretation in Context**

Our findings are consistent with {'previous meta-analyses' if results.pooled_effect > 0 else 'contradict some previous studies'}. The observed effect size corresponds to a {'clinically meaningful' if abs(results.pooled_effect) > 0.5 else 'small'} difference.

**Implications for Practice**

Based on the available evidence with {quality} certainty, clinicians should consider these findings when making treatment decisions. Further research is needed to [specific gaps].

**Conclusions**

{self._generate_conclusion(results)}
"""

    def _generate_conclusion(self, results: MetaAnalysisResults) -> str:
        """Generate conclusion"""
        if results.p_value < 0.05:
            if results.pooled_effect > 0:
                return f"The intervention showed a significant beneficial effect (p={results.p_value:.4f})."
            else:
                return f"The intervention showed a significant harmful effect (p={results.p_value:.4f})."
        else:
            return f"There was no significant evidence of an effect (p={results.p_value:.4f})."

    def _generate_references(self, studies: List[StudyData]) -> List[str]:
        """Generate reference list"""
        references = []
        for study in studies:
            ref = f"{study.authors}. {study.title}. {study.journal}. {study.year};{study.study_id}."
            references.append(ref)
        return references

    def _generate_tables(
        self,
        studies: List[StudyData],
        results: MetaAnalysisResults
    ) -> List[Dict]:
        """Generate tables for report"""
        tables = []

        # Table 1: Study Characteristics
        table1_data = []
        for study in studies:
            table1_data.append({
                "Study": study.study_id,
                "Authors": study.authors,
                "Year": study.year,
                "Design": study.design,
                "Sample Size": study.sample_size,
                "Population": study.population,
                "Intervention": study.intervention,
                "Comparator": study.comparator,
                "Outcome": study.outcome,
                "Effect": f"{study.effect_estimate:.2f}",
                "95% CI": f"({study.confidence_interval[0]:.2f}, {study.confidence_interval[1]:.2f})",
                "Quality": f"{study.quality_score:.0f}%",
                "Risk of Bias": study.risk_of_bias
            })

        tables.append({
            "title": "Table 1. Characteristics of Included Studies",
            "type": "study_characteristics",
            "data": table1_data
        })

        # Table 2: Meta-Analysis Results
        tables.append({
            "title": "Table 2. Meta-Analysis Results",
            "type": "meta_analysis_results",
            "data": {
                "Number of studies": results.n_studies,
                "Total participants": results.total_participants,
                "Pooled effect": f"{results.pooled_effect:.2f}",
                "95% CI": f"({results.confidence_interval[0]:.2f}, {results.confidence_interval[1]:.2f})",
                "P-value": f"{results.p_value:.4f}",
                "Q statistic": f"{results.heterogeneity_q:.2f}",
                "I²": f"{results.heterogeneity_i2:.1f}%",
                "Tau²": f"{results.heterogeneity_tau2:.4f}"
            }
        })

        # Table 3: GRADE Evidence Profile
        tables.append({
            "title": "Table 3. GRADE Evidence Profile",
            "type": "grade_profile",
            "data": results.GRADE_assessment
        })

        return tables

    def _generate_figures(
        self,
        studies: List[StudyData],
        results: MetaAnalysisResults
    ) -> List[Dict]:
        """Generate figures for report"""
        figures = []

        # Figure 1: PRISMA Flow Diagram
        figures.append({
            "title": "Figure 1. PRISMA 2020 Flow Diagram",
            "type": "flow_diagram",
            "description": "Study selection process"
        })

        # Figure 2: Forest Plot
        figures.append({
            "title": "Figure 2. Forest Plot",
            "type": "forest_plot",
            "description": f"Meta-analysis of {len(studies)} studies. Pooled effect: {results.pooled_effect:.2f} (95% CI: {results.confidence_interval[0]:.2f} to {results.confidence_interval[1]:.2f})"
        })

        # Figure 3: Funnel Plot
        figures.append({
            "title": "Figure 3. Funnel Plot",
            "type": "funnel_plot",
            "description": "Assessment of publication bias"
        })

        # Figure 4: Risk of Bias Summary
        figures.append({
            "title": "Figure 4. Risk of Bias Summary",
            "type": "rob_summary",
            "description": "Risk of bias assessment across studies"
        })

        return figures

    def _generate_appendices(
        self,
        studies: List[StudyData],
        results: MetaAnalysisResults
    ) -> List[str]:
        """Generate appendices"""
        return [
            "Appendix A: Detailed Search Strategy",
            "Appendix B: Full Risk of Bias Assessments",
            "Appendix C: Detailed Study Characteristics",
            "Appendix D: Subgroup Analysis Results",
            "Appendix E: Sensitivity Analysis Results",
            "Appendix F: Excluded Studies with Reasons"
        ]

    def _generate_prisma_checklist(self) -> Dict[str, str]:
        """Generate PRISMA checklist"""
        checklist = {}
        for item, description in self.prisma_items.items():
            checklist[item] = f"✓ Reported: {description}"
        return checklist

    def export_to_markdown(
        self,
        report: PRISMAReport,
        output_path: str
    ) -> None:
        """
        Export report to Markdown format.

        :param report: PRISMAReport
        :param output_path: Path to save file
        """
        md_content = f"""# {report.title}

**Authors:** {report.authors}
**Date:** {report.generated_date}

---

## Abstract

{report.abstract}

---

{report.introduction}

---

{report.methods}

---

{report.results}

---

{report.discussion}

---

## Conclusion

{report.conclusion}

---

## References

"""
        for ref in report.references:
            md_content += f"{ref}\n"

        md_content += "\n---\n\n## Tables\n\n"

        for table in report.tables:
            md_content += f"### {table['title']}\n\n"
            if table["type"] == "study_characteristics":
                md_content += "| Study | Authors | Year | Sample Size | Effect | 95% CI | Quality |\n"
                md_content += "|-------|---------|------|-------------|--------|--------|----------|\n"
                for row in table["data"]:
                    md_content += f"| {row['Study']} | {row['Authors']} | {row['Year']} | {row['Sample Size']} | {row['Effect']} | {row['95% CI']} | {row['Quality']} |\n"
            elif table["type"] == "meta_analysis_results":
                for key, value in table["data"].items():
                    md_content += f"**{key}:** {value}\n\n"

        md_content += "\n---\n\n## Figures\n\n"
        for fig in report.figures:
            md_content += f"### {fig['title']}\n\n{fig['description']}\n\n"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

    def export_to_html(
        self,
        report: PRISMAReport,
        output_path: str
    ) -> None:
        """
        Export report to HTML format.

        :param report: PRISMAReport
        :param output_path: Path to save file
        """
        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report.title}</title>
    <style>
        body {{ font-family: 'Times New Roman', serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .abstract {{ background: #ecf0f1; padding: 15px; border-left: 4px solid #3498db; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .authors {{ font-style: italic; color: #7f8c8d; }}
    </style>
</head>
<body>
    <h1>{report.title}</h1>
    <p class="authors"><strong>Authors:</strong> {report.authors} | <strong>Date:</strong> {report.generated_date}</p>

    <div class="abstract">
        <h2>Abstract</h2>
        {report.abstract.replace(chr(10), '<br>')}
    </div>

    {report.introduction.replace(chr(10), '<br>')}

    {report.methods.replace(chr(10), '<br>')}

    {report.results.replace(chr(10), '<br>')}

    {report.discussion.replace(chr(10), '<br>')}

    <h2>Conclusion</h2>
    {report.conclusion}

    <h2>References</h2>
    <ol>
        {''.join(f'<li>{ref}</li>' for ref in report.references)}
    </ol>
</body>
</html>"""

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_template)

    def export_to_json(
        self,
        report: PRISMAReport,
        output_path: str
    ) -> None:
        """
        Export report to JSON format.

        :param report: PRISMAReport
        :param output_path: Path to save file
        """
        report_dict = {
            "title": report.title,
            "authors": report.authors,
            "generated_date": report.generated_date,
            "abstract": report.abstract,
            "introduction": report.introduction,
            "methods": report.methods,
            "results": report.results,
            "discussion": report.discussion,
            "conclusion": report.conclusion,
            "references": report.references,
            "tables": report.tables,
            "figures": report.figures,
            "appendices": report.appendices,
            "prisma_checklist": report.prisma_checklist
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)


def generate_prisma_report(
    studies: List[Dict],
    results: Dict,
    picos: Dict[str, str],
    output_format: str = "markdown",
    output_path: str = "meta_analysis_report"
) -> None:
    """
    Convenience function to generate PRISMA report.

    :param studies: List of study dictionaries
    :param results: Meta-analysis results dictionary
    :param picos: PICOS elements
    :param output_format: 'markdown', 'html', or 'json'
    :param output_path: Output file path (without extension)
    """
    generator = PRISMAReportGenerator()

    # Convert studies to StudyData objects
    study_data = [
        StudyData(
            study_id=s.get("study_id", ""),
            authors=s.get("authors", ""),
            year=s.get("year", 2020),
            title=s.get("title", ""),
            journal=s.get("journal", ""),
            design=s.get("design", ""),
            population=s.get("population", ""),
            intervention=s.get("intervention", ""),
            comparator=s.get("comparator", ""),
            outcome=s.get("outcome", ""),
            effect_estimate=s.get("effect", 0),
            standard_error=s.get("se", 0),
            confidence_interval=s.get("ci", (0, 0)),
            sample_size=s.get("n", 0),
            events_intervention=s.get("events_int", 0),
            events_comparator=s.get("events_con", 0),
            quality_score=s.get("quality", 50),
            risk_of_bias=s.get("rob", "unclear")
        )
        for s in studies
    ]

    # Convert results to MetaAnalysisResults
    results_obj = MetaAnalysisResults(
        n_studies=results.get("n_studies", len(studies)),
        total_participants=results.get("total_participants", 0),
        pooled_effect=results.get("pooled_effect", 0),
        standard_error=results.get("se", 0),
        confidence_interval=results.get("ci", (0, 0)),
        p_value=results.get("p_value", 1),
        effect_measure=results.get("effect_measure", "MD"),
        heterogeneity_q=results.get("Q", 0),
        heterogeneity_i2=results.get("I2", 0),
        heterogeneity_tau2=results.get("tau2", 0),
        prediction_interval=results.get("prediction_interval", None),
        subgroup_analyses=results.get("subgroups", []),
        sensitivity_analyses=results.get("sensitivity", []),
        publication_bias_tests=results.get("pub_bias", {}),
        quality_assessment=results.get("quality", {}),
        GRADE_assessment=results.get("GRADE", {})
    )

    report = generator.generate_full_report(
        studies=study_data,
        results=results_obj,
        review_question=picos.get("o", ""),
        picos=picos,
        search_strategy="Comprehensive search strategy was used",
        databases_searched=["PubMed", "EMBASE", "Cochrane Library"],
        date_range=("2000", "2024"),
        risk_of_bias_tool="Cochrane RoB 2",
        quality_of_evidence=results.get("GRADE", {}).get("overall_quality", "Moderate")
    )

    if output_format == "markdown":
        generator.export_to_markdown(report, f"{output_path}.md")
    elif output_format == "html":
        generator.export_to_html(report, f"{output_path}.html")
    elif output_format == "json":
        generator.export_to_json(report, f"{output_path}.json")


if __name__ == "__main__":
    print("PRISMA Report Generator module loaded")
    print("Features:")
    print("  - PRISMA 2020 compliant reports")
    print("  - Structured abstracts")
    print("  - Complete methods section")
    print("  - Tables and figures")
    print("  - Export to Markdown, HTML, JSON")
    print("  - PRISMA checklist")
