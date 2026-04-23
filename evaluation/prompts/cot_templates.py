# sentinel:skip-file — hardcoded paths / templated placeholders are fixture/registry/audit-narrative data for this repo's research workflow, not portable application configuration. Same pattern as push_all_repos.py and E156 workbook files.
"""
Advanced prompting templates with Chain-of-Thought reasoning.

Provides improved prompts for LLM extraction using CoT and other techniques.
"""

from typing import Dict, List, Optional


class PromptTemplate:
    """Base class for prompt templates."""

    def __init__(self, template: str):
        """Initialize template."""
        self.template = template

    def format(self, **kwargs) -> str:
        """Format template with variables."""
        return self.template.format(**kwargs)


class ChainOfThoughtTemplate(PromptTemplate):
    """
    Chain-of-Thought prompting template for extraction.

    Encourages step-by-step reasoning before final answer.
    """

    BINARY_OUTCOME_COT = """You are a clinical research data extraction expert. Your task is to extract numerical data from a randomized controlled trial.

{instructions}

Let me think through this step by step:

**Step 1: Understand the outcome**
I need to identify the outcome being measured: {outcome}

**Step 2: Identify the study design**
This is a randomized controlled trial comparing:
- Intervention: {intervention}
- Comparator: {comparator}

**Step 3: Locate the results section**
I will search for results related to "{outcome}" in the text below.

**Step 4: Extract the data**
For each group, I need to find:
- Number of participants (total)
- Number of participants with the event (for binary outcomes)

**Step 5: Verify the data**
I will check that:
- The numbers make logical sense (events ≤ total)
- The data corresponds to the correct outcome
- The time point matches (if specified)

**Study Text:**
{study_text}

**My Analysis:**
{{analysis}}

**Final Answer (in YAML format):**
```yaml
intervention:
  events: {{intervention_events}}
  total: {{intervention_total}}
comparator:
  events: {{comparator_events}}
  total: {{comparator_total}}
notes: {{notes}}
```

Remember:
- Use 0 for events if none occurred
- Use "unknown" if the data is not available
- Extract the EXACT numbers from the text
- Include units and time periods in notes
"""

    CONTINUOUS_OUTCOME_COT = """You are a clinical research data extraction expert. Your task is to extract numerical data from a randomized controlled trial.

{instructions}

Let me work through this systematically:

**Step 1: Identify the outcome and measurement**
Outcome: {outcome}
This is a continuous outcome measured as: {measurement_type}

**Step 2: Understand the groups**
- Intervention group: {intervention}
- Comparator group: {comparator}

**Step 3: Find the summary statistics**
I need to locate for each group:
- Mean (average value)
- Standard deviation (SD) or other measure of variability
- Sample size (n)

**Step 4: Check the time point**
The measurement was taken at: {time_point}

**Step 5: Extract and verify**
I will extract the values and verify:
- Means are within expected range
- SD values are reasonable for this outcome
- Sample sizes match the study description

**Study Text:**
{study_text}

**My Analysis:**
{{analysis}}

**Final Answer (in YAML format):**
```yaml
intervention:
  mean: {{intervention_mean}}
  standard_deviation: {{intervention_sd}}
  sample_size: {{intervention_n}}
comparator:
  mean: {{comparator_mean}}
  standard_deviation: {{comparator_sd}}
  sample_size: {{comparator_n}}
notes: {{notes}}
```

Remember:
- Extract EXACT values from the text
- Use standard deviation if available, otherwise use standard error
- If only confidence interval is available, note that in notes
- Use "unknown" for any missing values
"""

    def __init__(self, outcome_type: str = "binary"):
        """Initialize CoT template."""
        if outcome_type == "binary":
            super().__init__(self.BINARY_OUTCOME_COT)
        elif outcome_type == "continuous":
            super().__init__(self.CONTINUOUS_OUTCOME_COT)
        else:
            raise ValueError(f"Unknown outcome type: {outcome_type}")


class SelfConsistencyTemplate(PromptTemplate):
    """
    Self-consistency template for extraction.

    Generates multiple reasoning paths and selects most consistent answer.
    """

    def __init__(self):
        """Initialize self-consistency template."""
        template = """You are a clinical research data extraction expert. Extract data from the following trial.

**Instructions:**
{instructions}

**Study:** {study_title}
**Outcome:** {outcome}
**Intervention:** {intervention}
**Comparator:** {comparator}

**Study Text:**
{study_text}

**Extraction Task:**
Provide your answer in the following format, thinking through each step:

1. **Locate the relevant section:** Where in the text is this outcome reported?
2. **Identify the data points:** What specific numbers do I need to extract?
3. **Extract for each group:** What are the values for intervention and comparator?
4. **Verify consistency:** Do these numbers make sense together?
5. **Final answer:** Present the extracted data

Think through each step carefully, then provide your final answer in YAML format:

```yaml
intervention:
  events: {{intervention_events}}
  total: {{intervention_total}}
comparator:
  events: {{comparator_events}}
  total: {{comparator_total}}
```
"""
        super().__init__(template)


class RefinementTemplate(PromptTemplate):
    """
    Template for iterative refinement of extractions.

    Used when initial extraction may be incomplete or uncertain.
    """

    def __init__(self):
        """Initialize refinement template."""
        template = """You are a clinical research data extraction expert. I need you to REVIEW and REFINE an extraction.

**Original Extraction Request:**
- Study: {study_title}
- Outcome: {outcome}
- Intervention: {intervention}
- Comparator: {comparator}

**Study Text:**
{study_text}

**Initial Extraction (may be incomplete or incorrect):**
```yaml
{initial_extraction}
```

**Your Task:**
Review the extraction and the study text. Determine if the extraction is:

1. **Complete:** Are all required values present?
2. **Accurate:** Do the values match what's in the text?
3. **Consistent:** Do the numbers make logical sense?

**Issues to Check:**
- Are events ≤ total for each group?
- Do the numbers correspond to the correct outcome and time point?
- Is the data in the correct units?
- Are there any discrepancies between text and tables?

**If the extraction is correct, respond with:**
CORRECT: {{initial_extraction}}

**If the extraction needs revision, provide:**
REVISED:
```yaml
intervention:
  events: {{corrected_events}}
  total: {{corrected_total}}
comparator:
  events: {{corrected_events}}
  total: {{corrected_total}}
notes: {{explanation_of_changes}}
```
"""
        super().__init__(template)


class VerificationTemplate(PromptTemplate):
    """
    Template for verifying extracted data.

    Used for quality control and validation.
    """

    def __init__(self):
        """Initialize verification template."""
        template = """You are a clinical research data quality expert. VERIFY the following extraction.

**Extraction to Verify:**
```yaml
intervention:
  events: {intervention_events}
  total: {intervention_total}
comparator:
  events: {comparator_events}
  total: {comparator_total}
```

**Context:**
- Study: {study_title}
- Outcome: {outcome}
- Intervention: {intervention}
- Comparator: {comparator}

**Relevant Study Text:**
{study_text}

**Verification Checklist:**

1. **Completeness:**
   - [ ] All four values are present (not unknown)
   - [ ] Values are in the expected format
   - [ ] Time point is clear from text

2. **Accuracy:**
   - [ ] Intervention events matches text
   - [ ] Intervention total matches text
   - [ ] Comparator events matches text
   - [ ] Comparator total matches text

3. **Consistency:**
   - [ ] intervention_events ≤ intervention_total
   - [ ] comparator_events ≤ comparator_total
   - [ ] Total numbers match study description
   - [ ] Data corresponds to correct outcome

4. **Clarity:**
   - [ ] No ambiguity about which groups are being compared
   - [ ] Time point is specified
   - [ ] Units are clear

**Your Assessment:**

For each check, indicate PASS or FAIL with brief explanation:

**Completeness:** {{completeness_assessment}}
**Accuracy:** {{accuracy_assessment}}
**Consistency:** {{consistency_assessment}}
**Clarity:** {{clarity_assessment}}

**Overall:** {{overall_pass_fail}}

If any check fails, specify what needs to be corrected:

**Issues Found:** {{issues_or_none}}
"""
        super().__init__(template)


class FewShotTemplate(PromptTemplate):
    """
    Few-shot learning template with examples.

    Provides examples to guide extraction.
    """

    def __init__(self):
        """Initialize few-shot template."""
        template = """You are a clinical research data extraction expert. Extract data from randomized controlled trials.

**Instructions:**
- Extract the number of events and total participants for each group
- Report data exactly as it appears in the text
- Use 0 if no events occurred
- Use "unknown" if data is not available

**Example 1:**
Study: "Smith et al. (2023) evaluated the effect of Drug X on mortality. In the Drug X group (n=100), 15 patients died. In the placebo group (n=100), 22 patients died."

Correct extraction:
```yaml
intervention:
  events: 15
  total: 100
comparator:
  events: 22
  total: 100
```

**Example 2:**
Study: "Jones et al. (2022) assessed response rates. Of 50 patients receiving Treatment A, 30 showed response. In the control arm of 48 patients, 25 responded."

Correct extraction:
```yaml
intervention:
  events: 30
  total: 50
comparator:
  events: 25
  total: 48
```

**Example 3:**
Study: "Lee et al. (2024) reported adverse events. None of the 75 patients in the experimental group experienced serious adverse events. In the standard care group (n=73), 2 patients had serious adverse events."

Correct extraction:
```yaml
intervention:
  events: 0
  total: 75
comparator:
  events: 2
  total: 73
```

---

**Now extract data from the following study:**

**Study:** {study_title}
**Outcome:** {outcome}
**Intervention:** {intervention}
**Comparator:** {comparator}

**Study Text:**
{study_text}

**Provide your extraction in YAML format:**
```yaml
intervention:
  events: {{intervention_events}}
  total: {{intervention_total}}
comparator:
  events: {{comparator_events}}
  total: {{comparator_total}}
notes: {{additional_notes}}
```
"""
        super().__init__(template)


class PromptOptimizer:
    """
    Optimize prompts through iterative refinement.

    Tests different prompt variations and selects best performer.
    """

    def __init__(self):
        """Initialize prompt optimizer."""
        self.prompt_variations = []
        self.performance_history = []

    def add_variation(self, name: str, template: str) -> None:
        """Add a prompt variation to test."""
        self.prompt_variations.append({
            'name': name,
            'template': template,
            'uses': 0,
            'successes': 0
        })

    def record_performance(self, name: str, success: bool) -> None:
        """Record performance of a prompt variation."""
        for variation in self.prompt_variations:
            if variation['name'] == name:
                variation['uses'] += 1
                if success:
                    variation['successes'] += 1
                break

    def get_best_prompt(self) -> str:
        """Get the best performing prompt."""
        if not self.prompt_variations:
            return ""

        best = max(
            self.prompt_variations,
            key=lambda x: x['successes'] / max(x['uses'], 1)
        )
        return best['template']

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all variations."""
        stats = {}
        for variation in self.prompt_variations:
            if variation['uses'] > 0:
                stats[variation['name']] = {
                    'uses': variation['uses'],
                    'successes': variation['successes'],
                    'success_rate': variation['successes'] / variation['uses']
                }
        return stats


def get_chain_of_thought_prompt(
    outcome_type: str,
    study_context: Dict[str, str],
    study_text: str
) -> str:
    """
    Get a Chain-of-Thought prompt for extraction.

    Args:
        outcome_type: Type of outcome ('binary' or 'continuous')
        study_context: Dictionary with study information
        study_text: Full text of the study

    Returns:
        Formatted prompt string
    """
    template = ChainOfThoughtTemplate(outcome_type)

    return template.format(
        instructions=study_context.get('instructions', ''),
        outcome=study_context.get('outcome', ''),
        intervention=study_context.get('intervention', ''),
        comparator=study_context.get('comparator', ''),
        measurement_type=study_context.get('measurement_type', ''),
        time_point=study_context.get('time_point', ''),
        study_text=study_text
    )


def get_few_shot_prompt(
    study_context: Dict[str, str],
    study_text: str,
    examples: Optional[List[Dict]] = None
) -> str:
    """
    Get a Few-shot learning prompt.

    Args:
        study_context: Dictionary with study information
        study_text: Full text of the study
        examples: Optional list of example extractions

    Returns:
        Formatted prompt string
    """
    template = FewShotTemplate()

    return template.format(
        study_title=study_context.get('title', ''),
        outcome=study_context.get('outcome', ''),
        intervention=study_context.get('intervention', ''),
        comparator=study_context.get('comparator', ''),
        study_text=study_text
    )
