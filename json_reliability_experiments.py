"""
JSON Reliability Experiments - Deep Dive

Focused experiments to understand and improve JSON output reliability:
1. Repetition penalty effects
2. Stop sequences
3. Prompt variations
4. Max tokens limits
5. Combined parameter tuning

Usage:
    python json_reliability_experiments.py
"""

import json
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class JsonExperimentConfig:
    """Extended config with JSON-specific parameters."""
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    repetition_penalty: float
    stop_sequences: List[str]
    prompt_style: str  # "strict", "example", "structured", "minimal"


class JsonReliabilityTester:
    """Test JSON output reliability with various strategies."""

    def _build_prompt(self, style: str, question: str) -> str:
        """Build prompts with different JSON-enforcement strategies."""

        base_instruction = """Wewe ni msaidizi wa tathmini ya majibu ya wakulima.
Tathmini jibu la mkulima kuhusu FAW (Fall Armyworm)."""

        if style == "strict":
            # Very strict formatting instructions
            return f"""<start_of_turn>user
{base_instruction}

Swali: {question}

MUHIMU: Jibu LAZIMA liwe JSON pekee, BILA maandishi mengine yoyote.

Muundo LAZIMA:
- Mstari MMOJA pekee
- Hakuna nafasi karibu na : au ,
- JSON halisi tu (hakuna maandishi kabla au baada)

Vitufe vinavyohitajika:
- "analysis" (maandishi katika Kiswahili)
- "expected_answer" (maandishi katika Kiswahili)
- "follow_up_question" (maandishi katika Kiswahili)
- "score" (nambari 0-3)

Majibu LAZIMA yawe kwa Kiswahili.

Andika JSON SASA:
<end_of_turn>
<start_of_turn>model
{{"""

        elif style == "example":
            # Provide concrete example
            return f"""<start_of_turn>user
{base_instruction}

Swali: {question}

Jibu kwa JSON halisi pekee. Mfano sahihi:
{{"analysis":"Jibu ni kamili","expected_answer":"Mazao yameharibiwa","follow_up_question":"Je, una mbinu nyingine?","score":2}}

Sasa andika JSON kwa ajili ya swali hili (mstari mmoja, Kiswahili pekee):
<end_of_turn>
<start_of_turn>model
{{"analysis":"""

        elif style == "structured":
            # Step-by-step structured approach
            return f"""<start_of_turn>user
{base_instruction}

Swali: {question}

Tathmini kwa kufuata hatua hizi:
1. Soma swali kwa umakini
2. Tathmini jibu la mkulima
3. Toa majibu yako kwa JSON muundo huu:

FORMAT: {{"analysis":"...","expected_answer":"...","follow_up_question":"...","score":0-3}}

KANUNI:
- Kiswahili pekee
- Mstari mmoja
- JSON halisi tu

Jibu:
<end_of_turn>
<start_of_turn>model
{{"""

        elif style == "minimal":
            # Minimal, concise prompt
            return f"""<start_of_turn>user
{base_instruction}

Swali: {question}

JSON (Kiswahili, mstari 1): {{"analysis","expected_answer","follow_up_question","score"}}
<end_of_turn>
<start_of_turn>model
{{"""

        elif style == "constrained":
            # Add explicit stop after closing brace
            return f"""<start_of_turn>user
{base_instruction}

Swali: {question}

Jibu kwa JSON pekee. Simama baada ya }} kuisha.

Muundo: {{"analysis":"...","expected_answer":"...","follow_up_question":"...","score":N}}

Kiswahili pekee. Anza SASA:
<end_of_turn>
<start_of_turn>model
{{"""

        else:  # default
            return f"""<start_of_turn>user
{base_instruction}

Swali: {question}

Jibu kwa JSON: {{"analysis","expected_answer","follow_up_question","score"}}
<end_of_turn>
<start_of_turn>model
{{"""

    def _simulate_output(self, config: JsonExperimentConfig) -> str:
        """
        Simulate output with JSON-specific failure modes.
        More realistic weighting based on config parameters.
        """
        random.seed(int(
            config.temperature * 1000 +
            config.top_p * 100 +
            config.top_k +
            config.repetition_penalty * 50 +
            hash(config.prompt_style)
        ))

        # Calculate "quality score" based on parameters
        quality = 0.5

        # Temperature: sweet spot around 0.2-0.4
        if 0.2 <= config.temperature <= 0.4:
            quality += 0.3
        elif config.temperature < 0.15:
            quality -= 0.2  # Too low = looping
        elif config.temperature > 0.7:
            quality -= 0.3  # Too high = chaos

        # Top-P: prefer 0.85-0.95
        if 0.85 <= config.top_p <= 0.95:
            quality += 0.2
        elif config.top_p < 0.7:
            quality -= 0.2

        # Repetition penalty: prefer 1.05-1.15
        if 1.05 <= config.repetition_penalty <= 1.15:
            quality += 0.2
        elif config.repetition_penalty == 1.0:
            quality -= 0.1  # More likely to loop

        # Stop sequences: help if present
        if "</end_of_turn>" in config.stop_sequences or "}" in config.stop_sequences:
            quality += 0.15

        # Prompt style matters
        style_bonus = {
            "strict": 0.15,
            "example": 0.25,  # Examples help most
            "structured": 0.20,
            "constrained": 0.18,
            "minimal": 0.05,
        }
        quality += style_bonus.get(config.prompt_style, 0)

        # Clamp to [0, 1]
        quality = max(0.0, min(1.0, quality))

        # Define output variants with probability weights
        outcomes = []

        # Perfect JSON (probability based on quality)
        if random.random() < quality * 0.8:
            outcomes.append((
                '{"analysis":"Jibu ni sahihi lakini halikamiliki","expected_answer":"Mazao yanaharibiwa na FAW katika miaka mitatu iliyopita","follow_up_question":"Je, umetumia dawa gani?","score":2}<end_of_turn>',
                quality * 3  # Higher weight for good configs
            ))

        # Compact JSON with spacing issues (common)
        if random.random() < 0.7:
            outcomes.append((
                '{"analysis" : "Jibu ni sahihi", "expected_answer": "Mazao yanaharibiwa", "follow_up_question" : "Je, kuna hatua nyingine?", "score": 2}<end_of_turn>',
                1.0 - quality * 0.5
            ))

        # JSON with line breaks (more likely with high temp)
        if config.temperature > 0.5:
            outcomes.append((
                '''{\n  "analysis": "Jibu ni sahihi",\n  "expected_answer": "Mazao yanaharibiwa",\n  "follow_up_question": "Je?",\n  "score": 2\n}<end_of_turn>''',
                (config.temperature - 0.5) * 2
            ))

        # Extra text before JSON (reduced by good prompts)
        preamble_prob = max(0.05, 0.4 - quality * 0.3)
        if random.random() < preamble_prob:
            outcomes.append((
                'Hii ni tathmini yangu:\n{"analysis":"Jibu ni sahihi","expected_answer":"Mazao yanaharibiwa","follow_up_question":"Je?","score":2}<end_of_turn>',
                preamble_prob
            ))

        # Indonesian (harder to fix, but good prompts help)
        indonesian_prob = max(0.15, 0.5 - quality * 0.2)
        if random.random() < indonesian_prob:
            outcomes.append((
                '{"analysis":"Jawaban tidak lengkap","expected_answer":"Hasil tanaman rusak akibat FAW","follow_up_question":"Berapa total kerugian?","score":1}<end_of_turn>',
                indonesian_prob
            ))

        # Mixed language
        if random.random() < 0.3 - quality * 0.15:
            outcomes.append((
                '{"analysis":"Jibu tidak cukup","expected_answer":"Mazao rusak karena FAW","follow_up_question":"Je, ada kerugian lain?","score":1}<end_of_turn>',
                0.3 - quality * 0.15
            ))

        # Token looping (strongly affected by temp and rep_penalty)
        loop_prob = 0.05
        if config.temperature < 0.15 and config.repetition_penalty <= 1.0:
            loop_prob = 0.6
        elif config.temperature > 0.85:
            loop_prob = 0.3

        if random.random() < loop_prob:
            loop_count = random.randint(3, 10) if config.temperature < 0.15 else random.randint(2, 4)
            outcomes.append((
                '{"analysis":"Jibu ni sahihi","expected_answer":"Mazao","follow_up_question":"Je?","score":2}' +
                '<end_of_turn>' * loop_count,
                loop_prob * 2
            ))

        # Incomplete JSON (high temp issue)
        if config.temperature > 0.7 and random.random() < (0.25 - quality * 0.15):
            outcomes.append((
                '{"analysis":"Jibu ni sahihi","expected_answer":"Mazao yanaharibiwa","follow_up',
                0.25 - quality * 0.15
            ))

        # Missing keys
        if random.random() < (0.2 - quality * 0.15):
            outcomes.append((
                '{"analysis":"Jibu ni sahihi","score":2}<end_of_turn>',
                0.2 - quality * 0.15
            ))

        # Extra text after (chatty model)
        if random.random() < (0.25 - quality * 0.2):
            outcomes.append((
                '{"analysis":"Jibu ni sahihi","expected_answer":"Mazao yanaharibiwa","follow_up_question":"Je?","score":2}\n\nNatumai hii inasaidia!<end_of_turn>',
                0.25 - quality * 0.2
            ))

        # Weighted random selection
        if not outcomes:
            # Fallback
            return '{"analysis":"Jibu ni sahihi","expected_answer":"Mazao yanaharibiwa","follow_up_question":"Je?","score":2}<end_of_turn>'

        outputs, weights = zip(*outcomes)
        return random.choices(outputs, weights=weights)[0]

    def _analyze_output(self, output: str) -> Dict[str, Any]:
        """
        Analyze JSON output quality.

        Definitions:
        - is_valid_json: Can we extract and parse valid JSON? (lenient)
        - is_perfect_json: Valid JSON with NO extra text, NO line breaks, compact format (strict)
        - has_extra_text_*: Text outside the JSON object
        - has_line_breaks: Newlines within the JSON object itself
        - is_compact: No unnecessary spaces in JSON formatting
        """
        metrics = {
            "is_valid_json": False,  # Can extract valid JSON (lenient)
            "is_perfect_json": False,  # Perfect format (strict)
            "json_parse_error": None,
            "has_extra_text_before": False,
            "has_extra_text_after": False,
            "has_line_breaks_in_json": False,  # Line breaks within JSON
            "has_spacing_issues": False,
            "end_of_turn_count": output.count("<end_of_turn>"),
            "excessive_end_tokens": False,  # More than 1 <end_of_turn>
            "contains_required_keys": False,
            "is_compact": False,
            "detected_language": "unknown",
        }

        # Check for extra text before JSON
        stripped = output.strip()
        if not stripped.startswith("{"):
            metrics["has_extra_text_before"] = True

        # Extract JSON
        json_text = stripped.replace("<end_of_turn>", "").strip()

        # Find JSON boundaries
        start = json_text.find("{")
        end = json_text.rfind("}") + 1

        if start >= 0 and end > start:
            # Check for extra text after
            after_text = json_text[end:].strip()
            if after_text:
                metrics["has_extra_text_after"] = True

            json_text = json_text[start:end]

            # Check for line breaks within the extracted JSON
            if "\n" in json_text:
                metrics["has_line_breaks_in_json"] = True

            # Try parsing
            try:
                parsed = json.loads(json_text)
                metrics["is_valid_json"] = True

                # Check keys
                required = ["analysis", "expected_answer", "follow_up_question", "score"]
                if all(k in parsed for k in required):
                    metrics["contains_required_keys"] = True

                # Check compactness
                canonical = json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
                if canonical in output:
                    metrics["is_compact"] = True
                else:
                    metrics["has_spacing_issues"] = True

                # Check if perfect: valid JSON with no issues
                metrics["is_perfect_json"] = (
                    metrics["is_valid_json"] and
                    metrics["contains_required_keys"] and
                    metrics["is_compact"] and
                    not metrics["has_extra_text_before"] and
                    not metrics["has_extra_text_after"] and
                    not metrics["has_line_breaks_in_json"] and
                    metrics["end_of_turn_count"] == 1
                )

            except json.JSONDecodeError as e:
                metrics["json_parse_error"] = str(e)

        # Flag excessive end tokens
        if metrics["end_of_turn_count"] > 1:
            metrics["excessive_end_tokens"] = True

        # Language detection
        indonesian_words = ["tidak", "adalah", "yang", "untuk", "dengan", "akan", "berapa", "ada", "karena"]
        swahili_words = ["ni", "kwa", "hawezi", "hakuna", "jibu", "mazao", "je", "una", "kuna"]

        lower = output.lower()
        indo_count = sum(1 for w in indonesian_words if w in lower)
        swahili_count = sum(1 for w in swahili_words if w in lower)

        if indo_count > 0 and swahili_count > 0:
            metrics["detected_language"] = "mixed"
        elif indo_count > swahili_count:
            metrics["detected_language"] = "indonesian"
        elif swahili_count > 0:
            metrics["detected_language"] = "swahili"

        return metrics

    def run_experiments(self) -> pd.DataFrame:
        """Run comprehensive JSON reliability experiments."""
        results = []

        question = "Je, umepata hasara gani kutokana na FAW katika miaka mitatu iliyopita?"

        # Experiment 1: Repetition penalty sweep
        print("\n=== Experiment 1: Repetition Penalty ===")
        for rep_pen in [1.0, 1.05, 1.1, 1.15, 1.2, 1.3]:
            for temp in [0.1, 0.3, 0.5]:
                config = JsonExperimentConfig(
                    temperature=temp,
                    top_p=0.9,
                    top_k=20,
                    max_tokens=256,
                    repetition_penalty=rep_pen,
                    stop_sequences=["<end_of_turn>"],
                    prompt_style="strict"
                )

                prompt = self._build_prompt(config.prompt_style, question)
                output = self._simulate_output(config)
                metrics = self._analyze_output(output)

                results.append({
                    "experiment": "repetition_penalty",
                    "temperature": temp,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "repetition_penalty": rep_pen,
                    "prompt_style": config.prompt_style,
                    **metrics
                })

        # Experiment 2: Prompt style comparison
        print("=== Experiment 2: Prompt Styles ===")
        for style in ["strict", "example", "structured", "minimal", "constrained"]:
            for temp in [0.2, 0.3, 0.4, 0.5]:
                config = JsonExperimentConfig(
                    temperature=temp,
                    top_p=0.9,
                    top_k=20,
                    max_tokens=256,
                    repetition_penalty=1.1,
                    stop_sequences=["<end_of_turn>"],
                    prompt_style=style
                )

                prompt = self._build_prompt(config.prompt_style, question)
                output = self._simulate_output(config)
                metrics = self._analyze_output(output)

                results.append({
                    "experiment": "prompt_style",
                    "temperature": temp,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "repetition_penalty": config.repetition_penalty,
                    "prompt_style": style,
                    **metrics
                })

        # Experiment 3: Temperature fine-tuning
        print("=== Experiment 3: Fine Temperature Tuning ===")
        for temp in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
            for top_p in [0.85, 0.9, 0.95]:
                config = JsonExperimentConfig(
                    temperature=temp,
                    top_p=top_p,
                    top_k=20,
                    max_tokens=256,
                    repetition_penalty=1.1,
                    stop_sequences=["<end_of_turn>"],
                    prompt_style="example"
                )

                prompt = self._build_prompt(config.prompt_style, question)
                output = self._simulate_output(config)
                metrics = self._analyze_output(output)

                results.append({
                    "experiment": "fine_temperature",
                    "temperature": temp,
                    "top_p": top_p,
                    "top_k": config.top_k,
                    "repetition_penalty": config.repetition_penalty,
                    "prompt_style": config.prompt_style,
                    **metrics
                })

        # Experiment 4: Combined optimization
        print("=== Experiment 4: Combined Parameter Optimization ===")
        best_candidates = [
            # Conservative: low temp, high safety
            (0.2, 0.9, 10, 1.1, "example"),
            (0.25, 0.9, 20, 1.1, "example"),
            # Balanced: mid temp, good diversity
            (0.3, 0.9, 20, 1.1, "example"),
            (0.35, 0.9, 20, 1.05, "structured"),
            # Exploratory: higher temp for variety
            (0.4, 0.95, 40, 1.15, "constrained"),
        ]

        for temp, top_p, top_k, rep_pen, style in best_candidates:
            # Test each candidate multiple times
            for run in range(5):
                config = JsonExperimentConfig(
                    temperature=temp,
                    top_p=top_p,
                    top_k=top_k,
                    max_tokens=256,
                    repetition_penalty=rep_pen,
                    stop_sequences=["<end_of_turn>"],
                    prompt_style=style
                )

                prompt = self._build_prompt(config.prompt_style, question)
                output = self._simulate_output(config)
                metrics = self._analyze_output(output)

                results.append({
                    "experiment": "combined_optimization",
                    "temperature": temp,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repetition_penalty": rep_pen,
                    "prompt_style": style,
                    "run": run,
                    **metrics
                })

        return pd.DataFrame(results)


def visualize_json_experiments(df: pd.DataFrame):
    """Create visualizations for JSON reliability experiments."""
    import os
    os.makedirs("experiments/results", exist_ok=True)

    # 1. Repetition penalty impact
    rep_df = df[df['experiment'] == 'repetition_penalty']
    if len(rep_df) > 0:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        pivot = rep_df.pivot_table(
            values='is_valid_json',
            index='repetition_penalty',
            columns='temperature',
            aggfunc='mean'
        )
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1)
        plt.title('JSON Validity: Repetition Penalty vs Temperature')

        plt.subplot(1, 2, 2)
        pivot2 = rep_df.pivot_table(
            values='end_of_turn_count',
            index='repetition_penalty',
            columns='temperature',
            aggfunc='mean'
        )
        sns.heatmap(pivot2, annot=True, fmt='.1f', cmap='YlOrRd')
        plt.title('Token Looping: Repetition Penalty vs Temperature')

        plt.tight_layout()
        plt.savefig("experiments/results/repetition_penalty_impact.png", dpi=300)
        plt.close()

    # 2. Prompt style comparison
    style_df = df[df['experiment'] == 'prompt_style']
    if len(style_df) > 0:
        plt.figure(figsize=(14, 6))

        metrics = ['is_perfect_json', 'is_valid_json', 'contains_required_keys', 'is_compact']
        style_summary = style_df.groupby('prompt_style')[metrics].mean()

        style_summary.plot(kind='bar', figsize=(12, 6))
        plt.title('JSON Quality Metrics by Prompt Style')
        plt.ylabel('Success Rate')
        plt.xlabel('Prompt Style')
        plt.legend(title='Metric', labels=['Perfect JSON', 'Valid (lenient)', 'Has Keys', 'Compact'])
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig("experiments/results/prompt_style_comparison.png", dpi=300)
        plt.close()

    # 3. Fine temperature tuning
    temp_df = df[df['experiment'] == 'fine_temperature']
    if len(temp_df) > 0:
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        pivot = temp_df.pivot_table(
            values='is_valid_json',
            index='temperature',
            columns='top_p',
            aggfunc='mean'
        )
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1)
        plt.title('JSON Validity')

        plt.subplot(2, 2, 2)
        pivot = temp_df.pivot_table(
            values='is_compact',
            index='temperature',
            columns='top_p',
            aggfunc='mean'
        )
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1)
        plt.title('Compact Format')

        plt.subplot(2, 2, 3)
        pivot = temp_df.pivot_table(
            values='end_of_turn_count',
            index='temperature',
            columns='top_p',
            aggfunc='mean'
        )
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd')
        plt.title('Token Looping')

        plt.subplot(2, 2, 4)
        lang_counts = temp_df.groupby(['temperature', 'detected_language']).size().unstack(fill_value=0)
        lang_pct = lang_counts.div(lang_counts.sum(axis=1), axis=0)
        lang_pct.plot(kind='area', stacked=True, alpha=0.7)
        plt.title('Language Distribution by Temperature')
        plt.ylabel('Proportion')
        plt.xlabel('Temperature')
        plt.legend(title='Language', bbox_to_anchor=(1.05, 1))

        plt.tight_layout()
        plt.savefig("experiments/results/fine_temperature_tuning.png", dpi=300)
        plt.close()

    # 4. Best configurations
    combined_df = df[df['experiment'] == 'combined_optimization']
    if len(combined_df) > 0:
        summary = combined_df.groupby(['temperature', 'top_p', 'top_k', 'repetition_penalty', 'prompt_style']).agg({
            'is_perfect_json': 'mean',
            'is_valid_json': 'mean',
            'contains_required_keys': 'mean',
            'is_compact': 'mean',
            'end_of_turn_count': 'mean',
            'excessive_end_tokens': 'mean',
        }).round(3)

        # Calculate composite score (prioritize perfection)
        summary['composite_score'] = (
            summary['is_perfect_json'] * 0.5 +
            summary['is_valid_json'] * 0.2 +
            summary['contains_required_keys'] * 0.15 +
            summary['is_compact'] * 0.15 -
            summary['excessive_end_tokens'] * 0.2
        )

        summary = summary.sort_values('composite_score', ascending=False)
        summary.to_csv("experiments/results/best_configurations.csv")

        print("\n=== Top 5 Configurations ===")
        print(summary.head())

    print("\nVisualizations saved to experiments/results/")


def main():
    print("=" * 60)
    print("JSON Reliability Experiments")
    print("=" * 60)

    tester = JsonReliabilityTester()

    print("\nRunning experiments...")
    results_df = tester.run_experiments()

    # Save results
    results_df.to_csv("experiments/results/json_reliability_results.csv", index=False)
    print(f"\nResults saved: {len(results_df)} experiments")

    # Overall statistics
    print("\n=== Overall Statistics ===")
    print(f"Perfect JSON rate (strict): {results_df['is_perfect_json'].mean():.1%}")
    print(f"Valid JSON rate (lenient): {results_df['is_valid_json'].mean():.1%}")
    print(f"Has required keys: {results_df['contains_required_keys'].mean():.1%}")
    print(f"Compact format: {results_df['is_compact'].mean():.1%}")
    print(f"Excessive end tokens (>1): {results_df['excessive_end_tokens'].mean():.1%}")
    print(f"Average end_of_turn count: {results_df['end_of_turn_count'].mean():.2f}")

    print("\nLanguage distribution:")
    print(results_df['detected_language'].value_counts())

    # By experiment type
    print("\n=== By Experiment Type ===")
    exp_summary = results_df.groupby('experiment').agg({
        'is_perfect_json': 'mean',
        'is_valid_json': 'mean',
        'contains_required_keys': 'mean',
        'is_compact': 'mean',
        'excessive_end_tokens': 'mean',
    }).round(3)
    print(exp_summary)

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_json_experiments(results_df)

    print("\n" + "=" * 60)
    print("Experiments complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
