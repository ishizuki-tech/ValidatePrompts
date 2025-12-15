"""
Temperature and Top-P Experimentation Script for Gemma LiteRT Models

This script helps test different sampling parameter combinations to understand
their impact on:
- JSON output reliability
- Token looping behavior (repeated <end_of_turn>)
- Language mixing (Swahili vs Indonesian)
- Output diversity vs consistency

Usage:
    python temperature_top_p_experiments.py

Requirements:
    pip install transformers torch pandas matplotlib seaborn

Note: This uses Hugging Face Transformers as a proxy for understanding model
behavior. Results should be validated against actual MediaPipe LiteRT behavior
on Android, but this provides faster iteration for parameter tuning.
"""

import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Uncomment when running with actual model
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    repetition_penalty: float = 1.0


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    config: ExperimentConfig
    prompt: str
    output: str
    execution_time: float

    # Metrics
    is_valid_json: bool
    json_parse_error: Optional[str]
    has_extra_text: bool
    has_line_breaks: bool
    has_spacing_issues: bool
    end_of_turn_count: int
    token_count: int
    detected_language: str  # "swahili", "indonesian", "mixed", "unknown"

    # Additional metrics
    contains_required_keys: bool
    output_preview: str  # First 100 chars for quick inspection


class ModelExperiment:
    """
    Wrapper for running systematic experiments with different sampling parameters.
    """

    def __init__(self, model_name: str = "google/gemma-2b-it"):
        """
        Initialize experiment runner.

        Args:
            model_name: Hugging Face model identifier
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

        # Uncomment when running with actual model
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     torch_dtype=torch.float16,
        #     device_map="auto"
        # )

    def _build_prompt(self, question: str, instructions: str) -> str:
        """
        Build a prompt following Gemma chat template.

        Args:
            question: The survey question/content
            instructions: System-level instructions

        Returns:
            Formatted prompt string
        """
        return f"""<start_of_turn>user
{instructions}

Question: {question}

Respond with ONLY a JSON object. Required keys:
- analysis (string, in Swahili)
- expected_answer (string, in Swahili)
- follow_up_question (string, in Swahili)
- score (integer 0-3)

Format: RAW JSON only, ONE LINE, COMPACT (no spaces around : and ,)
All values in Swahili.
<end_of_turn>
<start_of_turn>model
"""

    def _analyze_output(
        self,
        output: str,
        required_keys: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze model output for common issues.

        Args:
            output: Raw model output
            required_keys: Expected JSON keys

        Returns:
            Dictionary of metrics
        """
        if required_keys is None:
            required_keys = ["analysis", "expected_answer", "follow_up_question", "score"]

        metrics = {
            "is_valid_json": False,
            "json_parse_error": None,
            "has_extra_text": False,
            "has_line_breaks": "\n" in output.strip(),
            "has_spacing_issues": False,
            "end_of_turn_count": output.count("<end_of_turn>"),
            "token_count": len(output.split()),
            "detected_language": "unknown",
            "contains_required_keys": False,
        }

        # Try to extract JSON
        json_text = output.strip()

        # Remove <end_of_turn> tokens if present
        json_text = json_text.replace("<end_of_turn>", "").strip()

        # Check for extra text before/after JSON
        if not (json_text.startswith("{") and json_text.endswith("}")):
            metrics["has_extra_text"] = True
            # Try to extract JSON from mixed content
            start = json_text.find("{")
            end = json_text.rfind("}") + 1
            if start >= 0 and end > start:
                json_text = json_text[start:end]

        # Try parsing JSON
        try:
            parsed = json.loads(json_text)
            metrics["is_valid_json"] = True

            # Check for required keys
            if all(key in parsed for key in required_keys):
                metrics["contains_required_keys"] = True

            # Check for spacing issues in original output
            # (compare with canonical JSON)
            canonical = json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
            if canonical not in output:
                metrics["has_spacing_issues"] = True

        except json.JSONDecodeError as e:
            metrics["json_parse_error"] = str(e)

        # Detect language (simple heuristic)
        # Indonesian: "tidak", "adalah", "yang", "untuk"
        # Swahili: "ni", "na", "kwa", "ya"
        indonesian_words = ["tidak", "adalah", "yang", "untuk", "dengan", "akan"]
        swahili_words = ["ni", "kwa", "hawezi", "hakuna", "jibu"]

        lower_output = output.lower()
        indonesian_count = sum(1 for word in indonesian_words if word in lower_output)
        swahili_count = sum(1 for word in swahili_words if word in lower_output)

        if indonesian_count > 0 and swahili_count > 0:
            metrics["detected_language"] = "mixed"
        elif indonesian_count > swahili_count:
            metrics["detected_language"] = "indonesian"
        elif swahili_count > 0:
            metrics["detected_language"] = "swahili"

        return metrics

    def run_single_experiment(
        self,
        config: ExperimentConfig,
        prompt: str
    ) -> ExperimentResult:
        """
        Run a single experiment with given config.

        Args:
            config: Sampling parameters
            prompt: Input prompt

        Returns:
            ExperimentResult with metrics
        """
        start_time = time.time()

        # Simulate model output for demonstration
        # Replace this with actual model.generate() call
        output = self._simulate_output(config)

        # Uncomment for actual model inference:
        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        # outputs = self.model.generate(
        #     **inputs,
        #     max_new_tokens=config.max_tokens,
        #     temperature=config.temperature,
        #     top_p=config.top_p,
        #     top_k=config.top_k,
        #     repetition_penalty=config.repetition_penalty,
        #     do_sample=config.temperature > 0,
        # )
        # output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        # output = output[len(prompt):]  # Remove prompt from output

        execution_time = time.time() - start_time

        # Analyze output
        metrics = self._analyze_output(output)

        return ExperimentResult(
            config=config,
            prompt=prompt,
            output=output,
            execution_time=execution_time,
            output_preview=output[:100] + "..." if len(output) > 100 else output,
            **metrics
        )

    def _simulate_output(self, config: ExperimentConfig) -> str:
        """
        Simulate model output for testing (replace with actual inference).
        Simulates various failure modes based on sampling parameters.
        """
        import random
        random.seed(int(config.temperature * 1000 + config.top_p * 100 + config.top_k))

        # Base responses with different issues
        variants = []

        # 1. Perfect compact JSON (rare with high temp)
        if config.temperature < 0.3 and random.random() < 0.7:
            variants.append(
                '{"analysis":"Jibu ni sahihi","expected_answer":"Mazao yanaharibiwa","follow_up_question":"Je, kuna hatua?","score":2}<end_of_turn>'
            )

        # 2. JSON with spacing issues (common with mid temp)
        if config.temperature > 0.3 and config.temperature < 0.7:
            variants.append(
                '{"analysis" : "Jibu ni sahihi", "expected_answer": "Mazao yanaharibiwa", "follow_up_question" : "Je, kuna hatua nyingine?", "score": 2}\n<end_of_turn>'
            )

        # 3. JSON with line breaks (high temp)
        if config.temperature > 0.6:
            variants.append(
                '''{\n  "analysis": "Jibu ni sahihi",\n  "expected_answer": "Mazao yanaharibiwa",\n  "follow_up_question": "Je, kuna hatua?",\n  "score": 2\n}<end_of_turn>'''
            )

        # 4. Extra text before JSON (high temp, low top_p)
        if config.temperature > 0.6 and config.top_p < 0.85:
            variants.append(
                'Here is my evaluation:\n{"analysis":"Jibu ni sahihi","expected_answer":"Mazao yanaharibiwa","follow_up_question":"Je?","score":2}<end_of_turn>'
            )

        # 5. Indonesian mixing (common issue)
        if random.random() < 0.4:
            variants.append(
                '{"analysis":"Jawaban tidak lengkap","expected_answer":"Hasil tanaman rusak akibat FAW","follow_up_question":"Berapa kerugian total?","score":1}<end_of_turn>'
            )

        # 6. Mixed Swahili/Indonesian
        if config.temperature > 0.5 and random.random() < 0.3:
            variants.append(
                '{"analysis":"Jibu tidak lengkap","expected_answer":"Mazao rusak karena FAW","follow_up_question":"Je, ada kerugian lain?","score":1}<end_of_turn>'
            )

        # 7. Token looping (very low temp or very high temp)
        if config.temperature < 0.15 or config.temperature > 0.85:
            loop_count = random.randint(3, 8) if config.temperature < 0.15 else random.randint(2, 4)
            variants.append(
                '{"analysis":"Jibu ni sahihi","expected_answer":"Mazao","follow_up_question":"Je?","score":2}' +
                '<end_of_turn>' * loop_count
            )

        # 8. Incomplete JSON (high temp)
        if config.temperature > 0.7 and random.random() < 0.2:
            variants.append(
                '{"analysis":"Jibu ni sahihi","expected_answer":"Mazao yanaharibiwa","follow_up_question":"Je, kuna'
            )

        # 9. Missing keys (mid-high temp)
        if config.temperature > 0.5 and random.random() < 0.25:
            variants.append(
                '{"analysis":"Jibu ni sahihi","score":2}<end_of_turn>'
            )

        # 10. Extra text after JSON (common)
        if config.temperature > 0.4 and random.random() < 0.3:
            variants.append(
                '{"analysis":"Jibu ni sahihi","expected_answer":"Mazao yanaharibiwa","follow_up_question":"Je?","score":2}\n\nI hope this helps!<end_of_turn>'
            )

        # Select variant or fallback
        if variants:
            return random.choice(variants)
        else:
            # Fallback: decent JSON with one <end_of_turn>
            return '{"analysis":"Jibu ni sahihi","expected_answer":"Mazao yanaharibiwa","follow_up_question":"Je, kuna hatua?","score":2}<end_of_turn>'

    def run_grid_search(
        self,
        temperatures: List[float],
        top_ps: List[float],
        top_ks: List[int],
        max_tokens: int = 256,
        test_prompts: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Run experiments across a grid of parameters.

        Args:
            temperatures: List of temperature values to test
            top_ps: List of top_p values to test
            top_ks: List of top_k values to test
            max_tokens: Maximum tokens per generation
            test_prompts: List of prompts to test (uses default if None)

        Returns:
            DataFrame with all results
        """
        if test_prompts is None:
            test_prompts = [
                self._build_prompt(
                    "Je, umepata hasara gani kutokana na FAW katika miaka mitatu iliyopita?",
                    "Wewe ni msaidizi wa tathmini ya majibu ya wafanyabiashara. Tathmini jibu la mkulima."
                )
            ]

        results = []

        total_experiments = len(temperatures) * len(top_ps) * len(top_ks) * len(test_prompts)
        current = 0

        for temp in temperatures:
            for top_p in top_ps:
                for top_k in top_ks:
                    config = ExperimentConfig(
                        temperature=temp,
                        top_p=top_p,
                        top_k=top_k,
                        max_tokens=max_tokens
                    )

                    for prompt_idx, prompt in enumerate(test_prompts):
                        current += 1
                        print(f"Running experiment {current}/{total_experiments} "
                              f"(T={temp}, p={top_p}, k={top_k}, prompt={prompt_idx+1})")

                        result = self.run_single_experiment(config, prompt)

                        # Flatten result for DataFrame
                        result_dict = {
                            "temperature": config.temperature,
                            "top_p": config.top_p,
                            "top_k": config.top_k,
                            "max_tokens": config.max_tokens,
                            "prompt_idx": prompt_idx,
                            "execution_time": result.execution_time,
                            "is_valid_json": result.is_valid_json,
                            "has_extra_text": result.has_extra_text,
                            "has_line_breaks": result.has_line_breaks,
                            "has_spacing_issues": result.has_spacing_issues,
                            "end_of_turn_count": result.end_of_turn_count,
                            "token_count": result.token_count,
                            "detected_language": result.detected_language,
                            "contains_required_keys": result.contains_required_keys,
                            "output_preview": result.output_preview,
                            "full_output": result.output,
                        }
                        results.append(result_dict)

        return pd.DataFrame(results)


def visualize_results(df: pd.DataFrame, output_dir: str = "experiments/results"):
    """
    Create visualizations of experiment results.

    Args:
        df: Results DataFrame from run_grid_search
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 1. Heatmap: JSON validity vs temperature and top_p
    plt.figure(figsize=(10, 8))
    pivot = df.pivot_table(
        values='is_valid_json',
        index='temperature',
        columns='top_p',
        aggfunc='mean'
    )
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1)
    plt.title('JSON Validity Rate: Temperature vs Top-P')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/json_validity_heatmap.png", dpi=300)
    plt.close()

    # 2. Bar plot: Language detection by temperature
    plt.figure(figsize=(12, 6))
    lang_by_temp = df.groupby(['temperature', 'detected_language']).size().unstack(fill_value=0)
    lang_by_temp.plot(kind='bar', stacked=True)
    plt.title('Detected Language by Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Count')
    plt.legend(title='Language')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/language_by_temperature.png", dpi=300)
    plt.close()

    # 3. Scatter: end_of_turn count vs temperature
    plt.figure(figsize=(10, 6))
    plt.scatter(df['temperature'], df['end_of_turn_count'], alpha=0.6)
    plt.xlabel('Temperature')
    plt.ylabel('End-of-turn Token Count')
    plt.title('Token Looping: End-of-turn Count vs Temperature')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/looping_vs_temperature.png", dpi=300)
    plt.close()

    # 4. Summary statistics by temperature
    summary = df.groupby('temperature').agg({
        'is_valid_json': 'mean',
        'has_extra_text': 'mean',
        'has_line_breaks': 'mean',
        'end_of_turn_count': 'mean',
        'contains_required_keys': 'mean',
    }).round(3)

    print("\n=== Summary Statistics by Temperature ===")
    print(summary)
    summary.to_csv(f"{output_dir}/summary_by_temperature.csv")

    print(f"\nVisualizations saved to {output_dir}/")


def main():
    """
    Main experiment runner.
    """
    print("=" * 60)
    print("Temperature & Top-P Experimentation for Gemma LiteRT")
    print("=" * 60)

    # Define parameter grid
    temperatures = [0.1, 0.3, 0.5, 0.7, 0.9]
    top_ps = [0.7, 0.85, 0.95]
    top_ks = [10, 20, 40]

    print(f"\nParameter grid:")
    print(f"  Temperatures: {temperatures}")
    print(f"  Top-P values: {top_ps}")
    print(f"  Top-K values: {top_ks}")
    print(f"  Total combinations: {len(temperatures) * len(top_ps) * len(top_ks)}")

    # Initialize experiment
    print("\n[WARNING] Using simulated outputs for demonstration.")
    print("Update _simulate_output() or uncomment model loading for real experiments.\n")

    experiment = ModelExperiment()

    # Run grid search
    print("\nRunning grid search...\n")
    results_df = experiment.run_grid_search(
        temperatures=temperatures,
        top_ps=top_ps,
        top_ks=top_ks,
        max_tokens=256
    )

    # Save results
    import os
    output_dir = "experiments/results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/experiment_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Print key findings
    print("\n=== Key Findings ===")
    print(f"Total experiments run: {len(results_df)}")
    print(f"Overall JSON validity rate: {results_df['is_valid_json'].mean():.1%}")
    print(f"Outputs with extra text: {results_df['has_extra_text'].mean():.1%}")
    print(f"Outputs with line breaks: {results_df['has_line_breaks'].mean():.1%}")
    print(f"Average end_of_turn count: {results_df['end_of_turn_count'].mean():.2f}")

    print("\nLanguage distribution:")
    print(results_df['detected_language'].value_counts())

    # Best configuration
    valid_results = results_df[results_df['is_valid_json'] & results_df['contains_required_keys']]
    if len(valid_results) > 0:
        best = valid_results.nsmallest(1, 'end_of_turn_count').iloc[0]
        print(f"\n=== Best Configuration (valid JSON, fewest loops) ===")
        print(f"Temperature: {best['temperature']}")
        print(f"Top-P: {best['top_p']}")
        print(f"Top-K: {best['top_k']}")
        print(f"End-of-turn count: {best['end_of_turn_count']}")
        print(f"Language: {best['detected_language']}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_results(results_df)

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
