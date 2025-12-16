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
import os
import random
import hashlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class JsonExperimentConfig:
    """Extended config with JSON-specific parameters."""
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    repetition_penalty: float
    stop_sequences: List[str]
    prompt_style: str  # "strict", "example", "structured", "minimal", "constrained"


class JsonReliabilityTester:
    """Test JSON output reliability with various strategies."""

    def _build_prompt(self, style: str, question: str) -> str:
        """Build prompts with different JSON-enforcement strategies."""
        base_instruction = """Wewe ni msaidizi wa tathmini ya majibu ya wakulima.
Tathmini jibu la mkulima kuhusu FAW (Fall Armyworm)."""

        if style == "strict":
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

        if style == "example":
            return f"""<start_of_turn>user
{base_instruction}

Swali: {question}

Jibu kwa JSON halisi pekee. Mfano sahihi:
{{"analysis":"Jibu ni kamili","expected_answer":"Mazao yameharibiwa","follow_up_question":"Je, una mbinu nyingine?","score":2}}

Sasa andika JSON kwa ajili ya swali hili (mstari mmoja, Kiswahili pekee):
<end_of_turn>
<start_of_turn>model
{{"analysis":"""

        if style == "structured":
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

        if style == "minimal":
            return f"""<start_of_turn>user
{base_instruction}

Swali: {question}

JSON (Kiswahili, mstari 1): {{"analysis","expected_answer","follow_up_question","score"}}
<end_of_turn>
<start_of_turn>model
{{"""

        if style == "constrained":
            return f"""<start_of_turn>user
{base_instruction}

Swali: {question}

Jibu kwa JSON pekee. Simama baada ya }} kuisha.

Muundo: {{"analysis":"...","expected_answer":"...","follow_up_question":"...","score":N}}

Kiswahili pekee. Anza SASA:
<end_of_turn>
<start_of_turn>model
{{"""

        return f"""<start_of_turn>user
{base_instruction}

Swali: {question}

Jibu kwa JSON: {{"analysis","expected_answer","follow_up_question","score"}}
<end_of_turn>
<start_of_turn>model
{{"""

    def _stable_seed(self, config: JsonExperimentConfig, run_id: int, question: str) -> int:
        """Create a stable, reproducible seed independent of Python's hash randomization."""
        payload = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "max_tokens": config.max_tokens,
            "repetition_penalty": config.repetition_penalty,
            "stop_sequences": config.stop_sequences,
            "prompt_style": config.prompt_style,
            "run_id": run_id,
            "question": question,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return int(digest[:16], 16)

    def _apply_max_tokens(self, text: str, max_tokens: int, approx_chars_per_token: int = 4) -> Tuple[str, bool]:
        """Apply a rough max_tokens cap by truncating characters (simulation-only)."""
        if max_tokens <= 0:
            return text, False
        max_chars = max_tokens * approx_chars_per_token
        if len(text) > max_chars:
            return text[:max_chars], True
        return text, False

    def _apply_stop_sequences(self, text: str, stop_sequences: List[str], include_stop: bool = True) -> Tuple[str, bool, Optional[str]]:
        """Cut output at the earliest stop sequence occurrence (simulation-only)."""
        if not stop_sequences:
            return text, False, None

        earliest_idx: Optional[int] = None
        earliest_seq: Optional[str] = None
        for seq in stop_sequences:
            if not seq:
                continue
            idx = text.find(seq)
            if idx != -1 and (earliest_idx is None or idx < earliest_idx):
                earliest_idx = idx
                earliest_seq = seq

        if earliest_idx is None or earliest_seq is None:
            return text, False, None

        cut = earliest_idx + (len(earliest_seq) if include_stop else 0)
        return text[:cut], True, earliest_seq

    def _simulate_output(self, config: JsonExperimentConfig, question: str, run_id: int = 0) -> Tuple[str, Dict[str, Any]]:
        """
        Simulate output with JSON-specific failure modes.
        Returns (output, meta).
        """
        seed = self._stable_seed(config, run_id=run_id, question=question)
        rng = random.Random(seed)

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
            quality -= 0.1

        # Stop sequences: help if present
        if "<end_of_turn>" in config.stop_sequences or "}" in config.stop_sequences:
            quality += 0.15

        # Prompt style matters
        style_bonus = {
            "strict": 0.15,
            "example": 0.25,
            "structured": 0.20,
            "constrained": 0.18,
            "minimal": 0.05,
        }
        quality += style_bonus.get(config.prompt_style, 0.0)

        # Clamp to [0, 1]
        quality = max(0.0, min(1.0, quality))

        outcomes: List[Tuple[str, float]] = []

        # Perfect JSON
        if rng.random() < quality * 0.8:
            outcomes.append((
                '{"analysis":"Jibu ni sahihi lakini halikamiliki","expected_answer":"Mazao yanaharibiwa na FAW katika miaka mitatu iliyopita","follow_up_question":"Je, umetumia dawa gani?","score":2}<end_of_turn>',
                quality * 3
            ))

        # Spacing issues
        if rng.random() < 0.7:
            outcomes.append((
                '{"analysis" : "Jibu ni sahihi", "expected_answer": "Mazao yanaharibiwa", "follow_up_question" : "Je, kuna hatua nyingine?", "score": 2}<end_of_turn>',
                1.0 - quality * 0.5
            ))

        # Pretty-printed JSON
        if config.temperature > 0.5:
            outcomes.append((
                '{\n  "analysis": "Jibu ni sahihi",\n  "expected_answer": "Mazao yanaharibiwa",\n  "follow_up_question": "Je?",\n  "score": 2\n}<end_of_turn>',
                (config.temperature - 0.5) * 2
            ))

        # Extra text before JSON
        preamble_prob = max(0.05, 0.4 - quality * 0.3)
        if rng.random() < preamble_prob:
            outcomes.append((
                'Hii ni tathmini yangu:\n{"analysis":"Jibu ni sahihi","expected_answer":"Mazao yanaharibiwa","follow_up_question":"Je?","score":2}<end_of_turn>',
                preamble_prob
            ))

        # Indonesian leakage
        indonesian_prob = max(0.15, 0.5 - quality * 0.2)
        if rng.random() < indonesian_prob:
            outcomes.append((
                '{"analysis":"Jawaban tidak lengkap","expected_answer":"Hasil tanaman rusak akibat FAW","follow_up_question":"Berapa total kerugian?","score":1}<end_of_turn>',
                indonesian_prob
            ))

        # Mixed language
        mixed_prob = max(0.0, 0.3 - quality * 0.15)
        if rng.random() < mixed_prob:
            outcomes.append((
                '{"analysis":"Jibu tidak cukup","expected_answer":"Mazao rusak kwa sababu ya FAW","follow_up_question":"Je, kuna kerugian nyingine?","score":1}<end_of_turn>',
                mixed_prob
            ))

        # Token looping
        loop_prob = 0.05
        if config.temperature < 0.15 and config.repetition_penalty <= 1.0:
            loop_prob = 0.6
        elif config.temperature > 0.85:
            loop_prob = 0.3

        if rng.random() < loop_prob:
            loop_count = rng.randint(3, 10) if config.temperature < 0.15 else rng.randint(2, 4)
            outcomes.append((
                '{"analysis":"Jibu ni sahihi","expected_answer":"Mazao","follow_up_question":"Je?","score":2}' +
                '<end_of_turn>' * loop_count,
                loop_prob * 2
            ))

        # Incomplete JSON (max_tokens / high temp)
        incomplete_prob = 0.0
        if config.temperature > 0.7:
            incomplete_prob = max(0.0, 0.25 - quality * 0.15)
        if rng.random() < incomplete_prob:
            outcomes.append((
                '{"analysis":"Jibu ni sahihi","expected_answer":"Mazao yanaharibiwa","follow_up',
                incomplete_prob
            ))

        # Missing keys
        missing_prob = max(0.0, 0.2 - quality * 0.15)
        if rng.random() < missing_prob:
            outcomes.append((
                '{"analysis":"Jibu ni sahihi","score":2}<end_of_turn>',
                missing_prob
            ))

        # Extra text after
        after_prob = max(0.0, 0.25 - quality * 0.2)
        if rng.random() < after_prob:
            outcomes.append((
                '{"analysis":"Jibu ni sahihi","expected_answer":"Mazao yanaharibiwa","follow_up_question":"Je?","score":2}\n\nNatumai hii inasaidia!<end_of_turn>',
                after_prob
            ))

        if not outcomes:
            base = '{"analysis":"Jibu ni sahihi","expected_answer":"Mazao yanaharibiwa","follow_up_question":"Je?","score":2}<end_of_turn>'
        else:
            outputs, weights = zip(*outcomes)
            base = rng.choices(list(outputs), weights=list(weights))[0]

        # Apply simulated max_tokens and stop sequences (order: max_tokens then stop)
        out1, was_truncated = self._apply_max_tokens(base, config.max_tokens)
        out2, was_stopped, stop_used = self._apply_stop_sequences(out1, config.stop_sequences, include_stop=True)

        meta = {
            "seed": seed,
            "quality": quality,
            "was_truncated": was_truncated,
            "was_stopped": was_stopped,
            "stop_used": stop_used,
        }
        return out2, meta

    def _analyze_output(self, output: str) -> Dict[str, Any]:
        """
        Analyze JSON output quality.

        Definitions:
        - is_valid_json: Can we extract and parse valid JSON? (lenient)
        - is_perfect_json: Valid JSON with NO extra text, NO line breaks, compact format (strict)
        - has_extra_text_*: Text outside the JSON object
        - has_line_breaks: Newlines within the JSON object itself
        - is_compact: Extracted JSON text equals canonical compact JSON
        """
        metrics: Dict[str, Any] = {
            "is_valid_json": False,
            "is_perfect_json": False,
            "json_parse_error": None,
            "has_extra_text_before": False,
            "has_extra_text_after": False,
            "has_line_breaks_in_json": False,
            "has_spacing_issues": False,
            "end_of_turn_count": output.count("<end_of_turn>"),
            "excessive_end_tokens": False,
            "contains_required_keys": False,
            "is_compact": False,
            "detected_language": "unknown",
        }

        stripped = output.strip()
        if not stripped.startswith("{"):
            metrics["has_extra_text_before"] = True

        # Remove end tokens for JSON extraction
        json_carrier = stripped.replace("<end_of_turn>", "").strip()

        # Locate first JSON object start
        start = json_carrier.find("{")
        if start < 0:
            metrics["json_parse_error"] = "No '{' found"
        else:
            candidate = json_carrier[start:]
            decoder = json.JSONDecoder()

            try:
                parsed, idx = decoder.raw_decode(candidate)
                json_text = candidate[:idx]
                after_text = candidate[idx:].strip()
                if after_text:
                    metrics["has_extra_text_after"] = True

                if "\n" in json_text:
                    metrics["has_line_breaks_in_json"] = True

                metrics["is_valid_json"] = True

                required = ["analysis", "expected_answer", "follow_up_question", "score"]
                if isinstance(parsed, dict) and all(k in parsed for k in required):
                    metrics["contains_required_keys"] = True

                canonical = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
                metrics["is_compact"] = (json_text == canonical)
                metrics["has_spacing_issues"] = not metrics["is_compact"]

                metrics["is_perfect_json"] = (
                    metrics["is_valid_json"]
                    and metrics["contains_required_keys"]
                    and metrics["is_compact"]
                    and not metrics["has_extra_text_before"]
                    and not metrics["has_extra_text_after"]
                    and not metrics["has_line_breaks_in_json"]
                    and metrics["end_of_turn_count"] == 1
                )

            except json.JSONDecodeError as e:
                metrics["json_parse_error"] = str(e)

        if metrics["end_of_turn_count"] > 1:
            metrics["excessive_end_tokens"] = True

        # Language detection (heuristic)
        indonesian_words = ["tidak", "adalah", "yang", "untuk", "dengan", "akan", "berapa", "ada", "karena", "jawaban"]
        swahili_words = ["ni", "kwa", "hakuna", "jibu", "mazao", "je", "una", "kuna", "tathmini", "funza"]

        lower = output.lower()
        indo_count = sum(1 for w in indonesian_words if w in lower)
        sw_count = sum(1 for w in swahili_words if w in lower)

        if indo_count > 0 and sw_count > 0:
            metrics["detected_language"] = "mixed"
        elif indo_count > sw_count:
            metrics["detected_language"] = "indonesian"
        elif sw_count > 0:
            metrics["detected_language"] = "swahili"

        return metrics

    def run_experiments(self) -> pd.DataFrame:
        """Run comprehensive JSON reliability experiments."""
        results: List[Dict[str, Any]] = []

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
                    prompt_style="strict",
                )
                _ = self._build_prompt(config.prompt_style, question)
                output, meta = self._simulate_output(config, question=question, run_id=0)
                metrics = self._analyze_output(output)

                results.append({
                    "experiment": "repetition_penalty",
                    "temperature": temp,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "max_tokens": config.max_tokens,
                    "repetition_penalty": rep_pen,
                    "prompt_style": config.prompt_style,
                    "stop_sequences": "|".join(config.stop_sequences),
                    **meta,
                    **metrics,
                })

        # Experiment 2: Stop sequences
        print("=== Experiment 2: Stop Sequences ===")
        stop_sets = [
            [],
            ["<end_of_turn>"],
            ["}"],
            ["}<end_of_turn>"],
            ["<end_of_turn>", "}"],
        ]
        for stop_seqs in stop_sets:
            for temp in [0.2, 0.3, 0.4, 0.5]:
                config = JsonExperimentConfig(
                    temperature=temp,
                    top_p=0.9,
                    top_k=20,
                    max_tokens=256,
                    repetition_penalty=1.1,
                    stop_sequences=stop_seqs,
                    prompt_style="example",
                )
                _ = self._build_prompt(config.prompt_style, question)
                output, meta = self._simulate_output(config, question=question, run_id=0)
                metrics = self._analyze_output(output)

                results.append({
                    "experiment": "stop_sequences",
                    "temperature": temp,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "max_tokens": config.max_tokens,
                    "repetition_penalty": config.repetition_penalty,
                    "prompt_style": config.prompt_style,
                    "stop_sequences": "|".join(config.stop_sequences) if config.stop_sequences else "(none)",
                    **meta,
                    **metrics,
                })

        # Experiment 3: Prompt style comparison
        print("=== Experiment 3: Prompt Styles ===")
        for style in ["strict", "example", "structured", "minimal", "constrained"]:
            for temp in [0.2, 0.3, 0.4, 0.5]:
                config = JsonExperimentConfig(
                    temperature=temp,
                    top_p=0.9,
                    top_k=20,
                    max_tokens=256,
                    repetition_penalty=1.1,
                    stop_sequences=["<end_of_turn>"],
                    prompt_style=style,
                )
                _ = self._build_prompt(config.prompt_style, question)
                output, meta = self._simulate_output(config, question=question, run_id=0)
                metrics = self._analyze_output(output)

                results.append({
                    "experiment": "prompt_style",
                    "temperature": temp,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "max_tokens": config.max_tokens,
                    "repetition_penalty": config.repetition_penalty,
                    "prompt_style": style,
                    "stop_sequences": "|".join(config.stop_sequences),
                    **meta,
                    **metrics,
                })

        # Experiment 4: Max tokens limits
        print("=== Experiment 4: Max Tokens ===")
        for max_t in [64, 96, 128, 192, 256, 384]:
            for temp in [0.2, 0.3, 0.5, 0.8]:
                config = JsonExperimentConfig(
                    temperature=temp,
                    top_p=0.9,
                    top_k=20,
                    max_tokens=max_t,
                    repetition_penalty=1.1,
                    stop_sequences=["<end_of_turn>"],
                    prompt_style="example",
                )
                _ = self._build_prompt(config.prompt_style, question)
                output, meta = self._simulate_output(config, question=question, run_id=0)
                metrics = self._analyze_output(output)

                results.append({
                    "experiment": "max_tokens",
                    "temperature": temp,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "max_tokens": max_t,
                    "repetition_penalty": config.repetition_penalty,
                    "prompt_style": config.prompt_style,
                    "stop_sequences": "|".join(config.stop_sequences),
                    **meta,
                    **metrics,
                })

        # Experiment 5: Combined optimization
        print("=== Experiment 5: Combined Parameter Optimization ===")
        best_candidates = [
            (0.2, 0.9, 10, 1.1, "example", ["<end_of_turn>"]),
            (0.25, 0.9, 20, 1.1, "example", ["<end_of_turn>"]),
            (0.3, 0.9, 20, 1.1, "example", ["<end_of_turn>"]),
            (0.35, 0.9, 20, 1.05, "structured", ["<end_of_turn>"]),
            (0.4, 0.95, 40, 1.15, "constrained", ["}<end_of_turn>"]),
        ]

        for temp, top_p, top_k, rep_pen, style, stop_seqs in best_candidates:
            for run in range(5):
                config = JsonExperimentConfig(
                    temperature=temp,
                    top_p=top_p,
                    top_k=top_k,
                    max_tokens=256,
                    repetition_penalty=rep_pen,
                    stop_sequences=stop_seqs,
                    prompt_style=style,
                )
                _ = self._build_prompt(config.prompt_style, question)
                output, meta = self._simulate_output(config, question=question, run_id=run)
                metrics = self._analyze_output(output)

                results.append({
                    "experiment": "combined_optimization",
                    "temperature": temp,
                    "top_p": top_p,
                    "top_k": top_k,
                    "max_tokens": config.max_tokens,
                    "repetition_penalty": rep_pen,
                    "prompt_style": style,
                    "stop_sequences": "|".join(stop_seqs) if stop_seqs else "(none)",
                    "run": run,
                    **meta,
                    **metrics,
                })

        return pd.DataFrame(results)


def visualize_json_experiments(df: pd.DataFrame):
    """Create visualizations for JSON reliability experiments."""
    os.makedirs("experiments/results", exist_ok=True)

    # 1) Repetition penalty impact
    rep_df = df[df["experiment"] == "repetition_penalty"]
    if len(rep_df) > 0:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        pivot = rep_df.pivot_table(
            values="is_valid_json",
            index="repetition_penalty",
            columns="temperature",
            aggfunc="mean",
        )
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1)
        plt.title("JSON Validity: Repetition Penalty vs Temperature")

        plt.subplot(1, 2, 2)
        pivot2 = rep_df.pivot_table(
            values="end_of_turn_count",
            index="repetition_penalty",
            columns="temperature",
            aggfunc="mean",
        )
        sns.heatmap(pivot2, annot=True, fmt=".1f", cmap="YlOrRd")
        plt.title("Token Looping: Repetition Penalty vs Temperature")

        plt.tight_layout()
        plt.savefig("experiments/results/repetition_penalty_impact.png", dpi=300)
        plt.close()

    # 2) Stop sequences comparison
    stop_df = df[df["experiment"] == "stop_sequences"]
    if len(stop_df) > 0:
        plt.figure(figsize=(12, 6))
        summary = stop_df.groupby("stop_sequences")[["is_perfect_json", "is_valid_json", "excessive_end_tokens"]].mean()
        summary = summary.sort_values("is_perfect_json", ascending=False)
        summary.plot(kind="bar", figsize=(12, 6))
        plt.title("Stop Sequences: JSON Metrics")
        plt.ylabel("Rate")
        plt.xlabel("Stop Sequences")
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig("experiments/results/stop_sequences_comparison.png", dpi=300)
        plt.close()

    # 3) Prompt style comparison
    style_df = df[df["experiment"] == "prompt_style"]
    if len(style_df) > 0:
        metrics = ["is_perfect_json", "is_valid_json", "contains_required_keys", "is_compact"]
        style_summary = style_df.groupby("prompt_style")[metrics].mean()

        style_summary.plot(kind="bar", figsize=(12, 6))
        plt.title("JSON Quality Metrics by Prompt Style")
        plt.ylabel("Success Rate")
        plt.xlabel("Prompt Style")
        plt.legend(title="Metric", labels=["Perfect JSON", "Valid (lenient)", "Has Keys", "Compact"])
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig("experiments/results/prompt_style_comparison.png", dpi=300)
        plt.close()

    # 4) Max tokens impact
    max_df = df[df["experiment"] == "max_tokens"]
    if len(max_df) > 0:
        plt.figure(figsize=(12, 6))
        pivot = max_df.pivot_table(
            values="is_valid_json",
            index="max_tokens",
            columns="temperature",
            aggfunc="mean",
        )
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1)
        plt.title("Max Tokens vs JSON Validity")
        plt.tight_layout()
        plt.savefig("experiments/results/max_tokens_validity.png", dpi=300)
        plt.close()

        plt.figure(figsize=(10, 5))
        trunc = max_df.groupby("max_tokens")["was_truncated"].mean()
        trunc.plot(kind="line", marker="o")
        plt.title("Max Tokens vs Truncation Rate (simulation)")
        plt.ylabel("Truncation Rate")
        plt.xlabel("max_tokens")
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("experiments/results/max_tokens_truncation.png", dpi=300)
        plt.close()

    # 5) Best configurations
    combined_df = df[df["experiment"] == "combined_optimization"]
    if len(combined_df) > 0:
        summary = combined_df.groupby(
            ["temperature", "top_p", "top_k", "repetition_penalty", "prompt_style", "stop_sequences"]
        ).agg({
            "is_perfect_json": "mean",
            "is_valid_json": "mean",
            "contains_required_keys": "mean",
            "is_compact": "mean",
            "end_of_turn_count": "mean",
            "excessive_end_tokens": "mean",
            "was_truncated": "mean",
        }).round(3)

        summary["composite_score"] = (
            summary["is_perfect_json"] * 0.5
            + summary["is_valid_json"] * 0.2
            + summary["contains_required_keys"] * 0.15
            + summary["is_compact"] * 0.15
            - summary["excessive_end_tokens"] * 0.2
            - summary["was_truncated"] * 0.1
        )

        summary = summary.sort_values("composite_score", ascending=False)
        summary.to_csv("experiments/results/best_configurations.csv")

        print("\n=== Top 5 Configurations ===")
        print(summary.head())

    print("\nVisualizations saved to experiments/results/")


def main():
    print("=" * 60)
    print("JSON Reliability Experiments")
    print("=" * 60)

    os.makedirs("experiments/results", exist_ok=True)

    tester = JsonReliabilityTester()

    print("\nRunning experiments...")
    results_df = tester.run_experiments()

    results_df.to_csv("experiments/results/json_reliability_results.csv", index=False)
    print(f"\nResults saved: {len(results_df)} experiments")

    print("\n=== Overall Statistics ===")
    print(f"Perfect JSON rate (strict): {results_df['is_perfect_json'].mean():.1%}")
    print(f"Valid JSON rate (lenient): {results_df['is_valid_json'].mean():.1%}")
    print(f"Has required keys: {results_df['contains_required_keys'].mean():.1%}")
    print(f"Compact format: {results_df['is_compact'].mean():.1%}")
    print(f"Excessive end tokens (>1): {results_df['excessive_end_tokens'].mean():.1%}")
    print(f"Average end_of_turn count: {results_df['end_of_turn_count'].mean():.2f}")

    print("\nLanguage distribution:")
    print(results_df["detected_language"].value_counts())

    print("\n=== By Experiment Type ===")
    exp_summary = results_df.groupby("experiment").agg({
        "is_perfect_json": "mean",
        "is_valid_json": "mean",
        "contains_required_keys": "mean",
        "is_compact": "mean",
        "excessive_end_tokens": "mean",
        "was_truncated": "mean",
    }).round(3)
    print(exp_summary)

    print("\nGenerating visualizations...")
    visualize_json_experiments(results_df)

    print("\n" + "=" * 60)
    print("Experiments complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
