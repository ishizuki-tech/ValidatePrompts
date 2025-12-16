# SLM Prompt Composer + JSON Reliability Experiments (Gemma / LiteRT-LM)

Tools for:

1. composing Gemma-style chat prompts from a YAML survey/SLM config, and
2. running offline experiments to understand structured JSON output reliability (format adherence, looping `<end_of_turn>`, language mixing).

This repo assumes you are targeting **MediaPipe Tasks GenAI (`LlmInference` / `LlmInferenceSession`)** with a **Gemma LiteRT model**, but the experiment scripts can also be used as a fast offline proxy (simulated outputs by default).

---

## Contents

* `compose_prompt.py`

  * Compose a final prompt string (Gemma chat turn markers) from a YAML config.
  * Prints SLM sampling settings by default (**to stderr**) as `key=value` (or YAML/JSON).
  * Supports stdin config (`-`) and watchdog mode (`--empty`).

* `json_reliability_experiments.py`

  * Focused experiments for JSON reliability: repetition penalty, stop sequences, prompt styles, max token limits, and combined tuning.
  * Generates CSV summaries and plots.

* `temperature_top_p_experiments.py`

  * Grid search across sampling parameters (`temperature`, `top_p`, `top_k`).
  * Tracks JSON validity, looping behavior (repeated `<end_of_turn>`), and language mixing.

---

## Requirements

* Python 3.10+ recommended (3.9+ should work)

---

## Install Dependencies

### Option A: Install directly with pip (recommended)

```bash
python -m pip install -U pip
python -m pip install -U pyyaml pandas matplotlib seaborn
```

### Option B: Use a virtual environment (clean & reproducible)

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows PowerShell

python -m pip install -U pip
python -m pip install -U pyyaml pandas matplotlib seaborn
```

### Optional: Real model inference with Hugging Face Transformers (instead of simulation)

If you want to replace the simulated generation with real HF generation (for quick iteration on a desktop), install:

```bash
python -m pip install -U "transformers>=4.55.0" "accelerate>=0.30.0" torch
```

### (Optional) `requirements.txt` approach

Create `requirements.txt`:

```txt
pyyaml
pandas
matplotlib
seaborn
```

Install:

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

---

## Directory Layout (suggested)

```text
.
├─ compose_prompt.py
├─ json_reliability_experiments.py
├─ temperature_top_p_experiments.py
├─ experiments/
│  └─ results/                # CSV + plots are written here
└─ config/
   └─ survey_config.yaml       # your YAML config (example name)
```

Create the output folder once:

```bash
mkdir -p experiments/results
```

---

## 1) Prompt Composition: `compose_prompt.py`

### What it does

* Loads a YAML config from a file path OR stdin (`-`)
* Finds a node’s `question` from `graph.nodes` by `id`
* Finds a node prompt template from `prompts` by `nodeId`
* Assembles a final prompt using `slm.*` blocks (`preamble`, `key_contract`, etc.)
* Wraps it with Gemma chat markers by default:

  * `<start_of_turn>user`
  * `<end_of_turn>`
  * `<start_of_turn>model`
* Prints SLM sampling settings by default to **stderr** (safe for piping):

  * disable with `--no-show-slm`
  * change format with `--slm-format kv|yaml|json`

### Expected YAML shape (minimum)

```yaml
graph:
  nodes:
    - id: Q1
      question: "Your question text here"

prompts:
  - nodeId: Q1
    prompt: |
      Question: {{QUESTION}}
      Answer: {{ANSWER}}

slm:
  user_turn_prefix: "<start_of_turn>user"
  model_turn_prefix: "<start_of_turn>model"
  turn_end: "<end_of_turn>"

  preamble: |
    You are a well-known farmer survey expert. Read the Question and the Answer.

  key_contract: |
    OUTPUT KEYS:
    - analysis
    - expected answer
    - follow-up question
    - score

  length_budget: |
    Keep it short and concrete.

  scoring_rule: |
    Scoring rule: Judge ONLY content relevance/completeness/accuracy.

  strict_output: |
    STRICT OUTPUT:
    - RAW JSON only, ONE LINE.

  empty_json_instruction: "Respond with an empty JSON object: {}"

  # Optional sampling settings for printing/debug
  accelerator: "GPU"
  max_tokens: 256
  top_k: 20
  top_p: 0.9
  temperature: 0.3
```

### Usage

#### Compose prompt for a node (question resolved from `graph`)

```bash
python compose_prompt.py configs/survey_config10.yaml --node Q1 --answer "Test answer"
```

#### Compose prompt for Q1–Q6 quickly (example answers)

```bash
python compose_prompt.py configs/survey_config10.yaml --node Q1 --answer "Over the last 3 seasons, I lose about 20% of my maize yield to FAW on average (about 3 bags per acre)."
python compose_prompt.py configs/survey_config10.yaml --node Q2 --answer "I would give up up to 5% of yield (about 1 bag per acre) to harvest 10 days earlier."
python compose_prompt.py configs/survey_config10.yaml --node Q3 --answer "If 30% or more of plants or ears show moderate-to-severe pest/disease damage, especially around tasseling, I would stop using the variety and switch."
python compose_prompt.py configs/survey_config10.yaml --node Q4 --answer "A replacement variety should mature in 110–120 days (DTM), have strong FAW tolerance, tolerate drought at tasseling/silking, and yield about 18–20 bags per acre in a normal season."
python compose_prompt.py configs/survey_config10.yaml --node Q5 --answer "In a bad year, I would still replant if I get at least 50% of my usual harvest (around 8 bags per acre); below that I would not plant it again."
python compose_prompt.py configs/survey_config10.yaml --node Q6 --answer "Drought is most devastating at tasseling/silking because it reduces pollination and grain set."


python compose_prompt.py configs/survey_config20.yaml --node Q1 --answer "Katika misimu mitatu iliyopita, kwa wastani napoteza takribani 20% ya mavuno ya mahindi kutokana na FAW (takribani magunia 3 kwa ekari)."
python compose_prompt.py configs/survey_config20.yaml --node Q2 --answer "Ningekubali kupoteza hadi 5% ya mavuno (takribani gunia 1 kwa ekari) ili kuvuna siku 10 mapema."
python compose_prompt.py configs/survey_config20.yaml --node Q3 --answer "Ikiwa 30% au zaidi ya mimea au masikio yanaonyesha uharibifu wa wastani hadi mkali wa wadudu au magonjwa, hasa wakati wa kutoa maua, ningeacha aina hiyo na kubadili."
python compose_prompt.py configs/survey_config20.yaml --node Q4 --answer "Aina mbadala inapaswa kukomaa ndani ya siku 110-120 (DTM), iwe na uvumilivu mkubwa dhidi ya FAW, ivumilie ukame wakati wa kutoa maua/kuchanua, na itoe takribani magunia 18-20 kwa ekari katika msimu wa kawaida."
python compose_prompt.py configs/survey_config20.yaml --node Q5 --answer "Katika mwaka mbaya, ningeendelea kupanda tena ikiwa napata angalau 50% ya mavuno yangu ya kawaida (takribani magunia 8 kwa ekari); chini ya hapo nisingepanda tena."
python compose_prompt.py configs/survey_config20.yaml --node Q6 --answer "Ukame huathiri zaidi wakati wa kutoa maua/kuchanua kwa sababu hupunguza uchavushaji na kutengenezwa kwa punje."


python compose_prompt.py configs/survey_config30.yaml --node Q1 --answer "Katika misimu mitatu iliyopita, kwa wastani napoteza takribani 20% ya mavuno ya mahindi kutokana na FAW (takribani magunia 3 kwa ekari)."
python compose_prompt.py configs/survey_config30.yaml --node Q2 --answer "Ningekubali kupoteza hadi 5% ya mavuno (takribani gunia 1 kwa ekari) ili kuvuna siku 10 mapema."
python compose_prompt.py configs/survey_config30.yaml --node Q3 --answer "Ikiwa 30% au zaidi ya mimea au masikio yanaonyesha uharibifu wa wastani hadi mkali wa wadudu au magonjwa, hasa wakati wa kutoa maua, ningeacha aina hiyo na kubadili."
python compose_prompt.py configs/survey_config30.yaml --node Q4 --answer "Aina mbadala inapaswa kukomaa ndani ya siku 110-120 (DTM), iwe na uvumilivu mkubwa dhidi ya FAW, ivumilie ukame wakati wa kutoa maua/kuchanua, na itoe takribani magunia 18-20 kwa ekari katika msimu wa kawaida."
python compose_prompt.py configs/survey_config30.yaml --node Q5 --answer "Katika mwaka mbaya, ningeendelea kupanda tena ikiwa napata angalau 50% ya mavuno yangu ya kawaida (takribani magunia 8 kwa ekari); chini ya hapo nisingepanda tena."
python compose_prompt.py configs/survey_config30.yaml --node Q6 --answer "Ukame huathiri zaidi wakati wa kutoa maua/kuchanua kwa sababu hupunguza uchavushaji na kutengenezwa kwa punje."
```

#### Override the question text explicitly

```bash
python compose_prompt.py configs/survey_config10.yaml --node Q1 \
  --question "Override question here" \
  --answer "Test answer"
```

#### Force empty JSON instruction (watchdog fallback tests)

```bash
python compose_prompt.py configs/survey_config10.yaml --node Q1 --answer "Anything" --empty
```

#### Disable SLM settings printing (keep prompt output clean)

```bash
python compose_prompt.py configs/survey_config10.yaml --node Q1 --answer "Test" --no-show-slm
```

#### Change SLM settings print format (stderr)

```bash
python compose_prompt.py configs/survey_config10.yaml --node Q1 --answer "Test" --slm-format yaml
python compose_prompt.py configs/survey_config10.yaml --node Q1 --answer "Test" --slm-format json
```

#### Dump only SLM settings (stdout) and exit

```bash
python compose_prompt.py configs/survey_config10.yaml --dump-slm --slm-format json
```

#### Read YAML from stdin

```bash
cat configs/survey_config10.yaml | python compose_prompt.py - --node Q1 --answer "Test"
```

#### Omit Gemma turn markers (for debugging/experiments)

```bash
python compose_prompt.py configs/survey_config10.yaml --node Q1 --answer "Test" --no-turn-markers
```

#### Keep ONLY the prompt on stdout (drop the settings)

```bash
python compose_prompt.py configs/survey_config10.yaml --node Q1 --answer "Test" 2>/dev/null
```

---

## 2) JSON Reliability Deep-Dive: `json_reliability_experiments.py`

### What it measures

The analyzer computes metrics like:

* `is_valid_json`: can we extract and parse valid JSON (lenient)
* `is_perfect_json`: strict format (compact, no extra text, one-line, single end token)
* `contains_required_keys`
* `has_extra_text_before` / `has_extra_text_after`
* `has_line_breaks_in_json`
* `has_spacing_issues` (not compact separators)
* `end_of_turn_count` and `excessive_end_tokens`
* language heuristic: `swahili` / `indonesian` / `mixed`

### Run

```bash
python json_reliability_experiments.py
```

### Outputs

Written to:

* `experiments/results/json_reliability_results.csv`
* `experiments/results/repetition_penalty_impact.png`
* `experiments/results/prompt_style_comparison.png`
* `experiments/results/fine_temperature_tuning.png`
* `experiments/results/best_configurations.csv`

### Notes

* Current version uses `_simulate_output()` for controlled failure modes.
* Replace `_simulate_output()` with real MediaPipe/Transformers inference when ready.
* Keep `_analyze_output()` unchanged so your metrics stay comparable.

---

## 3) Temperature/Top-P Grid Search: `temperature_top_p_experiments.py`

### Run

```bash
python temperature_top_p_experiments.py
```

### Outputs

Written to:

* `experiments/results/experiment_results.csv`
* `experiments/results/json_validity_heatmap.png`
* `experiments/results/language_by_temperature.png`
* `experiments/results/looping_vs_temperature.png`
* `experiments/results/summary_by_temperature.csv`

### Notes

* By default it prints a warning that outputs are simulated.
* There may be scaffolded Transformers code (commented) if you want to switch to real inference.

---

## Recommended Workflow (practical)

1. Use `compose_prompt.py` to generate the exact prompt string that will be used on Android.
2. Mirror the prompt style(s) in the experiment scripts to compare reliability.
3. Use experiment results to choose:

   * temperature/top_p/top_k ranges
   * repetition penalty
   * whether to use an “example JSON” prompt style (usually best for structure)
4. Validate the “best config” on-device with MediaPipe, because LiteRT runtime behavior can differ.

---

## Tips for MediaPipe (Android) Alignment

* Treat `<end_of_turn>` as a logical delimiter, but still design for cases where:

  * `<end_of_turn>` repeats
  * final callback may arrive with empty partial text

* Keep your output parser robust:

  * strip `<end_of_turn>`
  * extract `{ ... }` by first `{` and last `}`
  * parse JSON leniently, then re-emit canonical compact JSON for storage

* For strict compact output, always canonicalize:

  * Python: `json.dumps(obj, separators=(',', ':'), ensure_ascii=False)`
  * Kotlin/Java: use your JSON library’s compact serialization

---

## Troubleshooting

### `Missing dependency: pyyaml`

```bash
python -m pip install -U pyyaml
```

### `No module named seaborn`

```bash
python -m pip install -U seaborn
```

### Plots not saved

Ensure output folder exists:

```bash
mkdir -p experiments/results
```

---

## License

MIT License (see file headers).
