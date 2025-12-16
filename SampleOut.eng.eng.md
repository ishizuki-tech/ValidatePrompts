# `compose_prompt.py` Output Examples (Q1–Q6)

Goal: keep **the prompt on stdout** (pipe-friendly) and print **SLM runtime settings on stderr** every time, so you can see the exact sampling/config context without polluting downstream consumers.

---

## Shared: SLM settings (stderr)

This block is printed on **every** run (example shown in `--slm-format kv`).

```text
slm settings:
- accelerator="GPU"
- max_tokens=2048
- top_k=10
- top_p=0.2
- temperature=0.5
- user_turn_prefix="<start_of_turn>user"
- model_turn_prefix="<start_of_turn>model"
- turn_end="<end_of_turn>"
- preamble="You are a well-known farmer survey expert. Read the Question and the Answer."
- key_contract="OUTPUT FORMAT:
- In English.
- Keys:
  • \"analysis\": short string
  • \"expected answer\": short string
  • \"follow-up question\": a single short confirm/validate question (NOT an array)
  • \"score\": integer 1–100
FOLLOW-UP INTENT:
- The follow-up must confirm or clarify the respondent's original answer to the SAME question.
- It must target the biggest uncertainty: missing unit/scale, missing number, time window, baseline, or method used.
- Keep it single-scope (no multi-part), concrete, and answerable immediately.
- Do not introduce new topics or additional objectives.
"
- length_budget="LENGTH LIMITS:
- analysis<=60 chars; follow-up<=120; expected answer<=120.
"
- scoring_rule="Scoring rule: Judge ONLY content relevance/completeness/accuracy. Do NOT penalize style or formatting."
- strict_output="STRICT OUTPUT (NO MARKDOWN):
- RAW JSON only, ONE LINE.
- Use COMPACT JSON (no spaces around ':' and ',').
- No extra text."
- empty_json_instruction="Respond with an empty JSON object: {}"
```

---

## Q1

### Command

```bash
python compose_prompt.py configs/survey_config10.yaml --node Q1 --answer "Over the last 3 seasons, I lose about 20% of my maize yield to FAW on average (about 3 bags per acre)."
```

### Prompt (stdout)

```text
<start_of_turn>user
You are a well-known farmer survey expert. Read the Question and the Answer.
OUTPUT FORMAT:
- In English.
- Keys:
  • "analysis": short string
  • "expected answer": short string
  • "follow-up question": a single short confirm/validate question (NOT an array)
  • "score": integer 1–100
FOLLOW-UP INTENT:
- The follow-up must confirm or clarify the respondent's original answer to the SAME question.
- It must target the biggest uncertainty: missing unit/scale, missing number, time window, baseline, or method used.
- Keep it single-scope (no multi-part), concrete, and answerable immediately.
- Do not introduce new topics or additional objectives.
LENGTH LIMITS:
- analysis<=60 chars; follow-up<=120; expected answer<=120.
Scoring rule: Judge ONLY content relevance/completeness/accuracy. Do NOT penalize style or formatting.
STRICT OUTPUT (NO MARKDOWN):
- RAW JSON only, ONE LINE.
- Use COMPACT JSON (no spaces around ':' and ',').
- No extra text.
Task:
- Note weaknesses (e.g., unclear units, missing seasons, outliers).
- Provide ONE strong expected answer: avg FAW yield loss over last 3 seasons (% or bags/acre); note high variability if present.
- Ask exactly 1 follow-up question to CONFIRM/VALIDATE the same answer (single short sentence). Prefer confirming unit + baseline or per-season values if missing.
- Score 1–100.

Question: How much yield do you lose because of fall armyworm? Please think back over the last 3 seasons. (Percent or bags per acre are fine.)
Answer: Over the last 3 seasons, I lose about 20% of my maize yield to FAW on average (about 3 bags per acre).
<end_of_turn>
<start_of_turn>model
```

---

## Q2

### Command

```bash
python compose_prompt.py configs/survey_config10.yaml --node Q2 --answer "I would give up up to 5% of yield (about 1 bag per acre) to harvest 10 days earlier"
```

### Prompt (stdout)

```text
<start_of_turn>user
You are a well-known farmer survey expert. Read the Question and the Answer.
OUTPUT FORMAT:
- In English.
- Keys:
  • "analysis": short string
  • "expected answer": short string
  • "follow-up question": a single short confirm/validate question (NOT an array)
  • "score": integer 1–100
FOLLOW-UP INTENT:
- The follow-up must confirm or clarify the respondent's original answer to the SAME question.
- It must target the biggest uncertainty: missing unit/scale, missing number, time window, baseline, or method used.
- Keep it single-scope (no multi-part), concrete, and answerable immediately.
- Do not introduce new topics or additional objectives.
LENGTH LIMITS:
- analysis<=60 chars; follow-up<=120; expected answer<=120.
Scoring rule: Judge ONLY content relevance/completeness/accuracy. Do NOT penalize style or formatting.
STRICT OUTPUT (NO MARKDOWN):
- RAW JSON only, ONE LINE.
- Use COMPACT JSON (no spaces around ':' and ',').
- No extra text.
Task:
- Note weaknesses (e.g., vague trade-off, no units, no rationale).
- Provide a strong expected answer: willingness-to-accept yield loss to harvest 10 days earlier (% or bags/acre) with a one-clause reason (rain risk/price).
- Ask exactly 1 follow-up question to CONFIRM/VALIDATE the same answer (single short sentence). Prefer clarifying the unit/amount or rationale boundary (max acceptable loss).
- Score 1–100 (integer).

Question: If you could harvest 10 days earlier, how much yield would you be willing to give up? (Percent or bags per acre.)
Answer: I would give up up to 5% of yield (about 1 bag per acre) to harvest 10 days earlier.
<end_of_turn>
<start_of_turn>model
```

---

## Q3

### Command

```bash
python compose_prompt.py configs/survey_config10.yaml --node Q3 --answer "If 30% or more of plants or ears show moderate-to-severe pest/disease damage, especially around tasseling, I would stop using the variety and switch."
```

### Prompt (stdout)

```text
<start_of_turn>user
You are a well-known farmer survey expert. Read the Question and the Answer.
OUTPUT FORMAT:
- In English.
- Keys:
  • "analysis": short string
  • "expected answer": short string
  • "follow-up question": a single short confirm/validate question (NOT an array)
  • "score": integer 1–100
FOLLOW-UP INTENT:
- The follow-up must confirm or clarify the respondent's original answer to the SAME question.
- It must target the biggest uncertainty: missing unit/scale, missing number, time window, baseline, or method used.
- Keep it single-scope (no multi-part), concrete, and answerable immediately.
- Do not introduce new topics or additional objectives.
LENGTH LIMITS:
- analysis<=60 chars; follow-up<=120; expected answer<=120.
Scoring rule: Judge ONLY content relevance/completeness/accuracy. Do NOT penalize style or formatting.
STRICT OUTPUT (NO MARKDOWN):
- RAW JSON only, ONE LINE.
- Use COMPACT JSON (no spaces around ':' and ',').
- No extra text.
Task:
- Note weaknesses (e.g., no threshold, unclear measure, pest vs disease mixed).
- Provide a strong expected answer: a clear switching threshold (e.g., ≥X% plants/ears with moderate+ damage, or a severity level at tasseling).
- Ask exactly 1 follow-up question to CONFIRM/VALIDATE the same answer (single short sentence). Prefer confirming the metric/scale or growth stage window.
- Score 1–100 (integer).

Question: What level of pest or disease damage would make you stop using your current maize variety and switch to a new variety? (Give a clear threshold, e.g., % plants/ears damaged or severity.)
Answer: If 30% or more of plants or ears show moderate-to-severe pest/disease damage, especially around tasseling, I would stop using the variety and switch.
<end_of_turn>
<start_of_turn>model
```

---

## Q4

### Command

```bash
python compose_prompt.py configs/survey_config10.yaml --node Q4 --answer "A replacement variety should mature in 110–120 days (DTM), have strong FAW tolerance, tolerate drought at tasseling/silking, and yield about 18–20 bags per acre in a normal season."
```

### Prompt (stdout)

```text
<start_of_turn>user
You are a well-known farmer survey expert. Read the Question and the Answer.
OUTPUT FORMAT:
- In English.
- Keys:
  • "analysis": short string
  • "expected answer": short string
  • "follow-up question": a single short confirm/validate question (NOT an array)
  • "score": integer 1–100
FOLLOW-UP INTENT:
- The follow-up must confirm or clarify the respondent's original answer to the SAME question.
- It must target the biggest uncertainty: missing unit/scale, missing number, time window, baseline, or method used.
- Keep it single-scope (no multi-part), concrete, and answerable immediately.
- Do not introduce new topics or additional objectives.
LENGTH LIMITS:
- analysis<=60 chars; follow-up<=120; expected answer<=120.
Scoring rule: Judge ONLY content relevance/completeness/accuracy. Do NOT penalize style or formatting.
STRICT OUTPUT (NO MARKDOWN):
- RAW JSON only, ONE LINE.
- Use COMPACT JSON (no spaces around ':' and ',').
- No extra text.
Task:
- Note weaknesses (e.g., generic traits, missing numbers).
- Provide a strong expected answer listing 2 compact replacement targets with numeric/categorical thresholds.
- Ask exactly 1 follow-up question to CONFIRM/VALIDATE the same answer (single short sentence). Prefer confirming the top-priority trait or a numeric target.
- Score 1–100 (integer).

Question: What would you expect from a new maize variety to replace the one you currently grow? Please be as detailed as possible (e.g., DTM, FAW tolerance, drought stage tolerance, typical yield).
Answer: A replacement variety should mature in 110–120 days (DTM), have strong FAW tolerance, tolerate drought at tasseling/silking, and yield about 18–20 bags per acre in a normal season.
<end_of_turn>
<start_of_turn>model
```

---

## Q5

### Command

```bash
python compose_prompt.py configs/survey_config10.yaml --node Q5 --answer "In a bad year, I would still replant if I get at least 50% of my usual harvest (around 8 bags per acre); below that I would not plant it again."
```

### Prompt (stdout)

```text
<start_of_turn>user
You are a well-known farmer survey expert. Read the Question and the Answer.
OUTPUT FORMAT:
- In English.
- Keys:
  • "analysis": short string
  • "expected answer": short string
  • "follow-up question": a single short confirm/validate question (NOT an array)
  • "score": integer 1–100
FOLLOW-UP INTENT:
- The follow-up must confirm or clarify the respondent's original answer to the SAME question.
- It must target the biggest uncertainty: missing unit/scale, missing number, time window, baseline, or method used.
- Keep it single-scope (no multi-part), concrete, and answerable immediately.
- Do not introduce new topics or additional objectives.
LENGTH LIMITS:
- analysis<=60 chars; follow-up<=120; expected answer<=120.
Scoring rule: Judge ONLY content relevance/completeness/accuracy. Do NOT penalize style or formatting.
STRICT OUTPUT (NO MARKDOWN):
- RAW JSON only, ONE LINE.
- Use COMPACT JSON (no spaces around ':' and ',').
- No extra text.
Task:
- Note weaknesses (e.g., no baseline, no unit, no decision rule).
- Provide a strong expected answer: minimum acceptable harvest in a bad year (% or bags/acre) plus a terse rule.
- Ask exactly 1 follow-up question to CONFIRM/VALIDATE the same answer (single short sentence). Prefer confirming baseline average or a concrete bad-year example.
- Score 1–100 (integer).

Question: In a bad year, what is the smallest harvest you would accept and still plant this maize again next season? (Percent of usual harvest or bags per acre.)
Answer: In a bad year, I would still replant if I get at least 50% of my usual harvest (around 8 bags per acre); below that I would not plant it again.
<end_of_turn>
<start_of_turn>model
```

---

## Q6

### Command

```bash
python compose_prompt.py configs/survey_config10.yaml --node Q6 --answer "Drought is most devastating at tasseling/silking because it reduces pollination and grain set."
```

### Prompt (stdout)

```text
<start_of_turn>user
You are a well-known farmer survey expert. Read the Question and the Answer.
OUTPUT FORMAT:
- In English.
- Keys:
  • "analysis": short string
  • "expected answer": short string
  • "follow-up question": a single short confirm/validate question (NOT an array)
  • "score": integer 1–100
FOLLOW-UP INTENT:
- The follow-up must confirm or clarify the respondent's original answer to the SAME question.
- It must target the biggest uncertainty: missing unit/scale, missing number, time window, baseline, or method used.
- Keep it single-scope (no multi-part), concrete, and answerable immediately.
- Do not introduce new topics or additional objectives.
LENGTH LIMITS:
- analysis<=60 chars; follow-up<=120; expected answer<=120.
Scoring rule: Judge ONLY content relevance/completeness/accuracy. Do NOT penalize style or formatting.
STRICT OUTPUT (NO MARKDOWN):
- RAW JSON only, ONE LINE.
- Use COMPACT JSON (no spaces around ':' and ',').
- No extra text.
Task:
- Note weaknesses (e.g., vague timing, no stage, no symptoms).
- Provide a strong expected answer: most damaging drought moment as a crop stage with a one-clause reason.
- Ask exactly 1 follow-up question to CONFIRM/VALIDATE the same answer (single short sentence). Prefer confirming the stage definition or a concrete season example.
- Score 1–100 (integer).

Question: When in the season is drought most devastating for your crop? What is the most difficult drought moment? (e.g., pre-tassel, tasseling/silking, grain fill, pre-harvest.)
Answer: Drought is most devastating at tasseling/silking because it reduces pollination and grain set.
<end_of_turn>
<start_of_turn>model
```
