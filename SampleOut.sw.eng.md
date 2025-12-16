# `compose_prompt.py` Output Examples (Q1–Q6) — Swahili (`survey_config20.yaml`)

Goal: keep **the prompt on stdout** (pipe-friendly) and print **SLM runtime settings on stderr** every time, so you can see the exact sampling/config context without polluting downstream consumers.

---

## Shared: SLM settings (stderr)

This block is printed on **every** run (example shown in `--slm-format kv`).

```text
slm settings:
- accelerator="GPU"
- max_tokens=4096
- top_k=10
- top_p=0.2
- temperature=0.3
- user_turn_prefix="<start_of_turn>user"
- model_turn_prefix="<start_of_turn>model"
- turn_end="<end_of_turn>"
- preamble="You are a well-known farmer survey expert. Read the Question and the Answer."
- key_contract="OUTPUT FORMAT:\n- All Values in Swahili.\n- Keys:\n  • \"analysis\": short string\n  • \"expected answer\": short string\n  • \"follow-up question\": a single short confirm/validate question (NOT an array)\n  • \"score\": integer 1–100\nFOLLOW-UP INTENT:\n- The follow-up must confirm or clarify the respondent's original answer to the SAME question.\n- It must target the biggest uncertainty: missing unit/scale, missing number, time window, baseline, or method used.\n- Keep it single-scope (no multi-part), concrete, and answerable immediately.\n- Do not introduce new topics or additional objectives.\n"
- length_budget="LENGTH LIMITS:\n- analysis<=60 chars; follow-up<=120; expected answer<=120.\n"
- scoring_rule="Scoring rule: Judge ONLY content relevance/completeness/accuracy. Do NOT penalize style or formatting."
- strict_output="STRICT OUTPUT (NO MARKDOWN):\n- RAW JSON only, ONE LINE.\n- Use COMPACT JSON (no spaces around ':' and ',').\n- No extra text."
- empty_json_instruction="Respond with an empty JSON object: {}"
```

---

## Q1

### Command

```bash
python compose_prompt.py configs/survey_config20.yaml --node Q1 --answer "Katika misimu mitatu iliyopita, kwa wastani napoteza takribani 20% ya mavuno ya mahindi kutokana na FAW (takribani magunia 3 kwa ekari)."
```

### Prompt (stdout)

```text
<start_of_turn>user
You are a well-known farmer survey expert. Read the Question and the Answer.
OUTPUT FORMAT:
- All Values in Swahili.
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

Question: Unapoteza mavuno kiasi gani kutokana na funza wa jeshi vamizi (FAW)? Tafadhali fikiria misimu mitatu iliyopita. (Asilimia au magunia kwa ekari ni sawa.)
Answer: Katika misimu mitatu iliyopita, kwa wastani napoteza takribani 20% ya mavuno ya mahindi kutokana na FAW (takribani magunia 3 kwa ekari).
<end_of_turn>
<start_of_turn>model
```

---

## Q2

### Command

```bash
python compose_prompt.py configs/survey_config20.yaml --node Q2 --answer "Ningekubali kupoteza hadi 5% ya mavuno (takribani gunia 1 kwa ekari) ili kuvuna siku 10 mapema."
```

### Prompt (stdout)

```text
<start_of_turn>user
You are a well-known farmer survey expert. Read the Question and the Answer.
OUTPUT FORMAT:
- All Values in Swahili.
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

Question: Ikiwa ungeweza kuvuna siku 10 mapema, ungekuwa tayari kupunguza mavuno kiasi gani? (Asilimia au magunia kwa ekari.)
Answer: Ningekubali kupoteza hadi 5% ya mavuno (takribani gunia 1 kwa ekari) ili kuvuna siku 10 mapema.
<end_of_turn>
<start_of_turn>model
```

---

## Q3

### Command

```bash
python compose_prompt.py configs/survey_config20.yaml --node Q3 --answer "Ikiwa 30% au zaidi ya mimea au masikio yanaonyesha uharibifu wa wastani hadi mkali wa wadudu au magonjwa, hasa wakati wa kutoa maua, ningeacha aina hiyo na kubadili."
```

### Prompt (stdout)

```text
<start_of_turn>user
You are a well-known farmer survey expert. Read the Question and the Answer.
OUTPUT FORMAT:
- All Values in Swahili.
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

Question: Ni kiwango gani cha uharibifu wa wadudu au magonjwa kingekufanya uache kutumia aina yako ya sasa ya mahindi na ubadilishe kwenda aina mpya? (Toa kigezo wazi, kwa mfano: asilimia ya mimea au masikio yaliyoharibika, au kiwango cha ukali.)
Answer: Ikiwa 30% au zaidi ya mimea au masikio yanaonyesha uharibifu wa wastani hadi mkali wa wadudu au magonjwa, hasa wakati wa kutoa maua, ningeacha aina hiyo na kubadili.
<end_of_turn>
<start_of_turn>model
```

---

## Q4

### Command

```bash
python compose_prompt.py configs/survey_config20.yaml --node Q4 --answer "Aina mbadala inapaswa kukomaa ndani ya siku 110-120 (DTM), iwe na uvumilivu mkubwa dhidi ya FAW, ivumilie ukame wakati wa kutoa maua/kuchanua, na itoe takribani magunia 18-20 kwa ekari katika msimu wa kawaida."
```

### Prompt (stdout)

```text
<start_of_turn>user
You are a well-known farmer survey expert. Read the Question and the Answer.
OUTPUT FORMAT:
- All Values in Swahili.
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

Question: Ungetarajia nini kutoka kwa aina mpya ya mahindi ili kuchukua nafasi ya ile unayolima sasa? Tafadhali toa maelezo kwa undani kadiri inavyowezekana (kwa mfano, muda wa kukomaa (DTM), uvumilivu wa FAW, uvumilivu wa ukame katika hatua fulani, mavuno ya kawaida).
Answer: Aina mbadala inapaswa kukomaa ndani ya siku 110-120 (DTM), iwe na uvumilivu mkubwa dhidi ya FAW, ivumilie ukame wakati wa kutoa maua/kuchanua, na itoe takribani magunia 18-20 kwa ekari katika msimu wa kawaida.
<end_of_turn>
<start_of_turn>model
```

---

## Q5

### Command

```bash
python compose_prompt.py configs/survey_config20.yaml --node Q5 --answer "Katika mwaka mbaya, ningeendelea kupanda tena ikiwa napata angalau 50% ya mavuno yangu ya kawaida (takribani magunia 8 kwa ekari); chini ya hapo nisingepanda tena."
```

### Prompt (stdout)

```text
<start_of_turn>user
You are a well-known farmer survey expert. Read the Question and the Answer.
OUTPUT FORMAT:
- All Values in Swahili.
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

Question: Katika mwaka mbaya, ni mavuno madogo kiasi gani ambayo ungeyakubali na bado ukapande mahindi haya tena msimu unaofuata? (Asilimia ya mavuno ya kawaida au magunia kwa ekari.)
Answer: Katika mwaka mbaya, ningeendelea kupanda tena ikiwa napata angalau 50% ya mavuno yangu ya kawaida (takribani magunia 8 kwa ekari); chini ya hapo nisingepanda tena.
<end_of_turn>
<start_of_turn>model
```

---

## Q6

### Command

```bash
python compose_prompt.py configs/survey_config20.yaml --node Q6 --answer "Ukame huathiri zaidi wakati wa kutoa maua/kuchanua kwa sababu hupunguza uchavushaji na kutengenezwa kwa punje."
```

### Prompt (stdout)

```text
<start_of_turn>user
You are a well-known farmer survey expert. Read the Question and the Answer.
OUTPUT FORMAT:
- All Values in Swahili.
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

Question: Ni kipindi gani katika msimu ambapo ukame unaathiri mazao yako kwa kiwango kikubwa zaidi? Ni wakati gani wa ukame ambao ni mgumu zaidi kwako? (Kwa mfano, kabla ya kutoa maua, wakati wa kutoa maua/kuchanua, kujaza punje, kabla ya kuvuna.)
Answer: Ukame huathiri zaidi wakati wa kutoa maua/kuchanua kwa sababu hupunguza uchavushaji na kutengenezwa kwa punje.
<end_of_turn>
<start_of_turn>model
```
