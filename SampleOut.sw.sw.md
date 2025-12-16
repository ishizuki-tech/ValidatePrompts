# `compose_prompt.py` Output Examples (Swahili, Q1–Q6)

Goal: keep **the prompt on stdout** (pipe-friendly) and print **SLM runtime settings on stderr** every time, so you can see the exact sampling/config context without polluting downstream consumers.

Config used in these examples: `configs/survey_config30.yaml`

---

## Shared: SLM settings (stderr)

This block is printed on **every** run (example shown in `--slm-format kv`).

```text
slm settings:
- accelerator="GPU"
- max_tokens=256
- top_k=20
- top_p=0.9
- temperature=0.2
- user_turn_prefix="<start_of_turn>user"
- model_turn_prefix="<start_of_turn>model"
- turn_end="<end_of_turn>"
- preamble="Wewe ni mtaalamu mashuhuri wa tafiti za wakulima. Soma Swali na Jibu."
- key_contract="MUUNDO WA MATOKEO:\n- Thamani zote ziwe kwa Kiswahili.\n- Vifunguo (keys):\n  • \"analysis\": sentensi fupi\n  • \"expected_answer\": sentensi fupi\n  • \"follow_up_question\": swali 1 fupi la kuthibitisha/kukagua (SI orodha/array)\n  • \"score\": nambari kamili 1–100\n\nDHAMIRA YA SWALI LA UFUATILIAJI:\n- Swali la ufuatiliaji lazima lithibitishe au lifafanue jibu la mhojiwa kuhusu swali HILO HILO.\n- Lilengwe pengo kubwa zaidi: kipimo/kiasi hakipo, namba haipo, kipindi cha muda hakijawekwa, msingi wa kulinganisha (baseline) haupo, au njia iliyotumika haijaelezwa.\n- Liwe la jambo moja tu (si swali lenye sehemu nyingi), liwe dhahiri, na lijibiwe mara moja.\n- Usilete mada mpya au malengo mengine ya ziada.\n"
- length_budget="MIPAKA YA UREFU:\n- analysis<=60 herufi; follow-up<=120 herufi; expected_answer<=120 herufi.\n"
- scoring_rule="KANUNI YA ALAMA: Pima TU uhusiano na swali, ukamilifu, na usahihi wa maudhui. USIADHIBU mtindo au uundaji wa maandishi."
- strict_output="MATOKEO MAKALI (BILA MARKDOWN):\n- JSON tupu pekee, mstari MMOJA.\n- Tumia JSON iliyobanwa (hakuna nafasi karibu na ':' au ',').\n- Hakuna maandishi ya ziada."
- empty_json_instruction="Jibu kwa kitu tupu cha JSON: {}"
```

---

## Q1

### Command

```bash
python compose_prompt.py configs/survey_config30.yaml --node Q1 --answer "Katika misimu mitatu iliyopita, kwa wastani napoteza takribani 20% ya mavuno ya mahindi kutokana na FAW (takribani magunia 3 kwa ekari)."
```

### Prompt (stdout)

```text
<start_of_turn>user
Wewe ni mtaalamu mashuhuri wa tafiti za wakulima. Soma Swali na Jibu.
MUUNDO WA MATOKEO:
- Thamani zote ziwe kwa Kiswahili.
- Vifunguo (keys):
  • "analysis": sentensi fupi
  • "expected_answer": sentensi fupi
  • "follow_up_question": swali 1 fupi la kuthibitisha/kukagua (SI orodha/array)
  • "score": nambari kamili 1–100

DHAMIRA YA SWALI LA UFUATILIAJI:
- Swali la ufuatiliaji lazima lithibitishe au lifafanue jibu la mhojiwa kuhusu swali HILO HILO.
- Lilengwe pengo kubwa zaidi: kipimo/kiasi hakipo, namba haipo, kipindi cha muda hakijawekwa, msingi wa kulinganisha (baseline) haupo, au njia iliyotumika haijaelezwa.
- Liwe la jambo moja tu (si swali lenye sehemu nyingi), liwe dhahiri, na lijibiwe mara moja.
- Usilete mada mpya au malengo mengine ya ziada.
MIPAKA YA UREFU:
- analysis<=60 herufi; follow-up<=120 herufi; expected_answer<=120 herufi.
KANUNI YA ALAMA: Pima TU uhusiano na swali, ukamilifu, na usahihi wa maudhui. USIADHIBU mtindo au uundaji wa maandishi.
MATOKEO MAKALI (BILA MARKDOWN):
- JSON tupu pekee, mstari MMOJA.
- Tumia JSON iliyobanwa (hakuna nafasi karibu na ':' au ',').
- Hakuna maandishi ya ziada.
Kazi:
- Bainisha udhaifu (mfano: vipimo/viwango havijaelezwa, misimu haijatajwa, majibu ya kupindukia).
- Toa jibu moja bora linalotarajiwa: wastani wa hasara ya mavuno kutokana na FAW katika misimu 3 iliyopita (% au magunia kwa ekari); taja kama kuna mabadiliko makubwa kati ya misimu.
- Uliza swali 1 tu la ufuatiliaji ili KUTHIBITISHA/KUHAKIKI jibu hilo hilo (sentensi moja fupi). Pendelea kuthibitisha kipimo (asilimia vs magunia/ekari) na msingi wa kulinganisha au thamani kwa kila msimu kama haipo.
- Toa alama 1–100 (nambari kamili).

Swali: Unapoteza mavuno kiasi gani kutokana na funza wa jeshi vamizi (FAW)? Tafadhali fikiria misimu mitatu iliyopita. (Asilimia au magunia kwa ekari ni sawa.)
Jibu: Katika misimu mitatu iliyopita, kwa wastani napoteza takribani 20% ya mavuno ya mahindi kutokana na FAW (takribani magunia 3 kwa ekari).
<end_of_turn>
<start_of_turn>model
```

---

## Q2

### Command

```bash
python compose_prompt.py configs/survey_config30.yaml --node Q2 --answer "Ningekubali kupoteza hadi 5% ya mavuno (takribani gunia 1 kwa ekari) ili kuvuna siku 10 mapema."
```

### Prompt (stdout)

```text
<start_of_turn>user
Wewe ni mtaalamu mashuhuri wa tafiti za wakulima. Soma Swali na Jibu.
MUUNDO WA MATOKEO:
- Thamani zote ziwe kwa Kiswahili.
- Vifunguo (keys):
  • "analysis": sentensi fupi
  • "expected_answer": sentensi fupi
  • "follow_up_question": swali 1 fupi la kuthibitisha/kukagua (SI orodha/array)
  • "score": nambari kamili 1–100

DHAMIRA YA SWALI LA UFUATILIAJI:
- Swali la ufuatiliaji lazima lithibitishe au lifafanue jibu la mhojiwa kuhusu swali HILO HILO.
- Lilengwe pengo kubwa zaidi: kipimo/kiasi hakipo, namba haipo, kipindi cha muda hakijawekwa, msingi wa kulinganisha (baseline) haupo, au njia iliyotumika haijaelezwa.
- Liwe la jambo moja tu (si swali lenye sehemu nyingi), liwe dhahiri, na lijibiwe mara moja.
- Usilete mada mpya au malengo mengine ya ziada.
MIPAKA YA UREFU:
- analysis<=60 herufi; follow-up<=120 herufi; expected_answer<=120 herufi.
KANUNI YA ALAMA: Pima TU uhusiano na swali, ukamilifu, na usahihi wa maudhui. USIADHIBU mtindo au uundaji wa maandishi.
MATOKEO MAKALI (BILA MARKDOWN):
- JSON tupu pekee, mstari MMOJA.
- Tumia JSON iliyobanwa (hakuna nafasi karibu na ':' au ',').
- Hakuna maandishi ya ziada.
Kazi:
- Bainisha udhaifu (mfano: ubadilishanaji hauko wazi, hakuna kipimo, hakuna sababu).
- Toa jibu bora linalotarajiwa: kiasi cha kupunguza mavuno ambacho mkulima yuko tayari kukubali ili avune siku 10 mapema (% au magunia/ekari), pamoja na sababu fupi ya kipengele kimoja (mfano: hatari ya mvua/bei sokoni).
- Uliza swali 1 tu la ufuatiliaji ili KUTHIBITISHA/KUHAKIKI jibu hilo hilo (sentensi moja fupi). Pendelea kufafanua kipimo/kiasi au mpaka wa juu wa hasara inayokubalika.
- Toa alama 1–100 (nambari kamili).

Swali: Ikiwa ungeweza kuvuna siku 10 mapema, ungekuwa tayari kupunguza mavuno kiasi gani? (Asilimia au magunia kwa ekari.)
Jibu: Ningekubali kupoteza hadi 5% ya mavuno (takribani gunia 1 kwa ekari) ili kuvuna siku 10 mapema.
<end_of_turn>
<start_of_turn>model
```

---

## Q3

### Command

```bash
python compose_prompt.py configs/survey_config30.yaml --node Q3 --answer "Ikiwa 30% au zaidi ya mimea au masikio yanaonyesha uharibifu wa wastani hadi mkali wa wadudu au magonjwa, hasa wakati wa kutoa maua, ningeacha aina hiyo na kubadili."
```

### Prompt (stdout)

```text
<start_of_turn>user
Wewe ni mtaalamu mashuhuri wa tafiti za wakulima. Soma Swali na Jibu.
MUUNDO WA MATOKEO:
- Thamani zote ziwe kwa Kiswahili.
- Vifunguo (keys):
  • "analysis": sentensi fupi
  • "expected_answer": sentensi fupi
  • "follow_up_question": swali 1 fupi la kuthibitisha/kukagua (SI orodha/array)
  • "score": nambari kamili 1–100

DHAMIRA YA SWALI LA UFUATILIAJI:
- Swali la ufuatiliaji lazima lithibitishe au lifafanue jibu la mhojiwa kuhusu swali HILO HILO.
- Lilengwe pengo kubwa zaidi: kipimo/kiasi hakipo, namba haipo, kipindi cha muda hakijawekwa, msingi wa kulinganisha (baseline) haupo, au njia iliyotumika haijaelezwa.
- Liwe la jambo moja tu (si swali lenye sehemu nyingi), liwe dhahiri, na lijibiwe mara moja.
- Usilete mada mpya au malengo mengine ya ziada.
MIPAKA YA UREFU:
- analysis<=60 herufi; follow-up<=120 herufi; expected_answer<=120 herufi.
KANUNI YA ALAMA: Pima TU uhusiano na swali, ukamilifu, na usahihi wa maudhui. USIADHIBU mtindo au uundaji wa maandishi.
MATOKEO MAKALI (BILA MARKDOWN):
- JSON tupu pekee, mstari MMOJA.
- Tumia JSON iliyobanwa (hakuna nafasi karibu na ':' au ',').
- Hakuna maandishi ya ziada.
Kazi:
- Bainisha udhaifu (mfano: hakuna kizingiti, kipimo hakiko wazi, wadudu na magonjwa vimechanganywa).
- Toa jibu bora linalotarajiwa: kizingiti wazi cha kubadili aina (mfano: ≥X% ya mimea/masikio yenye uharibifu wa kiwango cha wastani au zaidi, au kiwango cha ukali katika hatua fulani ya ukuaji).
- Uliza swali 1 tu la ufuatiliaji ili KUTHIBITISHA/KUHAKIKI jibu hilo hilo (sentensi moja fupi). Pendelea kuthibitisha kipimo/kiwango kinachotumika au dirisha la hatua ya ukuaji.
- Toa alama 1–100 (nambari kamili).

Swali: Ni kiwango gani cha uharibifu wa wadudu au magonjwa kingekufanya uache kutumia aina yako ya sasa ya mahindi na ubadilishe kwenda aina mpya? (Toa kigezo wazi, kwa mfano: asilimia ya mimea au masikio yaliyoharibika, au kiwango cha ukali.)
Jibu: Ikiwa 30% au zaidi ya mimea au masikio yanaonyesha uharibifu wa wastani hadi mkali wa wadudu au magonjwa, hasa wakati wa kutoa maua, ningeacha aina hiyo na kubadili.
<end_of_turn>
<start_of_turn>model
```

---

## Q4

### Command

```bash
python compose_prompt.py configs/survey_config30.yaml --node Q4 --answer "Aina mbadala inapaswa kukomaa ndani ya siku 110-120 (DTM), iwe na uvumilivu mkubwa dhidi ya FAW, ivumilie ukame wakati wa kutoa maua/kuchanua, na itoe takribani magunia 18-20 kwa ekari katika msimu wa kawaida."
```

### Prompt (stdout)

```text
<start_of_turn>user
Wewe ni mtaalamu mashuhuri wa tafiti za wakulima. Soma Swali na Jibu.
MUUNDO WA MATOKEO:
- Thamani zote ziwe kwa Kiswahili.
- Vifunguo (keys):
  • "analysis": sentensi fupi
  • "expected_answer": sentensi fupi
  • "follow_up_question": swali 1 fupi la kuthibitisha/kukagua (SI orodha/array)
  • "score": nambari kamili 1–100

DHAMIRA YA SWALI LA UFUATILIAJI:
- Swali la ufuatiliaji lazima lithibitishe au lifafanue jibu la mhojiwa kuhusu swali HILO HILO.
- Lilengwe pengo kubwa zaidi: kipimo/kiasi hakipo, namba haipo, kipindi cha muda hakijawekwa, msingi wa kulinganisha (baseline) haupo, au njia iliyotumika haijaelezwa.
- Liwe la jambo moja tu (si swali lenye sehemu nyingi), liwe dhahiri, na lijibiwe mara moja.
- Usilete mada mpya au malengo mengine ya ziada.
MIPAKA YA UREFU:
- analysis<=60 herufi; follow-up<=120 herufi; expected_answer<=120 herufi.
KANUNI YA ALAMA: Pima TU uhusiano na swali, ukamilifu, na usahihi wa maudhui. USIADHIBU mtindo au uundaji wa maandishi.
MATOKEO MAKALI (BILA MARKDOWN):
- JSON tupu pekee, mstari MMOJA.
- Tumia JSON iliyobanwa (hakuna nafasi karibu na ':' au ',').
- Hakuna maandishi ya ziada.
Kazi:
- Bainisha udhaifu (mfano: sifa za jumla mno, hakuna namba/viwango).
- Toa jibu bora linalotarajiwa: malengo 2 mafupi ya kuchukua nafasi ya aina ya sasa, kila moja likiwa na kigezo cha namba au cha kundi (mfano: DTM, uvumilivu wa FAW, uvumilivu wa ukame katika hatua fulani, mavuno ya kawaida).
- Uliza swali 1 tu la ufuatiliaji ili KUTHIBITISHA/KUHAKIKI jibu hilo hilo (sentensi moja fupi). Pendelea kuthibitisha sifa ya kipaumbele cha kwanza au lengo la namba.
- Toa alama 1–100 (nambari kamili).

Swali: Ungetarajia nini kutoka kwa aina mpya ya mahindi ili kuchukua nafasi ya ile unayolima sasa? Tafadhali toa maelezo kwa undani kadiri inavyowezekana (kwa mfano, muda wa kukomaa (DTM), uvumilivu wa FAW, uvumilivu wa ukame katika hatua fulani, mavuno ya kawaida).
Jibu: Aina mbadala inapaswa kukomaa ndani ya siku 110-120 (DTM), iwe na uvumilivu mkubwa dhidi ya FAW, ivumilie ukame wakati wa kutoa maua/kuchanua, na itoe takribani magunia 18-20 kwa ekari katika msimu wa kawaida.
<end_of_turn>
<start_of_turn>model
```

---

## Q5

### Command

```bash
python compose_prompt.py configs/survey_config30.yaml --node Q5 --answer "Katika mwaka mbaya, ningeendelea kupanda tena ikiwa napata angalau 50% ya mavuno yangu ya kawaida (takribani magunia 8 kwa ekari); chini ya hapo nisingepanda tena."
```

### Prompt (stdout)

```text
<start_of_turn>user
Wewe ni mtaalamu mashuhuri wa tafiti za wakulima. Soma Swali na Jibu.
MUUNDO WA MATOKEO:
- Thamani zote ziwe kwa Kiswahili.
- Vifunguo (keys):
  • "analysis": sentensi fupi
  • "expected_answer": sentensi fupi
  • "follow_up_question": swali 1 fupi la kuthibitisha/kukagua (SI orodha/array)
  • "score": nambari kamili 1–100

DHAMIRA YA SWALI LA UFUATILIAJI:
- Swali la ufuatiliaji lazima lithibitishe au lifafanue jibu la mhojiwa kuhusu swali HILO HILO.
- Lilengwe pengo kubwa zaidi: kipimo/kiasi hakipo, namba haipo, kipindi cha muda hakijawekwa, msingi wa kulinganisha (baseline) haupo, au njia iliyotumika haijaelezwa.
- Liwe la jambo moja tu (si swali lenye sehemu nyingi), liwe dhahiri, na lijibiwe mara moja.
- Usilete mada mpya au malengo mengine ya ziada.
MIPAKA YA UREFU:
- analysis<=60 herufi; follow-up<=120 herufi; expected_answer<=120 herufi.
KANUNI YA ALAMA: Pima TU uhusiano na swali, ukamilifu, na usahihi wa maudhui. USIADHIBU mtindo au uundaji wa maandishi.
MATOKEO MAKALI (BILA MARKDOWN):
- JSON tupu pekee, mstari MMOJA.
- Tumia JSON iliyobanwa (hakuna nafasi karibu na ':' au ',').
- Hakuna maandishi ya ziada.
Kazi:
- Bainisha udhaifu (mfano: hakuna msingi wa kulinganisha, hakuna kipimo, hakuna kanuni ya uamuzi).
- Toa jibu bora linalotarajiwa: mavuno ya chini kabisa yanayokubalika katika mwaka mbaya (% ya mavuno ya kawaida au magunia/ekari) pamoja na kanuni fupi ya uamuzi (mfano: “nikipata angalau X, nitapanda tena”).
- Uliza swali 1 tu la ufuatiliaji ili KUTHIBITISHA/KUHAKIKI jibu hilo hilo (sentensi moja fupi). Pendelea kuthibitisha wastani wa mavuno ya kawaida (baseline) au mfano wa mwaka mbaya kwa namba.
- Toa alama 1–100 (nambari kamili).

Swali: Katika mwaka mbaya, ni mavuno madogo kiasi gani ambayo ungeyakubali na bado ukapande mahindi haya tena msimu unaofuata? (Asilimia ya mavuno ya kawaida au magunia kwa ekari.)
Jibu: Katika mwaka mbaya, ningeendelea kupanda tena ikiwa napata angalau 50% ya mavuno yangu ya kawaida (takribani magunia 8 kwa ekari); chini ya hapo nisingepanda tena.
<end_of_turn>
<start_of_turn>model
```

---

## Q6

### Command

```bash
python compose_prompt.py configs/survey_config30.yaml --node Q6 --answer "Ukame huathiri zaidi wakati wa kutoa maua/kuchanua kwa sababu hupunguza uchavushaji na kutengenezwa kwa punje."
```

### Prompt (stdout)

```text
<start_of_turn>user
Wewe ni mtaalamu mashuhuri wa tafiti za wakulima. Soma Swali na Jibu.
MUUNDO WA MATOKEO:
- Thamani zote ziwe kwa Kiswahili.
- Vifunguo (keys):
  • "analysis": sentensi fupi
  • "expected_answer": sentensi fupi
  • "follow_up_question": swali 1 fupi la kuthibitisha/kukagua (SI orodha/array)
  • "score": nambari kamili 1–100

DHAMIRA YA SWALI LA UFUATILIAJI:
- Swali la ufuatiliaji lazima lithibitishe au lifafanue jibu la mhojiwa kuhusu swali HILO HILO.
- Lilengwe pengo kubwa zaidi: kipimo/kiasi hakipo, namba haipo, kipindi cha muda hakijawekwa, msingi wa kulinganisha (baseline) haupo, au njia iliyotumika haijaelezwa.
- Liwe la jambo moja tu (si swali lenye sehemu nyingi), liwe dhahiri, na lijibiwe mara moja.
- Usilete mada mpya au malengo mengine ya ziada.
MIPAKA YA UREFU:
- analysis<=60 herufi; follow-up<=120 herufi; expected_answer<=120 herufi.
KANUNI YA ALAMA: Pima TU uhusiano na swali, ukamilifu, na usahihi wa maudhui. USIADHIBU mtindo au uundaji wa maandishi.
MATOKEO MAKALI (BILA MARKDOWN):
- JSON tupu pekee, mstari MMOJA.
- Tumia JSON iliyobanwa (hakuna nafasi karibu na ':' au ',').
- Hakuna maandishi ya ziada.
Kazi:
- Bainisha udhaifu (mfano: muda ni wa jumla sana, hakuna hatua ya ukuaji, hakuna dalili/athari).
- Toa jibu bora linalotarajiwa: hatua ya zao ambapo ukame ni hatari zaidi (mfano: kabla ya kutoa maua, wakati wa kutoa maua/kuchanua, kujaza punje, kabla ya kuvuna) pamoja na sababu fupi ya kipengele kimoja.
- Uliza swali 1 tu la ufuatiliaji ili KUTHIBITISHA/KUHAKIKI jibu hilo hilo (sentensi moja fupi). Pendelea kuthibitisha maana ya hatua hiyo au mfano wa msimu mmoja maalum.
- Toa alama 1–100 (nambari kamili).

Swali: Ni kipindi gani katika msimu ambapo ukame unaathiri mazao yako kwa kiwango kikubwa zaidi? Ni wakati gani wa ukame ambao ni mgumu zaidi kwako? (Kwa mfano, kabla ya kutoa maua, wakati wa kutoa maua/kuchanua, kujaza punje, kabla ya kuvuna.)
Jibu: Ukame huathiri zaidi wakati wa kutoa maua/kuchanua kwa sababu hupunguza uchavushaji na kutengenezwa kwa punje.
<end_of_turn
```
