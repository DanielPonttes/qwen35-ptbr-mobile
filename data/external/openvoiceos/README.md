---
license: apache-2.0
task_categories:
  - text-classification
  - token-classification
language:
  - en
  - pt
  - es
  - fr
  - de
  - it
  - nl
  - ca
  - gl
  - da
  - eu
multilinguality:
  - multilingual
size_categories:
  - 10K<n<100K
pretty_name: OVOS Intent Benchmark
tags:
  - intent-classification
  - slot-filling
  - voice-assistant
  - ovos
  - multilingual
configs:
  - config_name: en-US-templates
    data_files:
      - split: train
        path: en-US/train_templates.jsonl
  - config_name: en-US-keywords
    data_files:
      - split: train
        path: en-US/train_keywords.jsonl
  - config_name: en-US-test
    data_files:
      - split: test
        path: en-US/test.jsonl
  - config_name: pt-PT-templates
    data_files:
      - split: train
        path: pt-PT/train_templates.jsonl
  - config_name: pt-PT-keywords
    data_files:
      - split: train
        path: pt-PT/train_keywords.jsonl
  - config_name: pt-PT-test
    data_files:
      - split: test
        path: pt-PT/test.jsonl
  - config_name: pt-BR-templates
    data_files:
      - split: train
        path: pt-BR/train_templates.jsonl
  - config_name: pt-BR-keywords
    data_files:
      - split: train
        path: pt-BR/train_keywords.jsonl
  - config_name: pt-BR-test
    data_files:
      - split: test
        path: pt-BR/test.jsonl
  - config_name: es-ES-templates
    data_files:
      - split: train
        path: es-ES/train_templates.jsonl
  - config_name: es-ES-keywords
    data_files:
      - split: train
        path: es-ES/train_keywords.jsonl
  - config_name: es-ES-test
    data_files:
      - split: test
        path: es-ES/test.jsonl
  - config_name: fr-FR-templates
    data_files:
      - split: train
        path: fr-FR/train_templates.jsonl
  - config_name: fr-FR-keywords
    data_files:
      - split: train
        path: fr-FR/train_keywords.jsonl
  - config_name: fr-FR-test
    data_files:
      - split: test
        path: fr-FR/test.jsonl
  - config_name: de-DE-templates
    data_files:
      - split: train
        path: de-DE/train_templates.jsonl
  - config_name: de-DE-keywords
    data_files:
      - split: train
        path: de-DE/train_keywords.jsonl
  - config_name: de-DE-test
    data_files:
      - split: test
        path: de-DE/test.jsonl
  - config_name: it-IT-templates
    data_files:
      - split: train
        path: it-IT/train_templates.jsonl
  - config_name: it-IT-keywords
    data_files:
      - split: train
        path: it-IT/train_keywords.jsonl
  - config_name: it-IT-test
    data_files:
      - split: test
        path: it-IT/test.jsonl
  - config_name: nl-NL-templates
    data_files:
      - split: train
        path: nl-NL/train_templates.jsonl
  - config_name: nl-NL-keywords
    data_files:
      - split: train
        path: nl-NL/train_keywords.jsonl
  - config_name: nl-NL-test
    data_files:
      - split: test
        path: nl-NL/test.jsonl
  - config_name: ca-ES-templates
    data_files:
      - split: train
        path: ca-ES/train_templates.jsonl
  - config_name: ca-ES-keywords
    data_files:
      - split: train
        path: ca-ES/train_keywords.jsonl
  - config_name: ca-ES-test
    data_files:
      - split: test
        path: ca-ES/test.jsonl
  - config_name: gl-ES-templates
    data_files:
      - split: train
        path: gl-ES/train_templates.jsonl
  - config_name: gl-ES-keywords
    data_files:
      - split: train
        path: gl-ES/train_keywords.jsonl
  - config_name: gl-ES-test
    data_files:
      - split: test
        path: gl-ES/test.jsonl
  - config_name: da-DK-templates
    data_files:
      - split: train
        path: da-DK/train_templates.jsonl
  - config_name: da-DK-keywords
    data_files:
      - split: train
        path: da-DK/train_keywords.jsonl
  - config_name: da-DK-test
    data_files:
      - split: test
        path: da-DK/test.jsonl
  - config_name: eu-ES-templates
    data_files:
      - split: train
        path: eu-ES/train_templates.jsonl
  - config_name: eu-ES-keywords
    data_files:
      - split: train
        path: eu-ES/train_keywords.jsonl
  - config_name: eu-ES-test
    data_files:
      - split: test
        path: eu-ES/test.jsonl
---

> **Purpose.** This dataset was collected specifically for **intent-parser benchmarking**, *independently from any OVOS skill*. Skill-derived utterances tend to overfit the exact phrasings a plugin was tuned on; this data is drawn from a disjoint source so it measures whether an OVOS intent plugin **generalizes** rather than memorizes. It is part of the [OVOS intent-classification datasets](https://huggingface.co/collections/OpenVoiceOS/intent-classification-datasets) used by the [OVOS Plugin Arena](https://github.com/OpenVoiceOS/ovos-plugin-arena) intent benchmark.

## Funding

Developed by [TigreGotico](https://tigregotico.pt) for [OpenVoiceOS](https://openvoiceos.org) as part of **OpenVoiceOS — From Beta to Breakthrough**, funded through the [NGI0 Commons Fund](https://nlnet.nl/commonsfund/), a fund established by [NLnet](https://nlnet.nl) with financial support from the European Commission's [Next Generation Internet](https://ngi.eu) programme (grant agreement No 101135429).


# OVOS Intent Benchmark

A paradigm-neutral benchmark for voice-assistant intent classification and slot extraction, covering **50 intents** across **10 domains** in **12 languages**. Designed to let keyword-based (Adapt, Palavreado), template-based (Padatious, Padacioso, Nebulento) and embedding-based (M2V, hierarchical-KNN) OVOS intent engines be scored on the same data.

See [`SPEC.md`](SPEC.md) for the full specification.

## At a glance

| | |
|---|---|
| Intents | 50 (across 10 domains) |
| Languages | 12 (en-US, pt-PT, pt-BR, es-ES, fr-FR, de-DE, it-IT, nl-NL, ca-ES, gl-ES, da-DK, eu-ES) |
| Train (templates) | 1 000 rows / lang (20 templates × 50 intents) — **12 000 total** |
| Train (keywords)  | 50 rows / lang (one Adapt-style rule per intent) — **600 total** |
| Test rows | 1 750 rows / lang — **21 000 total** |
| **Grand total** | **33 600 rows** (32 400 authored + 1 200 mechanically derived) |

## Three-file layout per language

```
{lang}/
  train_templates.jsonl   ← used by padacioso, padatious, nebulento, m2v, hknn
  train_keywords.jsonl    ← used by adapt, palavreado
  test.jsonl              ← shared by every engine
```

- **`train_templates.jsonl`** carries `{slot}`-placeholder templates with a slot schema (name / type / required / examples). 20 rows per intent.
- **`train_keywords.jsonl`** carries one complete Adapt-style keyword rule per intent — `required_vocab` groups (must match) + `optional_vocab` groups (boost score, slot-value vocab lives here). 1 row per intent.
- **`test.jsonl`** carries fully realised natural sentences with gold intent + gold slots. Shared by every engine.

### Test buckets per language

| Bucket | Count | Notes |
|---|---|---|
| `template` | 500 | Surface variants near a training template |
| `paraphrase` | 700 | Naturalistic rewordings of the same intent |
| `near_ood` | 400 | Same domain, different intent — measures inter-intent confusion |
| `far_ood` | 50 | Intent-agnostic chitchat / nonsense / out-of-scope (`expected_intent=null`) |
| `asr_noise` | 50 | ASR-style mistranscriptions: homophones, dropped function words, word-boundary breaks, filler insertions. Mechanically derived from `paraphrase` seeds; gold labels preserved. |
| `typos` | 50 | Keyboard / chat typos: 2–4 corruptions per utterance (adjacent-key swaps, transposed/dropped/doubled letters, case flips). Same `paraphrase` seeds as `asr_noise` so the two are directly comparable. |

`far_ood` is a shared pool per language (not per-intent), used to measure false-positive rate across the whole engine.

## Domains

`media`, `timers_alarms`, `smarthome`, `communication`, `navigation`, `search_qa`, `weather`, `calendar`, `system_control`, `news`.

## Schemas

### `train_templates.jsonl`

```json
{
  "intent_id": "media:play_song",
  "domain": "media",
  "lang": "en-US",
  "template": "play {song} by {artist}",
  "slots": [
    {"name": "song",   "type": "phrase",       "required": true,  "examples": ["..."]},
    {"name": "artist", "type": "named_entity", "required": false, "examples": ["..."]}
  ]
}
```

- `template` uses `{slot_name}` placeholders; padacioso-style `(alt|alt)` and `[opt]` are allowed.
- `slots[].type ∈ {word, phrase, int, float, date, time, duration, named_entity}`.

### `train_keywords.jsonl`

```json
{
  "intent_id": "media:play_song",
  "domain": "media",
  "lang": "en-US",
  "required_vocab": {
    "PlayKw": ["play", "put on", "throw on", "queue", "spin", "stream", "hear", "listen", "..."]
  },
  "optional_vocab": {
    "ByKw":     ["by", "from", "of"],
    "PleaseKw": ["please", "can you", "could you"],
    "TrackKw":  ["song", "track", "tune", "music", "record"],
    "song":     ["bohemian rhapsody", "smells like teen spirit", "africa", "hey jude"],
    "artist":   ["queen", "nirvana", "toto", "the beatles"]
  }
}
```

- `required_vocab` groups must all match (at least one word from each).
- `optional_vocab` groups are score-boosters; slot-name groups (`song`, `artist`, …) double as the slot's example value pool.

### `test.jsonl`

```json
{
  "utterance": "play yesterday by the beatles",
  "expected_intent": "media:play_song",
  "expected_slots": {"song": "yesterday", "artist": "the beatles"},
  "split": "template",
  "domain": "media",
  "lang": "en-US"
}
```

- Test utterances are **fully realised natural sentences** — never templates.
- For `far_ood`, `expected_intent` and `expected_slots` are `null`.

## Loading

Each language exposes **three configs** — `{lang}-templates`, `{lang}-keywords`, and `{lang}-test`. They have different row schemas, hence separate configs:

```python
from datasets import load_dataset

templates = load_dataset("OpenVoiceOS/intents-for-eval", "en-US-templates", split="train")
keywords  = load_dataset("OpenVoiceOS/intents-for-eval", "en-US-keywords",  split="train")
test      = load_dataset("OpenVoiceOS/intents-for-eval", "en-US-test",      split="test")

templates[0]   # → template rows for padatious/nebulento/m2v/hknn
keywords[0]    # → keyword rules for adapt/palavreado
test[0]        # → labelled test utterances
```

## Adapter contract

An engine claims conformance by implementing:

```python
train(train_jsonl: Path, lang: str) -> model_handle
predict(model_handle, utterance: str) -> {
    "intent_id": str | None,
    "confidence": float,
    "slots": dict[str, str],
}
```

Reference adapters and benchmark runner live in <https://github.com/OpenVoiceOS/ovos-intent-benchmark>.

## Metrics

### Intent classification
- Overall accuracy, macro-F1, micro-F1.
- False-positive rate on `far_ood`.
- **Per-bucket breakdown** (`template` / `paraphrase` / `near_ood` / `far_ood`) — the critical signal.
- Per-intent precision / recall / F1 / support; top-K confusion matrix.
- Latency: median, p95, p99 (ms); RTF.

### Slot extraction
Reported on rows where the engine predicted the correct intent (also report joint):
- Slot precision / recall / F1 (token-level, BIO-style).
- Slot exact-match (per slot, per utterance).
- Joint intent + slot exact-match.
- Per-slot-type breakdown.

## Languages

`en-US`, `pt-PT`, `pt-BR`, `es-ES`, `fr-FR`, `de-DE`, `it-IT`, `nl-NL`, `ca-ES`, `gl-ES`, `da-DK`, `eu-ES`.

Each language is an independent dataset — locale-specific entities (holidays, currencies, news outlets, etc.) are encouraged.

## Dataset generation

Generated by **Claude Opus under heavy human guidance**, working from `SPEC.md` and a fixed taxonomy of 50 intents across 10 domains. Important properties — kept here so they are reproducible and auditable:

### Authoring rules

- **Hand-authored, one (intent, lang) cell at a time.** No procedural English-template-then-translate pipeline. Each language gets in-language vocabulary (Spanish *pon* not *play*; Basque *jarri* not *play*) — that's the whole reason a per-language dataset exists.
- **Batched with QA gates.** Authoring proceeded one batch at a time (one intent × 12 langs per batch, 50 batches total). After each batch `scripts/validate.py` + a per-language coverage plot ran; errors were corrected before the next batch began. Roughly 20 % of batches needed correction at the gate.
- **No reuse of OVOS skill intent names.** Taxonomy choices exercise edge cases: slotless action intents (pause, mute, restart), slot-heavy intents (set_timer, create_event), and lexically-collision-prone sibling pairs within domains (next_story / previous_story).

### Test-bucket authoring

- **`template` (500/lang):** training templates with their `{slot}` placeholders filled in from the slot's example values.
- **`paraphrase` (700/lang):** hand-authored. Each must be lexically dissimilar from every training template while preserving intent and slot values — this is the bucket where engines diverge.
- **`near_ood` (400/lang):** the row's gold intent is a **sibling intent in the same domain**, not the row's host. Tests inter-intent confusion.
- **`far_ood` (50/lang):** hand-authored chitchat / nonsense / out-of-scope across six categories (politeness, philosophy, nonsense, feelings, factual non-questions, off-domain requests). `expected_intent = null`.

### Keyword-rule authoring

For each (intent, lang) cell in `train_keywords.jsonl`:

- At least one **required group** named with the `Kw` suffix (e.g. `PlayKw`, `LockKw`), large enough that on its own it distinguishes the intent against every sibling in the domain. Typical size: 8–15 surface forms.
- `optional_vocab` always carries a `PleaseKw` group in the target language (politeness markers — common test-set noise the engine should tolerate).
- Slot-name groups (lower-cased, e.g. `song`, `artist`, `destination`) under `optional_vocab` populated with the slot's `examples`. Adapt/Palavreado adapters register these as named-entity vocabularies so slot extraction works without an external NER.

### Known limitations

- **No native-speaker pass yet.** Multilingual content was authored by a multilingual model without per-language editor sign-off — this is the highest-leverage backlog item.
- **`far_ood` is small** (50/lang) — statistically noisy FPR.
- **Slot value pools are shared between train and test.** Real-world open-vocabulary slot extraction is harder than the numbers suggest.

## Versioning

Spec and dataset are versioned independently:
- Spec: see `SPEC.md` (semver; breaking schema changes bump major).
- Dataset: tagged on this repo (`dataset v0.x`).

## License

Apache-2.0.

## Citation

```
@misc{ovos-intent-benchmark,
  title  = {OVOS Intent Benchmark},
  author = {OpenVoiceOS contributors},
  year   = {2026},
  url    = {https://github.com/OpenVoiceOS/ovos-intent-benchmark}
}
```
