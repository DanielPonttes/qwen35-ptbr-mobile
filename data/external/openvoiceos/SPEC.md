# OVOS Intent Benchmark Specification

**Status:** Draft — `spec v0.3`
**Maintainer:** OpenVoiceOS
**License:** Apache-2.0

> Change log:
> - **v0.2 → v0.3:** two new resilience buckets — `asr_noise` (ASR-style mistranscriptions)
>   and `typos` (keyboard / chat typos), 50 rows per bucket per language. See §5.
> - **v0.1 → v0.2:** the single `train.jsonl` was split into two paradigm-specific files
>   (`train_templates.jsonl` for template/embedding engines, `train_keywords.jsonl` for
>   keyword engines), and `paradigm_hints` was removed from the row schema. See §4.

---

## 1. Goals & Non-Goals

### Goals

OVOS ships at least seven intent engines spanning three architectural paradigms. Each paradigm encodes intents differently — keyword vocabularies, surface templates, or dense embeddings — and each currently ships an ad-hoc benchmark (`nebulento/benchmark/`, `palavreado/benchmark/`) with incompatible dataset shapes. This specification defines a **single, paradigm-neutral benchmark format** that lets any current or future OVOS intent engine be scored on the same data.

The benchmark is designed to:

- Compare **keyword-based**, **template-based**, and **embedding-based** engines fairly on the same utterances.
- Score **slot extraction** as a first-class metric, not an afterthought.
- Stress-test engines with **controlled out-of-distribution** (OOD) data, not just held-out in-domain samples.
- Cover the **full OVOS multilingual surface**, not just English.
- Be **engine-agnostic**: the taxonomy and dataset do not reuse any specific skill's intent names.

### Non-Goals

- This is **not a leaderboard service**. Result reporting is structured but unhosted.
- This is **not training data for production skills**. Intents are synthetic and chosen to exercise edge cases.
- **Streaming / partial utterances** are out of scope.
- **ASR error modelling** is out of scope; all utterances are clean text.
- **Multi-turn dialog / context** is out of scope; every test row is a single utterance.

---

## 2. Engine Paradigms Covered

The benchmark targets three intent-engine families. Each requires a thin **adapter** that maps the benchmark dataset onto the engine's native API.

### 2.1 Keyword-based (Adapt / Palavreado)

Intents are defined by required and optional named keyword groups; matching is rule-based on the presence of registered vocabulary. The adapter reads **`train_keywords.jsonl`** — one row per intent, each row carries the complete Adapt-style rule (`required_vocab` + `optional_vocab` groups). Slot example values live in `optional_vocab` under the slot's name and double as the engine's slot-entity vocabulary.

Reference engines:
- [`ovos-adapt-pipeline-plugin`](https://github.com/OpenVoiceOS/ovos-adapt-pipeline-plugin)
- [`palavreado`](https://github.com/OpenVoiceOS/palavreado)

### 2.2 Example/template-based (Padatious / Padacioso / Nebulento / padaos)

Intents are defined by phrase templates containing `{slot}` placeholders, optional `[brackets]`, and `(alt|alt)` alternations. Matching is either neural (Padatious) or fuzzy/exact string matching (Padacioso, Nebulento, padaos). The adapter reads **`train_templates.jsonl`** and feeds the `template` field directly — the syntax is shared.

Reference engines:
- [`ovos-padatious-pipeline-plugin`](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin)
- [`padacioso`](https://github.com/OpenVoiceOS/padacioso)
- [`nebulento`](https://github.com/OpenVoiceOS/nebulento)
- [`padaos`](https://github.com/MycroftAI/padaos)

### 2.3 Embedding-based (M2V / hierarchical-KNN)

Intents are defined by a corpus of example utterances; matching is dense-vector similarity (cosine k-NN or learned classifier head). The adapter reads **`train_templates.jsonl`** and expands every template into N realised sentences by substituting the slot's `examples` values. Realised pairs `(utterance, intent_id)` are then encoded and indexed (or used to train a classifier head).

Reference engines:
- [`ovos-m2v-pipeline`](https://github.com/OpenVoiceOS/ovos-m2v-pipeline)
- [`ovos-hierarchical-knn-pipeline`](https://github.com/OpenVoiceOS/ovos-hierarchical-knn-pipeline)

### 2.4 Adapter Contract

To claim conformance, an engine ships an adapter implementing:

```python
def train(train_rows: Iterable[dict], lang: str) -> ModelHandle: ...
def predict(model: ModelHandle, utterance: str) -> Prediction: ...
```

where `Prediction` is `{"intent_id": str | None, "confidence": float, "slots": dict[str, str]}`. Returning `None` for `intent_id` signals an explicit no-match (required for far-OOD).

`train_rows` are read from whichever file fits the paradigm — `train_templates.jsonl` for template/embedding engines, `train_keywords.jsonl` for keyword engines. An engine that wants to consume both is free to load both files; most won't.

---

## 3. Taxonomy

A 3-level hierarchy: **`domain → intent → slots`**.

### 3.1 Domains (v1.0)

`media`, `timers_alarms`, `smarthome`, `communication`, `navigation`, `search_qa`, `weather`, `calendar`, `system_control`, `news`.

A separate `chitchat`-style pool of unattributed `far_ood` utterances is shipped per language but lives only in `test.jsonl` — no in-domain intents are declared for it.

### 3.2 Intent IDs

`domain:snake_case_intent` — e.g. `media:play_song`, `smarthome:lights_on`, `timers_alarms:set_timer`. Globally unique.

### 3.3 Slot Schema

Each intent declares a slot schema (carried per-row in `train_templates.jsonl`):

```json
{
  "name": "duration",
  "type": "duration",
  "required": true,
  "examples": ["5 minutes", "1 hour", "half an hour"]
}
```

`examples` is **mandatory** and MUST contain at least 3 entries (see §4.2 — non-template engines depend on this to synthesise training sentences, and keyword engines lift the same values into `train_keywords.jsonl`'s `optional_vocab`).

`type` is one of: `word`, `phrase`, `int`, `float`, `date`, `time`, `duration`, `named_entity`.

**Engine-native subset (template engines):** only `int`, `float`, and `word` are natively understood by [padacioso](https://github.com/OpenVoiceOS/padacioso) and (via the padatious-compat layer in `padacioso/bracket_expansion.py::translate_padatious`) by [nebulento](https://github.com/OpenVoiceOS/nebulento). Padatious itself expresses these via its `:0` wildcard convention, which the other engines auto-translate.

**Benchmark extensions:** `phrase`, `date`, `time`, `duration`, and `named_entity` are **not** native to any current OVOS template engine. They exist in this spec to give slot extraction a meaningful evaluation surface (date/duration parsing is a real production concern even if no current engine handles it inside a template). Adapters MUST treat these as opaque `phrase`-like captures unless the engine has external slot resolvers. Slot scoring (§8.3) canonicalises typed values regardless of how the engine produced them.

`examples` doubles as the canonicalisation target set for slot scoring (§8.3) where applicable, and as the slot-entity vocab for keyword adapters (§4.3).

The taxonomy is **OVOS-skill-independent**: it must not reuse any specific skill's intent names. This keeps the benchmark a fair test rather than a regression suite for any one engine.

---

## 4. Dataset Schema

Three JSONL files per language: `train_templates.jsonl`, `train_keywords.jsonl`, `test.jsonl`. UTF-8, one JSON object per line.

```
datasets/{lang}/
  train_templates.jsonl   ← padacioso, padatious, nebulento, m2v, hknn
  train_keywords.jsonl    ← adapt, palavreado
  test.jsonl              ← shared by every engine
```

The split exists because keyword paradigms model an intent as **one rule, not many templates**: replicating the same rule across 20 template rows is wasteful, and template engines never read `paradigm_hints` anyway. Two files keep both shapes first-class.

### 4.1 Why a separate `train_keywords.jsonl`?

In `spec v0.1` keyword hints lived alongside templates as a per-row `paradigm_hints` field. In practice this is wrong:

- A keyword rule for an intent is one coherent definition (required keyword groups + optional groups + slot entity vocab) — not 20 fragments scraped from 20 template surface forms.
- Authoring per-row hints encourages collecting whichever 1–2 keywords appear in *that* template, which produces useless rules. Authoring one rule per intent forces the curator to think about the full vocabulary that should distinguish this intent from every sibling intent in the domain.
- Template engines never consumed `paradigm_hints` and were obliged to ignore it; keeping it out of `train_templates.jsonl` makes the file 30 % smaller and clarifies what each engine actually reads.

### 4.2 `train_templates.jsonl`

```json
{
  "intent_id": "media:play_song",
  "domain": "media",
  "lang": "en-US",
  "template": "(play|put on) {song} [by {artist}]",
  "slots": [
    {"name": "song",   "type": "phrase",       "required": true,
     "examples": ["bohemian rhapsody", "smells like teen spirit", "africa"]},
    {"name": "artist", "type": "named_entity", "required": false,
     "examples": ["queen", "nirvana", "toto"]}
  ]
}
```

- `template` uses `{slot_name}` placeholders, `(alt|alt)` alternations, and `[opt]` optional groups. This subset is **natively supported** by padacioso, nebulento, and padatious. Padatious-specific `:0` wildcards are not part of the spec syntax — use named `{slot}` captures instead.
- Multiple rows MAY share the same `intent_id` — each row is one template variant. **20 rows per intent** is the minimum (§5).
- No `paradigm_hints` field — that lives in `train_keywords.jsonl`.
- **Every slot MUST declare `examples`: a list of at least 3 valid placeholder values.** Embedding-based engines cannot consume `{slot}` placeholders directly — they expand each template into realised sentences by substituting `examples` values. Three is the minimum; more is better. Values should be representative of the slot's natural distribution in the target language, not edge cases.

### 4.3 `train_keywords.jsonl`

```json
{
  "intent_id": "media:play_song",
  "domain": "media",
  "lang": "en-US",
  "required_vocab": {
    "PlayKw": ["play", "put on", "throw on", "queue", "spin",
               "stream", "hear", "listen", "blast", "hit play"]
  },
  "optional_vocab": {
    "ByKw":     ["by", "from", "of"],
    "PleaseKw": ["please", "can you", "could you"],
    "TrackKw":  ["song", "track", "tune", "music", "record"],
    "song":     ["bohemian rhapsody", "smells like teen spirit",
                 "africa", "hey jude"],
    "artist":   ["queen", "nirvana", "toto", "the beatles"]
  }
}
```

- **Exactly one row per `(intent_id, lang)`.** 50 intents × 12 langs = 600 rows total.
- `required_vocab` and `optional_vocab` mirror Palavreado's `.require()` / `.optionally()` distinction (see `palavreado/builder.py:68-100`) and Adapt's required/optional keyword slots.
- **Required-vocab semantics:** the engine must match **at least one word from each required group** for the intent to fire. A typical intent has 1–3 required groups (e.g. an action-verb group + a target-noun group).
- **Optional-vocab semantics:** matched optional words contribute to the confidence score but are not gating. The slot-name groups (`song`, `artist`, …) under `optional_vocab` double as the slot's **example value pool** — Adapt/Palavreado adapters register them as named entities so the engine recognises slot fillers in the utterance and can return them as slot extractions.
- A keyword rule MUST distinguish the intent against every other intent in the same domain (the `near_ood` threat model). The authoring guideline: if you can swap this intent's required groups with a sibling's and the utterance still matches both, the rule is too thin.
- An adapter that lacks the required/optional distinction is free to collapse both into a single vocabulary; an adapter that has it MUST honour it.

### 4.4 `test.jsonl`

```json
{
  "utterance": "put on bohemian rhapsody by queen please",
  "expected_intent": "media:play_song",
  "expected_slots": {"song": "bohemian rhapsody", "artist": "queen"},
  "split": "paraphrase",
  "domain": "media",
  "lang": "en-US"
}
```

- `utterance` is a **fully realised natural sentence** — never a template, never a `{slot}` placeholder.
- `expected_intent` is `null` only for `far_ood` rows.
- `expected_slots` is `null` when `expected_intent` is `null`, otherwise a `{slot_name: value}` dict for **required slots that appear in the utterance**. Missing optional slots are simply absent from the dict.
- `split` is one of `template`, `paraphrase`, `near_ood`, `far_ood` (see §5).
- `near_ood` rows have `expected_intent` set to a sibling intent in the same domain (NOT null) — the row is testing whether the engine confuses the surface form with the wrong sibling. The "near-OOD" name refers to the relationship between the utterance and the *target* intent of the dataset row's neighbour, not to the labelling.

---

## 5. Splits & Balance

The test set is partitioned into six buckets, with fixed proportions:

| Bucket       | Count / lang | Description                                                                       |
|--------------|--------------|-----------------------------------------------------------------------------------|
| `template`   | 500          | Surface variants close to a training template; lexical overlap is high.           |
| `paraphrase` | 700          | Naturalistic rewordings of the same intent; lexical overlap is moderate to low.   |
| `near_ood`   | 400          | Same domain, different intent — tests inter-intent confusion.                     |
| `far_ood`    | 50           | Unrelated chitchat / garbage / off-topic — tests false-positive rate. `expected_intent` must be `null`. |
| `asr_noise`  | 50           | Resilience against ASR mistranscriptions: homophone swaps, dropped function words, word-boundary breaks, filler insertions. Derived by perturbing `paraphrase` rows — gold labels are preserved. |
| `typos`      | 50           | Resilience against keyboard / chat typos: 2–4 corruptions per utterance (adjacent-key swaps, transposed letters, dropped/doubled letters, case noise). Derived by perturbing the same `paraphrase` rows as `asr_noise`. |

`far_ood` is a **shared pool per language** (one set of 50 chitchat utterances), not per-intent.

`asr_noise` and `typos` share their seed utterances — each row in one bucket has a sibling row in the other with the same gold intent and gold slots, so a side-by-side comparison isolates "does the engine survive surface-level noise vs deep word-boundary noise". See §10.6 for the perturbation methodology.

### Per-intent minimums (train_templates)

- 20 templates per intent per language.

### Per-intent minimums (train_keywords)

- Exactly 1 row per intent per language.
- At least one non-empty `required_vocab` group.
- Every slot declared in the intent's schema MUST appear as a key in `optional_vocab`, populated with at least 3 example values.

### Per-test attribution

- Per-intent test rows are split **10 / 14 / 8** across the first three buckets (template / paraphrase / near_ood). 50 intents × 32 = 1 600 attributed rows + 50 unattributed `far_ood` = **1 650 test rows per language**.

### Per-language minimums

- **Intents**: ≥ 50 distinct `intent_id`s.
- **Domains**: ≥ 8 of the 10 declared domains represented.

### Ratio enforcement

Per-intent ratios are enforced **at validation time** (not at training time — engines never see split labels). A dataset that deviates by more than ±1 utterance per bucket per intent is non-conformant.

---

## 6. Languages

v1.0 required languages (BCP-47 tags):

`en-US`, `pt-PT`, `pt-BR`, `es-ES`, `fr-FR`, `de-DE`, `it-IT`, `nl-NL`, `ca-ES`, `gl-ES`, `da-DK`, `eu-ES`.

Each language is an **independent dataset** with its own three files. Not every intent must exist in every language — datasets declare a **coverage matrix** in `coverage.json`:

```json
{"en-US": ["media:play_song", "smarthome:lights_on", "..."], "pt-BR": ["..."]}
```

Locale-specific entities are encouraged: Portuguese datasets may use *fado*-genre songs and Iberian holidays; English datasets may use US dollars; etc. Templates should reflect natural phrasing in the target language rather than translating English structures verbatim.

---

## 7. Metrics — Intent Classification

### 7.1 Overall (per language)

- **Accuracy** (fraction of test rows where `predicted_intent == expected_intent`, with both-`null` counted as correct).
- **Macro-F1** over all intents (treats every intent equally).
- **Micro-F1** over all intents (weighted by support).
- **False-positive rate on `far_ood`** — fraction of far-OOD rows where the engine returned a non-null intent. This is the single most diagnostic number for production safety.

### 7.2 Per-bucket breakdown

Each of accuracy, macro-F1, and false-positive rate **must also be reported per bucket** (`template`, `paraphrase`, `near_ood`, `far_ood`). Aggregate scores hide the most interesting signal — paraphrase recall and near-OOD precision are usually where engines diverge.

### 7.3 Per-intent

Precision, recall, F1, support — one row per intent. Required for the published `results.json`.

### 7.4 Confusion

Top-K confusion pairs `(predicted, expected, count)` with K = 20. Helps spot near-OOD leakage and intent-pair collisions.

### 7.5 Latency

Single-utterance inference latency on a fixed reference machine (see §9.2): **median, p95, p99** in milliseconds, plus **RTF** (latency / 1 s).

### 7.6 Resource cost

- Peak RSS during training (MB).
- Trained model size on disk (MB).
- Index build time (s) — for embedding engines.

### 7.7 Confidence thresholds

Each engine has a native confidence-tier configuration (`conf_high`, `conf_medium`, `conf_low`) drawn from its upstream OPM plugin defaults. Results should be reported at **each** tier the engine supports, not just one operating point — this is what lets a deployer pick a cascade configuration. The reference benchmark runner stores raw `(utterance, prediction, confidence)` rows and computes metrics for every tier in post-processing.

---

## 8. Metrics — Slot Extraction

Slot scoring is reported **separately** from intent scoring. By default, slot metrics are computed only over rows where the engine predicted the correct intent (otherwise slot quality is double-penalised by intent error). A joint score is also reported.

### 8.1 Slot-level

- **Precision / recall / F1** at the slot-token level, BIO-aligned over the utterance.
- **Per-slot-type breakdown** (per type from §3.3): `int`, `duration`, `date`, etc. — different engines have very different strengths here.

### 8.2 Utterance-level

- **Slot exact-match**: 1.0 iff every required slot's predicted value equals the expected value after canonicalisation (case-insensitive, whitespace-normalised; numeric types compared semantically).
- **Joint intent+slot exact-match**: 1.0 iff intent is correct **and** slot exact-match is 1.0. The single strictest scalar in the benchmark.

### 8.3 Canonicalisation

For typed slots (`int`, `float`, `date`, `time`, `duration`), comparison is **semantic**: `"five"` == `"5"`, `"half an hour"` == `"30 minutes"` == `PT30M`. For `word`, `phrase`, and `named_entity`, comparison is string-normalised (lowercase, trimmed, internal whitespace collapsed). Canonicalisation is performed by the benchmark runner, not the engine.

---

## 9. Reporting Format

### 9.1 `results.json`

Every conformant run produces a `results.json` with this schema:

```json
{
  "spec_version": "0.2",
  "dataset_version": "0.3",
  "engine": {"name": "padacioso", "version": "1.2.0", "variant": "strict"},
  "lang": "en-US",
  "env": {"cpu": "...", "ram_gb": 32, "os": "...", "python": "3.11.6"},
  "intent_metrics": {
    "@high_thr_0.95": {
      "overall": {"accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0, "far_ood_fpr": 0.0},
      "per_bucket": {"template": {...}, "paraphrase": {...}, "near_ood": {...}, "far_ood": {...}}
    },
    "@medium_thr_0.50": { ... },
    "@low_thr_0.10":    { ... },
    "per_intent": [{"intent_id": "...", "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}],
    "confusion_top_k": [["predicted", "expected", 42]]
  },
  "slot_metrics": {
    "slot_f1": 0.0, "slot_precision": 0.0, "slot_recall": 0.0,
    "slot_exact_match": 0.0, "joint_exact_match": 0.0,
    "per_slot_type": {"duration": {...}, "named_entity": {...}}
  },
  "latency_ms": {"median": 0.0, "p95": 0.0, "p99": 0.0, "rtf": 0.0},
  "cost": {"peak_rss_mb": 0, "model_size_mb": 0, "index_build_s": 0.0}
}
```

### 9.2 `report.md`

A human-readable summary alongside `results.json`, with the headline numbers, per-bucket table, top confusions, and the engine's hardware/environment footprint. Template provided in the repo.

### 9.3 Reference machine

Conformant reports should be produced on a documented reference profile (e.g. GitHub Actions `ubuntu-latest`, 4 vCPU, 16 GB RAM) for cross-engine comparability. Other profiles are allowed but must be declared in `env`.

---

## 10. Dataset Generation Methodology

The v0.2 dataset was generated by **Claude Opus under heavy human guidance**, working from this spec and a fixed taxonomy of 50 intents across 10 domains. Generation rules — recorded here so they are reproducible, comparable across future regenerations, and auditable by reviewers:

### 10.1 Authoring constraints

- **No procedural template expansion.** Every train template, every test paraphrase, every keyword rule was written by the model and reviewed by a human, one (intent, lang) cell at a time. There is no script that takes a single English template and translates it to 12 languages; each language is authored in-language, with locale-specific vocabulary.
- **No LLM batch generation of full files.** Authoring proceeded one batch at a time (one intent × 12 languages per batch, 50 batches total). After each batch the runner runs `scripts/validate.py` and emits a small progress plot; errors were corrected before the next batch began. This keeps drift small and makes per-batch review tractable.
- **Native-speaker check absent.** The model used in v0.2 was multilingual-trained but not human-reviewed for fluency in every language. Native-speaker review is an open backlog item (see §11).
- **No reuse of OVOS skill intent names.** The 50 intents are taxonomy-only choices designed to exercise edge cases: slotless action intents (pause, mute, restart), slot-heavy intents (set_timer, create_event), and lexically-collision-prone sibling pairs within domains (next_story / previous_story / next_event).

### 10.2 Test bucket authoring rules

- **`template` (500/lang):** 10 rows per intent. Each row picks one training template at random and substitutes its `{slot}` placeholders with the slot's example values. This is the lexical-overlap-high bucket.
- **`paraphrase` (700/lang):** 14 rows per intent, hand-authored. Each must be lexically dissimilar to every training template (no shared content words beyond unavoidable function words like "the" / "a") while still preserving the intent and any required slots. This is the hard bucket; it dominates real production traffic.
- **`near_ood` (400/lang):** 8 rows per intent. Each row's gold label is a **sibling intent in the same domain** — not the row's "host" intent. The author writes an utterance for the sibling, and the row is filed under the host's batch so the dataset's near-OOD threat model is balanced (every intent has 8 sibling utterances pointing to it). The row's surface form is allowed to lexically overlap with the host's templates — that's the whole point of testing inter-intent confusion.
- **`far_ood` (50/lang):** 50 rows per language, hand-authored, chitchat / nonsense / out-of-scope. Six broad categories (politeness, philosophy, nonsense, feelings, factual non-questions, off-domain requests) cover the production threat surface.

### 10.3 Keyword rule authoring rules

For each (intent, lang) cell in `data/keyword_intents.py`:

- **At least one required group named with the `Kw` suffix** (e.g. `PlayKw`, `LockKw`, `WeatherKw`). The required group is the smallest set of action verbs / signature nouns that on its own distinguishes the intent against every sibling in the domain.
- **Optional groups always include `PleaseKw`** (politeness markers in the target language) — this is the most common test-set noise word that an engine should be allowed but not required to match.
- **Slot-name groups under `optional_vocab` populated with the slot's `examples`.** Adapt/Palavreado adapters register these as the slot's entity vocabulary so the engine can return the slot value alongside the intent. Slot groups are intentionally lower-cased to distinguish them from `Kw`-suffixed keyword groups when downstream tooling needs to separate the two.
- **Hand-translated, not transliterated.** Spanish doesn't get `play` as a `PlayKw` entry; it gets `pon` / `reproduce` / `dale`. Basque doesn't get `play` either; it gets `jarri` / `jo`. Per-language vocabulary is the whole reason a separate keyword file exists.

### 10.4 Per-batch QA gate

After each (intent × 12 langs) batch was authored, the pipeline ran:

1. `python scripts/validate.py` — schema validity, slot example counts, near-OOD target intent membership.
2. `python scripts/metrics.py` — emits per-language coverage plots and a progress snapshot. Used to spot drops in row counts (a sign of a malformed batch) early.

The two-stage gate caught roughly 20 % of batches needing correction before the next batch began — mostly slot-example dictionaries that lost a value to a typo, or near-OOD rows that pointed to a non-existent sibling intent.

### 10.5 Why this matters for results

The dataset is **small but dense**: 32 400 hand-authored rows across 12 languages, with deliberate near-OOD pressure and a shared far-OOD pool. Compared to procedurally-generated corpora of similar headline size, these properties hold:

- **Lexical diversity per intent is much higher.** A typical procedurally-generated intent ships 1–2 verb forms; this dataset ships 10–15 hand-authored action verbs per intent per language inside `required_vocab`. That diversity is why keyword engines (Adapt / Palavreado) score noticeably above their usual benchmark baselines here — they see real authoring of the keyword groups, not extracted scraps.
- **Paraphrase rows are adversarial.** Because they are explicitly de-overlapped against training templates, an engine that relies on surface n-grams (Padacioso strict, Nebulento ratio-mode) drops sharply on this bucket — that's the desired signal, not a bug.
- **Far-OOD is small enough to overfit to.** 50/lang means the FPR is statistically noisy; we accept this in exchange for the authoring time it took to write 12 × 50 hand-crafted chitchat samples in 12 languages. Future versions may expand this pool.

### 10.6 `asr_noise` and `typos` perturbation methodology

The two resilience buckets are **mechanically derived** from `paraphrase` rows, not hand-authored. The choice is deliberate:

- A human author can't fake keyboard-adjacency biases reliably — typos in production have a *systematic* distribution that machine perturbation captures more honestly than ad-hoc human guessing.
- ASR noise patterns (homophones, dropped function words, word-boundary breakdowns, filler insertions) are well-studied; we encode the common patterns once and apply them deterministically.
- Both buckets sample from the same 50 paraphrase seeds per language, so a side-by-side comparison directly answers *"which kind of noise breaks the engine — phonetic or typographic?"*

The generator lives at `scripts/_add_noise_buckets.py` in this repo and is fully reproducible (seed = 0).

**`asr_noise` operations**, applied additively per utterance:

1. **Homophone swap** (prob 0.55) — one pair from the language's homophone table (`their/there`, `e/é`, `dass/das`, `son/sont`, etc.). The table for each language lives in the generator.
2. **Dropped function words** (1–2 per utterance) — drops articles, prepositions, politeness markers from a language-specific stop set. ASR misses these constantly in noisy audio.
3. **Word-boundary collapse** (prob 0.35) — merge two adjacent words into one (e.g. `set a timer` → `setatimer`). Captures the worst of fast speech.
4. **Filler insertion** (prob 0.45) — splice in one hesitation word (`um`, `uh`, `humm`, `eh`, `ähm`, depending on language).

**`typos` operations**, 2–4 applied per utterance from this menu:

1. **Adjacent-key swap** — replace a letter with a QWERTY-adjacent one (e.g. `r` → `t`).
2. **Letter transposition** — swap two consecutive letters (`the` → `teh`).
3. **Letter drop** — delete one letter.
4. **Letter double** — duplicate one letter.
5. **Case flip** — uppercase/lowercase a single letter.

QWERTY adjacency is used for every language regardless of native keyboard layout — most users of the 12 target languages use QWERTY-derived layouts in practice, and modelling each AZERTY/QWERTZ/Dvorak variant would dilute the signal.

**What these buckets do NOT capture:**

- **ASR-specific phoneme confusions** beyond simple homophones (e.g. /θ/ → /f/ swaps in non-native English). A real ASR-output corpus would dwarf this synthetic bucket.
- **Autocorrect repairs** — modern keyboards mask many typos before the text reaches an intent engine. The benchmark assumes raw text-as-typed.
- **Code-switching** — utterances that mix languages mid-sentence.

These omissions are intentional; expanding into them is a v1.1 question.

---

## 11. Versioning

- **Spec version** follows semver. Backwards-incompatible schema changes bump the major (e.g. `1.0 → 2.0`). The v0.1 → v0.2 split into `train_templates.jsonl` + `train_keywords.jsonl` is a breaking change to the dataset shape but not yet a stable major.
- **Dataset version** is independent of spec version. Adding new intents, languages, or utterances within the existing schema bumps the dataset minor. Reshuffling splits bumps the dataset major.
- `results.json` records both versions; engines must declare which dataset version they were scored on.

---

## 12. Open Questions

- Whether to include a small **noisy-text** evaluation track (typos, run-on words) as a v1.1 add-on. Probably yes, but out of v1.0.
- **Native-speaker review pass** — every language other than en-US is currently single-author (Claude Opus). Per-language editor sign-off is the highest-leverage backlog item for v0.3.
- How to handle code-switched utterances ("toca a *Bohemian Rhapsody*") — defer to per-language dataset judgement for v1.0.
- Whether `train_keywords.jsonl` should ship a `required_vocab[group].weight` field for engines that support weighted-keyword scoring (currently no OVOS engine does).

---

## 13. Appendix A — Sample Intents

### A.1 Template-shaped: `media:play_song`

**`train_templates.jsonl` row:**

```json
{"intent_id": "media:play_song", "domain": "media", "lang": "en-US",
 "template": "(play|put on|throw on) {song} [by {artist}]",
 "slots": [
   {"name": "song",   "type": "phrase",       "required": true,
    "examples": ["bohemian rhapsody", "smells like teen spirit", "africa", "hey jude"]},
   {"name": "artist", "type": "named_entity", "required": false,
    "examples": ["queen", "nirvana", "toto", "the beatles"]}
 ]}
```

**`train_keywords.jsonl` row (same intent, same language):**

```json
{"intent_id": "media:play_song", "domain": "media", "lang": "en-US",
 "required_vocab": {
   "PlayKw": ["play", "put on", "throw on", "queue", "spin",
              "stream", "hear", "listen", "blast", "hit play",
              "crank up", "start"]
 },
 "optional_vocab": {
   "ByKw":     ["by", "from", "of"],
   "MeKw":     ["me", "for me", "to me"],
   "PleaseKw": ["please", "can you", "could you", "would you"],
   "TrackKw":  ["song", "track", "tune", "music", "record"],
   "song":     ["bohemian rhapsody", "smells like teen spirit", "africa", "hey jude"],
   "artist":   ["queen", "nirvana", "toto", "the beatles"]
 }}
```

**Sample `test.jsonl` rows:**

```json
{"utterance": "play bohemian rhapsody",                            "expected_intent": "media:play_song",  "expected_slots": {"song": "bohemian rhapsody"},                  "split": "template",   "domain": "media", "lang": "en-US"}
{"utterance": "could you throw on some queen for me",              "expected_intent": "media:play_song",  "expected_slots": {"artist": "queen"},                            "split": "paraphrase", "domain": "media", "lang": "en-US"}
{"utterance": "pause the music",                                   "expected_intent": "media:pause_playback", "expected_slots": null,                                       "split": "near_ood",   "domain": "media", "lang": "en-US"}
{"utterance": "what's the capital of finland",                     "expected_intent": null,               "expected_slots": null,                                            "split": "far_ood",    "domain": null,    "lang": "en-US"}
```

### A.2 Slotless: `media:pause_playback`

```json
// train_templates.jsonl (20 rows; one shown)
{"intent_id": "media:pause_playback", "domain": "media", "lang": "en-US",
 "template": "pause the music", "slots": []}

// train_keywords.jsonl (1 row)
{"intent_id": "media:pause_playback", "domain": "media", "lang": "en-US",
 "required_vocab": {
   "PauseKw": ["pause", "hold", "halt", "freeze", "hit pause"],
   "StopKw":  ["stop", "rest", "hold on"]
 },
 "optional_vocab": {
   "MusicKw":  ["music", "song", "track", "audio", "playback", "playing"],
   "PleaseKw": ["please", "can you", "could you", "would you"]
 }}
```

A slotless intent has no `optional_vocab` slot-name groups — the keyword rule is the whole identification surface.

### A.3 Slot-heavy: `timers_alarms:set_timer`

```json
// train_templates.jsonl
{"intent_id": "timers_alarms:set_timer", "domain": "timers_alarms", "lang": "en-US",
 "template": "(set|start) a {duration} timer [(called|named) {label}]",
 "slots": [
   {"name": "duration", "type": "duration", "required": true,
    "examples": ["5 minutes", "25 minutes", "half an hour"]},
   {"name": "label",    "type": "phrase",   "required": false,
    "examples": ["pomodoro", "tea", "laundry"]}
 ]}

// train_keywords.jsonl
{"intent_id": "timers_alarms:set_timer", "domain": "timers_alarms", "lang": "en-US",
 "required_vocab": {
   "SetKw":   ["set", "start", "begin", "create", "make"],
   "TimerKw": ["timer", "countdown"]
 },
 "optional_vocab": {
   "ForKw":     ["for", "of"],
   "CalledKw":  ["called", "named", "labelled"],
   "PleaseKw":  ["please", "can you", "could you"],
   "duration":  ["5 minutes", "10 minutes", "1 hour", "30 seconds",
                 "two hours", "ninety seconds"],
   "label":     ["pasta", "tea", "laundry", "cookies", "meditation", "study"]
 }}

// test.jsonl
{"utterance": "start a 25 minute timer called pomodoro",
 "expected_intent": "timers_alarms:set_timer",
 "expected_slots": {"duration": "25 minutes", "label": "pomodoro"},
 "split": "paraphrase", "domain": "timers_alarms", "lang": "en-US"}
```

---

## 14. Appendix B — Mapping Existing Benchmarks

For continuity, every field used by the existing nebulento and palavreado benchmarks maps cleanly into this spec:

| Existing concept                               | This spec                                                |
|------------------------------------------------|----------------------------------------------------------|
| `INTENTS` dict (nebulento)                     | `train_templates.jsonl` rows grouped by `intent_id`      |
| `.require()` keyword slots (palavreado/adapt)  | `train_keywords.jsonl` `required_vocab`                  |
| `.optionally()` keyword slots                  | `train_keywords.jsonl` `optional_vocab`                  |
| Slot value pools (named entities)              | `train_keywords.jsonl` `optional_vocab[<slot_name>]`     |
| `TEST_MATCH` utterances                        | `test.jsonl` rows with `split ∈ {template, paraphrase}`  |
| `NO_MATCH_UTTERANCES`                          | `test.jsonl` rows with `split = far_ood`                 |
| Per-intent FP / FN counts                      | §7.3 per-intent precision / recall                       |
| Latency RTF                                    | §7.5 latency block                                       |

A one-shot migration script can lift the existing benchmarks into the new format without information loss; `near_ood` (§4.4) and slot-level metrics (§8) are the genuinely new contributions.
