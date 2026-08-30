# From Human Speech Transcripts to Conservative Android Function Calls

## A leakage-audited evaluation of a small language model

**Working paper — ERAMIA-RS 2026**
**Status:** phase-1 experimental draft, 30 August 2026

## Abstract

Voice interfaces for mobile devices must distinguish a supported command from an unsupported or ambiguous request before any Android API is invoked. This paper presents a reproducible, deliberately narrow evaluation of a Qwen3.5-2B language model as a text-to-function router for two Android-aligned operations: media play/pause and volume up/down. To obtain human-origin English command text without claiming that an existing corpus is an Android corpus, we derive an auditable benchmark from Fluent Speech Commands (FSC). Only four native FSC semantic combinations are mapped to calls; all other source examples are retained as explicit policy-abstention negatives. We compare zero-shot prompting with a LoRA adapter on two split protocols: the official speaker-disjoint split and a stricter phrase-disjoint split that prevents normalized templates from crossing train, development, and test. The experiment consumes transcriptions, not waveforms, and therefore does not evaluate automatic speech recognition, physical Android execution, or on-device inference. The speaker-disjoint run illustrates why the lexical control is required: the zero-shot model reaches 39.97\% exact match, whereas the adapted model reaches 100\% on the same test, despite substantial template overlap between splits. On the phrase-disjoint test, zero-shot reaches 40.33\% exact match and LoRA reaches 100\%. The contribution is therefore a measurement protocol and an auditable baseline, not a claim of general Android command understanding.

**Keywords:** function calling; Android; speech commands; small language models; abstention; data leakage; Brazilian Portuguese research infrastructure.

## 1. Motivation and scope

The original project goal was a specialized small language model for Android commands in Brazilian Portuguese. The available human command corpus, however, is not a native Android/PT-BR corpus. The present phase makes that limitation explicit and uses English human speech data as a temporary benchmark. Portuguese synthetic data remains a historical harness for the repository; it is not presented as human evidence.

The operational question is narrower than “can a language model control Android?” We ask whether a small model can transform a human-origin command transcription into a validated canonical object, or abstain when the source semantics do not belong to the evaluated contract. No generated object is executed. The Android API names in the registry identify the intended integration boundary only.

The phase has four research questions:

1. Can a Qwen3.5-2B model produce valid canonical JSON for a two-tool Android-aligned contract?
2. What improvement does a LoRA adapter provide over the same model and prompt without adaptation?
3. How much does performance change when normalized phrase templates are held out, rather than only speakers?
4. Does the abstention policy prevent unsupported FSC semantics from being silently converted into Android calls?

## 2. Corpus and conservative semantic bridge

We use Fluent Speech Commands, a human-recorded English speech-command corpus with 30,043 recordings, 97 speakers, and 31 native intents. The source provides official train, validation, and test files with speaker separation. The source archive is stored outside the Git repository and is tracked by SHA-256 in `data/external/fluent_speech_commands_manifest.json`.

The corpus is not an Android command dataset. We therefore define a two-tool evaluation registry and a deterministic mapping that accepts only four native semantic combinations:

| Native FSC action | Native object | Derived Android-aligned target |
|---|---|---|
| activate | music | `media_control(action=play)` |
| deactivate | music | `media_control(action=pause)` |
| increase | volume | `volume_adjust(direction=up)` |
| decrease | volume | `volume_adjust(direction=down)` |

Every other FSC example becomes `{"action":"abstain","tool":null,"arguments":{}}`. This is a derived policy label, not a native FSC gold annotation. The benchmark balances supported calls and unsupported policy negatives independently within each split. This prevents the much larger set of unrelated smart-home intents from dominating the aggregate score.

The registry is intentionally narrow. `media_control` represents the Android `MediaSessionManager` boundary and accepts `play` or `pause`; `volume_adjust` represents the `AudioManager.adjustStreamVolume` boundary and accepts `up` or `down`. The experiment does not request permissions, call either API, or claim that the model is safe to deploy without a separate validator and policy layer.

## 3. Leakage-audited splits

We preserve the official speaker-disjoint split and also construct a phrase-disjoint split. A template is the Unicode-normalized, case-folded, whitespace-normalized, punctuation-stripped transcription; its stable SHA-256 prefix is stored as `template_id`. A template group is assigned to only one split within its native FSC label. Speakers may occur in several splits in the phrase-disjoint protocol by design.

| Protocol | Train | Dev | Test | Speaker overlap | Template overlap |
|---|---:|---:|---:|---|---|
| Official speaker-disjoint, balanced | 11,442 | 1,580 | 1,934 | none | train/dev/test overlap exists |
| Phrase-disjoint, balanced | 10,458 | 2,202 | 2,296 | allowed by design | none |

Both derived files contain 7,478 call examples and 7,478 abstention examples. The official test contains 967 calls and 967 policy negatives. The phrase-disjoint test contains 1,148 calls and 1,148 policy negatives. Dataset validation checks schema, target validity, duplicate IDs, expected locale, split assignment, and the selected group-disjointness invariant.

## 4. Model and training protocol

The base checkpoint is Qwen3.5-2B loaded from the local Hugging Face cache. The model receives an English system instruction, the compact two-tool catalog, and one transcription. It must emit exactly `action`, `tool`, and `arguments` as JSON. The baseline uses the canonical prompt without adapter weights.

The adapted model uses LoRA with rank 16, alpha 32, dropout 0.05, and all-linear target modules. Training uses two epochs, batch size 4, gradient accumulation 4, learning rate $2\times10^{-4}$, maximum sequence length 1,024, bf16 autocast, and seed 20260830. The official run used 11,442 training records and 1,432 optimizer steps per epoch on an NVIDIA RTX 5090 with 32 GB VRAM. The phrase-disjoint run uses the same hyperparameters and hardware, with only the split protocol changed.

The model is evaluated as a server-side text router. There is no microphone input, ASR stage, Android APK execution, permission flow, battery test, thermal test, or physical-device latency result in this phase. Those are separate gates for a later experiment.

## 5. Metrics

For every test item we report:

- JSON validity and canonical validity;
- exact match of the canonical target;
- action accuracy;
- tool-selection accuracy on call gold items;
- argument exactness overall and conditional on the selected tool;
- abstention precision, recall, and F1;
- the same measurements grouped by the explicit mapping rule.

The evaluator also reports missing/extra predictions, a sample of invalid outputs, deterministic bootstrap percentile intervals for abstention F1, and Wilson 95\% intervals for proportions. Latency is recorded per generated request as an engineering observation, not as an on-device claim.

## 6. Results

### 6.1 Official speaker-disjoint test

The zero-shot baseline produced 1,934 predictions. It achieved 100\% JSON parseability but only 68.77\% canonical validity and 39.97\% exact match. Abstention F1 was 61.08\%. Call selection was uneven: exact match was 57.85\% for play, 31.43\% for pause, and 0\% for both volume directions. On unsupported policy negatives, abstention recall was 69.29\%.

The LoRA adapter produced 1,934 valid predictions and achieved 100\% on every reported point estimate in this test: exact match, canonical validity, action, tool, arguments, and abstention F1. The Wilson lower bound for exact match is 99.80\% with 1,934 observations.

This apparent result is not sufficient evidence of broad generalization. The official split is speaker-disjoint but has 244--247 normalized templates in common across split pairs. The adapted model can therefore exploit lexical repetition while learning the mapping. This is the motivation for the phrase-disjoint run.

The recorded GPU generation latency was 361.7 ms on average for the adapted model (median 359.0 ms, 95th percentile 396.6 ms) and 312.5 ms on average for the baseline. These are server measurements for single-request text generation and must not be interpreted as smartphone performance.

### 6.2 Phrase-disjoint test

The phrase-disjoint adapter was trained and evaluated with the same code and hyperparameters. On 2,296 test items, the zero-shot baseline reached 74.09\% canonical validity, 40.33\% exact match, tool selection of 10.63\%, and abstention F1 of 58.97\%. The LoRA adapter reached 100\% on all point estimates: JSON validity, canonical validity, exact match, action, tool, arguments, and abstention F1. The Wilson lower bound for exact match is 99.83\%.

The per-rule pattern is informative. Zero-shot exact match was 100\% for play, 0\% for pause, 0\% for decrease-volume, and 0\% for increase-volume; abstention recall on derived negatives was 70.03\%. LoRA reached 100\% on all four mapped rules and on the derived abstention policy.

The two split protocols therefore agree on the narrow task, but they do not establish Android command understanding beyond the four manually declared semantic bridges. The absence of template overlap removes one important leakage channel; it does not replace an independently annotated Android test.

## 7. Discussion

The experiment demonstrates the value of separating three questions that are often conflated. First, a human-origin speech corpus can provide command text, but it does not become an Android corpus merely because a researcher writes a mapping. Second, a model can learn a strict JSON contract while still failing semantic routing, as the zero-shot volume results show. Third, a high score on an official speaker split can reflect repeated lexical templates rather than robustness to new wording.

The abstention target is useful as a safety-oriented interface, but it is also the most assumption-sensitive part of the benchmark. Unsupported FSC examples are not human-annotated Android refusals; they are policy negatives created by the experimenter. The correct claim is therefore “abstention under a declared derived policy,” not “safe refusal for Android.”

The adapter's official-split perfect score should be treated as an overfit signal until compared with the phrase-disjoint protocol and, ultimately, with a held-out human evaluation designed for Android commands. Even after lexical control, text-only evaluation leaves the ASR error channel unmeasured. A production pipeline needs an ASR model, confidence thresholding, schema validation, permission checks, and a policy gate between generated JSON and Android APIs.

## 8. Reproducibility and data governance

The repository records the mapping contract, generator, source-file hashes, split summaries, validator, training script, baseline script, evaluator, tests, and aggregate metrics. The raw FSC archive and derived JSONL files remain server-local. Because the source is distributed under CC BY-NC-ND 4.0 for academic research, the derived transcripts/labels and per-example predictions are ignored by Git and are not redistributed with the repository. The manifest preserves provenance without shipping the restricted data.

The main commands are:

```text
python scripts/prepare_fsc_android_fc.py ...
python scripts/validate_fc_dataset.py ... --expected-locale en-US --allow-text-duplicates --group-field speaker_id
python scripts/run_qwen_fc_baseline.py ... --locale en-US --prompt-mode canonical
python scripts/train_qwen_fc_lora.py ... --locale en-US
python scripts/fc_eval.py ...
```

All phase-1 tests pass locally on the Neuromancer environment. The training and evaluation jobs use the RTX 5090; the Android phone is not required for this phase.

## 9. Limitations and next gates

1. The only human corpus used in this phase is English FSC; no PT-BR human command set is claimed.
2. FSC is not an Android corpus; four mappings are manually specified and all other targets are derived abstentions.
3. The input is transcription text, not audio; ASR robustness is unmeasured.
4. The official result is vulnerable to lexical repetition; the phrase-disjoint control was executed, but an independent Android test is still required before publication.
5. The contract covers only play/pause and volume up/down.
6. No Android API is called, no permission is exercised, and no safety guarantee is established.
7. No on-device or physical-phone result is reported. The Qwen3.5-2B experiment runs on the RTX 5090.
8. The source license requires an explicit redistribution audit before any public data release.

The next critical gates are: add an independently annotated Android-command test; insert an ASR stage; benchmark the validator and policy layer; and only then connect the phone for physical execution and on-device measurements.

## 10. Current conclusion

This phase establishes an auditable benchmark and an honest experimental boundary for the first article. The model reaches 100\% on both declared split protocols, while the zero-shot baseline remains near 40\% exact match; this is evidence that the model learns the narrow declared contract, not evidence of general Android command understanding. The article should still include an independent human/Android evaluation or be framed explicitly as a protocol and preliminary engineering report.
