# Resultados da fase 1 — FSC humano em inglês

**Data da execução:** 30/08/2026
**Servidor:** Neuromancer / RTX 5090 32 GB
**Modelo:** Qwen3.5-2B; baseline zero-shot e LoRA (r=16, alpha=32)
**Entrada:** transcrição textual; não houve ASR, execução Android ou uso do celular.

## Escopo e dados

O Fluent Speech Commands (FSC) é um corpus humano em inglês, não um corpus Android. A ponte usada nesta fase mapeia somente quatro combinações nativas para o contrato Android estreito:

- `activate + music` → `media_control(action=play)`;
- `deactivate + music` → `media_control(action=pause)`;
- `increase + volume` → `volume_adjust(direction=up)`;
- `decrease + volume` → `volume_adjust(direction=down)`.

Todos os demais exemplos são negativos de política derivados e recebem `abstain`; não são anotações humanas de recusa Android.

| Protocolo | Treino | Dev | Teste | Controle | Interseção entre splits |
|---|---:|---:|---:|---|---|
| Oficial | 11.442 | 1.580 | 1.934 | locutor | 0 locutores; 244–247 templates |
| Phrase-disjoint | 10.458 | 2.202 | 2.296 | template | 0 templates; locutores podem repetir |

Cada dataset derivado tem 7.478 chamadas e 7.478 abstentions. O teste oficial contém 967 chamadas e 967 negativos de política; o teste phrase-disjoint contém 1.148 chamadas e 1.148 negativos.

## Métricas agregadas

| Split / modelo | JSON válido | Canônico válido | Exact match | Tool selection (calls) | Argumento exato | Abstention F1 |
|---|---:|---:|---:|---:|---:|---:|
| Oficial / zero-shot | 100,00% | 68,77% | 39,97% | 10,65% | 10,65% | 61,08% |
| Oficial / LoRA | 100,00% | 100,00% | 100,00% | 100,00% | 100,00% | 100,00% |
| Phrase / zero-shot | 100,00% | 74,09% | 40,33% | 10,63% | 10,63% | 58,97% |
| Phrase / LoRA | 100,00% | 100,00% | 100,00% | 100,00% | 100,00% | 100,00% |

Os intervalos Wilson de 95% para exact match do LoRA têm limites inferiores de 99,80% (oficial, n=1.934) e 99,83% (phrase-disjoint, n=2.296). O F1 de abstention do LoRA ficou em 100% nos dois testes; isso não deve ser interpretado como garantia de segurança, pois os negativos são derivados por política.

## Métricas por regra

No teste oficial, o baseline acertou `activate_music_to_media_play` em 57,85%, `deactivate_music_to_media_pause` em 31,43% e as duas regras de volume em 0%. Sua recall de abstention nos negativos derivados foi 69,29%.

No teste phrase-disjoint, o baseline acertou play em 100%, pause em 0%, decrease volume em 0% e increase volume em 0%. Sua recall de abstention nos negativos derivados foi 70,03%.

O LoRA alcançou 100% de exact match, validade canônica, seleção de ferramenta e argumentos em cada uma das quatro regras e 100% de abstention F1 nos dois protocolos.

## Latência de geração no servidor

Valores de geração de uma requisição textual, em GPU, sem pretensão de representar smartphone:

| Execução | Média (ms) | Mediana (ms) | P95 (ms) | Máximo (ms) |
|---|---:|---:|---:|---:|
| Oficial / zero-shot | 312,5 | 340,3 | 410,1 | 528,0 |
| Oficial / LoRA | 361,7 | 359,0 | 396,6 | 704,9 |
| Phrase / zero-shot | 308,4 | 332,1 | 402,9 | 518,0 |
| Phrase / LoRA | 368,5 | 364,5 | 403,1 | 694,8 |

## Configuração e rastreabilidade

- Dataset oficial: SHA-256 `5d5ed89b0108ff9821c9c1e44d50f8435623cdc1ada5dedfb73612bb4990b2a6`.
- Dataset phrase-disjoint: SHA-256 `bb551304122a8aa28199ad5226b48da29dd53bd9bd11c8aed5a9b27b728b4707`.
- Registry: `data/tools/android_tools_fsc.json`.
- Seed: `20260830`; épocas: 2; batch: 4; acumulação: 4; max length: 1024; learning rate: `2e-4`; bf16.
- Treino oficial: 1.789,257 s; 1.432 passos por época; dev loss final `0.0`.
- Treino phrase-disjoint: 1.661,871 s; 1.308 passos por época; dev loss final `0.034014`.
- Ambiente: PyTorch `2.10.0+cu128`, Transformers `5.3.0`, PEFT `0.20.0`.

## Interpretação

O resultado zero-shot mostra que produzir JSON não implica selecionar a ferramenta correta: no split phrase-disjoint, a validade canônica foi 74,09%, mas exact match apenas 40,33%; a falha se concentra nas regras de volume. O LoRA aprende o contrato e as quatro pontes declaradas, inclusive em templates não vistos. Ainda assim, a conclusão é restrita: não há evidência de desempenho em português, ASR, comandos Android reais, permissões, execução de API ou inferência no dispositivo.

O FSC é licenciado sob CC BY-NC-ND 4.0 para pesquisa acadêmica. Os JSONL derivados, adaptadores e previsões por exemplo permanecem server-local e estão protegidos pelo `.gitignore`; o Git recebe somente o código, contratos, manifestos, documentação e métricas agregadas.

## Próximos gates críticos

1. Construir um teste Android independente, anotado por humanos, para substituir a ponte derivada como evidência principal.
2. Inserir e medir ASR, incluindo erros de transcrição e limiar de confiança.
3. Implementar o validador/policy gate entre JSON e APIs Android.
4. Só então conectar o celular para medir execução física, permissões, térmica, bateria e on-device.
