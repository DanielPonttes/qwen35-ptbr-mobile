# Revisão adversarial e correção da Fase 1

**Data:** 2026-08-30
**Branch:** `codex/phase1-fc`
**HEAD da revisão inicial:** `5e311b5`
**Escopo:** desenho experimental, pipeline de function calling e evidências produzidas até a correção.

## 1. Veredito da revisão inicial

| Camada | Veredito inicial |
|---|---|
| Infraestrutura de engenharia | **PASS com ressalvas** |
| Desenho formal documentado | **CONDITIONAL PASS** |
| Evidência científica para artigo | **FAIL** |
| Prontidão do protótipo Android FC | **FAIL** |

O contrato JSON, o catálogo, o validador, o treinamento LoRA na RTX 5090, a exportação GGUF, o smoke test no `llama.cpp`, os testes automatizados e a compilação da base Android funcionavam. A revisão concluiu que isso não demonstrava generalização nem autorizava submissão do piloto como resultado final.

## 2. Revisores acionados

Foi enviado o mesmo dossiê factual, sem autorização para editar o repositório.

| Revisor | Configuração | Resultado |
|---|---|---|
| Gemini | `gemini-3.7-flash`, esforço alto, modo plan | **CONDITIONAL PASS** |
| Grok | `cursor-grok-4.6-xhigh` expirou; fallback `cursor-grok-4.6-xhigh-fast`, modo ask | **FAIL** |
| Muse | `opencode-go/muse-spark-1.2-contributor` e fallback gratuito | timeout sem parecer; não contado como aprovação |
| GLM solicitado em substituição ao Muse | `opencode-go/glm-5.3-flash` | modelo listado no Neuromancer, mas timeout tanto no dossiê completo quanto no teste mínimo `PONG`; sem parecer utilizável |

A indisponibilidade do GLM é registrada como limitação operacional da revisão, não como aprovação ou reprovação. As correções abaixo foram aplicadas com base no consenso Gemini/Grok e nas auditorias determinísticas do próprio pipeline.

## 3. Falhas encontradas

### P1 — leakage por identidade de caso

A versão antiga usava quatro variantes no treino, uma no desenvolvimento e uma no teste, mas todos os 120 `case_id` apareciam nos três splits. Também havia 41 targets repetidos entre os splits. O exact match de 99,17% era, portanto, um teste de pipeline em corpus controlado, não evidência válida de generalização.

### P1 — inferência estatística ausente

A avaliação anterior apresentava apenas métricas pontuais e uma seed. Não havia IC95%, comparação pareada, bootstrap de F1 ou verificação da variabilidade do treinamento.

### P1 — escopo maior que a evidência

O app compilado era a aplicação legada de chat. Não existiam parser FC Kotlin, camada de segurança, allowlist executável ou dispatcher Android integrado. Também não havia dados humanos, ruído, OOD, grid completo de quantização ou baselines externos.

## 4. Correções aplicadas

1. Dataset e resultados antigos foram preservados em `data/archive/fc_dataset_variant_holdout_v1/` e `results/archive/variant_holdout_v1/`.
2. O gerador passou a usar `fc-android-ptbr/0.2.0-case-split`, com 200 `case_id` únicos, 1.200 registros, split 720/240/240 e balanceamento chamada/abstenção. A validação agora falha se um caso cruzar splits ou se houver texto duplicado.
3. `scripts/fc_eval.py` passou a emitir IC95% de Wilson e bootstrap determinístico para F1 de abstenção.
4. `scripts/compare_fc_predictions.py` foi adicionado para alinhar predições por `id`, calcular IC95% e McNemar exato bilateral; p-values muito pequenos não são arredondados para zero.
5. O treinamento foi repetido com três seeds (`20260830`, `20260831`, `20260832`) no RTX 5090 e todos os adapters foram avaliados no mesmo teste corrigido.
6. O adapter representativo foi reexportado para GGUF e validado no `llama.cpp` com uma chamada e uma abstenção.
7. A documentação passou a separar explicitamente piloto sintético, integração de servidor e benchmark Android. O celular não foi usado.

## 5. Evidência após a correção

No teste de 240 exemplos, o B0 obteve 71,67% de exact match `[65,66%, 76,99%]` pelo IC95% Wilson. As três execuções LoRA obtiveram 99,17%, 99,17% e 97,08%, com IC95% `[97,01%, 99,77%]`, `[97,01%, 99,77%]` e `[94,10%, 98,58%]`, respectivamente.

O McNemar exato bilateral para exact match, comparando cada seed com B0, resultou em `p=2,71e-20`, `p=2,71e-20` e `p=6,80e-16`. Esses p-values confirmam diferença no corpus sintético pareado; não eliminam risco de viés de template, nem substituem avaliação humana.

O teste de integridade executou sete testes unitários, compilação Python, geração/validação do dataset, três treinamentos CUDA, quatro avaliações HF, três comparações pareadas, exportação GGUF e dois smoke tests `llama.cpp` válidos.

## 6. Veredito revisado

| Camada | Estado após correção |
|---|---|
| Infraestrutura de engenharia | **PASS com ressalvas** |
| Controle de leakage e estatística do piloto | **PASS para o escopo sintético corrigido** |
| Evidência para o artigo final | **CONDITIONAL / ainda não pronta** |
| Protótipo Android FC e claims on-device | **FAIL; fase posterior** |

O artigo pode descrever este material como piloto metodológico/engenharia, desde que não o apresente como validação de comandos Android reais. Para um artigo experimental completo ainda são necessários dataset humano independente, split por família semântica, teste OOD/noisy/compositional, baselines comparáveis, quantização e benchmark físico.

## 7. Registro de não aprovação

Nenhum revisor externo aprovou o artigo para submissão. Gemini deu aprovação condicional da infraestrutura, Grok apontou falha científica antes da correção e o GLM não entregou parecer por timeout. A aplicação das correções melhora a validade do piloto, mas não deve ser descrita como revisão posterior positiva sem um novo ciclo de revisão.
