# Revisão adversarial da Fase 1

**Data:** 2026-08-30
**Branch auditada:** `codex/phase1-fc`
**HEAD auditado:** `737bbf7233404e7b3d1fc738b25f6e647eba9daa`
**Escopo:** desenho experimental e evidências produzidas até agora

## 1. Veredito consolidado

| Camada | Veredito |
|---|---|
| Infraestrutura de engenharia | **PASS com ressalvas** |
| Desenho formal documentado | **CONDITIONAL PASS** |
| Evidência científica para artigo | **FAIL** |
| Prontidão do protótipo Android FC | **FAIL** |

O contrato JSON, o catálogo, o validador, o treinamento LoRA na RTX 5090, a exportação GGUF, o smoke test no `llama.cpp`, os testes automatizados e a compilação da base Android funcionam. Isso não torna o piloto uma demonstração de generalização nem permite submeter o resultado como artigo experimental.

## 2. Revisores acionados

Foi solicitada uma revisão independente do `HEAD` atual, sem edição de arquivos, com o mesmo dossiê e critérios.

| Revisor | Configuração | Resultado |
|---|---|---|
| Gemini | `gemini-3.7-flash`, esforço alto, modo plan | respondeu: **CONDITIONAL PASS** |
| Grok | `cursor-grok-4.6-xhigh-fast`, modo ask | respondeu: **FAIL** |
| Muse | `opencode-go/muse-spark-1.2-contributor` e fallback `opencode/muse-spark-1.2-contributor-free` | timeout sem parecer; não contado como aprovação |

O identificador exato `cursor-grok-4.6-xhigh` também expirou antes de responder; o fallback operacional foi usado e identificado acima. A indisponibilidade do Muse é uma falha de cobertura da revisão, não evidência positiva.

## 3. Evidências confirmadas

- dataset sintético: 720 registros, 360 chamadas e 360 abstenções;
- B0/B1 avaliados no mesmo teste de 120 exemplos;
- exact match: 64,17% no B0 e 99,17% no B1;
- LoRA BF16 treinado na NVIDIA GeForce RTX 5090;
- adapter GGUF F16 de 33.664.704 bytes;
- smoke `llama.cpp` com um caso de chamada e um caso de abstenção, ambos válidos;
- cinco testes automatizados, validação do dataset e `git diff --check` aprovados;
- projeto Android legado compilado com `test assembleDebug` após configurar SDK 35 no servidor;
- branch limpo, `master` inalterado e sem push.

Esses itens são evidências de execução de pipeline, não de desempenho on-device. Não há medições no A54, nem dispatcher FC no app atual.

## 4. Bloqueador crítico: leakage do dataset

O gerador usa quatro variantes de texto no treino, uma no dev e uma no teste. Essa separação é útil para testar superfície textual, mas não separa a identidade semântica do caso.

Auditoria direta do JSONL atual:

```text
unique_case_ids = 120
case_ids presentes em train, dev e test = 120
unique_targets = 41
targets presentes em train, dev e test = 41
variant 0–3 = train; variant 4 = dev; variant 5 = test
```

Assim, os mesmos casos, ferramentas e valores de argumentos atravessam os splits. O número de 99,17% deve ser tratado como **não interpretável para generalização**; no máximo, é um teste de pipeline em um corpus controlado. Não deve aparecer como resultado principal de artigo, mesmo acompanhado de uma ressalva.

## 5. Problemas metodológicos adicionais

### P1 — inferência estatística ausente

O `scripts/fc_eval.py` calcula métricas pontuais, mas não IC95%, McNemar, bootstrap, p-value ou effect size. Uma execução e uma seed não separam efeito de especialização, overfit e variação aleatória.

### P1 — escopo medido menor que o escopo declarado

As seis RQs cobrem especialização, quantidade de dados, quantização, eficiência on-device, robustez linguística e comparação com modelos menores. O estado atual só sustenta uma comparação preliminar B0/B1 em dataset sintético com holdout de formulação. Não sustenta conclusões sobre Android real, energia, OOD humano, quantização em lote ou baselines externos.

### P1 — app Android ainda não é um protótipo FC

O APK que compila é a aplicação legada de chat. Ainda não há carregamento do adapter no app, parser/validador Kotlin, camada de segurança, allowlist executável ou dispatcher de intents/APIs. O smoke test no servidor confirma somente a cadeia GGUF + `llama.cpp`.

### P2 — contrato de execução precisa de confirmação documental

O catálogo é adequado como contrato experimental, mas algumas superfícies (`settings_panel`, `direct_or_permissioned` e `media_session`) podem exigir interação, permissões ou serviços específicos nas versões atuais do Android. O artigo não deve chamar essas ferramentas de execução automática antes da verificação por API e aparelho.

### P2 — baselines e quantização incompletos

LFM2.5, Octopus-V2 e o grid Q8/Q4/Q3 ainda não foram avaliados em lote com o mesmo contrato. O GGUF e o smoke test são integração, não uma ablação de qualidade.

## 6. Pontos fortes

1. O projeto abandonou corretamente os resultados antigos de ENEM/chat como proxy de function calling.
2. B0 e B1 isolam o efeito do adapter no mesmo checkpoint, prompt e catálogo.
3. O contrato canônico, o validador em duas camadas e a política explícita de abstenção tornam a avaliação auditável.
4. O uso do RTX 5090, hashes, manifestos, branch separado e artefatos ignorados pelo Git está rastreável.
5. As limitações principais estão explicitamente registradas, reduzindo risco de overclaim.

## 7. Ações necessárias antes de declarar aprovação

1. Reconstruir o split com `case_id`, combinação de argumentos e, idealmente, famílias de templates disjuntas; classificar o teste atual como inválido para generalização.
2. Congelar um plano estatístico e implementar IC95%, comparação pareada e múltiplas seeds antes de voltar a treinar.
3. Reduzir o artigo a perguntas que possam ser respondidas pelos dados disponíveis ou executar os experimentos adicionais: conjunto humano/semântico, baselines, quantização e Android real.
4. Se o escopo incluir on-device, implementar e testar o dispatcher FC no APK antes de usar “execução Android” no título, resumo ou conclusão.

Até essas ações, o resultado deve ser descrito como **piloto de engenharia interno**. A revisão atual não aprova submissão ao ERAMIA nem autoriza tratar 99,17% como resultado científico final.
