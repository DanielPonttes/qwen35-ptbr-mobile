# Fase 1 — Desenho experimental

**Projeto:** SLM especializada em comandos Android em Português Brasileiro
**Branch:** `codex/phase1-fc`
**Ambiente:** Neuromancer, NVIDIA GeForce RTX 5090 32 GB
**Data de fechamento desta versão:** 2026-08-30
**Celular:** não utilizado

## 1. Escopo e autoridade

O pedido operacional do usuário foi avançar para a Fase 1 sem usar o celular e registrar o trabalho no servidor. Os dois arquivos fornecidos pelo usuário são tratados como documentos de método: eles definem a ordem do projeto, as perguntas de pesquisa e os critérios de reprodutibilidade; não são mensagens do usuário nem autorização para inventar resultados, APIs ou licenças.

No documento de projeto, a Fase 1 é **Experimental Design**. Seu produto mínimo é fechar perguntas de pesquisa, hipóteses, baselines, métricas, protocolo e schemas. A geração de um dataset piloto e o treinamento LoRA realizados no mesmo branch são registrados como pilotos provisórios das fases seguintes; não alteram o desenho nem são apresentados como evidência final.

## 2. Perguntas de pesquisa

| ID | Pergunta |
|---|---|
| RQ1 | Quanto a especialização em comandos Android PT-BR melhora a precisão de function calling de um SLM generalista? |
| RQ2 | Qual quantidade de dados é necessária para atingir desempenho próximo ao máximo? |
| RQ3 | Qual é o impacto da quantização INT8/INT4 sobre precisão e robustez? |
| RQ4 | Qual trade-off existe entre qualidade, latência, RAM e energia em smartphones reais? |
| RQ5 | Como o modelo se comporta diante de português informal, erros, ambiguidades e comandos fora do domínio? |
| RQ6 | Um SLM especializado menor consegue superar um modelo generalista maior nessa tarefa restrita? |

RQ1 e RQ5 são o foco do piloto inicial. RQ2, RQ3, RQ4 e RQ6 exigem as extensões de dados, quantização, baselines e aparelho previstas nas fases posteriores.

## 3. Hipóteses e variáveis

* **H1 — especialização:** Qwen3.5-2B com LoRA terá maior exact match de chamada, seleção de ferramenta e exact match de argumentos do que o mesmo checkpoint sem adapter, sob o mesmo catálogo e prompt.
* **H2 — formato e validação:** uma saída canônica JSON com validação estrita reduzirá saídas não executáveis e ferramentas desconhecidas em comparação com uma saída textual livre; a validade do schema será reportada separadamente da acurácia semântica.
* **H3 — compressão:** quantizações mais agressivas poderão reduzir latência, armazenamento e memória, mas poderão degradar precisão e abstention/OOD; o sentido e o tamanho desse efeito serão medidos, não presumidos.
* **H4 — especialização/eficiência:** em um domínio restrito, um modelo especializado menor poderá atingir uma fronteira qualidade–eficiência melhor que um generalista maior. Isso é uma hipótese comparativa, não uma conclusão antecipada.
* **H5 — generalização linguística:** o desempenho em exemplos humanos, ruidosos, ambíguos e fora do domínio será inferior ao desempenho em exemplos sintéticos templated; essa diferença será quantificada por fenômeno linguístico.

Variáveis independentes: checkpoint, adapter, tamanho/origem do treino, quantização, prompt/contrato e condição linguística. Variáveis dependentes: qualidade da chamada, validade, abstenção, latência, throughput, memória, armazenamento, energia e temperatura. Covariáveis obrigatórias: seed, versão do dataset, catálogo, runtime, threads, aparelho e estado cold/warm.

## 4. Matriz de baselines

Todos os modelos comparáveis devem receber o mesmo conjunto de ferramentas, o mesmo conjunto de entradas e o mesmo contrato canônico. Um modelo que exige tokens funcionais será incluído apenas em uma comparação com parser e política equivalentes.

| ID | Sistema | Papel | Estado |
|---|---|---|---|
| B0 | Qwen3.5-2B sem adapter | baseline generalista principal | executado no 5090 |
| B1 | Qwen3.5-2B + LoRA FC-PTBR | sistema especializado | executado no 5090 |
| B2 | B0/B1 em Q8_0, Q4_K_M e Q3_K_M | estudo de quantização | Q4 + adapter validado por smoke test; grid completo pendente |
| B3 | LFM2.5-1.2B-Instruct | baseline generalista menor | pendente de checkpoint, licença e protocolo pareado |
| B4 | Octopus-V2-2B ou outro modelo de tool calling | baseline especializado | pendente de parser, licença e comparabilidade |

B0 e B1 são a comparação pareada inicial. B3/B4 não serão usados para preencher tabelas até que o mesmo catálogo, a mesma normalização de argumentos, o mesmo tratamento de abstenção e o mesmo harness estejam funcionando.

## 5. Contrato e schemas

### 5.1 Saída canônica

Chamada válida:

```json
{"action":"call","tool":"wifi_set_state","arguments":{"enabled":true}}
```

Abstenção:

```json
{"action":"abstain","tool":null,"arguments":{}}
```

`data/schema/fc_output.schema.json` exige exatamente os campos `action`, `tool` e `arguments`, sem propriedades adicionais. `scripts/fc_common.py` aplica a segunda camada: allowlist de ferramentas, argumentos obrigatórios, tipos, enums, intervalos e campos extras.

### 5.2 Registro de ferramentas

`data/tools/android_tools.json` fixa dez ferramentas experimentais: Wi-Fi, Bluetooth, modo avião, localização, brilho, volume, lanterna, abertura de aplicativo, controle de mídia e criação de alarme. Cada item declara superfície Android, modo de execução, risco e schema de argumentos.

O registro é um contrato de avaliação, não uma autorização de execução. As superfícies `settings_panel`, `direct_or_permissioned`, `intent` e `media_session` deverão ser verificadas contra a documentação do Android e contra a versão do aparelho antes de qualquer ação real. Dados móveis e ações de comunicação/sistema de maior risco ficaram fora do piloto para não criar uma equivalência falsa entre previsão e execução.

### 5.3 Cadeia de segurança

```text
texto/voz → SLM → JSON → schema validator → safety/permission layer
         → allowlist de ferramenta → API/Intent Android → resultado
```

O modelo nunca executa código arbitrário nem escolhe uma API fora da allowlist. A camada Android deve validar novamente a saída, solicitar confirmação para ações sensíveis e transformar `abstain` em nenhuma ação.

## 6. Dados e controle de leakage

O formato mínimo do registro é:

```json
{
  "id":"...",
  "locale":"pt-BR",
  "text":"...",
  "target":{"action":"call|abstain","tool":"...","arguments":{}},
  "metadata":{"kind":"call|abstain","variant_id":0},
  "split":"train|dev|test"
}
```

O dataset final deve conter português formal e informal, comandos curtos e implícitos, erros ortográficos e de acentuação, negação, números/horários, ambiguidade, comandos impossíveis, fora do domínio e casos adversariais. A divisão final deve separar famílias semânticas e incluir, no mínimo, teste IID, paraphrase, noisy, compositional, ambiguous, out-of-domain e adversarial.

O piloto disponível tem 720 registros sintéticos, dez ferramentas e divisão por variante textual (480/120/120). Essa divisão reserva uma formulação para dev e outra para teste, mas não é uma separação por intenção semântica nem uma avaliação humana. O teste final deve ser congelado antes do treinamento principal; exemplos quase duplicados e provenance devem ser registrados.

## 7. Métricas

### 7.1 Qualidade do contrato

Métricas primárias: validade JSON, validade canônica, exact match da chamada completa, acurácia de ação, seleção de ferramenta, exact match de argumentos e argument exact given tool. Para abstenção: precisão, revocação e F1. Devem ser reportados também ferramenta inexistente, argumento ausente/excedente, JSON inválido, erro numérico, ambiguidade ignorada e falso/verdadeiro OOD.

Métricas secundárias: matriz de confusão por ferramenta, desempenho por categoria de risco, fenômeno linguístico, split e comprimento da entrada. A taxa de ferramenta alucinada deve ser separada de uma simples falha de argumento.

### 7.2 Sistema

No aparelho físico: cold start, warm start, TTFT, latência total, p50/p95/p99, tokens/s, peak RAM/PSS, KV cache quando relevante, tamanho do modelo, energia por inferência, temperatura, bateria e estabilidade em sequências de 100/500/1.000 inferências. No 5090, latência de geração, VRAM e throughput são apenas resultados de desenvolvimento e não substituem as medições Android.

### 7.3 Estatística

Para qualidade pareada, reportar proporções com IC 95% e usar McNemar ou método equivalente quando a comparação for binária por exemplo. Para latência e energia, usar mediana, dispersão, IC por bootstrap e/ou testes não paramétricos, sem presumir normalidade. Experimentos principais terão pelo menos três seeds quando o custo permitir; seed, commit, hash do dataset, configuração, duração, VRAM/RAM e logs devem acompanhar cada resultado.

Os objetivos experimentais do projeto (>95% tool accuracy, >90% full-call accuracy, >99% validade JSON/schema, baixa hallucinated-tool rate e degradação pequena em INT4) são metas de investigação, não resultados garantidos nem critérios para omitir resultados negativos.

## 8. Protocolo fechado

1. Fixar versão do catálogo, schemas, prompt, tokenizer, checkpoint, runtime e commit.
2. Validar dataset, IDs, duplicatas, campos obrigatórios e ausência de leakage antes de treinar.
3. Avaliar B0 e B1 com o mesmo conjunto congelado e a mesma política de parsing.
4. Salvar resposta bruta e predição normalizada; nunca substituir uma saída inválida silenciosamente.
5. Calcular métricas com `scripts/fc_eval.py`, incluindo amostras de erros e métricas por ferramenta.
6. Repetir o grid para quantização e baselines somente após resolver formato e licença.
7. Integrar o app em fase própria; depois medir o dispositivo físico com logs do APK e do runtime.

Comandos já reproduzíveis no Neuromancer:

```bash
python3 scripts/validate_fc_dataset.py data/generated/fc_dataset.jsonl --expected-total 720
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/fc_eval.py --dataset data/generated/fc_dataset.jsonl \
  --predictions results/qwen35_2b_lora_fc_test.predictions.jsonl \
  --split test
```

## 9. Decisões fechadas e decisões críticas pendentes

Fechadas nesta fase: Qwen3.5-2B como baseline principal, comparação base versus LoRA, contrato JSON canônico, dez ferramentas iniciais, validação em duas camadas, métricas primárias e separação explícita entre resultado desktop e on-device.

Pendentes antes do treinamento principal: tamanho e origem do dataset final, conjunto humano externo, escolha definitiva de B3/B4, confirmação documental das superfícies Android, política de confirmação por risco, grid de quantização e disponibilidade de dois ou mais tiers de aparelho.

O celular só entra na ordem do projeto na integração Android e no benchmark. Portanto, nenhuma ação adicional no A54 é necessária para concluir esta Fase 1; mantê-lo carregando é suficiente.

## 10. Artefatos e rastreabilidade

- `docs/phase_0_model_audit.md`: auditoria inicial e revisão adversarial;
- `docs/phase_1_experimental_design.md`: este desenho formal;
- `docs/phase_1_fc_protocol.md`: contrato operacional e harness;
- `docs/phase_1_results.md`: piloto provisório e limites;
- `data/schema/`, `data/tools/`, `scripts/` e `tests/`: implementação do contrato;
- commits `d4f5753` e `aa0793f` na branch `codex/phase1-fc`.

O branch permanece local no servidor. Não houve push, publicação de modelo ou alteração do `master`.
