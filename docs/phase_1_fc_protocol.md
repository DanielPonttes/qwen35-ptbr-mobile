# Fase 1 — Contrato, dataset e harness de function calling

**Projeto:** SLM especializada em comandos Android em Português Brasileiro
**Branch:** `codex/phase1-fc`
**Ambiente:** Neuromancer / RTX 5090
**Status:** contrato, dataset corrigido, piloto LoRA, estatística básica e smoke GGUF implementados; benchmark Android ainda pendente.

## 1. Objetivo desta fase

Esta fase transforma a hipótese do projeto em um protocolo executável antes de qualquer claim de desempenho. O resultado é um contrato canônico, um catálogo versionado de ferramentas, um dataset sintético de desenvolvimento e um avaliador independente do modelo.

O celular não é necessário para gerar, validar ou avaliar o dataset, nem para executar o piloto no 5090. Ele será necessário somente para validar empacotamento, execução local e métricas de aparelho.

## 2. Contrato de saída

Toda predição deve ser normalizada para:

```json
{"action":"call","tool":"wifi_set_state","arguments":{"enabled":true}}
```

Quando o comando for ambíguo, incompleto, fora do catálogo ou ultrapassar a política de segurança:

```json
{"action":"abstain","tool":null,"arguments":{}}
```

`data/schema/fc_output.schema.json` impõe os campos canônicos. `scripts/fc_common.py` valida adicionalmente se a ferramenta existe, se os argumentos obrigatórios estão presentes, se não há campos extras e se tipos, enums e limites são respeitados.

## 3. Catálogo de ferramentas

`data/tools/android_tools.json` contém dez ferramentas experimentais:

- conectividade: Wi-Fi, Bluetooth e modo avião;
- privacidade/dispositivo: localização, brilho e lanterna;
- mídia e aplicativos: volume, controle de mídia e abertura de aplicativo;
- produtividade: criação de alarme.

O catálogo é um contrato de avaliação, não uma autorização Android. Ações classificadas como `settings_panel` podem exigir interação do usuário ou permissões especiais em versões recentes do Android. O despachante deverá validar os argumentos novamente antes de executar qualquer ação.

## 4. Dataset de desenvolvimento corrigido

O gerador `scripts/generate_fc_dataset.py` cria, com seed fixa `20260830`, a versão `fc-android-ptbr/0.2.0-case-split`:

| Split | Registros | Chamadas | Abstenções |
|---|---:|---:|---:|
| train | 720 | 360 | 360 |
| dev | 240 | 120 | 120 |
| test | 240 | 120 | 120 |
| **total** | **1.200** | **600** | **600** |

O dataset tem 200 `case_id` únicos, seis formulações por caso e nenhuma sobreposição de caso entre splits. A validação também bloqueia textos duplicados. As seis formulações superficiais podem reaparecer em casos diferentes; logo, o piloto mede holdout por caso/valores, não holdout completo de template ou de família semântica.

As chamadas positivas cobrem dez casos por ferramenta; as abstenções cobrem dez casos por família: alvo ausente, ação fora do catálogo, domínio externo, referência ambígua, limite de segurança, falta de desambiguação, múltiplas ações, pedido contraditório, valor numérico ausente e estado não suportado.

O corpus é sintético e serve para validar código, schema, parser e treinamento piloto. Não deve ser apresentado como cobertura linguística humana.

## 5. Métricas e estatística

`scripts/fc_eval.py` calcula, por conjunto e por ferramenta:

- taxa de saída parseável e taxa JSON válida;
- validade canônica;
- exact match da chamada completa;
- acurácia de ação e seleção de ferramenta;
- exact match de argumentos, inclusive condicionado à ferramenta correta;
- precisão, revocação e F1 de abstenção;
- IC95% de Wilson para proporções e bootstrap determinístico para F1 de abstenção.

`scripts/compare_fc_predictions.py` alinha duas predições por `id`, calcula IC95% e o teste de McNemar exato bilateral para comparações binárias pareadas. A latência no desktop é extraída das predições, mas não é inferida como latência Android.

## 6. Comandos reprodutíveis no Neuromancer

Na raiz do projeto:

```bash
python3 scripts/generate_fc_dataset.py --output data/generated/fc_dataset.jsonl --seed 20260830
python3 scripts/validate_fc_dataset.py data/generated/fc_dataset.jsonl --expected-total 1200
python3 -m unittest discover -s tests -p 'test_*.py'
```

Para avaliar uma saída de modelo em JSONL:

```bash
python3 scripts/fc_eval.py \
  --dataset data/generated/fc_dataset.jsonl \
  --predictions results/qwen35_fc_test.predictions.jsonl \
  --registry data/tools/android_tools.json \
  --split test \
  --output results/qwen35_fc_test.metrics.json
```

Para uma comparação pareada:

```bash
python3 scripts/compare_fc_predictions.py \
  --dataset data/generated/fc_dataset.jsonl \
  --predictions-a results/base.predictions.jsonl \
  --predictions-b results/lora.predictions.jsonl \
  --registry data/tools/android_tools.json \
  --split test \
  --output results/mcnemar.json
```

Cada linha de predição deve conter `id` e `prediction` como objeto canônico, ou `id` e `raw` como string. O parser também reconhece o formato funcional observado no Qwen (`<function=...><parameter=...>`), mas essa conversão é contabilizada como não-JSON.

## 7. Integração de inferência sem celular

Três adapters LoRA foram treinados no mesmo dataset, com seeds `20260830`, `20260831` e `20260832`, e avaliados no teste corrigido. O exemplar usado para o smoke foi exportado para `results/qwen35_2b_lora_fc_seed20260830.gguf` em F16. O arquivo tem 33.664.736 bytes e SHA-256 `c9b497df1bb35c98d2125664abb01f243622412dddf38875fb2fde84a4e93a29`.

O wrapper `scripts/convert_qwen35_lora_gguf.py` corrige, em uma visão temporária somente de configuração, o campo `text_config.architectures=null` do checkpoint Qwen3.5; o cache original não é alterado.

O carregamento conjunto da quantização Q4_K_M do modelo base com o adapter foi verificado no `llama.cpp` commit `a66d50588`, com CUDA no RTX 5090. `scripts/run_llama_fc_smoke.py` confirmou uma chamada e uma abstenção, ambas com JSON válido e sem erros do contrato. O smoke é uma verificação de integração, não substitui a avaliação em lote nem o teste no aparelho.

Exemplo:

```bash
python3 scripts/convert_qwen35_lora_gguf.py \
  --base /caminho/para/Qwen3.5-2B \
  --lora results/qwen35_2b_lora_fc_seed20260830 \
  --outfile results/qwen35_2b_lora_fc_seed20260830.gguf \
  --outtype f16

PYTHONPATH=scripts python3 scripts/run_llama_fc_smoke.py \
  --base-hf /caminho/para/Qwen3.5-2B \
  --base-gguf /caminho/para/qwen35-2b-q4_k_m.gguf \
  --lora results/qwen35_2b_lora_fc_seed20260830.gguf \
  --record-id pos_bluetooth_set_state_08_00 \
  --output results/qwen35_2b_lora_fc_seed20260830_llama_smoke.json
```

O GGUF e os pesos dos adapters continuam ignorados pelo Git; hashes, seeds e manifestos acompanham o registro experimental.

## 8. Próximas atividades sem celular

1. Auditar manualmente uma amostra e complementar exemplos informais, ruidosos, composicionais e de famílias semânticas disjuntas.
2. Avaliar um conjunto humano/independente congelado antes do treinamento principal.
3. Implementar no app Android o parser, a validação duplicada, a camada de segurança e o despachante das dez ferramentas.
4. Compilar o APK e testar o caminho local no servidor; só instalar no aparelho quando o build estiver pronto e a bateria carregada.
5. Medir no A54 TTFT, tokens/s, latência fim a fim, RSS, temperatura e energia com logs rastreáveis.
6. Comparar LFM2.5/Octopus somente após resolver checkpoint, licença, parser e contrato equivalente.

## 9. Gates e limites

- Nenhuma métrica antiga de ENEM/chat/PPL será reutilizada como resultado de FC.
- Nenhum número de aparelho será publicado sem logs, configuração, versão do APK, modelo, quantização e dispositivo rastreáveis.
- O treinamento não prova capacidade de execução Android; apenas o benchmark de comandos e a validação no aparelho sustentam esse claim.
- A licença do checkpoint precisa ser registrada antes de distribuir modelo, adapter ou resultados derivados.
