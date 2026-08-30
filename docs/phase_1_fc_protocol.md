# Fase 1 — Contrato, dataset e harness de function calling

**Projeto:** SLM especializada em comandos Android em Português Brasileiro  
**Branch:** `codex/phase1-fc`  
**Ambiente de trabalho:** Neuromancer / RTX 5090  
**Status:** infraestrutura, piloto LoRA e smoke test GGUF implementados; benchmark Android ainda pendente.

## 1. Objetivo desta fase

Esta fase transforma a hipótese do projeto em um protocolo executável antes de qualquer claim de desempenho. O resultado é um contrato canônico, um catálogo versionado de ferramentas, um dataset sintético de desenvolvimento e um avaliador independente do modelo.

O celular não é necessário para gerar, validar ou avaliar o dataset, nem para executar o primeiro piloto no 5090. Ele será necessário somente para validar empacotamento, execução local e métricas de aparelho.

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

Dados móveis foi deliberadamente deixado fora da primeira versão para manter dez ferramentas e reduzir a mistura entre ações de configuração altamente restritas. Ele pode entrar em uma extensão posterior, desde que o despachante Android tenha uma rota compatível e a inclusão seja mantida igual para todos os modelos.

O catálogo é um contrato de avaliação, não uma autorização Android. Ações classificadas como `settings_panel` podem exigir interação do usuário ou permissões especiais em versões recentes do Android. O despachante deverá validar os argumentos novamente antes de executar qualquer ação.

O Octopus-V2-2B e outros modelos com tokens funcionais podem exigir um adaptador para este contrato JSON. Não serão comparados como equivalentes até que o parser e a política de argumentos estejam testados.

## 4. Dataset de desenvolvimento

O gerador `scripts/generate_fc_dataset.py` cria, com seed fixa `20260830`:

| Split | Registros | Chamadas | Abstenções |
|---|---:|---:|---:|
| train | 480 | 240 | 240 |
| dev | 120 | 60 | 60 |
| test | 120 | 60 | 60 |
| **total** | **720** | **360** | **360** |

As chamadas positivas cobrem seis cenários por ferramenta e seis formulações em PT-BR. As abstenções cobrem alvo ausente, ação fora do catálogo, domínio externo, referência ambígua, limite de segurança e falta de desambiguação.

O split reserva quatro variações textuais para treino, uma para desenvolvimento e uma para teste, mantendo os cenários balanceados em cada split. Isso mede generalização para formulações não vistas, mas não substitui uma avaliação humana independente nem uma divisão por intenção semântica. O corpus é sintético e serve inicialmente para validar código, schema, parser e treinamento piloto; não deve ser apresentado como evidência de cobertura linguística humana.

## 5. Métricas

`scripts/fc_eval.py` calcula, por conjunto e por ferramenta:

- taxa de saída parseável e taxa JSON válida;
- validade canônica;
- exact match da chamada completa;
- acurácia de ação e seleção de ferramenta;
- exact match de argumentos, inclusive condicionado à ferramenta correta;
- precisão, revocação e F1 de abstenção.

Latência, TTFT, tokens/s, RSS, temperatura e bateria não entram neste harness de desktop. Esses números exigem um protocolo separado no aparelho Android e não podem ser inferidos a partir do 5090.

## 6. Comandos reprodutíveis no Neuromancer

Na raiz do projeto:

```bash
python3 scripts/generate_fc_dataset.py
python3 scripts/validate_fc_dataset.py data/generated/fc_dataset.jsonl --expected-total 720
python3 -m unittest discover -s tests -p 'test_*.py'
```

Para avaliar uma saída de modelo em JSONL:

```bash
python3 scripts/fc_eval.py \
  --dataset data/generated/fc_dataset.jsonl \
  --predictions results/qwen35_fc_test.predictions.jsonl \
  --output results/qwen35_fc_test.metrics.json
```

Cada linha de predição deve conter `id` e `prediction` como objeto canônico, ou `id` e `raw` como string. O parser também reconhece o formato funcional observado no Qwen (`<function=...><parameter=...>`), mas essa conversão é contabilizada como não-JSON.

## 7. Integração de inferência sem celular

O adapter LoRA foi exportado para `results/qwen35_2b_lora_fc.gguf` com `scripts/convert_qwen35_lora_gguf.py`. O wrapper corrige, em uma visão temporária somente de configuração, o campo `text_config.architectures=null` do checkpoint Qwen3.5; o cache original não é alterado. O arquivo tem 33.664.704 bytes e SHA-256 `cc1b794d20220c9267a92cfac7b173e15e776fbac5ce01b6a5993ac9fc1c2ca6`.

O carregamento conjunto da quantização Q4_K_M do modelo base com o adapter foi verificado no `llama.cpp` do commit `a66d50588`, com CUDA habilitado no RTX 5090. `scripts/run_llama_fc_smoke.py` reproduz a validação usando o schema canônico e confirmou uma chamada e uma abstenção, ambas com JSON válido e sem erros do contrato. O smoke test é uma verificação de integração, não substitui a avaliação de 120 exemplos nem o teste no aparelho.

Exemplo:

```bash
python3 scripts/convert_qwen35_lora_gguf.py \
  --base /caminho/para/Qwen3.5-2B \
  --lora results/qwen35_2b_lora_fc \
  --outfile results/qwen35_2b_lora_fc.gguf \
  --outtype f16

PYTHONPATH=scripts python3 scripts/run_llama_fc_smoke.py \
  --base-hf /caminho/para/Qwen3.5-2B \
  --base-gguf /caminho/para/qwen35-2b-q4_k_m.gguf \
  --lora results/qwen35_2b_lora_fc.gguf \
  --record-id pos_bluetooth_set_state_04_05 \
  --output results/qwen35_2b_lora_fc_llama_smoke.json
```

O GGUF e os pesos do adapter continuam ignorados pelo Git; seus hashes e parâmetros devem acompanhar o registro experimental quando forem transferidos para outro servidor.

## 8. Próximas atividades sem celular

1. Auditar manualmente uma amostra do dataset e complementar exemplos de negação, variações regionais, comandos incompletos e composição.
2. Avaliar um conjunto humano/independente, com separação por intenção semântica e casos fora do template.
3. Implementar no app Android o parser, a validação duplicada e o despachante de dez ferramentas, sem executar ações não autorizadas.
4. Compilar o APK e testar o caminho local no servidor; só instalar no aparelho quando a bateria estiver carregada e o build estiver pronto.
5. Medir no A54 TTFT, tokens/s, latência fim a fim, RSS, temperatura e energia com versões e logs rastreáveis.
6. Comparar Qwen3.5-2B com LFM2.5 ou Octopus somente após resolver checkpoint, licença, parser e contrato equivalente.

## 9. Gates e limites

- Nenhuma métrica antiga de ENEM, chat, PPL ou projeção de banda será reutilizada como resultado de FC.
- Nenhum número de aparelho será publicado sem logs, configuração, versão do APK, modelo, quantização e dispositivo rastreáveis.
- O treinamento não prova capacidade de execução Android; apenas o benchmark de comandos e a validação no aparelho sustentam esse claim.
- A licença do checkpoint precisa ser registrada antes de distribuir modelo, adapter ou resultados derivados.
