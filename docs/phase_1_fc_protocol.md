# Fase 1 — Contrato, dataset e harness de function calling

**Projeto:** SLM especializada em comandos Android em Português Brasileiro  
**Branch:** `codex/phase1-fc`  
**Ambiente de trabalho:** Neuromancer / RTX 5090  
**Status:** infraestrutura da Fase 1 implementada; treinamento e benchmark Android ainda não iniciados.

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

## 7. Próximas atividades sem celular

1. Criar o ambiente de treinamento no Neuromancer com versões fixadas de PyTorch, Transformers, PEFT e Accelerate.
2. Auditar manualmente uma amostra do dataset e complementar exemplos de negação, variações regionais e comandos incompletos.
3. Executar baseline zero-shot no Qwen3.5-2B usando exatamente o catálogo e o prompt do protocolo.
4. Treinar um piloto LoRA/QLoRA no 5090, registrar seed, hash do dataset, hiperparâmetros e checkpoint.
5. Avaliar Qwen3.5-2B e o baseline generalista no mesmo harness; adicionar LFM2.5 ou Octopus somente após resolver checkpoint, licença e parser.
6. Converter o candidato aprovado para a rota Android e só então instalar a build no aparelho.

## 8. Gates e limites

- Nenhuma métrica antiga de ENEM, chat, PPL ou projeção de banda será reutilizada como resultado de FC.
- Nenhum número de aparelho será publicado sem logs, configuração, versão do APK, modelo, quantização e dispositivo rastreáveis.
- O treinamento não prova capacidade de execução Android; apenas o benchmark de comandos e a validação no aparelho sustentam esse claim.
- A licença do checkpoint precisa ser registrada antes de distribuir modelo, adapter ou resultados derivados.
