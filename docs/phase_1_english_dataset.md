# Fase 1 — Corpus humano em inglês e contrato Android restrito

**Data:** 2026-08-30
**Branch:** codex/phase1-fc
**Ambiente de treino:** Neuromancer, NVIDIA GeForce RTX 5090 32 GB
**Modelo:** Qwen3.5-2B
**Entrada do modelo:** transcrição textual; o modelo não recebe áudio

## Decisão metodológica

O corpus humano principal desta rodada é o **Fluent Speech Commands (FSC)**, em
inglês norte-americano. Ele contém 30.043 gravações humanas, 97 falantes, 31
intenções e divisão oficial por falante. A licença registrada na fonte é
CC BY-NC-ND 4.0, para pesquisa acadêmica; o corpus não pode ser usado para
produto comercial ou redistribuição de um derivado fora dos termos da licença.

O FSC não é um corpus Android. Para não inventar anotações, foi criada uma
derivação conservadora em
scripts/prepare_fsc_android_fc.py. Somente quatro rótulos nativos são
considerados semanticamente compatíveis:

| Rótulo FSC | Chamada derivada | Justificativa |
|---|---|---|
| activate / music / none | media_control(action=play) | iniciar a mídia ativa |
| deactivate / music / none | media_control(action=pause) | pausar a mídia ativa |
| increase / volume / none | volume_adjust(direction=up) | ajustar volume de mídia em um passo |
| decrease / volume / none | volume_adjust(direction=down) | ajustar volume de mídia em um passo |

Os 27 rótulos restantes são mantidos apenas como exemplos de
abstain/fora do escopo. Essa abstenção é uma **política derivada do
projeto**, não uma anotação Android original do FSC.

## Artefatos

- contrato restrito: data/tools/android_tools_fsc.json;
- derivação: scripts/prepare_fsc_android_fc.py;
- corpus oficial equilibrado: data/external/fluent_speech_commands_android_fc.jsonl;
- corpus com frases disjuntas: data/external/fluent_speech_commands_android_fc_phrase_disjoint.jsonl;
- auditoria e hashes: data/external/fluent_speech_commands_android_fc_manifest.json;
- validador: scripts/validate_fc_dataset.py.

Os dois JSONL derivados são artefatos privados de execução no servidor e estão
ignorados pelo Git. Eles não serão redistribuídos com o repositório: a licença
CC BY-NC-ND do FSC exige cautela adicional para qualquer transformação que
contenha transcrições ou áudio.

Cada derivação possui 14.956 registros equilibrados: 7.478 chamadas
compatíveis e 7.478 abstenções. O conjunto oficial preserva a divisão do FSC:
77 falantes no treino, 10 na validação e 10 no teste, sem interseção de
falantes. Como o FSC reutiliza as mesmas frases entre falantes, essa divisão
mede generalização a falantes, não a paráfrases.

A segunda derivação atribui grupos de frase a train/dev/test de modo
disjunto, preservando a mesma proporção de chamadas e abstenções. Ela mede
generalização lexical/por frase; falantes podem aparecer em mais de um split
por construção e essa propriedade é registrada no manifesto.

## Protocolo

1. Treinar um adapter LoRA usando somente o train da derivação selecionada.
2. Selecionar hiperparâmetros e checkpoint usando somente dev.
3. Congelar o teste antes de comparar os sistemas.
4. Executar o baseline Qwen3.5-2B sem adapter com o mesmo catálogo, prompt e
   parser.
5. Reportar separadamente:
   - exact match da chamada completa;
   - validade JSON e do contrato;
   - seleção da ferramenta;
   - acerto dos argumentos;
   - precisão, revocação e F1 de abstenção;
   - resultados por regra de mapeamento e por split.
6. Repetir em três seeds quando o custo permitir e agregar por seed, sem
   escolher a melhor seed olhando o teste.

O scripts/fc_eval.py avalia o contrato de forma independente da geração e
aceita saídas JSON ou a marcação funcional do Qwen. Transcrições repetidas são
permitidas, mas a derivação registra template_id para a auditoria de
leakage. A validação oficial exige speaker_id disjunto entre splits; a
validação lexical exige template_id disjunto.

## Escopo das conclusões

Esta rodada pode sustentar apenas conclusões sobre **transcrição em inglês
humana → chamada estruturada em um contrato Android restrito**. Ela não prova:

- compreensão de fala ponta a ponta;
- desempenho em português brasileiro;
- cobertura geral de comandos Android;
- equivalência entre smart-home e Android;
- execução real no aparelho;
- segurança operacional além da validade do contrato.

O LaPSMail, o OpenVoiceOS e o SLURP permanecem documentados como recursos
auxiliares ou históricos. SLURP não é misturado ao FSC nesta rodada; seu split
sintético continua separado.

## Próximos gates

Antes de declarar uma aplicação Android funcional, ainda são necessários:

- parser/validador/safety layer no APK;
- conversão e quantização do checkpoint com rastreabilidade;
- teste em aparelho físico, com memória, latência, temperatura e energia;
- verificação das APIs e permissões Android na versão do aparelho;
- análise de licença antes de publicar qualquer adapter ou checkpoint.

O celular não é necessário para a preparação, treinamento ou avaliação textual
desta fase.

## Resultados executados

Os quatro experimentos planejados foram concluídos: baseline zero-shot e LoRA
no split oficial, e baseline zero-shot e LoRA no split phrase-disjoint. Os
resultados detalhados, intervalos, latências e hashes estão em
`docs/phase_1_english_results.md`.

| Protocolo / modelo | Exact match | Validade canônica | F1 de abstention |
|---|---:|---:|---:|
| Oficial / zero-shot | 39,97% | 68,77% | 61,08% |
| Oficial / LoRA | 100,00% | 100,00% | 100,00% |
| Phrase-disjoint / zero-shot | 40,33% | 74,09% | 58,97% |
| Phrase-disjoint / LoRA | 100,00% | 100,00% | 100,00% |

O resultado perfeito do LoRA é válido para o contrato e as pontes declaradas,
mas não é uma prova de compreensão geral do Android. A entrada continuou
sendo texto transcrito e os alvos de abstention fora das quatro regras foram
gerados por política do experimento.
