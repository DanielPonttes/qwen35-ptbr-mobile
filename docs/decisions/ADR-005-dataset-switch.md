# ADR-005 — Corpus humano principal em inglês

## Decision

Usar o Fluent Speech Commands como corpus humano principal da rodada
experimental, em uma derivação Android-aligned conservadora. Manter o piloto
sintético PT-BR anterior como histórico de engenharia, sem apresentá-lo como
avaliação humana.

## Context

Os revisores apontaram que não havia um conjunto humano simultaneamente
PT-BR, independente, speaker-disjoint, Android-relevante e anotado como
chamada de função. O LaPSMail fornece fala humana PT-BR, mas seu domínio é
e-mail e suas frases são fixas. O FSC fornece áudio humano, transcrições,
rótulos nativos e split por falante, mas seu domínio é smart-home. A solução
adotada é reduzir o claim e usar somente mapeamentos semânticos diretos,
marcando o restante como abstenção derivada.

## Evidence

- arquivo de origem: data/external/fluent_speech_commands_manifest.json;
- archive SHA-256:
  c9fd67f2efa078daa84daddcad2de937eb96581c140e3131ed8cd06fbae9ba1b;
- fonte oficial:
  https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/;
- licença registrada: CC BY-NC-ND 4.0, pesquisa acadêmica;
- derivação e política: scripts/prepare_fsc_android_fc.py;
- contrato: data/tools/android_tools_fsc.json;
- saída equilibrada:
  14.956 registros, sendo 7.478 chamadas e 7.478 abstenções;
- os JSONL derivados permanecem privados no servidor e são ignorados pelo Git;
- interseção de falantes no split oficial: zero;
- interseção de frases no split oficial: alta e explicitamente registrada;
- split alternativo por frase: fluent_speech_commands_android_fc_phrase_disjoint.jsonl.

## Reviewer positions

### Turing

Aprovar com condições: usar o FSC para transferência/robustez, não como prova
de comandos Android PT-BR; exigir teste independente, métricas semânticas,
controle de leakage e separar ASR de function calling.

### Hegel

Aprovar com condições: a contribuição deve ser o protocolo reproduzível de
adaptação e avaliação, não um novo modelo. Exigir delimitação da cadeia
transcrição → função e não misturar datasets de domínios diferentes.

### Wegener

Não concluiu a rodada antes do encerramento operacional. Seus pontos
adversariais já foram incorporados a partir da especificação do protocolo:
leakage, incompatibilidade semântica, licenciamento, labels derivados e
claims de on-device.

## Disagreement

O FSC poderia ser usado como benchmark Android diretamente somente com
anotações adicionais. Para não transformar rótulos smart-home em uma falsa
verdade de Android, a implementação restringe a quatro mapeamentos e conserva
os demais registros como abstain de política.

## Final Decision

Prosseguir com o FSC como evidência humana em inglês para uma tarefa de
escopo restrito. O artigo não deve alegar especialização PT-BR humana nem
speech-to-function end-to-end nesta fase.

## Why

O FSC é o recurso humano disponível que combina áudio, transcrições, labels e
falantes novos. Sua divisão oficial permite um teste de generalização a
falantes; a derivação lexical permite medir separadamente o efeito de frases
não vistas. O contrato restrito torna explícita a distância semântica restante.

## Risks

- os labels Android são derivados, não nativos;
- os exemplos negativos dependem de uma política de abstenção;
- a licença impede usos comerciais e exige cuidado com artefatos derivados;
- o modelo recebe texto, não áudio;
- a execução física Android ainda não foi medida.

## Conditions for Reconsideration

Reconsiderar o corpus principal se for obtido um conjunto humano Android
independente com consentimento, falantes novos, paráfrases espontâneas e
anotação auditada de schema. Reconsiderar o claim on-device somente após
execução e medição em aparelho físico.
