# ADR-006 — Reenquadrar o artigo como benchmark de modelos ultrapequenos

**Status:** aceito
**Data:** 31 de agosto de 2026

## Contexto

Não há dataset humano PT-BR nativo de comandos Android disponível para a primeira submissão. O Fluent Speech Commands fornece transcrições humanas em inglês, mas seus rótulos são de assistente/smart-home e não de Android. Além disso, a revisão adversarial identificou sobreposição de templates no split oficial e ausência de baselines triviais.

## Decisão

O artigo deixa de alegar um modelo Android/PT-BR e passa a ser um benchmark de roteamento estruturado de transcrições humanas para modelos com no máximo 3 bilhões de parâmetros. O contrato é genérico, com duas operações (`media_control` e `volume_adjust`) e uma abstention de política derivada. A matriz principal é zero-shot e compara cinco checkpoints de quatro famílias; LoRA não é parte da comparação principal.

O benchmark usa dois protocolos:

1. oficial speaker-disjoint, mantido como diagnóstico de transferência de falante e de leakage lexical;
2. phrase-disjoint, com `template_id` exclusivo por split, como controle principal de lexical leakage.

Todo resultado inclui `always_abstain`, controle lexical, métricas de validade estrutural, exact match e cluster bootstrap por template. Nenhum output é executado e o celular fica fora desta fase.

## Consequências

- O artigo passa a ter uma pergunta empiricamente respondível sem hardware móvel.
- O resultado pode ser negativo sem ser ambíguo: o controle lexical supera os modelos zero-shot no contrato estrito.
- FSC não pode ser descrito como Android, e as abstentions não podem ser descritas como anotações humanas de segurança.
- ASR, PT-BR, execução e medição on-device tornam-se trabalhos futuros.
- O corte exclui Qwen2.5-3B (3.085.938.688 parâmetros) apesar do nome comercial “3B”.
