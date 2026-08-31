# Resultados do benchmark ultrapequeno

Os arquivos `fsc_ultrasmall_*_base_*` e `fsc_ultrasmall_*_base_*.metrics.json` são produzidos pelo runner da matriz e devem conter somente agregados versionáveis e logs locais. As predições `.jsonl` e os adapters são derivados do FSC e permanecem server-local conforme o `.gitignore`.

Cada relatório de métrica inclui:

- validade JSON e validade canônica;
- exact match, action accuracy, seleção de ferramenta e argumentos;
- precision/recall/F1 de abstention;
- métricas por regra de derivação;
- resumo macro e strict por `template_id`;
- intervalos Wilson por item e cluster-bootstrap por template.

Os arquivos `fsc_en_qwen35_2b_*` pertencem à fase anterior, baseada em um contrato Android-alinhado, e não são resultados principais do benchmark ultrapequeno atual.
