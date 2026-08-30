# Resultados experimentais

Os arquivos de predição e métricas desta pasta foram gerados no Neuromancer com a RTX 5090 e estão vinculados ao SHA-256 do dataset registrado em `data/generated/fc_dataset.manifest.json`.

Os arquivos `*_predictions.jsonl` preservam a resposta bruta do modelo e a latência por exemplo. Os arquivos `*_metrics.json` foram produzidos por `scripts/fc_eval.py`.

O adapter LoRA em `qwen35_2b_lora_fc/` é um artefato binário local de aproximadamente 109 MB e fica fora do commit Git. O treinamento está documentado em `training_manifest.json`. A pasta `archive/case_split/` contém os resultados da divisão anterior, mantidos apenas para rastreabilidade e não usados na comparação atual.

Todos os resultados atuais são de desktop/5090 sobre dataset sintético. Ainda não são evidência de desempenho Android.
