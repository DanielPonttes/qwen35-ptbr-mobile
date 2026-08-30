# Fase 0 — Auditoria de modelos e desenho do primeiro artigo

**Projeto:** SLM especializada em comandos Android em Português Brasileiro  
**Evento-alvo:** [ERAMIA-RS 2026](https://eramia-rs.sbc.org.br/2026/)  
**Data da auditoria:** 2026-08-30  
**Status:** Fase 0 concluída; implementação experimental ainda não iniciada.

## 1. Escopo e distinção entre fontes de instrução

Esta auditoria separa explicitamente três camadas:

1. **Solicitação do pesquisador:** escrever um artigo para o ERAMIA-RS 2026, utilizar a RTX 5090 configurada no Neuromancer e interromper apenas diante de um impeditivo de implementação ou de uma decisão crítica.
2. **Documento de projeto:** define a hipótese, a necessidade de uma auditoria de modelos e o artefato desta fase (`docs/phase_0_model_audit.md`). Ele é usado como protocolo de trabalho, não como substituto da solicitação do pesquisador.
3. **Protocolo de discussão adversarial:** define papéis de revisores e perguntas de red team. Ele orienta a revisão do desenho, mas as respostas dos revisores não são evidência experimental do artigo.

Consequência prática: números, claims e conclusões dos documentos anexos ou do projeto remoto não são transferidos automaticamente para o manuscrito. Só serão usados após rastreamento até checkpoint, código, entrada, configuração e medição reproduzível.

## 2. Restrições editoriais relevantes

O site oficial do evento informa artigos de até quatro páginas, incluindo referências, no formato SBC, em português ou inglês; a submissão é cega e o evento não aceita artigos de revisão, survey, mapeamento ou revisão sistemática. O artigo deve, portanto, ser uma avaliação experimental estreita de um sistema, e não um catálogo narrativo de modelos.

O desenho proposto nesta auditoria é compatível com essa restrição:

> **Pergunta:** qual combinação de SLM, quantização e decodificação estruturada maximiza a execução correta de comandos Android em PT-BR sob restrição de dispositivo?

O título de trabalho é **“Comandos Android Offline em Português Brasileiro: Avaliação Reprodutível de Function Calling em SLMs Quantizadas”**.

## 3. Inventário do Neuromancer

### 3.1 Infraestrutura confirmada

- Host SSH: `neuromancer` (`neuromancer.inf.ufpel.edu.br`, usuário `daniel`).
- GPU confirmada por `nvidia-smi`: **NVIDIA GeForce RTX 5090**, 32.607 MiB, driver 595.84.
- `llama.cpp` presente em `/home/daniel/llama.cpp`.
- Projeto remoto existente em `/home/daniel/qwen35-ptbr-mobile`.
- Ambiente Python com CUDA funcional em `/home/daniel/Área de trabalho/swarm-emotions-tag/python-ml/.venv`: PyTorch 2.10.0+cu128, CUDA 12.8 e `torch.cuda.is_available() == True`.
- Checkpoint Hugging Face do `Qwen/Qwen3.5-2B` disponível localmente no cache do Neuromancer.
- Ollama em contêineres e modelos Gemma disponíveis, mas nenhum deles foi tratado como evidência para o artigo.

### 3.2 Smoke test executado no 5090

O checkpoint local do Qwen3.5-2B foi carregado com Transformers no 5090. Com uma ferramenta `toggle_wifi(enabled: boolean)` e o comando “Ative o Wi-Fi.”, a saída foi:

```text
<tool_call>
<function=toggle_wifi>
<parameter=enabled>
True
</parameter>
</function>
</tool_call>
```

Isso confirma somente que o checkpoint, o tokenizer e o caminho de inferência funcionam no Neuromancer e que o formato de ferramenta do Qwen pode ser acionado. Não é resultado do artigo: foi uma única entrada, sem conjunto de teste, sem comparação, sem quantização e sem Android.

### 3.3 Impeditivos técnicos identificados

- O Neuromancer não possui `adb` nem uma rota USB direta. O aparelho agora está conectado ao computador local e autorizado pelo ADB oficial temporário: Samsung `SM-A546E`, Android 16, `arm64-v8a`. A bateria está em 5%; por isso as medições de TTFT, tok/s, RSS, temperatura e bateria continuam adiadas até a carga adequada.
- O binário x86 de `llama.cpp` compilado com CUDA não encontra `libcudart.so.12` no caminho dinâmico do sistema. A inferência PyTorch no 5090 funciona, mas o runtime x86 do `llama.cpp` precisa ser corrigido ou substituído antes de ser usado em benchmarks de servidor.
- O ambiente Python usado no smoke test não possui `accelerate` nem `peft`. LoRA/QLoRA exigirá um ambiente reprodutível com essas dependências; não instalei pacotes nem iniciei treinamento nesta fase.
- Os checkpoints dos candidatos LFM2.5, SmolLM3, Granite e xLAM não estão no cache Hugging Face observado. O download, a licença para uso e o espaço em disco precisam ser resolvidos antes da comparação.

## 4. Auditoria do projeto reaproveitado

O projeto `/home/daniel/qwen35-ptbr-mobile` é valioso como ponto de partida de engenharia, mas não como resultado pronto para o artigo proposto.

| Item | O que existe | Decisão |
|---|---|---|
| Modelo | Qwen3.5 0.8B ajustado para chat PT-BR | Reaproveitar código e experiência; não usar como evidência de FC sem novo teste |
| Treino | 2.248 conversas sintéticas, full fine-tuning BF16 | Não usar como dataset de ferramentas; projetar dataset FC separado |
| Android | Serviço `llama-server` e interface de chat | Adaptar somente depois de definir contrato de ferramentas |
| Benchmark | ENEM, PPL, `llama-bench` e projeções de banda | Não é proxy aceitável para seleção de ferramenta ou argumentos |
| Quantização | Q3/Q4/Q5/Q6/Q8 e IQ2 em artefatos existentes | Reavaliar nas métricas de FC; não transferir o “cliff” do ENEM |
| Velocidade | Um ponto A54 declarado como medido e outras linhas projetadas | Não reportar como nova medição sem logs e dispositivo disponíveis |
| Reprodutibilidade | Há divergências entre `PAPER.md`, `paper/paper.tex` e JSONs | Tratar números antigos como histórico não validado |

As revisões adversariais encontraram, em particular, inconsistências entre tamanhos/BPW, quantidade de questões, número de execuções, valores medidos e projetados, além da ausência de schema, parser, grammar e dataset de function calling no código atual. Esses problemas impedem a reutilização automática dos resultados.

## 5. Candidatos auditados

Os links abaixo são as fontes primárias de documentação dos modelos. A coluna de “evidência PT-BR” descreve apenas o que está declarado na documentação; não representa desempenho medido neste projeto.

| Candidato | Escala/licença declarada | PT-BR | Function calling | Rota de implantação | Papel provisório |
|---|---|---|---|---|---|
| [Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) / [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | 2B; Apache 2.0; arquitetura híbrida texto/visão | O card declara 201 línguas/dialetos | Tokenizer e card documentam formato de tools; smoke test no Neuromancer/5090 passou | Transformers no 5090; `llama.cpp`/Android ainda precisam de teste específico | **Principal provisório**, condicionado ao runtime Android e ao teste PT-BR |
| [LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) | 1,17B; licença LFM 1.0 | PT não aparece na lista linguística declarada | Function calling é documentado; há [GGUF](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF) e [ONNX](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-ONNX) | Forte candidato para edge; Android ainda exige smoke test | **Baseline especializado**, com revisão jurídica da licença |
| [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base) / [GGUF](https://huggingface.co/ggml-org/SmolLM3-3B-GGUF) | 3B; Apache 2.0 | Português está entre as línguas declaradas | Card documenta tool calling e integrações com `llama.cpp` | Boa rota GGUF; footprint maior | Baseline multilíngue, se o orçamento de memória permitir |
| [Granite 3.3-2B-Instruct](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct) | 2B; Apache 2.0 | Português declarado | Tarefas de function calling declaradas | Precisa verificar GGUF e Android no mesmo harness | Alternativa generalista |
| [Granite 4.0 H-Micro](https://huggingface.co/ibm-granite/granite-4.0-h-micro) | 3B; Apache 2.0; híbrido Attention/Mamba2 | Português declarado | Schema de tools documentado | A arquitetura híbrida pode complicar PEFT e runtime móvel; há [documentação IBM](https://www.ibm.com/granite/docs/models/granite4-0) | Candidato exploratório, não principal |
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | 1,5B; Apache 2.0 | Usar como controle generalista; medir PT-BR | Template genérico; capacidade de FC deve ser medida no harness | Simples e maduro no `llama.cpp` | **Baseline generalista** |
| [Octopus-V2-2B](https://huggingface.co/NexaAI/Octopus-v2) | 2B; baseado em Gemma-2B; CC BY-NC-4.0 | PT-BR não demonstrado na documentação consultada | Especializado em chamadas Android; usa tokens funcionais e suporta chamadas individuais, aninhadas e paralelas | Não é JSON-native: exige adaptador, validação de argumentos e teste de compatibilidade com o despachante | Baseline FC específico, condicionado à licença não comercial |
| [xLAM-1B-fc-r](https://huggingface.co/Salesforce/xLAM-1b-fc-r) | 1,35B; licença CC-BY-NC-4.0 e termos associados | Não há garantia PT-BR no card | Especializado em function calling | Avaliar somente após verificar licença e formato | Baseline especializado de pesquisa, condicionado à licença |
| [FunctionGemma](https://ai.google.dev/gemma/docs/mobile-actions?hl=en) | 270M; fora da faixa 1–3B; termos Gemma | Não é comparação de escala | Treinado para ações/function calling móvel | [LiteRT-LM](https://developers.google.com/edge/litert-lm/overview) é a rota Android de referência | Referência mobile, não baseline pareado |

### 5.1 Recomendação de shortlist

Para caber no limite de quatro páginas, a primeira rodada deve conter no máximo três modelos:

1. **Qwen3.5-2B-Instruct ou Base ajustado com LoRA/QLoRA**, como candidato principal. A variante exata será congelada após o teste de template e licença.
2. **LFM2.5-1.2B-Instruct**, como candidato especializado e de menor footprint, condicionado à licença LFM 1.0.
3. **Qwen2.5-1.5B-Instruct** como baseline generalista simples.

SmolLM3-3B pode substituir o Qwen2.5 se a prioridade for representação declarada de português e uma comparação maior. Octopus-V2-2B pode substituir o baseline generalista somente se o artigo aceitar a licença não comercial e implementar o adaptador de tokens funcionais; Granite, xLAM e FunctionGemma ficam como alternativas ou referências até que runtime, licença e orçamento de páginas estejam resolvidos.

## 6. Protocolo experimental mínimo

### 6.1 Dataset

Criar um conjunto separado de comandos Android em PT-BR, com 8–15 ferramentas, por exemplo `open_app`, `set_alarm`, `create_calendar_event`, `send_message`, `toggle_wifi`, `set_volume`, `read_battery` e `create_reminder`. Cada ferramenta terá um JSON Schema versionado.

O conjunto deve conter comandos diretos, paráfrases coloquiais, omissão de parâmetros, múltiplos parâmetros, nomes/acentos, comandos fora de escopo e pedidos que exigem abstenção. O split deve separar templates/paráfrases e, quando possível, geradores de exemplos; dividir apenas por linha favorece vazamento.

### 6.2 Condições

- Mesmo catálogo semântico de ferramentas e mesmos distratores para todos os modelos.
- Template de conversa nativo de cada modelo, versionado no repositório; diferenças de template devem ser reportadas.
- Decodificação livre versus grammar/schema estrito, sem reparador de JSON na métrica primária.
- FP16/BF16 de referência e Q4_K_M como condição principal; Q8_0 e Q3_K_M apenas se o tempo permitir.
- `temperature=0` para a métrica determinística principal e sementes fixadas para análises estocásticas.

### 6.3 Métricas

1. Seleção de ferramenta: acurácia e precisão/recall/F1.
2. Argumentos: F1 por chave/valor após normalização previamente especificada.
3. JSON estrito: taxa de saída aceita por parser e pelo JSON Schema, sem correção posterior.
4. Abstenção: precisão, recall e taxa de falsos acionamentos em pedidos fora de escopo.
5. Execução: taxa de comandos que passam pelo validador e podem ser encaminhados ao adaptador Android.
6. Eficiência no aparelho: TTFT, tok/s, pico de RSS, tempo de carregamento, temperatura e, se possível, energia.

### 6.4 Estatística e rastreabilidade

Usar os mesmos exemplos pareados entre modelos, reportar média e dispersão em pelo menos três seeds quando a condição for estocástica e aplicar teste pareado apropriado (por exemplo, McNemar para decisão binária ou bootstrap pareado para diferenças contínuas). Registrar SHA do checkpoint/GGUF, SHA do `llama.cpp`, template, grammar, prompt, configuração, seed e logs brutos.

## 7. Desenho de quatro páginas

- **Página 1:** problema, contribuição, catálogo de ferramentas e definição formal de saída válida.
- **Página 2:** dataset, split, modelos, template, grammar e protocolo.
- **Página 3:** tabela de FC (tool F1, argument F1, JSON válido, abstenção) e tabela de eficiência Android.
- **Página 4:** análise de erros, trade-off quantização/grammar, limitações e conclusão.

Cortar ENEM, PPL cruzado, matriz de 19 dispositivos e projeções de banda. Eles pertencem a outro estudo e consumiriam espaço sem responder à pergunta de function calling.

## 8. Decisões críticas e próximo gate

### Decisão A — validade do claim “em Android real”

**Estado:** parcialmente resolvido por implementação. Há um Samsung A54 real autorizado via ADB no computador local, mas ainda falta instalar uma build com function calling e executar o protocolo com bateria suficiente. Os números históricos do A54 não possuem rastreabilidade suficiente para serem reutilizados como nova evidência.

**Opções:**

- manter o aparelho conectado ao computador local, que fará a instalação, a coleta ADB e o controle do benchmark; ou
- restringir o artigo a avaliação no 5090 + validação de empacotamento/runtime, removendo claims de latência e consumo em aparelho.

### Decisão B — modelo principal

**Estado:** decisão ainda não congelada. A opção técnica mais conveniente é Qwen3.5-2B devido ao checkpoint local e ao smoke test no 5090. A opção potencialmente mais favorável à borda é LFM2.5-1.2B, mas a licença e a ausência de PT na lista declarada exigem verificação. A escolha deve ser feita por um piloto pareado, não por marketing de model card.

### Decisão C — início do treinamento

**Estado:** adiado corretamente. O dataset FC, o contrato de schema e o alvo móvel devem ser congelados antes de iniciar LoRA/QLoRA. A RTX 5090 será usada para triagem, treinamento e avaliação em lote; ela não substitui a medição Android.

## 9. Revisão adversarial registrada

Foram consultados, independentemente, os três papéis do protocolo:

- **Gemini 3.7 Flash (High), via Agy:** rejeitou o desenho atual por invalidar o constructo ao usar ENEM/chat como proxy de FC e exigiu métricas de tool selection, argumentos, JSON, abstenção e latência móvel.
- **Muse Spark 1.2, via OpenCode (`muse-spark-1.2-contributor`, variante xHigh solicitada):** confirmou que o repositório não contém FC, destacou a falta de dataset/harness e as inconsistências dos artefatos; recomendou dois ou três modelos e um aparelho real.
- **Grok 4.6, via Cursor Agent:** o identificador exato `cursor-grok-4.6-xhigh` está disponível e respondeu ao smoke test; a revisão longa com esse identificador excedeu o timeout sem saída, então o parecer registrado foi executado em `cursor-grok-4.6-xhigh-fast`. O red team encontrou o mesmo descompasso de tarefa, projeções tratadas como medições, divergências entre arquivos e ausência de grammar/schema.

Esses pareceres são controle de qualidade do desenho. Não são dados do artigo e não serão citados como resultados experimentais.

## 10. Gate da Fase 1

Antes de criar dataset ou treinar, confirmar apenas estas decisões críticas:

1. o aparelho Android real, já conectado, será usado com carga suficiente e build de function calling validada; e
2. o shortlist de três modelos e a licença de cada checkpoint estão aprovados.

Com essas duas decisões, a próxima entrega deve ser o contrato JSON Schema, o gerador/validador do dataset e o harness `fc_eval.py`; somente depois começa o LoRA/QLoRA no 5090.
