# Detecção e Reparo Automatizado de Resource Leaks no Ecossistema Android usando LLMs e Inteligência Artificial

Este repositório contém o arcabouço experimental da dissertação de Mestrado focada na avaliação, detecção e reparo automatizado de falhas de vazamento de recursos (*Resource Leaks*) em aplicativos Android. 

O projeto implementa uma abordagem comparativa dividida em uma **Tríade Metodológica** (Léxica, Estrutural e Semântica) para treinar classificadores ("Juízes") capazes de auditar de forma cega e automatizada a eficácia de grandes modelos de linguagem (LLMs) em um torneio de reparo de código baseado no dataset **DroidLeaks**.

---

## 🧬 A Tríade Metodológica

A validação dos reparos gerados pelas LLMs é sustentada por três representações distintas de inteligência:

1. **Abordagem Léxica (Texto):** Representação por frequência de termos via **TF-IDF** combinada com o classificador *Random Forest*. Serve como a linha de base (*Baseline*) do projeto.
2. **Abordagem Estrutural (Regras):** Extração e conversão da Árvore de Sintaxe Abstrata (**AST**) via parser compilador `javalang`. Mapeia o esqueleto lógico da gerência de recursos (blocos try-catch-finally e métodos de ciclo de vida Android como `release()`, `recycle()`, `close()`).
3. **Abordagem Semântica/Neural (Contexto):** Vetorização profunda de código-fonte usando o modelo estado da arte **GraphCodeBERT** (Microsoft), capturando o fluxo de dados (*Data Flow Graph*) em vetores densos de 768 dimensões (Token `[CLS]`).

---

## 📂 Estrutura do Repositório

```text
Meu_Mestrado_DroidLeaks/
│
├── data_bases/
│   ├── 04_final/
│   │   ├── dataset_sintetico_chatgpt.csv        # Dataset bruto expandido (2014 linhas)
│   │   ├── treino_validador_final.csv           # Divisão de treino oficial (1414 linhas)
│   │   ├── holdout_300_bugs_llm.csv             # 300 bugs isolados (Caderno de provas)
│   │   └── holdout_ast.csv                      # IDs dos bugs sobreviventes da AST
│   │
│   ├── ast/
│   │   ├── dataset_ast_completo.csv             # Features da AST balanceadas (1926 linhas)
│   │   ├── dataset_ast_treino.csv               # Treino da AST alinhado
│   │   ├── dataset_ast_teste.csv                # Teste da AST alinhado
│   │   └── respostas_llms_ast.csv               # Features AST extraídas das respostas das IAs
│   │
│   ├── graphcodebert/
│   │   └── embeddings_graphcodebert.csv         # Matriz densa de 768 dimensões dos códigos
│   │
│   └── 05_results/
│       └── resultados_torneio_llms_FILTRADO_AST.csv  # Checklist de chamadas de API alinhadas
│
├── models/
│   ├── tfidf/
│   │   ├── vectorizer.pkl                       # Vetorizador TF-IDF treinado
│   │   └── validador_resource_leak.pkl          # Classificador RF (TF-IDF)
│   ├── ast/
│   │   ├── vectorizer_ast.pkl                   # Vetorizador de vocabulário da AST
│   │   └── validador_ast_rf.pkl                 # Classificador RF (AST)
│   └── graphcodebert/
│       └── classificador_neural_gcbert.pkl      # Classificador Semântico
│
└── scripts/
    ├── 01_extrair_casos.py                      # Mineração do DroidLeaks original
    ├── 02_gerar_diffs.py                        # Isola a modificação do commit (Bug -> Fix)
    ├── 03_consolidar_dataset.py                 # Estrutura a tabela inicial de engenharia
    ├── 04_balancear_classes.py                  # Garante a proporção exata de 50% Bug / 50% Fix
    ├── 05_aumento_dados_openai.py               # Geração sintética via GPT para expansão da base
    ├── 05_b_auditor_sintaxe_javalang.py         # Diagnóstico e log de falhas gramaticais da LLM
    ├── 06_dividir_holdout.py                    # Separação randômica inicial de Treino/Teste (70/30)
    │
    ├── 07_a_treinar_validador_tfidf.py          # Treinamento da Baseline Léxica
    ├── 07_b_treinar_validador_ast.py            # Extração de features sintáticas e treino RF AST
    ├── 07_c_extrair_embeddings_gcbert.py        # Conversor Neural de código para 768 dimensões
    │
    ├── 08_a_orquestrar_torneio_llms.py          # Envio automático e tolerante a falhas para as APIs
    ├── 08_b_alinhar_split_ast.py                # Sincronizador de IDs para evitar vazamento de dados
    ├── 08_c_mesclar_resultados_ast.py           # Auditor de histórico para evitar re-chamadas pagas
    ├── 08_d_converter_respostas_ast.py          # Tradutor das correções das LLMs para texto AST
    │
    ├── 09_a_oraculo_torneio_tfidf.py            # Apuração de notas e placar final via TF-IDF
    ├── 09_b_oraculo_torneio_ast.py              # Apuração e geração de gráfico em 300 DPI via AST
    └── 09_c_oraculo_torneio_gcbert.py           # Apuração de notas e placar via Deep Learning
