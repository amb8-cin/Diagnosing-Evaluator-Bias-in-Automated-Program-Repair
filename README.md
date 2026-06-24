# Detecção e Reparo Automatizado de Resource Leaks no Ecossistema Android usando LLMs e Inteligência Artificial

Este repositório contém o arcabouço experimental da dissertação de Mestrado focada na avaliação, detecção e reparo automatizado de falhas de vazamento de recursos (*Resource Leaks*) em aplicativos Android. 

O projeto implementa uma abordagem comparativa dividida em uma **Tríade Metodológica** (Léxica, Estrutural e Semântica) para treinar classificadores ("Juízes") capazes de auditar de forma cega e automatizada a eficácia de grandes modelos de linguagem (LLMs) em um torneio de reparo de código baseado no dataset **DroidLeaks**.

---

## 🧬 A Tríade Metodológica

A validação dos reparos gerados pelas LLMs é sustentada por três representações instruídas de inteligência:

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
    ├── 08_a_orquestrar_torneio_llms.py          # Envio automático e tolerant a falhas para as APIs
    ├── 08_b_alinhar_split_ast.py                # Sincronizador de IDs para evitar vazamento de dados
    ├── 08_c_mesclar_resultados_ast.py           # Auditor de histórico para evitar re-chamadas pagas
    ├── 08_d_converter_respostas_ast.py          # Tradutor das correções das LLMs para texto AST
    │
    ├── 09_a_oraculo_torneio_tfidf.py            # Apuração de notas e placar final via TF-IDF
    ├── 09_b_oraculo_torneio_ast.py              # Apuração e geração de gráfico em 300 DPI via AST
    └── 09_c_oraculo_torneio_gcbert.py           # Apuração de notas e placar via Deep Learning
```

---

## ⚙️ Pré-requisitos

Certifique-se de ter o Python 3.10+ instalado. Instale as dependências executando:

```bash
pip install pandas numpy torch transformers javalang scikit-learn openai anthropic google-generativeai matplotlib seaborn python-dotenv joblib
```

Configure um arquivo `.env` na raiz do projeto com as suas credenciais de API:
```env
OPENAI_API_KEY="sua_chave_aqui"
ANTHROPIC_API_KEY="sua_chave_aqui"
GEMINI_API_KEY="sua_chave_aqui"
```

---

## 🚀 Esteira de Execução (Passo a Passo)

Para reproduzir os experimentos da dissertação na íntegra, execute os scripts seguindo a ordem lógica do pipeline de dados:

### Fase 1: Engenharia e Preparação Sintética
1. Execute a preparação e balanceamento dos dados (`01` ao `04`).
2. Gere o aumento de dados e valide a qualidade sintática do que a IA cuspiu:
   ```bash
   python scripts/05_aumento_dados_openai.py
   python scripts/05_b_auditor_sintaxe_javalang.py
   ```
   *O script `05_b` gerará o relatório técnico `erros_inspecao.txt` mapeando as limitações gramaticais da LLM.*

### Fase 2: Treinamento dos Oráculos (Os Juízes)
1. Treine a Baseline do TF-IDF usando o script `07_a`.
2. Realize a conversão estrutural completa e balanceada da AST:
   ```bash
   python scripts/07_b_treinar_validador_ast.py
   ```
3. Extraia as matrizes matemáticas de 768 dimensões com o GraphCodeBERT utilizando aceleração por GPU (CUDA):
   ```bash
   python scripts/07_c_extrair_embeddings_gcbert.py
   ```

### Fase 3: O Torneio e Alinhamento Científico
1. Rode o alinhamento de Split para garantir que a AST herde os mesmos grupos de teste das outras frentes, blindando a pesquisa contra *Data Leakage*:
   ```bash
   python scripts/08_b_alinhar_split_ast.py
   ```
2. Realize a mescla inteligente para auditar quais respostas das LLMs já existem no disco e quais precisam ser disparadas nas APIs:
   ```bash
   python scripts/08_c_mesclar_resultados_ast.py
   ```
3. Execute o orquestrador da arena para coletar as correções pendentes do GPT, Claude e Gemini:
   ```bash
   python scripts/08_a_orquestrar_torneio_llms.py
   ```

### Fase 4: Apuração de Resultados e Gráficos
1. Converta as respostas de código Java puro geradas pelas LLMs em assinaturas sintáticas legíveis:
   ```bash
   python scripts/08_d_converter_respostas_ast.py
   ```
2. Execute o juiz supremo para computar os acertos e gerar o ativo visual oficial da dissertação:
   ```bash
   python scripts/09_b_oraculo_torneio_ast.py
   ```
   *Este comando exibirá o desempenho comparativo no console e salvará a imagem `Figura_Placar_Torneio_AST.png` pronta para publicação em alta resolução (300 DPI).*

---

## 📝 Nota de Rigor Acadêmico

> ⚠️ **Aviso de Replicação:** Devido ao filtro rígido imposto pelo compilador sintático `javalang`, exemplos gerados por LLMs com cortes de contexto ou formatações inválidas são preservados no ambiente de teste com a flag `"ERRO_SINTAXE"`, penalizando os modelos geradores de forma justa e automatizada, refletindo com precisão as taxas empíricas de alucinação de código monitoradas no estudo.
