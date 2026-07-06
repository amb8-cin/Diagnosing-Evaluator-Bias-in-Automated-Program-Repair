# Automated Resource Leak Detection and Repair in the Android Ecosystem using LLMs and Artificial Intelligence

This repository contains the experimental framework of a Master's dissertation focused on the evaluation, detection, and automated repair of resource leak failures (*Resource Leaks*) in Android applications. 

The project implements a comparative approach divided into a **Methodological Triad** (Lexical, Structural, and Semantic) to train classifiers ("Judges") capable of blindly and automatically auditing the effectiveness of large language models (LLMs) in a code repair tournament based on the **DroidLeaks** dataset.

---

## 🧬 The Methodological Triad

The validation of LLM-generated repairs is supported by three instructed representations of intelligence:

1.  **Lexical Approach (Text):** Representation by term frequency via **TF-IDF** combined with the *Random Forest* classifier. Serves as the project's baseline.
2.  **Structural Approach (Rules):** Extraction and conversion of the Abstract Syntax Tree (**AST**) via the `javalang` compiler parser. Maps the logical skeleton of resource management (try-catch-finally blocks and Android lifecycle methods such as `release()`, `recycle()`, `close()`).
3.  **Semantic/Neural Approach (Context):** Deep source code vectorization using the state-of-the-art **GraphCodeBERT** model (Microsoft), capturing the data flow (*Data Flow Graph*) into dense 768-dimensional vectors (`[CLS]` Token).

---

## 📂 Repository Structure

```text
Meu_Mestrado_DroidLeaks/
│
├── data_bases/
│   ├── 04_final/
│   │   ├── dataset_synthetic_chatgpt.csv        # Raw expanded dataset (2014 rows)
│   │   ├── train_validator_final.csv           # Official training split (1414 rows)
│   │   ├── holdout_300_bugs_llm.csv             # 300 isolated bugs (Proofbook)
│   │   └── holdout_ast.csv                      # IDs of AST-surviving bugs
│   │
│   ├── ast/
│   │   ├── dataset_ast_complete.csv             # Balanced AST features (1926 rows)
│   │   ├── dataset_ast_train.csv               # Aligned AST training
│   │   ├── dataset_ast_test.csv                # Aligned AST testing
│   │   └── llm_responses_ast.csv               # AST features extracted from AI responses
│   │
│   ├── graphcodebert/
│   │   └── embeddings_graphcodebert.csv         # Dense 768-dimensional matrix of codes
│   │
│   └── 05_results/
│       └── llm_tournament_results_FILTERED_AST.csv  # Checklist of aligned API calls
│
├── models/
│   ├── tfidf/
│   │   ├── vectorizer.pkl                       # Trained TF-IDF vectorizer
│   │   └── resource_leak_validator.pkl          # RF Classifier (TF-IDF)
│   ├── ast/
│   │   ├── vectorizer_ast.pkl                   # AST vocabulary vectorizer
│   │   └── ast_rf_validator.pkl                 # RF Classifier (AST)
│   └── graphcodebert/
│       └── neural_gcbert_classifier.pkl      # Semantic Classifier
│
└── scripts/
    ├── 01_extract_cases.py                      # Mining of the original DroidLeaks
    ├── 02_generate_diffs.py                        # Isolates the commit modification (Bug -> Fix)
    ├── 03_consolidate_dataset.py                 # Structures the initial engineering table
    ├── 04_balance_classes.py                  # Ensures the exact proportion of 50% Bug / 50% Fix
    ├── 05_data_augmentation_openai.py               # Synthetic generation via GPT for base expansion
    ├── 05_b_javalang_syntax_auditor.py         # Diagnosis and log of LLM grammatical failures
    ├── 06_split_holdout.py                    # Initial random Train/Test split (70/30)
    │
    ├── 07_a_train_tfidf_validator.py          # Lexical Baseline Training
    ├── 07_b_train_ast_validator.py            # Syntactic feature extraction and AST RF training
    ├── 07_c_extract_gcbert_embeddings.py        # Neural code converter to 768 dimensions
    │
    ├── 08_a_orchestrate_llm_tournament.py          # Automatic and fault-tolerant submission to APIs
    ├── 08_b_align_ast_split.py                # ID synchronizer to prevent data leakage
    ├── 08_c_merge_ast_results.py           # History auditor to prevent paid re-calls
    ├── 08_d_convert_ast_responses.py          # Translator of LLM corrections to AST text
    │
    ├── 09_a_tfidf_tournament_oracle.py            # Score computation and final scoreboard via TF-IDF
    ├── 09_b_ast_tournament_oracle.py              # Score computation and 300 DPI graph generation via AST
    └── 09_c_gcbert_tournament_oracle.py           # Score computation and scoreboard via Deep Learning
```

---

## ⚙️ Prerequisites

Ensure you have Python 3.10+ installed. Install dependencies by running:

```bash
pip install pandas numpy torch transformers javalang scikit-learn openai anthropic google-generativeai matplotlib seaborn python-dotenv joblib
```

Configure a `.env` file in the project root with your API credentials:
```env
OPENAI_API_KEY="your_key_here"
ANTHROPIC_API_KEY="your_key_here"
GEMINI_API_KEY="your_key_here"
```

---

## 🚀 Execution Pipeline (Step-by-Step)

To reproduce the dissertation experiments in full, execute the scripts following the logical order of the data pipeline:

### Phase 1: Engineering and Synthetic Preparation
1.  Execute data preparation and balancing (`01` to `04`).
2.  Generate data augmentation and validate the syntactic quality of what the AI output:
    ```bash
    python scripts/05_data_augmentation_openai.py
    python scripts/05_b_javalang_syntax_auditor.py
    ```
    *Script `05_b` will generate the technical report `inspection_errors.txt` mapping the grammatical limitations of the LLM.*

### Phase 2: Training the Oracles (The Judges)
1.  Train the TF-IDF Baseline using script `07_a`.
2.  Perform the complete and balanced structural conversion of the AST:
    ```bash
    python scripts/07_b_train_ast_validator.py
    ```
3.  Extract the 768-dimensional mathematical matrices with GraphCodeBERT using GPU acceleration (CUDA):
    ```bash
    python scripts/07_c_extract_gcbert_embeddings.py
    ```

### Phase 3: The Tournament and Scientific Alignment
1.  Run the Split alignment to ensure the AST inherits the same test groups as the other fronts, safeguarding the research against *Data Leakage*:
    ```bash
    python scripts/08_b_align_ast_split.py
    ```
2.  Perform intelligent merging to audit which LLM responses already exist on disk and which need to be triggered on the APIs:
    ```bash
    python scripts/08_c_merge_ast_results.py
    ```
3.  Execute the arena orchestrator to collect pending corrections from GPT, Claude, and Gemini:
    ```bash
    python scripts/08_a_orchestrate_llm_tournament.py
    ```

### Phase 4: Results and Graph Computation
1.  Convert the pure Java code responses generated by the LLMs into readable syntactic signatures:
    ```bash
    python scripts/08_d_convert_ast_responses.py
    ```
2.  Execute the supreme judge to compute hits and generate the official visual asset of the dissertation:
    ```bash
    python scripts/09_b_ast_tournament_oracle.py
    ```
    *This command will display the comparative performance on the console and save the image `Figure_AST_Tournament_Scoreboard.png` ready for high-resolution publication (300 DPI).*

---

## 📝 Academic Rigor Note

> ⚠️ **Replication Warning:** Due to the strict filter imposed by the `javalang` syntactic compiler, examples generated by LLMs with context cuts or invalid formatting are preserved in the test environment with the flag `"SYNTAX_ERROR"`, penalizing the generating models fairly and automatically, accurately reflecting the empirical rates of code hallucination monitored in the study.