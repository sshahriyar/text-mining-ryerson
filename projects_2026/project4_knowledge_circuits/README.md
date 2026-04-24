# Knowledge Circuits in GPT-2: A Small-Scale Study

DS8008 course project. A reduced-scale reproduction of *Knowledge Circuits in Pretrained Transformers* (Yao et al., 2024), studying which attention heads and MLP layers in GPT-2 small most support factual recall.

## Approach

Zero-ablation of individual components, scored by **logit difference** on counterfactual prompt pairs from the CounterFact dataset. For each `(layer, head)` and each MLP layer, we compute the drop in the model's preference for the correct answer over a matched counterfactual answer, averaged across prompts. The paper's full method uses automatic edge-level circuit discovery (ACDC); this project uses a simpler node-level analysis as an approximation.

## Structure

```
├── knowledge_circuits.ipynb              Main report notebook
├── data/
│   ├── facts.json                        Filtered subset of CounterFact (1000 candidates)
│   └── ims_country_capital_city_0.01     Contains the Output files when running original code
├── src/
│   ├── model_utils.py                    Model loading, logit-difference, fact filtering
│   ├── ablation.py                       Head and MLP zero-ablation hooks + sweep loops
│   └── visualization.py                  Heatmap and bar-chart plotting
├── tests/                                Pytest tests for model_utils and ablation
├── scripts/
│   └── build_dataset.py                  Rebuild data/facts.json from HuggingFace
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.13. On a fresh clone:

```bash
python3.13 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

If you need to rebuild the dataset file, run `.venv/bin/python scripts/build_dataset.py` (downloads CounterFact from HuggingFace).

## Running the notebook

```bash
.venv/bin/jupyter notebook knowledge_circuits.ipynb
```

The head-ablation sweep runs 12 × 12 × N forward passes, where N is the number of filtered prompts (default 100). On a modern CPU this takes roughly 10–20 minutes. The MLP sweep is much faster. A CUDA GPU will accelerate both substantially.

## Running tests

```bash
.venv/bin/python -m pytest tests/ -v
```

Nine tests cover the `logit_diff` metric, tokenizer assertions, fact filtering, hook correctness, and sweep output shapes.

## References

1. Y. Yao et al. *Knowledge Circuits in Pretrained Transformers*, 2024.
2. K. Meng, D. Bau, A. Andonian, Y. Belinkov. *Locating and Editing Factual Associations in GPT* (ROME). NeurIPS 2022.
3. K. Wang et al. *Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small*. ICLR 2023.
4. CounterFact dataset (`NeelNanda/counterfact-tracing` on HuggingFace), derived from Meng et al.
