# DIxAI-Text

Extension of the DIxAI framework (Decision-Information Explanations) to Transformer-based NLP models.

This repository builds on the official DIxAI implementation by Haythem Ghazouani
(<https://github.com/haythemghazouani/decision_information_xai>). The original code (vision, tabular, medical
imaging) is kept, and DIxAI-Text is integrated into it as a new `dixai.text` subpackage.

Paper: *DIxAI-Text: Extension of the DIxAI Framework to the Explainability of Natural Language Processing Models*
(Ghazouani, Chaieb, Selmene, Mhiri, Kamel).

## Overview

For a given input text, DIxAI-Text learns a per-token mask on the contextual embeddings of a frozen BERT-style
classifier (Gumbel-Softmax relaxation), so that the masked input preserves the model decision with as few tokens
as possible. A sequential Total Variation penalty replaces the spatial one used for images and favors contiguous
spans. Explanations are evaluated with the ERASER metrics (sufficiency, comprehensiveness) on SST-2 and SNLI,
against LIME, SHAP, Integrated Gradients, attention and random selection.

## What was added, and where

| Location | Status | Content |
|---|---|---|
| `src/dixai/text/` | new | DIxAI-Text subpackage |
| `src/dixai/text/explainer_text.py` | new | text explainer (instance-wise mask optimization) |
| `src/dixai/text/masks_text.py` | new | Gumbel-Softmax token-level masks |
| `src/dixai/text/objective_text.py` | new | objective: sparsity, fidelity, sequential contiguity (TV) |
| `src/dixai/text/baselines_text.py` | new | baseline embeddings (mean, mask token, pad token) |
| `src/dixai/text/metrics_text.py` | new | ERASER sufficiency and comprehensiveness |
| `src/dixai/text/data.py`, `data_snli.py`, `data_eraser.py` | new | text datasets loading (including SNLI and ERASER) |
| `experiments/text/` | new | `pub_sst2.py`, `pub_snli.py` and their `results/` |

Everything outside these paths is original DIxAI code.


## Installation

```
git clone https://github.com/Eya-mhiri/DIxAI-Text.git
cd DIxAI-Text
pip install -e .
pip install transformers datasets lime shap matplotlib
```

## Reproducing the DIxAI-Text experiments

All text experiments use N = 50 examples, 3 seeds (0, 1, 2) and S = 200 optimization steps, on CPU.

```
# SST-2: comparison with the baselines
python experiments/text/pub_sst2.py

# SNLI: comparison with the baselines (real SNLI loaded through the datasets library)
python experiments/text/pub_snli.py
```

Ablation and comparison figures for SST-2 and SNLI are saved in `experiments/text/results/`.
Classifiers: `textattack/bert-base-uncased-SST-2` (SST-2) and `cross-encoder/nli-deberta-v3-small` (SNLI).

## Original DIxAI experiments

The vision, tabular and medical experiments of the original framework are in `experiments/benchmark`,
`experiments/tabular`, `experiments/medical` and `experiments/vision`. See the original repository for their
description.

## Project structure

```
src/dixai/            DIxAI library
src/dixai/text/       DIxAI-Text (this work)
experiments/          benchmark scripts
experiments/text/     DIxAI-Text experiments (this work)
figures/              figures
scripts/              helper scripts
```

## Citation

If you use the text extension, please cite:

```
@article{ghazouani2026dixaitext,
  title={DIxAI-Text: Extension of the DIxAI Framework to the Explainability of Natural Language Processing Models},
  author={Ghazouani, Haythem and Chaieb, Marouene and Selmene, Sarra and Mhiri, Eya and Kamel, Imen},
  journal={Preprint},
  year={2026},
  url={https://github.com/Eya-mhiri/DIxAI-Text}
}
```

and the original framework:

```
@article{ghazouani2026dixai,
  title={DIxAI: Decision-Information Conservation for Explainable AI},
  author={Ghazouani, Haythem},
  journal={arXiv preprint},
  year={2026}
}
```

## 📄 License
MIT License.
