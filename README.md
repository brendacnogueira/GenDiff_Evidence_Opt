# GenDiff_Evidence_Opt
**Genotype-Conditioned Molecular Generation via Evidence-Grounded Multi-Objective Latent Perturbation in Diffusion Models**

> Given a tumor genotype, `GenDiff_Evidence_Opt` generates and optimizes drug candidates jointly for predicted sensitivity, drug-likeness, synthetic accessibility, and biologically grounded relevance — guided by a three-agent LLM pipeline and NeST pathway-aware attention.

---

## Overview

<!-- pipeline overview diagram -->
<!-- ![Pipeline Overview](images/pipeline_overview.png) -->

`GenDiff_Evidence_Opt` builds on a pretrained genotype-conditioned diffusion model to perform multi-objective latent space optimization. A three-agent LLM pipeline (Biologist Agent → Chemistry Agent → Score Agent) provides biologically grounded reward signals that steer generation toward cancer cell-line-specific therapeutic candidates.

**Key contributions:**
- Multi-objective latent optimization
- NeST adjacency-aware hybrid re-ranking of model attention for genotype-specific gene prioritization
- Three-agent LLM scoring pipeline with NCI interaction analysis and literature retrieval
- End-to-end pipeline from tumor genotype to scored, interpretable drug candidates

---

### Three-Agent Pipeline

| Agent | Role |
|-------|------|
| **Biologist Agent** | Extracts top attention genes from the diffusion model for the input cell line genotype |
| **Chemistry Agent** | Performs NCI interaction analysis against known pharmacophores, grounded in the Biologist Agent's gene targets and web-retrieved literature |
| **Score Agent** | Synthesizes AUC prediction, descriptor-based drug-likeness, NCI overlap, and genotype exploitation rationale into a calibrated final score |

---

## Repository Structure

```
GenDiff_Evidence_Opt/
│
├── attention_analysis.py          # AttentionExtractor: hybrid NeST + attention re-ranking
├── llm_scorer.py                  # Three-agent LLM scoring pipeline
├── evaluation_known_binders.py    # Known binder/non-binder calibration evaluation
├── requirements.txt
├── llm_consistency_eval.py
│
├── src/
├── vae_package/
├── data/
|   ├──drug_response_data/
|       ├── DC_drug_response.csv
|       ├── DC_drug2smi.csv
│       ├── original_cell2mut.csv  # Mutation matrix (cell lines × 718 genes)
│       ├── original_cell2cna.csv  # Copy number amplification matrix
│       └── original_cell2cnd.csv  # Copy number deletion matrix
│   ├── model_ckpts/               # Pretrained model checkpoints
│   │   ├── 1229_512_adanorm_6layers_2474.ckpt   # Diffusion model
│   │   ├── 0104_predictor_[0-4]_*.pth            # Ensemble predictors
│   │   └── 250_lstm09.ckpt                       # VAE
│   ├── NeST_neighbor_adj.npy      # NeST co-membership adjacency matrix (718 × 718)
│   ├── chemicalVAE_tokens.txt     # VAE vocabulary
│
├── hparam_results/
│   └── best_hparams.json          # Best hyperparameters from tuning
│
└── images/                        # Figures for paper / README


```
### Important

You can download all processed datasets, model checkpoints in this google drive link.
[https://drive.google.com/file/d/1qk4Wwkqvwas7kpjcuFKbSCT8aPaP8RKI/view?usp=drive_link](https://drive.google.com/file/d/1qk4Wwkqvwas7kpjcuFKbSCT8aPaP8RKI/view?usp=drive_link)
You must unpack this zip file in the repository folder.

-- data <- Need to download from the link above.

-- src

-- other files...

```
---

## Installation

```bash
git clone https://github.com/yourusername/GenDiff_evidence_opt.git
cd GenDiff_evidence_opt

conda create -n GenDiff_Evidence_Opt python=3.8
conda activate GenDiff_Evidence_Opt

pip install -r requirement.txt --extra-index-url https://download.pytorch.org/whl/cu113
```

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Quick Start


### 1. Run latent optimization on test cell lines

```bash
python test_set_latent_opt.py \
    --diff_ckpt  ./data/model_ckpts/1229_512_adanorm_6layers_2474.ckpt \
    --pred_ckpt  data/model_ckpts/0104_predictor_0_194.pth \
                 data/model_ckpts/0104_predictor_1_191.pth \
                 data/model_ckpts/0104_predictor_2_198.pth \
                 data/model_ckpts/0104_predictor_3_185.pth \
                 data/model_ckpts/0104_predictor_4_191.pth \
    --mut_data   ./data/drug_response_data/original_cell2mut.csv \
    --cna_data   ./data/drug_response_data/original_cell2cna.csv \
    --cnd_data   ./data/drug_response_data/original_cell2cnd.csv \
    --vae_ckpt   data/model_ckpts/250_lstm09.ckpt \
    --vocab_path data/chemicalVAE_tokens.txt \
    --nest_adj   ./data/NeST_neighbor_adj.npy \
    --w_llm 1.0 --llm_every 10 --llm_score_top_n 25
```



### 2. Run known-binder calibration evaluation

```bash
python evaluation_known_binders.py \
    --diff_ckpt  ./data/model_ckpts/1229_512_adanorm_6layers_2474.ckpt \
    --pred_ckpt  data/model_ckpts/0104_predictor_[0-4]_*.pth \
    --mut_data   ./data/drug_response_data/original_cell2mut.csv \
    --cna_data   ./data/drug_response_data/original_cell2cna.csv \
    --cnd_data   ./data/drug_response_data/original_cell2cnd.csv
```


### 3. Consistency Analysis
```bash
python llm_scorer_consistency_eval.py \
  --diff_ckpt  ./data/model_ckpts/1229_512_adanorm_6layers_2474.ckpt \
  --cell_data  ./data/drug_response_data/DC_drug_response.csv \
  --drug2smi   ./data/drug_response_data/DC_drug2smi.csv \
  --mut_data   ./data/drug_response_data/original_cell2mut.csv \
  --cna_data   ./data/drug_response_data/original_cell2cna.csv \
  --cnd_data   ./data/drug_response_data/original_cell2cnd.csv \
  --n_rounds 5 --n_pairs 50 --device cuda --web_search

```
---

## License

MIT License. See `LICENSE` for details.
