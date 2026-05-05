"""
test_set_latent_opt.py
----------------------

Usage:
    python test_set_latent_opt.py \\
        --diff_ckpt ./data/model_ckpts/1229_512_adanorm_6layers_2474.ckpt \\
        --pred_ckpt data/model_ckpts/0104_predictor_0_194.pth \\
                    data/model_ckpts/0104_predictor_1_191.pth \\
                    data/model_ckpts/0104_predictor_2_198.pth \\
                    data/model_ckpts/0104_predictor_3_185.pth \\
                    data/model_ckpts/0104_predictor_4_191.pth \\
        --cell_data ./data/drug_response_data/DC_drug_response.csv \\
        --mut_data  ./data/drug_response_data/original_cell2mut.csv \\
        --cna_data  ./data/drug_response_data/original_cell2cna.csv \\
        --cnd_data  ./data/drug_response_data/original_cell2cnd.csv \\
        --vae_ckpt  data/model_ckpts/250_lstm09.ckpt \\
        --vocab_path data/chemicalVAE_tokens.txt \\
        --n_steps   200 --n_samples 32 --ddim_steps 50 --device cuda
"""

import os
import sys
import random
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# RDKit
from rdkit import Chem
from rdkit.Chem import QED
from rdkit import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

# VAE decoder
sys.path.insert(0, './vae_package')
sys.path.insert(0, './src')
from vae_package import vocab, vae_lstm_model, vae_tool

# Local imports — use the same GenoDataset the notebook uses
from src.g2d_diff_diff import Diffusion
from src.g2d_diff_pred import NCIPREDICTOR
from src.utils.g2d_diff_geno_dataset import GenoDataset, GenoCollator

try:
    from llm_scorer import  score_molecule
    _LLM_SCORER_AVAILABLE = True
except ImportError:
    _LLM_SCORER_AVAILABLE = False

from attention_analysis import AttentionExtractor
# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cell line sets (from notebook)
# ---------------------------------------------------------------------------

EV1_CELLS = [
    "HCT116_LARGE_INTESTINE",
    "LOXIMVI_SKIN",
    "K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE",
    "MCF7_BREAST",
    "CCRFCEM_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE",
]
EV2_CELLS = ["EKVX_LUNG", "SKMEL28_SKIN", "SKOV3_OVARY", "NCIH226_LUNG", "OVCAR4_OVARY"]
EV3_CELLS = ["TK10_KIDNEY", "OVCAR5_OVARY", "HOP92_LUNG", "SKMEL2_SKIN", "HS578T_BREAST"]

# Cell lines to re-run (from SKOV3_OVARY onwards)
RERUN_CELLS = [
    "SKMEL28_SKIN","SKOV3_OVARY", "NCIH226_LUNG", "OVCAR4_OVARY",
    "TK10_KIDNEY", "OVCAR5_OVARY", "HOP92_LUNG", "SKMEL2_SKIN", "HS578T_BREAST",
]

# AUC class 0 = most sensitive — best conditioning for optimization
SENSITIVE_CLASS = 0
PREDEFINED_GENOTYPES = ["mut", "cna", "cnd"]


# ---------------------------------------------------------------------------
# Molecular property helpers
# ---------------------------------------------------------------------------

def compute_mol_properties(smiles_list: list) -> list:
    results = []
    for smi in smiles_list:
        if not smi or not isinstance(smi, str):
            results.append({"valid": False, "qed": 0.0, "sas": 10.0, "smiles": ""})
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            results.append({"valid": False, "qed": 0.0, "sas": 10.0, "smiles": smi})
        else:
            try:
                qed_val = float(QED.qed(mol))
                sas_val = float(sascorer.calculateScore(mol))
            except Exception:
                qed_val, sas_val = 0.0, 10.0
            results.append({"valid": True, "qed": qed_val, "sas": sas_val, "smiles": smi})
    return results


def mol_summary(props: list) -> dict:
    valids = [p for p in props if p["valid"]]
    return {
        "validity": len(valids) / max(len(props), 1),
        "mean_qed": float(np.mean([p["qed"] for p in valids])) if valids else 0.0,
        "mean_sas": float(np.mean([p["sas"] for p in valids])) if valids else 10.0,
    }


# ---------------------------------------------------------------------------
# Online property surrogates (QED, SAS, LLM score)
# ---------------------------------------------------------------------------

import torch.nn as nn

class PropertySurrogate(nn.Module):
    """Tiny MLP: z_dim → 1, predicts a property that lives in [0, 1]."""
    def __init__(self, z_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),   nn.ReLU(),
            nn.Linear(64, 1),     nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


def update_surrogate(surrogate, surrogate_opt, z_detached, targets, n_inner=3):
    targets_t = torch.tensor(targets, dtype=torch.float32, device=z_detached.device)
    loss_val = 0.0
    for _ in range(n_inner):
        surrogate_opt.zero_grad()
        loss = nn.functional.mse_loss(surrogate(z_detached), targets_t)
        loss.backward()
        surrogate_opt.step()
        loss_val = loss.item()
    return loss_val


# ---------------------------------------------------------------------------
# Ensemble predictor
# ---------------------------------------------------------------------------

class EnsemblePredictor:
    def __init__(self, ckpt_paths: list, device: str):
        self.device = device
        self.predictors = []
        for i, path in enumerate(ckpt_paths):
            model = NCIPREDICTOR(
                num_of_genotypes=3,
                num_of_dcls=5,
                cond_dim=128,
                drug_dim=128,
                device=device,
            ).to(device).float()
            ckpt  = torch.load(path, map_location=device)
            state = ckpt.get("predictor_state_dict", ckpt)
            model.load_state_dict(state, strict=False)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self.predictors.append(model)
            log.info(f"  Loaded predictor [{i}] from {path}")
        log.info(f"Ensemble of {len(self.predictors)} predictors ready.")

    def predict(self, batch: dict) -> torch.Tensor:
        """Differentiable mean AUC (B, 1), clamped [0, 1]."""
        preds = [p(batch) for p in self.predictors]
        return torch.stack(preds, dim=0).mean(dim=0).clamp(0.0, 1.0)

    @torch.no_grad()
    def predict_no_grad(self, batch: dict) -> np.ndarray:
        """Evaluation-only mean AUC as numpy (B,)."""
        preds = [p(batch).squeeze(-1) for p in self.predictors]
        return torch.stack(preds, dim=0).mean(dim=0).clamp(0.0, 1.0).cpu().numpy()


# ---------------------------------------------------------------------------
# VAE decoder (identical to latent_opt_g2d.py)
# ---------------------------------------------------------------------------

class RNNVAEDecoder:
    def __init__(self, ckpt_path: str, vocab_path: str, device: str, batch_size: int = 32):
        self.device = device
        vo   = vocab.Vocabulary(init_from_file=vocab_path)
        smtk = vocab.SmilesTokenizer(vo)
        self.rvae    = vae_lstm_model.RNNVAE(vo, smtk, device=device, load_fn=ckpt_path)
        self.rvae.model.eval()
        self.sampler = vae_tool.RNNVAESampler(self.rvae, vo, batch_size=batch_size)
        log.info(f"Loaded VAE decoder from {ckpt_path}")

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> list:
        z_np = z.detach().cpu().numpy()
        try:
            token_tensor = self.sampler.sample_from_z(z_np, method='greedy')
            smiles_list  = self.sampler.gen_to_smiles(token_tensor.detach().cpu().numpy())
            return [smi if (smi and isinstance(smi, str)) else "" for smi in smiles_list]
        except Exception as e:
            log.warning(f"VAE decode failed: {e}")
            return [""] * len(z_np)


# ---------------------------------------------------------------------------
# Reward function (identical to latent_opt_g2d.py)
# ---------------------------------------------------------------------------

def compute_reward(
    pred_auc: torch.Tensor,
    props: list,
    w_auc: float,
    w_qed: float,
    w_sas: float,
) -> torch.Tensor:
    device = pred_auc.device
    auc   = pred_auc.squeeze(-1)
    qed_t = torch.tensor([p["qed"]        for p in props], dtype=torch.float32, device=device)
    sas_t = torch.tensor([p["sas"] / 10.0 for p in props], dtype=torch.float32, device=device)
    return -w_auc * auc + w_qed * qed_t - w_sas * sas_t


# ---------------------------------------------------------------------------
# Build a batch using GenoDataset + GenoCollator (same as notebook)
# ---------------------------------------------------------------------------

def build_cell_batch(
    cell_name: str,
    n_samples: int,
    cell2mut: pd.DataFrame,
    cell2cna: pd.DataFrame,
    cell2cnd: pd.DataFrame,
    device: str,
) -> dict:
    input_df = pd.DataFrame(
        [(cell_name, SENSITIVE_CLASS)] * n_samples,
        columns=["ccle_name", "auc_label"],
    ).astype({"auc_label": "int64"})

    dataset  = GenoDataset(input_df, cell2mut, cna=cell2cna, cnd=cell2cnd)
    collator = GenoCollator(genotypes=PREDEFINED_GENOTYPES)
    loader   = DataLoader(
        dataset, batch_size=n_samples, drop_last=False, collate_fn=collator
    )

    batch = next(iter(loader))

    for key in batch:
        if key == "genotype":
            for mut in batch[key]:
                batch[key][mut] = batch[key][mut].to(device)
        elif key == "cell_name":
            pass
        else:
            batch[key] = batch[key].to(device)

    return batch


# ---------------------------------------------------------------------------
# Baseline: sample from diffusion → decode → predict AUC (no optimization)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_baseline(
    batch: dict,
    diff_model: Diffusion,
    ensemble: EnsemblePredictor,
    vae_decoder: RNNVAEDecoder,
    ddim_steps: int,
) -> dict:
    z = diff_model.ddim_sample(batch, sampling_eta=0.0, sampling_time=ddim_steps)

    batch_z = {k: v for k, v in batch.items()}
    batch_z["drug"] = z
    auc_np = ensemble.predict_no_grad(batch_z)

    smiles_list = vae_decoder.decode(z)
    props = compute_mol_properties(smiles_list)
    stats = mol_summary(props)

    return {
        "mean_auc":       float(np.mean(auc_np)),
        "min_auc":        float(np.min(auc_np)),
        "std_auc":        float(np.std(auc_np)),
        "validity":       stats["validity"],
        "mean_qed":       stats["mean_qed"],
        "mean_sas":       stats["mean_sas"],
        "z_init":         z,
        "smiles":         smiles_list,
        "auc_per_sample": auc_np.tolist(),
        "props":          props,
    }


# ---------------------------------------------------------------------------
# Optimized: latent optimization starting from z_init
# ---------------------------------------------------------------------------

def run_optimized(
    z_init: torch.Tensor,
    batch: dict,
    ensemble: EnsemblePredictor,
    vae_decoder: RNNVAEDecoder,
    n_steps: int,
    lr: float,
    w_auc: float,
    w_qed: float,
    w_sas: float,
    l2_reg: float,
    log_every: int,
    surrogate_lr: float = 1e-3,
    surrogate_n_inner: int = 3,
    w_llm: float = 0.3,
    llm_every: int = 10,
    llm_score_top_n: int = 3,
    cell_name: str = None,
    gene_ctx: dict = None,
    att_summary:dict=None,
) -> dict:
    device = z_init.device
    B      = z_init.shape[0]
    z_dim  = z_init.shape[-1]

    z0 = z_init.clone().detach()
    z  = z_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)

    qed_surrogate = PropertySurrogate(z_dim).to(device)
    sas_surrogate = PropertySurrogate(z_dim).to(device)
    llm_surrogate = PropertySurrogate(z_dim).to(device)
    qed_opt = torch.optim.Adam(qed_surrogate.parameters(), lr=surrogate_lr)
    sas_opt = torch.optim.Adam(sas_surrogate.parameters(), lr=surrogate_lr)
    llm_opt = torch.optim.Adam(llm_surrogate.parameters(), lr=1e-3)

    SURROGATE_WARMUP = 5
    LLM_WARMUP_CALLS = 1

    llm_calls_done     = 0
    use_llm_surrogate  = False
    current_mean_llm   = float("nan")

    _llm_enabled = (
        _LLM_SCORER_AVAILABLE
        and bool(os.environ.get("ANTHROPIC_API_KEY"))
        and cell_name is not None
        and gene_ctx is not None
        and w_llm > 0.0
    )
    if _llm_enabled:
        log.info(f"    LLM surrogate ENABLED (every {llm_every} steps, w_llm={w_llm})")
    else:
        log.info("    LLM surrogate DISABLED (set ANTHROPIC_API_KEY + cell_name + w_llm>0)")

    best_reward_t    = torch.full((B,), -1e9, device=device)
    best_z           = z0.clone()
    best_smiles      = [""] * B
    best_auc_vals    = [1.0] * B
    best_reward_vals = [-1e9] * B
    history          = []

    for step in range(n_steps):
        optimizer.zero_grad()

        batch_z = {k: v for k, v in batch.items()}
        batch_z["drug"] = z
        pred_auc = ensemble.predict(batch_z)

        smiles_list = vae_decoder.decode(z)
        props       = compute_mol_properties(smiles_list)

        update_surrogate(qed_surrogate, qed_opt, z.detach(),
                         [p["qed"]        for p in props], surrogate_n_inner)
        update_surrogate(sas_surrogate, sas_opt, z.detach(),
                         [p["sas"] / 10.0 for p in props], surrogate_n_inner)

        if _llm_enabled and step > 0 and step % llm_every == 0:
            auc_np        = pred_auc.detach().squeeze(-1).cpu().numpy()
            validity_rate = mol_summary(props)["validity"]

            valid_indices = [i for i, p in enumerate(props) if p["valid"] and smiles_list[i]]
            ranked        = sorted(valid_indices, key=lambda i: auc_np[i])
            to_score      = ranked[:llm_score_top_n]

            if to_score:
                llm_targets = [None] * B
                scored_any  = False

                log.info(f"    [LLM surrogate] step {step}: scoring {len(to_score)} molecule(s)")
                for idx in to_score:
                    smi = smiles_list[idx]
                    try:
                        r         = score_molecule(
                            smi, cell_name, gene_ctx,
                            pred_auc=float(auc_np[idx]),
                            qed=props[idx]["qed"],
                            sas=props[idx]["sas"],
                            validity=validity_rate,
                            attention_summary=att_summary
                        )
                        llm_score        = float(r.get("final_score", 0.5))
                        llm_targets[idx] = llm_score
                        scored_any       = True
                        log.info(
                            f"      [{idx:2d}] {smi[:45]}... "
                            f"score={llm_score:.3f} "
                            f"conf={r.get('confidence', 0):.2f} "
                            f"auc={auc_np[idx]:.3f}"
                        )
                    except Exception as e:
                        log.warning(f"      [{idx:2d}] LLM call failed: {e}")

                if scored_any:
                    from rdkit.Chem import AllChem
                    from rdkit import DataStructs

                    fps = {}
                    for i in range(B):
                        if smiles_list[i] and props[i]["valid"]:
                            mol_i = Chem.MolFromSmiles(smiles_list[i])
                            if mol_i is not None:
                                fps[i] = AllChem.GetMorganFingerprintAsBitVect(
                                    mol_i, radius=2, nBits=2048)

                    scored_with_fps = [
                        (i, llm_targets[i]) for i in to_score
                        if llm_targets[i] is not None and i in fps
                    ]

                    for i in range(B):
                        if llm_targets[i] is not None:
                            continue
                        if i not in fps or not scored_with_fps:
                            llm_targets[i] = 0.5
                            continue
                        sims    = np.array([DataStructs.TanimotoSimilarity(fps[i], fps[j])
                                            for j, _ in scored_with_fps])
                        scores  = np.array([s for _, s in scored_with_fps])
                        sim_sum = sims.sum()
                        llm_targets[i] = (float(scores.mean()) if sim_sum < 1e-6
                                          else float((sims * scores).sum() / sim_sum))

                    final_targets    = [t if t is not None else 0.5 for t in llm_targets]
                    current_mean_llm = float(np.mean(final_targets))

                    update_surrogate(llm_surrogate, llm_opt, z.detach(), final_targets, n_inner=5)
                    llm_calls_done += len(scored_with_fps)
                    if llm_calls_done >= LLM_WARMUP_CALLS:
                        use_llm_surrogate = True
                    log.info(
                        f"    [LLM surrogate] updated — {len(scored_with_fps)} LLM scores + "
                        f"Tanimoto fill | mean_llm={current_mean_llm:.3f} "
                        f"(total calls: {llm_calls_done})"
                    )

        auc = pred_auc.squeeze(-1)

        if (step >= SURROGATE_WARMUP) and ((w_qed>0) or (w_qed>0)):
            pred_qed = qed_surrogate(z)
            pred_sas = sas_surrogate(z)
            reward   = -w_auc * auc + w_qed * pred_qed - w_sas * pred_sas
        else:
            reward = -w_auc * auc

        if use_llm_surrogate and _llm_enabled:
            pred_llm = llm_surrogate(z)
            reward   = reward + w_llm * pred_llm

        l2_penalty = l2_reg * (z - z0).pow(2).mean()
        loss       = -reward.mean() + l2_penalty

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            qed_t     = torch.tensor([p["qed"]        for p in props],
                                     dtype=torch.float32, device=device)
            sas_t     = torch.tensor([p["sas"] / 10.0 for p in props],
                                     dtype=torch.float32, device=device)
            selection = -w_auc * auc.detach() + w_qed * qed_t - w_sas * sas_t

            improved      = selection > best_reward_t
            best_reward_t = torch.where(improved, selection, best_reward_t)
            best_z[improved] = z[improved].detach()
            for i in range(B):
                if improved[i].item():
                    best_smiles[i]      = smiles_list[i]
                    best_auc_vals[i]    = pred_auc[i].item()
                    best_reward_vals[i] = selection[i].item()

        if step % log_every == 0 or step == n_steps - 1:
            stats    = mol_summary(props)
            mean_auc = pred_auc.detach().mean().item()

            surrogate_tag = ""
            if step >= SURROGATE_WARMUP:
                surrogate_tag += " [QED/SAS✓]"
            if use_llm_surrogate:
                surrogate_tag += " [LLM✓]"

            llm_str = f" | LLM={current_mean_llm:.3f}" if not np.isnan(current_mean_llm) else ""

            log.info(
                f"    step {step:4d}/{n_steps} | loss={loss.item():.4f} | "
                f"AUC={mean_auc:.3f} | QED={stats['mean_qed']:.3f} | "
                f"SAS={stats['mean_sas']:.2f} | valid={stats['validity']:.1%}"
                + llm_str + surrogate_tag
            )
            history.append({
                "step":             step,
                "loss":             loss.item(),
                "mean_auc":         mean_auc,
                "mean_qed":         stats["mean_qed"],
                "mean_sas":         stats["mean_sas"],
                "validity":         stats["validity"],
                "mean_reward":      reward.mean().item(),
                "mean_llm_score":   current_mean_llm,
                "qed_surrogate_on": step >= SURROGATE_WARMUP,
                "llm_surrogate_on": use_llm_surrogate,
                "llm_calls_done":   llm_calls_done,
            })

    with torch.no_grad():
        batch_best = {k: v for k, v in batch.items()}
        batch_best["drug"] = best_z
        best_auc_np = ensemble.predict_no_grad(batch_best)

    best_props = compute_mol_properties(best_smiles)
    best_stats = mol_summary(best_props)

    return {
        "mean_auc":       float(np.mean(best_auc_np)),
        "min_auc":        float(np.min(best_auc_np)),
        "std_auc":        float(np.std(best_auc_np)),
        "validity":       best_stats["validity"],
        "mean_qed":       best_stats["mean_qed"],
        "mean_sas":       best_stats["mean_sas"],
        "best_z":         best_z,
        "smiles":         best_smiles,
        "auc_per_sample": best_auc_np.tolist(),
        "props":          best_props,
        "history":        history,
    }


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_cell_comparison(cell_name: str, baseline: dict, optimized: dict) -> None:
    delta = optimized["mean_auc"] - baseline["mean_auc"]
    log.info(f"  {'─'*55}")
    log.info(f"  Cell: {cell_name}")
    log.info(f"  {'':28s} {'Baseline':>9s}   {'Optimized':>9s}   {'Δ':>8s}")
    log.info(f"  {'Mean AUC':28s} {baseline['mean_auc']:>9.3f}   {optimized['mean_auc']:>9.3f}   {delta:>+8.3f}")
    log.info(f"  {'Min AUC':28s} {baseline['min_auc']:>9.3f}   {optimized['min_auc']:>9.3f}")
    log.info(f"  {'Validity':28s} {baseline['validity']:>9.1%}   {optimized['validity']:>9.1%}")
    log.info(f"  {'Mean QED':28s} {baseline['mean_qed']:>9.3f}   {optimized['mean_qed']:>9.3f}")
    log.info(f"  {'Mean SAS':28s} {baseline['mean_sas']:>9.2f}   {optimized['mean_sas']:>9.2f}")


# ---------------------------------------------------------------------------
# Merge-save results
# Keeps existing rows for cells NOT in new_cell_results,
# replaces/adds rows for cells that ARE in new_cell_results.
# ---------------------------------------------------------------------------

def merge_save_results(new_cell_results: dict, out_dir: str) -> None:
    """
    Merges new_cell_results into existing CSVs/JSON in out_dir.
    - Rows for cells in new_cell_results are replaced.
    - Rows for all other cells are preserved from the existing files.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rerun_set = set(new_cell_results.keys())

    # ── 1. per_cell_results.csv ──────────────────────────────────────────
    per_cell_path = os.path.join(out_dir, "per_cell_results.csv")

    new_rows = []
    for cell_name, res in new_cell_results.items():
        b, o = res["baseline"], res["optimized"]
        new_rows.append({
            "cell_name":          cell_name,
            "baseline_auc_mean":  round(b["mean_auc"],  4),
            "baseline_auc_min":   round(b["min_auc"],   4),
            "baseline_auc_std":   round(b["std_auc"],   4),
            "baseline_validity":  round(b["validity"],  3),
            "baseline_qed":       round(b["mean_qed"],  3),
            "baseline_sas":       round(b["mean_sas"],  3),
            "optimized_auc_mean": round(o["mean_auc"],  4),
            "optimized_auc_min":  round(o["min_auc"],   4),
            "optimized_auc_std":  round(o["std_auc"],   4),
            "optimized_validity": round(o["validity"],  3),
            "optimized_qed":      round(o["mean_qed"],  3),
            "optimized_sas":      round(o["mean_sas"],  3),
            "auc_delta":          round(o["mean_auc"] - b["mean_auc"], 4),
        })
    new_df = pd.DataFrame(new_rows)

    if os.path.exists(per_cell_path):
        old_df = pd.read_csv(per_cell_path)
        # Drop rows whose cell_name will be replaced, then append new ones
        old_df = old_df[~old_df["cell_name"].isin(rerun_set)]
        merged_df = pd.concat([old_df, new_df], ignore_index=True)
        log.info(f"Merged per_cell_results: kept {len(old_df)} old rows, "
                 f"added/replaced {len(new_df)} rows.")
    else:
        merged_df = new_df
        log.info("No existing per_cell_results.csv found — writing fresh.")

    merged_df.to_csv(per_cell_path, index=False)
    log.info(f"per_cell_results → {per_cell_path}")

    # ── 2. best_molecules.csv ────────────────────────────────────────────
    mol_path = os.path.join(out_dir, "best_molecules.csv")

    new_mol_rows = []
    for cell_name, res in new_cell_results.items():
        o = res["optimized"]
        for smi, auc, prop in zip(o["smiles"], o["auc_per_sample"], o["props"]):
            new_mol_rows.append({
                "cell_name":    cell_name,
                "smiles":       smi,
                "valid":        prop["valid"],
                "ensemble_auc": round(auc, 4),
                "qed":          round(prop["qed"], 4),
                "sas":          round(prop["sas"], 4),
            })
    new_mol_df = pd.DataFrame(new_mol_rows)

    if os.path.exists(mol_path):
        old_mol_df = pd.read_csv(mol_path)
        old_mol_df = old_mol_df[~old_mol_df["cell_name"].isin(rerun_set)]
        merged_mol_df = pd.concat([old_mol_df, new_mol_df], ignore_index=True)
        log.info(f"Merged best_molecules: kept {len(old_mol_df)} old rows, "
                 f"added/replaced {len(new_mol_df)} rows.")
    else:
        merged_mol_df = new_mol_df
        log.info("No existing best_molecules.csv found — writing fresh.")

    merged_mol_df.to_csv(mol_path, index=False)
    log.info(f"best_molecules → {mol_path}")

    # ── 3. optimization_history.json ─────────────────────────────────────
    hist_path = os.path.join(out_dir, "optimization_history.json")

    if os.path.exists(hist_path):
        with open(hist_path, "r") as f:
            old_hist = json.load(f)
        # Remove stale entries for re-run cells
        for cell in rerun_set:
            old_hist.pop(cell, None)
        log.info(f"Merged optimization_history: removed {len(rerun_set)} stale entries, "
                 f"kept {len(old_hist)} existing.")
    else:
        old_hist = {}
        log.info("No existing optimization_history.json found — writing fresh.")

    new_hist = {cell: res["optimized"]["history"] for cell, res in new_cell_results.items()}
    merged_hist = {**old_hist, **new_hist}

    with open(hist_path, "w") as f:
        json.dump(merged_hist, f, indent=2)
    log.info(f"optimization_history → {hist_path}")


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

def final_summary(cell_results: dict) -> None:
    log.info("")
    log.info("=" * 65)
    log.info("FINAL SUMMARY — RE-RUN CELLS")
    log.info("=" * 65)
    log.info(f"  {'Cell line':38s} {'Base':>6s}  {'Opt':>6s}  {'Δ':>7s}")
    log.info(f"  {'─'*60}")

    deltas     = []
    base_aucs  = []
    opt_aucs   = []

    for cell_name, res in cell_results.items():
        b = res["baseline"]["mean_auc"]
        o = res["optimized"]["mean_auc"]
        d = o - b
        deltas.append(d)
        base_aucs.append(b)
        opt_aucs.append(o)
        log.info(f"  {cell_name:38s} {b:>6.3f}  {o:>6.3f}  {d:>+7.3f}")

    log.info(f"  {'─'*60}")
    log.info(
        f"  {'Average':38s} {np.mean(base_aucs):>6.3f}  "
        f"{np.mean(opt_aucs):>6.3f}  {np.mean(deltas):>+7.3f}"
    )
    log.info("")

    

    


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_diffusion_model(ckpt_path: str, device: str, cfgw: float) -> Diffusion:
    model = Diffusion(device=device, training=False, cfgw=cfgw).to(device).float()
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("diffusion_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log.info(f"Loaded diffusion model from {ckpt_path}  (cfgw={cfgw})")
    return model


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Track 0: Latent optimization on test cell lines (resume from SKOV3_OVARY)"
    )
    parser.add_argument("--diff_ckpt",  required=True)
    parser.add_argument("--pred_ckpt",  required=True, nargs="+")
    parser.add_argument("--cell_data",  required=True)
    parser.add_argument("--mut_data",   required=True)
    parser.add_argument("--cna_data",   required=True)
    parser.add_argument("--cnd_data",   required=True)
    parser.add_argument("--vae_ckpt",   default="data/model_ckpts/250_lstm09.ckpt")
    parser.add_argument("--vocab_path", default="data/chemicalVAE_tokens.txt")
    parser.add_argument("--out_dir",    default="./test_results")
    parser.add_argument("--nest_adj",   default="./data/NeST_neighbor_adj.npy")
    parser.add_argument("--nest_sets",  default="./data/NeST_gene_sets.json")

    # Optimization hyperparameters
    parser.add_argument("--n_steps",    type=int,   default=200)
    parser.add_argument("--n_samples",  type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=0.005)
    parser.add_argument("--w_auc",      type=float, default=1.0)
    parser.add_argument("--w_qed",      type=float, default=0.5)
    parser.add_argument("--w_sas",      type=float, default=0.5)
    parser.add_argument("--l2_reg",     type=float, default=0.05)
    parser.add_argument("--cfgw",       type=float, default=7.0)
    parser.add_argument("--ddim_steps", type=int,   default=50)
    parser.add_argument("--log_every",  type=int,   default=20)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--device",     default="cuda")

    # LLM surrogate options
    parser.add_argument("--w_llm",           type=float, default=1.0)
    parser.add_argument("--llm_every",       type=int,   default=10)
    parser.add_argument("--llm_score_top_n", type=int,   default=10)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    best_params = pd.read_json('hparam_results/best_hparams.json')

    args.lr         = best_params['hyperparameters']['lr']
    args.l2_reg     = best_params['hyperparameters']['l2_reg']
    args.n_steps    = int(best_params['hyperparameters']['n_steps'])
    args.ddim_steps = int(best_params['hyperparameters']['ddim_steps'])
    args.w_auc      = best_params['fixed_reward_weights']['w_auc']

    hp = best_params['hyperparameters']
    args.surrogate_lr      = float(hp.get('surrogate_lr',      1e-3))
    args.surrogate_n_inner = int(  hp.get('surrogate_n_inner', 3))
    log.info(
        f"Loaded best_hparams.json: lr={args.lr} l2={args.l2_reg} "
        f"steps={args.n_steps} ddim={args.ddim_steps} | "
        f"surrogate_lr={args.surrogate_lr} surrogate_n_inner={args.surrogate_n_inner}"
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    log.info(f"Device: {device}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # ── Fixed list: only the cells to re-run ────────────────────────────
    eval_cells = RERUN_CELLS
    log.info(f"Re-running cells: {eval_cells}")
    log.info("Results for all other cells will be preserved from existing output files.")

    # ── Load models ──────────────────────────────────────────────────────
    log.info("Loading diffusion model ...")
    diff_model = load_diffusion_model(args.diff_ckpt, device, args.cfgw)

    log.info(f"Loading ensemble of {len(args.pred_ckpt)} predictors ...")
    ensemble = EnsemblePredictor(args.pred_ckpt, device)

    log.info("Loading VAE decoder ...")
    vae_decoder = RNNVAEDecoder(args.vae_ckpt, args.vocab_path, device, args.n_samples)

    # ── Load genotype data ───────────────────────────────────────────────
    log.info("Loading genotype data ...")
    cell2mut = pd.read_csv(args.mut_data, index_col=0).rename(columns={"index": "ccle_name"})
    cell2cna = pd.read_csv(args.cna_data, index_col=0).rename(columns={"index": "ccle_name"})
    cell2cnd = pd.read_csv(args.cnd_data, index_col=0).rename(columns={"index": "ccle_name"})

    # ── Build AttentionExtractor once (shared across all cell lines) ─────
    _att_extractor = None
    if _LLM_SCORER_AVAILABLE and os.environ.get("ANTHROPIC_API_KEY") and args.w_llm > 0.0:
        try:
            _gene_names_att = load_gene_names(args.mut_data)
            _att_extractor = AttentionExtractor(
                diff_model=diff_model,
                gene_names=_gene_names_att,
                nest_adj_path=args.nest_adj,
                nest_sets_path=args.nest_sets,
            )
            log.info("AttentionExtractor ready.")
        except Exception as _e:
            log.warning(f"AttentionExtractor setup failed: {_e} — running without attention")
   
    log.info(
        f"\nConfiguration:\n"
        f"  Re-run cells       : {eval_cells}\n"
        f"  Samples/cell       : {args.n_samples}\n"
        f"  Opt steps          : {args.n_steps} | lr={args.lr} | l2={args.l2_reg}\n"
        f"  Reward weights     : AUC={args.w_auc} QED={args.w_qed} SAS={args.w_sas}\n"
        f"  QED/SAS surrogates : lr={args.surrogate_lr} n_inner={args.surrogate_n_inner}\n"
        f"  LLM surrogate      : w_llm={args.w_llm} every={args.llm_every} "
        f"top_n={args.llm_score_top_n}"
    )

    # ── Run per cell line ────────────────────────────────────────────────
    cell_results = {}

    for cell_name in eval_cells:
        log.info(f"\n{'='*65}")
        log.info(f"Cell: {cell_name}")
        log.info(f"{'='*65}")

        batch = build_cell_batch(
            cell_name, args.n_samples,
            cell2mut, cell2cna, cell2cnd, device
        )

        log.info("  Running baseline (no optimization) ...")
        baseline = run_baseline(batch, diff_model, ensemble, vae_decoder, args.ddim_steps)
        log.info(
            f"  Baseline: AUC={baseline['mean_auc']:.3f} | "
            f"validity={baseline['validity']:.1%} | QED={baseline['mean_qed']:.3f}"
        )

        log.info(f"  Running latent optimization ({args.n_steps} steps) ...")

        _gene_ctx = None
        if _LLM_SCORER_AVAILABLE and os.environ.get("ANTHROPIC_API_KEY") and args.w_llm > 0.0:
            try:
                from llm_scorer import load_gene_names, get_cell_gene_context
                _gene_names = load_gene_names(args.mut_data)
                _gene_ctx   = get_cell_gene_context(
                    cell_name, cell2mut, cell2cna, cell2cnd, _gene_names
                )
            except Exception as _e:
                log.warning(f"  gene_ctx build failed: {_e} — LLM surrogate disabled for this cell")

        # Build altered_mask and extract attention summary for this cell
        _att_summary = None
        if _att_extractor is not None:
            try:
              
                _att_result  = _att_extractor.extract(batch, cell_name=cell_name)
                _att_summary = _att_result.attention_summary
                log.info(
                    f"  [attention] {cell_name}: {len(_att_result.top_genes)} top genes, "
                    f"{len(_att_result.enriched_pathways)} enriched pathways"
                )
            except Exception as _e:
                log.warning(f"  Attention extraction failed: {_e} — scoring without attention")

        optimized = run_optimized(
            z_init=baseline["z_init"],
            batch=batch,
            ensemble=ensemble,
            vae_decoder=vae_decoder,
            n_steps=args.n_steps,
            lr=args.lr,
            w_auc=args.w_auc,
            w_qed=args.w_qed,
            w_sas=args.w_sas,
            l2_reg=args.l2_reg,
            log_every=args.log_every,
            surrogate_lr=args.surrogate_lr,
            surrogate_n_inner=args.surrogate_n_inner,
            w_llm=args.w_llm,
            llm_every=args.llm_every,
            llm_score_top_n=args.llm_score_top_n,
            cell_name=cell_name,
            gene_ctx=_gene_ctx,
            att_summary=_att_summary,
        )

        cell_results[cell_name] = {"baseline": baseline, "optimized": optimized}
        log_cell_comparison(cell_name, baseline, optimized)

    # ── Merge new results into existing output files ─────────────────────
    merge_save_results(cell_results, args.out_dir)
    final_summary(cell_results)


if __name__ == "__main__":
    main()