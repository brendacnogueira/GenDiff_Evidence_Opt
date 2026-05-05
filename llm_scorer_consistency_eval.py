"""
llm_scorer_consistency_eval.py
-------------------------------
Evaluates the consistency of the two-agent LLM scorer (ChemistryAgent +
BiochemistryAgent) across the training set molecules and cell lines.

For each molecule × cell combination sampled from the training set:
  - Runs score_molecule() N_ROUNDS times (default 5)
  - Records per-round: final_score, confidence, nci_score, descriptor_score,
    recommendation, top 2 attended genes + their attention scores, admet_flags,
    nci_similarity, genotype_exploitation mechanism
  - Computes consistency metrics across rounds:
      mean, std, min, max, range for all numeric scores
      modal recommendation (most frequent)
      gene stability (are the same top-2 genes returned each round?)

Why JSON over CSV:
  Each record has nested lists (per-round scores, gene lists, flags) that
  don't flatten cleanly. JSON preserves structure and is easy to load with
  pd.read_json(orient='records') for analysis.

Output files (in --out_dir):
  consistency_results.json   — full per-pair results with all rounds
  consistency_summary.csv    — one row per mol×cell pair, aggregated metrics
  failed_pairs.json          — any pairs that errored across all rounds

Usage:
    python llm_scorer_consistency_eval.py \\
        --diff_ckpt  ./data/model_ckpts/1229_512_adanorm_6layers_2474.ckpt \\
        --cell_data  ./data/drug_response_data/DC_drug_response.csv \\
        --drug2smi   ./data/drug_response_data/DC_drug2smi.csv \\
        --mut_data   ./data/drug_response_data/original_cell2mut.csv \\
        --cna_data   ./data/drug_response_data/original_cell2cna.csv \\
        --cnd_data   ./data/drug_response_data/original_cell2cnd.csv \\
        --out_dir    ./consistency_eval \\
        --n_rounds   5 \\
        --n_pairs    50 \\
        --device     cuda

Environment:
    ANTHROPIC_API_KEY must be set.
"""

import os
import sys
import json
import logging
import argparse
import random
from pathlib import Path
from datetime import datetime
from statistics import mode, StatisticsError

import numpy as np
import pandas as pd
import torch

from rdkit import Chem
from rdkit.Chem import QED
from rdkit import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

sys.path.insert(0, './vae_package')
sys.path.insert(0, './src')

from llm_scorer import (
    score_molecule,
    get_cell_gene_context,
    get_mol_descriptors,
    load_gene_names,
    fetch_literature_with_metadata,
    USE_WEB_SEARCH,
    DEFAULT_MODEL,
)
from attention_analysis import AttentionExtractor

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ── Cell splits ──────────────────────────────────────────────────────────────
EV2_CELLS = ['EKVX_LUNG', 'SKMEL28_SKIN', 'SKOV3_OVARY', 'NCIH226_LUNG', 'OVCAR4_OVARY']
EV3_CELLS = ['TK10_KIDNEY', 'OVCAR5_OVARY', 'HOP92_LUNG', 'SKMEL2_SKIN', 'HS578T_BREAST']
HELD_OUT  = set(EV2_CELLS + EV3_CELLS)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_diffusion_model(ckpt_path: str, device: str, cfgw: float = 7.0):
    from src.g2d_diff_diff import Diffusion
    model = Diffusion(device=device, training=False, cfgw=cfgw).to(device).float()
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("diffusion_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log.info(f"Loaded diffusion model from {ckpt_path}")
    return model


# ---------------------------------------------------------------------------
# Attention extraction per cell (cached across rounds)
# ---------------------------------------------------------------------------

def build_attention_cache(
    cell_names: list,
    extractor: AttentionExtractor,
    cell2mut: pd.DataFrame,
    cell2cna: pd.DataFrame,
    cell2cnd: pd.DataFrame,
    device: str,
) -> dict:
    """
    Run attention extraction once per unique cell line and cache results.
    Attention is deterministic — no need to repeat per round.
    """
    from src.utils.g2d_diff_geno_dataset import GenoDataset, GenoCollator
    from torch.utils.data import DataLoader

    cache = {}
    for cell_name in sorted(set(cell_names)):
        log.info(f"  Extracting attention for {cell_name} ...")
        try:
            input_df = pd.DataFrame(
                [(cell_name, 0)], columns=["ccle_name", "auc_label"]
            ).astype({"auc_label": "int64"})

            dataset  = GenoDataset(input_df, cell2mut, cna=cell2cna, cnd=cell2cnd)
            collator = GenoCollator(genotypes=["mut", "cna", "cnd"])
            loader   = DataLoader(dataset, batch_size=1, collate_fn=collator)
            batch    = next(iter(loader))

            for key in batch:
                if key == "genotype":
                    for k in batch[key]:
                        batch[key][k] = batch[key][k].to(device)
                elif key != "cell_name":
                    batch[key] = batch[key].to(device)

            att_result = extractor.extract(batch, cell_name=cell_name)
            cache[cell_name] = att_result.attention_summary

            log.info(
                f"    → {len(att_result.top_genes)} top genes, "
                f"{len(att_result.enriched_pathways)} enriched pathways"
            )
        except Exception as e:
            log.warning(f"  Attention failed for {cell_name}: {e}")
            cache[cell_name] = None

    return cache


# ---------------------------------------------------------------------------
# Single round scoring
# ---------------------------------------------------------------------------

def score_one_round(
    smiles: str,
    cell_name: str,
    gene_ctx: dict,
    pred_auc: float,
    qed: float,
    sas: float,
    attention_summary: dict,
    model: str,
    round_idx: int,
) -> dict:
    """
    Run score_molecule once and extract the fields we care about for
    consistency evaluation.
    """
    try:
        result = score_molecule(
            smiles=smiles,
            cell_name=cell_name,
            gene_ctx=gene_ctx,
            pred_auc=pred_auc,
            qed=qed,
            sas=sas,
            validity=1.0,
            model=model,
            attention_summary=attention_summary,
        )

        ch  = result.get("chemistry", {})
        nci = ch.get("nci_analysis", [])
        gex = ch.get("genotype_exploitation", {}) or {}

        # Top 2 genes from the NCI analysis (the genes the agent actually searched)
        top_genes_scored = []
        for item in nci[:2]:
            top_genes_scored.append({
                "gene":               item.get("gene"),
                "attention_score":    item.get("attention_score"),
                "in_enriched_pathway": item.get("in_enriched_pathway"),
                "known_hits":         item.get("known_hits", [])[:3],
                "nci_overlap":        item.get("candidate_nci_overlap"),
                "nci_gaps":           item.get("candidate_nci_gaps"),
            })

        return {
            "round":               round_idx,
            "success":             True,
            "error":               result.get("error"),

            # BiochemistryAgent scores
            "final_score":         result.get("final_score"),
            "confidence":          result.get("confidence"),
            "nci_score":           result.get("nci_score"),
            "descriptor_score":    result.get("descriptor_score"),
            "recommendation":      result.get("recommendation"),
            "admet_flags":         result.get("admet_flags", []),
            "summary":             result.get("summary", ""),
            "key_factors":         result.get("key_factors", []),
            "concerns":            result.get("concerns", []),

            # ChemistryAgent evidence
            "attention_grounded":          ch.get("attention_grounded", False),
            "nci_similarity":              ch.get("overall_nci_similarity"),
            "target_genes_with_docking":   ch.get("target_genes_with_docking_data", []),
            "top_2_genes":                 top_genes_scored,
            "genotype_mechanism":          gex.get("mechanism"),
            "genotype_key_gene":           gex.get("key_gene"),
            "known_therapy_analogy":       gex.get("known_therapy_analogy"),
        }

    except Exception as e:
        log.warning(f"    Round {round_idx} failed: {e}")
        return {
            "round":        round_idx,
            "success":      False,
            "error":        str(e),
            "final_score":  None,
            "confidence":   None,
            "nci_score":    None,
            "descriptor_score": None,
            "recommendation": None,
        }


# ---------------------------------------------------------------------------
# Consistency metrics across rounds
# ---------------------------------------------------------------------------

def compute_consistency(rounds: list) -> dict:
    """
    Compute consistency statistics across N rounds for one mol×cell pair.
    """
    successful = [r for r in rounds if r.get("success") and r.get("final_score") is not None]
    n_success  = len(successful)

    if n_success == 0:
        return {"n_successful_rounds": 0, "all_failed": True}

    def stats(key):
        vals = [r[key] for r in successful if r.get(key) is not None]
        if not vals:
            return {}
        return {
            "mean":  round(float(np.mean(vals)), 4),
            "std":   round(float(np.std(vals)),  4),
            "min":   round(float(np.min(vals)),  4),
            "max":   round(float(np.max(vals)),  4),
            "range": round(float(np.max(vals) - np.min(vals)), 4),
        }

    # Modal recommendation
    recs = [r["recommendation"] for r in successful if r.get("recommendation")]
    try:
        modal_rec = mode(recs)
    except StatisticsError:
        modal_rec = recs[0] if recs else None

    # Gene stability — are the same top-2 genes returned each round?
    all_gene_pairs = []
    for r in successful:
        genes = tuple(sorted(g["gene"] for g in r.get("top_2_genes", []) if g.get("gene")))
        all_gene_pairs.append(genes)
    unique_gene_pairs = set(all_gene_pairs)
    gene_stable = len(unique_gene_pairs) == 1

    # Most common gene pair
    from collections import Counter
    gene_pair_counter = Counter(all_gene_pairs)
    modal_gene_pair   = list(gene_pair_counter.most_common(1)[0][0]) if gene_pair_counter else []

    # NCI similarity stability
    nci_sims = [r.get("nci_similarity") for r in successful if r.get("nci_similarity")]
    try:
        modal_nci_sim = mode(nci_sims)
    except StatisticsError:
        modal_nci_sim = nci_sims[0] if nci_sims else None

    # Mechanism stability
    mechs = [r.get("genotype_mechanism") for r in successful if r.get("genotype_mechanism")]
    try:
        modal_mechanism = mode(mechs)
    except StatisticsError:
        modal_mechanism = mechs[0] if mechs else None

    return {
        "n_successful_rounds":   n_success,
        "n_failed_rounds":       len(rounds) - n_success,
        "all_failed":            False,

        "final_score":           stats("final_score"),
        "confidence":            stats("confidence"),
        "nci_score":             stats("nci_score"),
        "descriptor_score":      stats("descriptor_score"),

        "modal_recommendation":  modal_rec,
        "recommendation_stable": len(set(recs)) == 1,

        "gene_stable":           gene_stable,
        "modal_gene_pair":       modal_gene_pair,
        "n_unique_gene_pairs":   len(unique_gene_pairs),

        "modal_nci_similarity":  modal_nci_sim,
        "modal_mechanism":       modal_mechanism,
    }


# ---------------------------------------------------------------------------
# Summary CSV row builder
# ---------------------------------------------------------------------------

def _get_gene_meta(pair_result: dict, gene_idx: int, field: str):
    """Helper to safely extract per-gene literature metadata."""
    lit = pair_result.get("literature_retrieval", {})
    per_gene = lit.get("per_gene", {})
    genes = list(per_gene.keys())
    if gene_idx >= len(genes):
        return None
    return per_gene[genes[gene_idx]].get(field)


def build_summary_row(pair_result: dict) -> dict:
    """Flatten a pair result into one CSV row."""
    c = pair_result["consistency"]
    fs = c.get("final_score", {})
    cf = c.get("confidence", {})
    ns = c.get("nci_score", {})
    ds = c.get("descriptor_score", {})

    mol_ctx = pair_result["mol_context"]
    genes   = c.get("modal_gene_pair", [])

    return {
        "smiles":                 pair_result["smiles"],
        "cell_name":              pair_result["cell_name"],
        "auc_label":              mol_ctx.get("auc_label"),
        "pred_auc":               mol_ctx.get("pred_auc"),
        "qed":                    mol_ctx.get("qed"),
        "sas":                    mol_ctx.get("sas"),
        "tissue":                 pair_result["cell_name"].split("_", 1)[-1] if "_" in pair_result["cell_name"] else "",
        "attention_grounded":     pair_result.get("attention_grounded", False),
        "n_rounds":               pair_result["n_rounds"],
        "n_successful":           c.get("n_successful_rounds", 0),

        # Score consistency
        "final_score_mean":       fs.get("mean"),
        "final_score_std":        fs.get("std"),
        "final_score_range":      fs.get("range"),
        "confidence_mean":        cf.get("mean"),
        "confidence_std":         cf.get("std"),
        "nci_score_mean":         ns.get("mean"),
        "nci_score_std":          ns.get("std"),
        "descriptor_score_mean":  ds.get("mean"),
        "descriptor_score_std":   ds.get("std"),

        # Stability flags
        "recommendation":         c.get("modal_recommendation"),
        "recommendation_stable":  c.get("recommendation_stable"),
        "gene_stable":            c.get("gene_stable"),
        "modal_gene_1":           genes[0] if len(genes) > 0 else None,
        "modal_gene_2":           genes[1] if len(genes) > 1 else None,
        "n_unique_gene_pairs":    c.get("n_unique_gene_pairs"),
        "modal_nci_similarity":   c.get("modal_nci_similarity"),
        "modal_mechanism":        c.get("modal_mechanism"),

        # Literature retrieval metadata
        "lit_web_search_used":    bool(pair_result.get("literature_retrieval")),
        "lit_total_papers":       pair_result.get("literature_retrieval", {}).get("total_papers", 0),
        "lit_total_fulltext":     pair_result.get("literature_retrieval", {}).get("total_fulltext", 0),
        "lit_total_abstract":     pair_result.get("literature_retrieval", {}).get("total_abstract_only", 0),
        "lit_genes_searched":     "|".join(pair_result.get("literature_retrieval", {}).get("genes_searched", [])),
        "lit_gene1_source":       _get_gene_meta(pair_result, 0, "source"),
        "lit_gene1_n_papers":     _get_gene_meta(pair_result, 0, "n_papers"),
        "lit_gene1_query":        _get_gene_meta(pair_result, 0, "query_used"),
        "lit_gene2_source":       _get_gene_meta(pair_result, 1, "source"),
        "lit_gene2_n_papers":     _get_gene_meta(pair_result, 1, "n_papers"),
        "lit_gene2_query":        _get_gene_meta(pair_result, 1, "query_used"),
    }


# ---------------------------------------------------------------------------
# Training set sampler
# ---------------------------------------------------------------------------

def sample_train_pairs(
    cell_data_path: str,
    drug2smi_path: str,
    n_pairs: int,
    seed: int,
) -> pd.DataFrame:
    """
    Sample n_pairs (smiles, cell_name, auc_label) rows from the training set.
    Training set = all rows whose cell_name is NOT in EV2 or EV3.
    Filters to valid SMILES only.
    """
    nci_data = pd.read_csv(cell_data_path).dropna()
    drug2smi = pd.read_csv(drug2smi_path)

    # Keep only training cells
    train_data = nci_data[~nci_data["ccle_name"].isin(HELD_OUT)].copy()

    # Merge with SMILES
    smi_col = [c for c in drug2smi.columns if "smiles" in c.lower() or "smi" in c.lower()]
    id_col  = [c for c in drug2smi.columns if c not in smi_col]
    if not smi_col:
        raise ValueError(f"No SMILES column found in drug2smi. Columns: {drug2smi.columns.tolist()}")

    drug2smi = drug2smi.rename(columns={smi_col[0]: "smiles"})
    drug_id_col = [c for c in nci_data.columns if c not in ("ccle_name", "auc_label")][0]
    merged = train_data.merge(drug2smi, on=drug_id_col, how="inner")
    merged = merged.dropna(subset=["smiles"])

    # Filter to valid SMILES
    valid_mask = merged["smiles"].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
    merged     = merged[valid_mask].reset_index(drop=True)

    log.info(
        f"Training pool: {len(merged)} valid mol×cell pairs "
        f"across {merged['ccle_name'].nunique()} cell lines"
    )

    # Sample
    rng    = random.Random(seed)
    n      = min(n_pairs, len(merged))
    sample = merged.sample(n=n, random_state=seed).reset_index(drop=True)
    return sample[["smiles", "ccle_name", "auc_label"]]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_consistency_eval(args):
    import llm_scorer as _llm_scorer_module
    _llm_scorer_module.USE_WEB_SEARCH = args.web_search
    if args.web_search:
        log.info("Literature search ENABLED (PubMed + Semantic Scholar)")
    else:
        log.info("Literature search DISABLED — using LLM training knowledge only")

    device = args.device if torch.cuda.is_available() else "cpu"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("ANTHROPIC_API_KEY not set. Exiting.")
        sys.exit(1)

    # ── Load data ──────────────────────────────────────────────────────────
    log.info("Loading genotype data...")
    cell2mut   = pd.read_csv(args.mut_data, index_col=0).rename(columns={"index": "ccle_name"})
    cell2cna   = pd.read_csv(args.cna_data, index_col=0).rename(columns={"index": "ccle_name"})
    cell2cnd   = pd.read_csv(args.cnd_data, index_col=0).rename(columns={"index": "ccle_name"})
    gene_names = load_gene_names(args.mut_data)

    # ── Sample training pairs ──────────────────────────────────────────────
    log.info("Sampling training pairs...")
    pairs_df = sample_train_pairs(
        args.cell_data, args.drug2smi, args.n_pairs, args.seed
    )
    log.info(f"Sampled {len(pairs_df)} pairs for evaluation")

    # ── Load diffusion model + attention extractor ─────────────────────────
    log.info("Loading diffusion model...")
    diff_model = load_diffusion_model(args.diff_ckpt, device, args.cfgw)
    extractor  = AttentionExtractor(
        diff_model=diff_model,
        gene_names=gene_names,
        nest_adj_path=args.nest_adj,
    )

    # ── Pre-compute attention for all unique cell lines ────────────────────
    unique_cells = pairs_df["ccle_name"].unique().tolist()
    log.info(f"Pre-computing attention for {len(unique_cells)} cell lines...")
    attention_cache = build_attention_cache(
        unique_cells, extractor, cell2mut, cell2cna, cell2cnd, device
    )

    # ── Main evaluation loop ───────────────────────────────────────────────
    all_results  = []
    failed_pairs = []

    for idx, row in pairs_df.iterrows():
        smiles     = str(row["smiles"])
        cell_name  = str(row["ccle_name"])
        auc_label  = int(row["auc_label"])

        log.info(f"\n[{idx+1}/{len(pairs_df)}] {cell_name} | AUC class={auc_label} | {smiles[:55]}...")

        # Molecular properties
        mol     = Chem.MolFromSmiles(smiles)
        mol_qed = float(QED.qed(mol)) if mol else 0.0
        mol_sas = float(sascorer.calculateScore(mol)) if mol else 10.0

        # Gene context
        gene_ctx = get_cell_gene_context(
            cell_name, cell2mut, cell2cna, cell2cnd, gene_names, top_k=15
        )

        # Attention summary (pre-computed, None if extraction failed)
        att_summary = attention_cache.get(cell_name)

        # ── Literature retrieval (once per pair, shared across all rounds) ─
        # Pre-fetch outside the rounds loop — same papers used for all rounds,
        # which is what we want: consistency measures LLM variance, not
        # search variance.
        lit_text     = ""
        lit_metadata = {}
        if args.web_search:
            if att_summary and att_summary.get("top_genes"):
                search_genes = att_summary["top_genes"][:2]
            else:
                search_genes = gene_ctx.get("mutated_genes", [])[:2]
            try:
                lit_text, lit_metadata = fetch_literature_with_metadata(
                    search_genes, model=args.model
                )
                log.info(
                    f"  Literature: {lit_metadata.get('total_papers', 0)} paper(s) "
                    f"({lit_metadata.get('total_fulltext', 0)} full-text, "
                    f"{lit_metadata.get('total_abstract_only', 0)} abstract-only)"
                )
            except Exception as e:
                log.warning(f"  Literature fetch failed: {e}")

        # ── Run N rounds ──────────────────────────────────────────────────
        rounds = []
        for r in range(args.n_rounds):
            log.info(f"  Round {r+1}/{args.n_rounds}...")
            round_result = score_one_round(
                smiles=smiles,
                cell_name=cell_name,
                gene_ctx=gene_ctx,
                pred_auc=float(auc_label),   # use AUC label as proxy pred_auc
                qed=mol_qed,
                sas=mol_sas,
                attention_summary=att_summary,
                model=args.model,
                round_idx=r + 1,
            )
            rounds.append(round_result)
            log.info(
                f"    final_score={round_result.get('final_score','?')}  "
                f"confidence={round_result.get('confidence','?')}  "
                f"recommendation={round_result.get('recommendation','?')}"
            )

        # ── Compute consistency ────────────────────────────────────────────
        consistency = compute_consistency(rounds)

        pair_result = {
            "pair_id":          idx,
            "smiles":           smiles,
            "cell_name":        cell_name,
            "n_rounds":         args.n_rounds,
            "attention_grounded": att_summary is not None,
            "literature_retrieval": lit_metadata,
            "mol_context": {
                "auc_label":    auc_label,
                "pred_auc":     float(auc_label),
                "qed":          round(mol_qed, 4),
                "sas":          round(mol_sas, 4),
                "mutated_genes":   gene_ctx.get("mutated_genes", [])[:8],
                "amplified_genes": gene_ctx.get("amplified_genes", []),
                "deleted_genes":   gene_ctx.get("deleted_genes", []),
            },
            "attention_context": {
                "top_genes":         att_summary.get("top_genes", [])[:10] if att_summary else [],
                "enriched_pathways": [
                    {"pathway": p["pathway"], "p_value": p["p_value"], "enrichment": p["enrichment"]}
                    for p in (att_summary.get("enriched_pathways", [])[:3] if att_summary else [])
                ],
            },
            "rounds":           rounds,
            "consistency":      consistency,
        }

        all_results.append(pair_result)

        # Check if all rounds failed
        if consistency.get("all_failed"):
            failed_pairs.append({"pair_id": idx, "smiles": smiles, "cell_name": cell_name})
            log.warning(f"  All {args.n_rounds} rounds failed for this pair")
        else:
            fs = consistency.get("final_score", {})
            log.info(
                f"  Consistency: final_score={fs.get('mean','?')}±{fs.get('std','?')} "
                f"range={fs.get('range','?')} "
                f"recommendation_stable={consistency.get('recommendation_stable')} "
                f"gene_stable={consistency.get('gene_stable')}"
            )

        # ── Save incrementally after each pair ────────────────────────────
        _save_results(all_results, failed_pairs, args.out_dir)

    # ── Final summary ──────────────────────────────────────────────────────
    _print_summary(all_results)
    log.info(f"\nAll results saved to {args.out_dir}/")


def _save_results(all_results: list, failed_pairs: list, out_dir: str):
    """Save all outputs — called after every pair so partial results survive crashes."""

    # Full JSON
    json_path = os.path.join(out_dir, "consistency_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "n_pairs":      len(all_results),
            "results":      all_results,
        }, f, indent=2, default=str)

    # Summary CSV — one row per pair
    summary_rows = [build_summary_row(r) for r in all_results]
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(out_dir, "consistency_summary.csv"), index=False
    )

    # Failed pairs
    if failed_pairs:
        with open(os.path.join(out_dir, "failed_pairs.json"), "w") as f:
            json.dump(failed_pairs, f, indent=2)


def _print_summary(all_results: list):
    successful = [r for r in all_results
                  if not r["consistency"].get("all_failed")]
    if not successful:
        log.info("No successful evaluations to summarize.")
        return

    final_scores = [
        r["consistency"]["final_score"]["mean"]
        for r in successful
        if r["consistency"].get("final_score", {}).get("mean") is not None
    ]
    final_stds = [
        r["consistency"]["final_score"]["std"]
        for r in successful
        if r["consistency"].get("final_score", {}).get("std") is not None
    ]
    rec_stable = [r["consistency"].get("recommendation_stable") for r in successful]
    gene_stable = [r["consistency"].get("gene_stable") for r in successful]

    log.info("\n" + "="*65)
    log.info("CONSISTENCY EVALUATION SUMMARY")
    log.info("="*65)
    log.info(f"  Pairs evaluated:              {len(all_results)}")
    log.info(f"  Pairs with ≥1 success:        {len(successful)}")
    log.info(f"  Pairs fully failed:           {len(all_results) - len(successful)}")
    if final_scores:
        log.info(f"  Mean final_score (avg):       {np.mean(final_scores):.3f}")
        log.info(f"  Mean intra-pair std:          {np.mean(final_stds):.4f}")
        log.info(f"  Max intra-pair std:           {np.max(final_stds):.4f}")
    if rec_stable:
        pct = 100 * sum(rec_stable) / len(rec_stable)
        log.info(f"  Recommendation stable:        {pct:.0f}% of pairs")
    if gene_stable:
        pct = 100 * sum(gene_stable) / len(gene_stable)
        log.info(f"  Gene pair stable:             {pct:.0f}% of pairs")
    log.info("="*65)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM scorer consistency evaluation on training set"
    )
    parser.add_argument("--diff_ckpt",  required=True)
    parser.add_argument("--cell_data",  required=True,
                        help="DC_drug_response.csv")
    parser.add_argument("--drug2smi",   required=True,
                        help="DC_drug2smi.csv")
    parser.add_argument("--mut_data",   required=True)
    parser.add_argument("--cna_data",   required=True)
    parser.add_argument("--cnd_data",   required=True)
    parser.add_argument("--out_dir",    default="./consistency_eval")
    parser.add_argument("--nest_adj",   default="./data/NeST_neighbor_adj.npy")

    parser.add_argument("--n_rounds",   type=int, default=5,
                        help="Number of scoring rounds per mol×cell pair")
    parser.add_argument("--n_pairs",    type=int, default=50,
                        help="Number of mol×cell pairs to evaluate")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--cfgw",       type=float, default=7.0)
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--web_search", action="store_true", default=False,
                        help="Enable PubMed/Semantic Scholar literature search "
                             "for each mol×cell pair.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_consistency_eval(args)