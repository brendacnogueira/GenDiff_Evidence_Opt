"""
attention_analysis.py
---------------------
Extracts and analyzes attention coefficients from G2D-Diff's condition encoder
transformer blocks to identify genes and pathways the model focuses on.

Based on the interpretability approach described in the G2D-Diff paper:
  - Attention coefficients are extracted from three transformer blocks:
      T_neigh (NeST sibling propagation)
      T_whole (whole-gene propagation)
      T_reout (response-class output)
  - Top-attended genes are identified per attention head
  - Gene set enrichment against NeST ontology identifies enriched pathways
  - This gives a data-driven signal of which pathways the model believes
    are relevant for this cell line and condition

This output feeds into two agents in llm_scorer.py:
  AlignmentAgent  — uses the enriched pathways as primary evidence
  PathwayAgent    — uses enriched pathways to reason about molecule-target fit

Usage (standalone):
    from attention_analysis import AttentionExtractor, enrich_pathways

    extractor = AttentionExtractor(
        diff_model=diff_model,     # Diffusion model with get_att=True
        gene_names=gene_names,     # list of 718 gene names
        nest_path="./data/NeST_neighbor_adj.npy",
        nest_gene_sets=nest_gene_sets,  # dict: {pathway: [gene, ...]}
    )

    result = extractor.extract(batch, condition_class=0)
    # result.top_genes         — list of top-attended genes
    # result.enriched_pathways — list of (pathway, p_value, genes) tuples
    # result.attention_summary — dict ready to pass to agents

Integration:
    Pass result.attention_summary into AlignmentAgent.user_prompt()
    and PathwayAgent.user_prompt() to ground them in model evidence
    rather than LLM prior knowledge alone.
"""

import os
import sys
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class for attention results
# ---------------------------------------------------------------------------

@dataclass
class AttentionResult:
    cell_name:          str
    condition_class:    int
    top_genes:          list         # top-attended gene names
    top_gene_scores:    list         # corresponding attention scores
    enriched_pathways:  list         # list of dicts: {pathway, p_value, overlap_genes, enrichment}
    layer_attentions:   dict         # raw per-layer attention (optional, for debugging)
    attention_summary:  dict         # compact dict ready for LLM agent prompts


# ---------------------------------------------------------------------------
# Condition encoder with attention extraction
# ---------------------------------------------------------------------------

class ConditionEncoderWithAttention(torch.nn.Module):
    """
    Wrapper around G2D-Diff's Condition_Encoder that enables attention extraction.

    The diffusion model instantiates Condition_Encoder with get_att=False, which
    discards attention weights. This wrapper creates a second instance with
    get_att=True sharing the same weights, then parses the 4-tuple return value:
        (out1, cond_embedding, att_list, out4)

    att_list contains attention tensors from the three transformer blocks
    [T_neigh, T_whole, T_reout], each (B, n_heads, n_tokens, n_tokens).
    """

    def __init__(self, diff_model):
        """
        diff_model: loaded Diffusion model instance.
        """
        super().__init__()

        orig_enc = diff_model.model.condition_encoder
        device   = next(orig_enc.parameters()).device

        from src.g2d_diff_ce import Condition_Encoder

        self.encoder = Condition_Encoder(
            num_of_genotypes=3,
            num_of_dcls=5,
            device=device,
            get_att=True,
        ).to(device).float()

        self.encoder.load_state_dict(orig_enc.state_dict(), strict=False)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        self._device = device
        log.info("ConditionEncoderWithAttention: created get_att=True encoder copy")

    @torch.no_grad()
    def forward_with_attention(self, batch: dict) -> dict:
        """
        Call Condition_Encoder(get_att=True) and parse its 4-tuple return.

        Returns:
            {
              "t_neigh": np.ndarray or None,
              "t_whole": np.ndarray or None,
              "t_reout": np.ndarray or None,
              "cond":    np.ndarray (B, emb_dim)
            }
        """
        result = self.encoder(batch)

        # Condition_Encoder(get_att=True) returns:
        #   (_fg, out, attention, whole_att_list)
        #   [0] _fg          : full output sequence (B, 719, 128)
        #   [1] out          : CLS embedding used as condition (B, 128)
        #   [2] attention    : last-layer CLS row, already extracted
        #   [3] whole_att_list: list of 3 full att tensors [T_neigh, T_whole, T_reout]
        #                       each (B, n_heads, n_tokens, n_tokens), already on CPU
        if isinstance(result, (tuple, list)) and len(result) == 4:
            _fg, out, attention, whole_att_list = result
        else:
            raise ValueError(
                f"Expected 4-tuple from Condition_Encoder(get_att=True), "
                f"got {type(result)} len={len(result) if hasattr(result,'__len__') else '?'}"
            )

        def _to_np(t):
            if t is None:
                return None
            if isinstance(t, torch.Tensor):
                return t.detach().cpu().numpy()
            return np.array(t)

        # Use out (position 1) as the condition embedding
        cond_np  = out.detach().cpu().numpy() if isinstance(out, torch.Tensor) \
                   else np.array(out)
        # Use whole_att_list (position 3) — all three layers
        att_list = whole_att_list if isinstance(whole_att_list, list) else []

        return {
            "t_neigh": _to_np(att_list[0]) if len(att_list) > 0 else None,
            "t_whole": _to_np(att_list[1]) if len(att_list) > 1 else None,
            "t_reout": _to_np(att_list[2]) if len(att_list) > 2 else None,
            "cond":    cond_np,
        }


# ---------------------------------------------------------------------------
# Gene-level attention aggregation
# ---------------------------------------------------------------------------

def aggregate_gene_attention(
    attention_dict: dict,
    n_genes: int,
    top_pct: float = 0.10,
    adj: np.ndarray = None,
    altered_mask: np.ndarray = None,
    hybrid_alpha: float = 0.6,
) -> tuple:
    """
    Aggregate CLS-token attention from T_reout into per-gene importance scores.

    Follows the paper (Supp. Fig 10): attention scores above the uniform
    baseline (1/719 ≈ 0.0014) are considered meaningful.

    Optionally applies a hybrid re-ranking that blends raw attention with a
    NeST-graph prior: for each gene, the prior counts how many of the cell
    line's actually-altered genes are its NeST neighbors.  This suppresses
    spurious high-attention genes that are unconnected to the input genotype
    (e.g. TSHR, GABRA6) and promotes genes like HDAC1, AKT1 that are not
    directly mutated but sit inside the same NeST systems as mutated genes.

    Args:
        attention_dict : output of forward_with_attention()
        n_genes        : 718
        top_pct        : fraction of genes to return (default 10%)
        adj            : (n_genes, n_genes) NeST adjacency as float/bool array.
                         Pass None to skip hybrid re-ranking.
        altered_mask   : (n_genes,) bool array — True for genes altered in this
                         cell line (MUT | CNA | CND).  Required when adj is given.
        hybrid_alpha   : weight for attention signal; (1-alpha) goes to graph
                         prior.  Default 0.6 / 0.4 split.

    Returns:
        gene_scores  (np.ndarray, shape n_genes): final per-gene score used for ranking
        top_indices  (np.ndarray): indices of top genes, sorted descending
    """
    uniform_value = 1.0 / (n_genes + 1)

    # ── 1. Use only T_reout (final layer) ────────────────────────────────────
    # T_neigh is structurally masked (sparse), T_whole is intermediate.
    # T_reout is where the fully-propagated signal reaches the CLS token.
    att = attention_dict.get("t_reout")
    if att is None:
        log.warning("t_reout not found — falling back to t_whole")
        att = attention_dict.get("t_whole")
    if att is None:
        log.warning("No attention layer found — returning uniform scores")
        return np.ones(n_genes) / n_genes, np.arange(n_genes)

    att = np.array(att)
    if att.ndim == 3:          # squeezed batch dim (B=1)
        att = att[np.newaxis, ...]

    # CLS token is at position -1; read its attention toward each gene
    att_to_genes = att[0, :, -1, :n_genes]   # (n_heads, n_genes)

    # Average over attention heads (paper approach)
    raw_att = att_to_genes.mean(axis=0)       # (n_genes,)

    # ── 2. Optional hybrid re-ranking ────────────────────────────────────────
    if adj is not None and altered_mask is not None:
        # How many of the cell line's altered genes are NeST neighbors of each gene?
        neighbor_support = adj @ altered_mask.astype(float)   # (n_genes,)

        # Normalize both signals to [0, 1]
        att_norm = raw_att / (raw_att.max() + 1e-8)
        sup_norm = neighbor_support / (neighbor_support.max() + 1e-8)

        gene_scores = hybrid_alpha * att_norm + (1.0 - hybrid_alpha) * sup_norm
        log.debug("Hybrid re-ranking applied (alpha=%.2f)", hybrid_alpha)
    else:
        gene_scores = raw_att

    # ── 3. Select top genes above uniform baseline ────────────────────────────
    above_uniform = gene_scores > uniform_value
    top_indices   = np.where(above_uniform)[0]
    top_indices   = top_indices[np.argsort(gene_scores[top_indices])[::-1]]

    # Trim to top_pct if too many pass the threshold
    n_top       = max(1, int(n_genes * top_pct))
    top_indices = top_indices[:n_top]

    return gene_scores, top_indices


# ---------------------------------------------------------------------------
# NeST gene set enrichment
# ---------------------------------------------------------------------------

def load_nest_gene_sets(nest_adj_path: str, gene_names: list) -> dict:
    """
    Build NeST pathway gene sets from the adjacency matrix.

    The NeST adjacency matrix encodes which genes are neighbors (siblings)
    in the NeST hierarchy. We use connected components as approximate
    pathway/system groupings.

    For a proper NeST ontology you would load the full term-gene mapping.
    This is a fallback that uses the adjacency structure directly.

    Returns:
        dict: {system_id_str: [gene_name, ...]}
    """
    adj = np.load(nest_adj_path)   # (n_genes, n_genes) boolean

    # Simple connected-component extraction as pathway proxies
    visited  = set()
    gene_sets = {}
    set_id    = 0

    for start in range(len(gene_names)):
        if start in visited:
            continue
        component = []
        stack     = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            neighbors = np.where(adj[node])[0]
            stack.extend(n for n in neighbors if n not in visited)

        if len(component) >= 3:   # ignore trivial singletons/pairs
            gene_sets[f"NeST_system_{set_id}"] = [gene_names[i] for i in component]
            set_id += 1

    log.info(f"Loaded {len(gene_sets)} NeST gene sets from {nest_adj_path}")
    return gene_sets


def load_nest_gene_sets_from_file(nest_sets_path: str) -> dict:
    """
    Load pre-computed NeST pathway gene sets from a JSON file.
    Format: {"pathway_name": ["GENE1", "GENE2", ...], ...}
    """
    with open(nest_sets_path) as f:
        return json.load(f)


def run_gene_set_enrichment(
    top_gene_names: list,
    all_gene_names: list,
    gene_sets: dict,
    n_top_pathways: int = 10,
    p_threshold: float = 0.05,
) -> list:
    """
    Fisher's exact test enrichment of top-attended genes against NeST gene sets.

    For each gene set, tests whether the top genes are enriched using a
    hypergeometric / Fisher's exact test (same statistical framework as GSEA).

    Returns list of dicts sorted by p-value:
        [{pathway, p_value, overlap_count, overlap_genes, enrichment_ratio}, ...]
    """
    top_set    = set(top_gene_names)
    N          = len(all_gene_names)        # total genes
    K          = len(top_set)               # total top genes
    results    = []

    for pathway, pathway_genes in gene_sets.items():
        pathway_set  = set(pathway_genes) & set(all_gene_names)
        overlap      = top_set & pathway_set
        n_pathway    = len(pathway_set)     # size of gene set
        n_overlap    = len(overlap)         # overlap

        if n_pathway < 3 or n_overlap == 0:
            continue

        # Fisher's exact test (2×2 contingency table)
        # [[overlap, K - overlap], [n_pathway - overlap, N - K - n_pathway + overlap]]
        a = n_overlap
        b = K - n_overlap
        c = n_pathway - n_overlap
        d = N - K - n_pathway + n_overlap

        if a < 0 or b < 0 or c < 0 or d < 0:
            continue

        _, p_value = stats.fisher_exact([[a, b], [c, d]], alternative="greater")

        # Enrichment ratio: observed / expected overlap
        expected    = K * n_pathway / N
        enrichment  = n_overlap / expected if expected > 0 else 0.0

        results.append({
            "pathway":         pathway,
            "p_value":         float(p_value),
            "overlap_count":   n_overlap,
            "overlap_genes":   sorted(overlap),
            "pathway_size":    n_pathway,
            "enrichment":      round(enrichment, 2),
        })

    # Sort by p-value, return top pathways below threshold
    results.sort(key=lambda x: x["p_value"])
    significant = [r for r in results if r["p_value"] < p_threshold]

    return significant[:n_top_pathways]


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class AttentionExtractor:
    """
    Full pipeline: batch → attention extraction → top genes → pathway enrichment.

    Usage:
        extractor = AttentionExtractor(
            diff_model=diff_model,
            gene_names=gene_names,
            nest_adj_path="./data/NeST_neighbor_adj.npy",
        )

        result = extractor.extract(batch)
        # result.top_genes          — ["TP53", "PTEN", ...]
        # result.enriched_pathways  — [{pathway, p_value, ...}, ...]
        # result.attention_summary  — dict for LLM agent prompts
    """

    def __init__(
        self,
        diff_model,
        gene_names: list,
        nest_adj_path: str = "./data/NeST_neighbor_adj.npy",
        nest_sets_path: str = None,   # JSON file with pathway→genes mapping
        top_pct: float = 0.10,
        n_top_pathways: int = 10,
        p_threshold: float = 0.05,
        hybrid_alpha: float = 0.6,    # weight for attention vs. graph prior
    ):
        self.gene_names     = gene_names
        self.n_genes        = len(gene_names)
        self.top_pct        = top_pct
        self.n_top_pathways = n_top_pathways
        self.p_threshold    = p_threshold
        self.hybrid_alpha   = hybrid_alpha

        # Pass full diff_model so wrapper can re-instantiate with get_att=True
        self.att_encoder = ConditionEncoderWithAttention(diff_model)

        # Load and store the NeST adjacency matrix for hybrid re-ranking
        if os.path.exists(nest_adj_path):
            self.adj = np.load(nest_adj_path).astype(float)  # (n_genes, n_genes)
            log.info(f"NeST adjacency loaded for hybrid re-ranking: {self.adj.shape}")
        else:
            self.adj = None
            log.warning(f"NeST adjacency not found at {nest_adj_path} — hybrid re-ranking disabled")

        # Load NeST gene sets (for pathway enrichment)
        if nest_sets_path and os.path.exists(nest_sets_path):
            self.gene_sets = load_nest_gene_sets_from_file(nest_sets_path)
            log.info(f"Loaded NeST gene sets from {nest_sets_path}: {len(self.gene_sets)} pathways")
        elif self.adj is not None:
            self.gene_sets = load_nest_gene_sets(nest_adj_path, gene_names)
        else:
            log.warning(f"NeST file not found at {nest_adj_path} — pathway enrichment disabled")
            self.gene_sets = {}

    def extract(
        self,
        batch: dict,
        cell_name: str = None,
        condition_class: int = 0,
        altered_mask: np.ndarray = None,
    ) -> AttentionResult:
        """
        Extract attention for a batch and compute top genes + enriched pathways.

        Args:
            batch          : standard G2D-Diff batch dict (on device)
            cell_name      : for labeling the result
            condition_class: AUC class (0 = most sensitive)
            altered_mask   : (n_genes,) bool array — True for genes altered in
                             this cell line (MUT | CNA | CND).  When provided,
                             enables hybrid re-ranking that promotes NeST
                             neighbors of mutated genes (e.g. HDAC1, AKT1 for
                             HS578T) and suppresses unrelated high-attention
                             genes (e.g. TSHR, GABRA6).
                             Build it with:
                               altered_mask = (mut_vec | cna_vec | cnd_vec)
                             where each vec is a (n_genes,) bool numpy array
                             from the cell2mut / cell2cna / cell2cnd CSVs.

        Returns:
            AttentionResult with top genes and enriched pathways
        """
        # Run forward with attention collection
        att_dict = self.att_encoder.forward_with_attention(batch)

        # Aggregate to per-gene scores, with optional hybrid re-ranking
        gene_scores, top_indices = aggregate_gene_attention(
            att_dict,
            self.n_genes,
            self.top_pct,
            adj=self.adj,
            altered_mask=altered_mask,
            hybrid_alpha=self.hybrid_alpha,
        )

        top_genes       = [self.gene_names[i] for i in top_indices]
        top_gene_scores = [float(gene_scores[i]) for i in top_indices]

        # Pathway enrichment
        enriched = []
        if self.gene_sets:
            enriched = run_gene_set_enrichment(
                top_gene_names=top_genes,
                all_gene_names=self.gene_names,
                gene_sets=self.gene_sets,
                n_top_pathways=self.n_top_pathways,
                p_threshold=self.p_threshold,
            )

        # Build compact summary for LLM agents
        attention_summary = self._build_summary(
            cell_name, condition_class, top_genes, gene_scores, enriched
        )

        return AttentionResult(
            cell_name=cell_name or "unknown",
            condition_class=condition_class,
            top_genes=top_genes,
            top_gene_scores=top_gene_scores,
            enriched_pathways=enriched,
            layer_attentions={k: v for k, v in att_dict.items() if k != "cond"},
            attention_summary=attention_summary,
        )

    def _build_summary(
        self,
        cell_name: str,
        condition_class: int,
        top_genes: list,
        gene_scores: np.ndarray,
        enriched: list,
    ) -> dict:
        """
        Build the compact dict that gets passed into LLM agent prompts.
        """
        # Top 15 genes with scores
        top15 = [
            {"gene": g, "attention": round(float(gene_scores[self.gene_names.index(g)]), 4)}
            for g in top_genes[:15]
            if g in self.gene_names
        ]

        # Top enriched pathways
        top_pathways = [
            {
                "pathway":       e["pathway"],
                "p_value":       round(e["p_value"], 4),
                "enrichment":    e["enrichment"],
                "overlap_genes": e["overlap_genes"][:8],   # show top 8 genes
            }
            for e in enriched[:5]
        ]

        return {
            "cell_name":         cell_name,
            "condition_class":   condition_class,
            "top_genes":         [g for g in top_genes[:20]],
            "top_genes_detail":  top15,
            "enriched_pathways": top_pathways,
            "n_top_genes":       len(top_genes),
            "n_enriched":        len(enriched),
        }


# ---------------------------------------------------------------------------
# Updated agent prompts using attention evidence
# ---------------------------------------------------------------------------
# These replace the corresponding user_prompt() methods in llm_scorer.py
# when attention data is available.

def alignment_agent_prompt_with_attention(
    smiles: str,
    cell_name: str,
    gene_ctx: dict,
    attention_summary: dict,
) -> str:
    """
    AlignmentAgent prompt grounded in model attention evidence.

    Instead of asking the LLM to guess which pathways matter, we provide
    the actual pathways the model's attention highlighted, plus the
    top-attended genes. The LLM's role is to reason about whether the
    molecule targets these model-identified vulnerabilities.
    """
    top_genes = ", ".join(attention_summary.get("top_genes", [])[:15]) or "none"
    pathways  = attention_summary.get("enriched_pathways", [])

    pathway_lines = []
    for p in pathways[:5]:
        pathway_lines.append(
            f"  • {p['pathway']}  "
            f"(enrichment={p['enrichment']}x, p={p['p_value']:.3e}, "
            f"genes: {', '.join(p['overlap_genes'][:5])})"
        )
    pathway_block = "\n".join(pathway_lines) if pathway_lines else "  None identified"

    return f"""Cell line: {cell_name}
Condition class: {attention_summary.get('condition_class', 0)} (0 = most sensitive)

GENOTYPE (from data):
  Mutated genes:   {', '.join(gene_ctx.get('mutated_genes', [])[:10]) or 'none'}
  Amplified genes: {', '.join(gene_ctx.get('amplified_genes', [])) or 'none'}
  Deleted genes:   {', '.join(gene_ctx.get('deleted_genes', [])) or 'none'}

MODEL ATTENTION EVIDENCE (from G2D-Diff condition encoder):
  Top-attended genes: {top_genes}

  Enriched pathways (Fisher's exact, p < 0.05):
{pathway_block}

Molecule SMILES: {smiles}

The MODEL ATTENTION EVIDENCE above shows which genes and pathways the diffusion
model actually focused on when generating molecules for this cell line and condition.
This is more reliable than guessing from genotype alone.

Does this molecule plausibly target the pathways the model highlighted?
Consider: direct target inhibition, synthetic lethality with top-attended genes,
mechanism overlap with enriched pathways.

Return ONLY valid JSON:
{{
  "alignment_score": <float 0-1>,
  "key_gene": "<most relevant attended gene>",
  "mechanism": "<synthetic_lethality | oncogene_inhibition | pathway_dependency | unknown>",
  "matched_pathway": "<most relevant enriched pathway or null>",
  "rationale": "<one sentence — reference specific attended genes or pathways>"
}}"""


def pathway_agent_prompt_with_attention(
    smiles: str,
    cell_name: str,
    gene_ctx: dict,
    desc: dict,
    attention_summary: dict,
) -> str:
    """
    PathwayAgent prompt that uses model attention to focus its reasoning.

    The attention tells us which pathways the model associated with sensitive
    responses for this cell line. The agent's job is to evaluate whether the
    molecule could inhibit those specific model-highlighted pathways.
    """
    top_pathways = attention_summary.get("enriched_pathways", [])
    pathway_names = [p["pathway"] for p in top_pathways[:3]]
    pathway_str   = ", ".join(pathway_names) if pathway_names else "unknown (no enrichment found)"

    top_genes = ", ".join(attention_summary.get("top_genes", [])[:12]) or "none"

    return f"""Cell line: {cell_name}

MODEL-IDENTIFIED PATHWAYS (from G2D-Diff attention analysis):
  Top enriched pathways: {pathway_str}
  Top-attended genes:    {top_genes}

GENOTYPE:
  Mutated: {', '.join(gene_ctx.get('mutated_genes', [])[:8]) or 'none'}
  Deleted: {', '.join(gene_ctx.get('deleted_genes', [])) or 'none'}

Molecule SMILES: {smiles}
MW={desc.get('mw','?')}, LogP={desc.get('logp','?')}, AromaticRings={desc.get('arom_rings','?')}

The model's attention mechanism identified these pathways as relevant for sensitivity
in this cell line. Evaluate whether this molecule could inhibit these specific pathways.
Do not guess new pathways — focus on the model-identified ones above.

Return ONLY valid JSON:
{{
  "score": <float 0-1>,
  "pathway": "<which of the above pathways is most likely targeted>",
  "reasoning": "<one sentence referencing the specific model-highlighted pathway>",
  "known_analogues": "<approved drug targeting same pathway, or null>"
}}"""


# ---------------------------------------------------------------------------
# Convenience: build full attention_summary string for agent prompts
# ---------------------------------------------------------------------------

def format_attention_for_prompt(attention_summary: dict) -> str:
    """
    Format AttentionResult.attention_summary as a compact string
    suitable for embedding in any agent prompt.
    """
    if not attention_summary:
        return "No attention data available."

    top = ", ".join(attention_summary.get("top_genes", [])[:12])
    pathways = attention_summary.get("enriched_pathways", [])

    lines = [f"Top-attended genes ({attention_summary.get('n_top_genes',0)} total): {top}"]
    if pathways:
        lines.append("Enriched pathways:")
        for p in pathways[:5]:
            lines.append(
                f"  {p['pathway']}  "
                f"enrichment={p['enrichment']}x  p={p['p_value']:.3e}  "
                f"genes=[{', '.join(p['overlap_genes'][:4])}]"
            )
    else:
        lines.append("No significantly enriched pathways (p < 0.05)")

    return "\n".join(lines)