"""
llm_scorer.py
-------------
Two-agent LLM scoring for G2D-Diff latent optimization.

Architecture:

    ┌─────────────────────────────────────────────────────────────┐
    │                  G2D-Diff model (biology agent)             │
    │                                                             │
    │  Provides: pred_AUC, QED, SAS, validity, MW, LogP,         │
    │  HBD, HBA, TPSA, RotBonds, AromaticRings                   │
    │  + attention analysis → top genes + enriched NeST pathways │
    │  + gene context: MUT, CNA, CND per cell line                │
    └──────────────────────────────┬──────────────────────────────┘
                                   │  model biology output
                    ┌──────────────▼──────────────┐
                    │       ChemistryAgent         │
                    │                              │
                    │  Receives model-identified   │
                    │  target genes + attention    │
                    │  scores + enriched pathways  │
                    │                              │
                    │  Searches literature for     │
                    │  AlphaFold/docking studies   │
                    │  on those specific targets   │
                    │                              │
                    │  Reports key NCIs of known   │
                    │  hit molecules vs candidate: │
                    │  pi-pi, H-bond, halogen,     │
                    │  electrostatic, hydrophobic  │
                    │                              │
                    │  NO SCORE                    │
                    └──────────────┬───────────────┘
                                   │  NCI literature report
                    ┌──────────────▼──────────────┐
                    │     BiochemistryAgent        │
                    │                              │
                    │  Receives:                   │
                    │  - ChemistryAgent NCI report │
                    │  - All model biology output  │
                    │    (AUC, QED, SAS, desc,     │
                    │     pathways, genes)         │
                    │                              │
                    │  Produces: final_score (0-1) │
                    │  for latent optimization     │
                    └──────────────────────────────┘

The G2D-Diff model IS the biology agent — its attention mechanism,
predicted AUC, and molecular descriptors replace the previous four
specialist LLM agents (PathwayAgent, ADMETAgent, AlignmentAgent,
SelectivityAgent). These are removed. The model's own output is more
reliable than LLM guesses about pathways and drug-likeness.

Integration with latent_opt_g2d.py:
    See patch instructions at the bottom of this file.

Standalone:
    python llm_scorer.py \\
        --smiles "CCc1ccc(Nc2ccnc(N)c2)cc1" \\
        --cell_name MCF7_BREAST \\
        --pred_auc 0.09 --qed 0.52 --sas 3.1 --validity 1.0 \\
        --mut_data ./data/drug_response_data/original_cell2mut.csv \\
        --cna_data ./data/drug_response_data/original_cell2cna.csv \\
        --cnd_data ./data/drug_response_data/original_cell2cnd.csv

Environment:
    ANTHROPIC_API_KEY must be set.
"""

import os
import sys
import json
import logging
import hashlib
import argparse
import time

import numpy as np
import pandas as pd
import requests

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED
from rdkit import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

log = logging.getLogger(__name__)

# Optional: format_attention_for_prompt used in ChemistryAgent
try:
    from attention_analysis import format_attention_for_prompt
    _ATTENTION_AVAILABLE = True
except ImportError:
    _ATTENTION_AVAILABLE = False

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL     = "claude-sonnet-4-6"

# Set to True to enable real-time web search in ChemistryAgent.
# Requires a plan that supports web_search_20250305 and has sufficient
# rate limits. Disable if you hit persistent 429 errors.
# Can also be toggled via --web_search / --no_web_search CLI flags.
USE_WEB_SEARCH = True


# ---------------------------------------------------------------------------
# Gene context helpers
# ---------------------------------------------------------------------------

def load_gene_names(mut_path: str) -> list:
    """Return the 718 gene column names from the mutation CSV."""
    df = pd.read_csv(mut_path, index_col=0, nrows=1)
    df = df.rename(columns={"index": "ccle_name"})
    return [c for c in df.columns if c != "ccle_name"]


def get_cell_gene_context(
    cell_name: str,
    cell2mut: pd.DataFrame,
    cell2cna: pd.DataFrame,
    cell2cnd: pd.DataFrame,
    gene_names: list,
    top_k: int = 15,
) -> dict:
    """
    Return the top mutated, amplified, and deleted genes for one cell line.
    Uses all three genotype modalities: MUT, CNA, and CND.
    CND is critical for detecting tumor suppressor deletions (TP53, PTEN, RB1).
    """
    mut_idx = cell2mut.set_index("ccle_name") if "ccle_name" in cell2mut.columns else cell2mut
    cna_idx = cell2cna.set_index("ccle_name") if "ccle_name" in cell2cna.columns else cell2cna
    cnd_idx = cell2cnd.set_index("ccle_name") if "ccle_name" in cell2cnd.columns else cell2cnd

    try:
        mut_vec = mut_idx.loc[cell_name].values.astype(float)
        cna_vec = cna_idx.loc[cell_name].values.astype(float)
        cnd_vec = cnd_idx.loc[cell_name].values.astype(float)
    except KeyError:
        return {"mutated_genes": [], "amplified_genes": [], "deleted_genes": []}

    mut_order = np.argsort(mut_vec)[::-1]
    mutated   = [gene_names[i] for i in mut_order[:top_k] if mut_vec[i] > 0]

    cna_order = np.argsort(cna_vec)[::-1]
    amplified = [gene_names[i] for i in cna_order[:top_k // 2] if cna_vec[i] > 0.5]

    # Deleted: CND (positive = deleted) + strongly negative CNA
    cnd_order     = np.argsort(cnd_vec)[::-1]
    deleted_cnd   = [gene_names[i] for i in cnd_order[:top_k // 2] if cnd_vec[i] > 0.5]
    cna_del_order = np.argsort(cna_vec)
    deleted_cna   = [gene_names[i] for i in cna_del_order[:top_k // 2] if cna_vec[i] < -0.5]
    seen, deleted = set(), []
    for g in deleted_cnd + deleted_cna:
        if g not in seen:
            seen.add(g)
            deleted.append(g)

    return {"mutated_genes": mutated, "amplified_genes": amplified, "deleted_genes": deleted}


def get_mol_descriptors(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    try:
        return {
            "mw":         round(Descriptors.MolWt(mol), 1),
            "logp":       round(Descriptors.MolLogP(mol), 2),
            "hbd":        rdMolDescriptors.CalcNumHBD(mol),
            "hba":        rdMolDescriptors.CalcNumHBA(mol),
            "tpsa":       round(Descriptors.TPSA(mol), 1),
            "rotbonds":   rdMolDescriptors.CalcNumRotatableBonds(mol),
            "arom_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "rings":      rdMolDescriptors.CalcNumRings(mol),
            "qed":        round(QED.qed(mol), 3),
            "sas":        round(sascorer.calculateScore(mol), 2),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Low-level API caller
# ---------------------------------------------------------------------------

def _call_api(system: str, user: str, model: str = DEFAULT_MODEL, max_tokens: int = 300) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    resp = requests.post(
        ANTHROPIC_API_URL,
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
        },
        json={
            "model":      model,
            "max_tokens": max_tokens,
            "system":     system,
            "messages":   [{"role": "user", "content": user}],
        },
        timeout=45,
    )
    resp.raise_for_status()
    blocks = resp.json().get("content", [])
    return "".join(b["text"] for b in blocks if b.get("type") == "text").strip()


def _fetch_pmc_fulltext(pmid: str, max_chars: int = 6000) -> str:
    """
    Try to fetch full-text XML from PubMed Central for an open-access paper.
    Returns plain text of Methods + Results + Discussion sections, or "" if
    the paper is not in PMC or is paywalled.

    PMC full-text covers binding site descriptions, key residue contacts,
    NCI details, and docking protocol — much richer than the abstract alone.
    """
    import re
    try:
        # Step 1: convert PMID → PMCID via elink
        link_resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi",
            params={
                "dbfrom": "pubmed",
                "db":     "pmc",
                "id":     pmid,
                "retmode":"json",
            },
            headers={"User-Agent": "G2D-Diff-research/1.0 (academic use)"},
            timeout=10,
        )
        link_resp.raise_for_status()
        link_data = link_resp.json()
        linksets  = link_data.get("linksets", [])
        pmcid = None
        for ls in linksets:
            for lsd in ls.get("linksetdbs", []):
                if lsd.get("dbto") == "pmc":
                    ids = lsd.get("links", [])
                    if ids:
                        pmcid = str(ids[0])
                        break

        if not pmcid:
            return ""   # not in PMC

        # Step 2: fetch full-text XML from PMC
        ft_resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={
                "db":      "pmc",
                "id":      pmcid,
                "retmode": "xml",
            },
            headers={"User-Agent": "G2D-Diff-research/1.0 (academic use)"},
            timeout=30,
        )
        ft_resp.raise_for_status()
        xml = ft_resp.text

        # Extract text from relevant sections — prioritise binding/docking content
        sections_text = []
        for sec_xml in re.findall(r"<sec[^>]*>(.*?)</sec>", xml, re.DOTALL):
            title_m = re.search(r"<title>(.*?)</title>", sec_xml, re.DOTALL)
            sec_title = re.sub(r"<[^>]+>", "", title_m.group(1)).strip().lower() if title_m else ""
            # Keep sections relevant to docking/binding
            keep_keywords = {"dock", "bind", "interact", "result", "discussion",
                             "method", "nci", "contact", "residue", "pharmacophore",
                             "structure", "affinit", "inhibit"}
            if any(kw in sec_title for kw in keep_keywords) or not sec_title:
                raw_text = re.sub(r"<[^>]+>", " ", sec_xml)
                raw_text = re.sub(r"\s+", " ", raw_text).strip()
                if raw_text:
                    sections_text.append(raw_text)

        full_text = "\n\n".join(sections_text)
        # Cap to max_chars to stay within ChemistryAgent context budget
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "... [truncated]"

        return full_text

    except Exception as e:
        log.debug(f"  [pmc_fulltext] failed for pmid={pmid}: {e}")
        return ""


def _search_pubmed(query: str, max_results: int = 10) -> list:
    """
    Search PubMed via E-utilities (free, no API key).
    For each paper, tries to fetch full text from PMC.
    Falls back to abstract if full text is unavailable.

    Returns list of dicts: {title, abstract, full_text, year, pmid}.
    """
    import re
    try:
        search_resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={
                "db":      "pubmed",
                "term":    query,
                "retmax":  max_results,
                "retmode": "json",
                "sort":    "relevance",
            },
            headers={"User-Agent": "G2D-Diff-research/1.0 (academic use)"},
            timeout=15,
        )
        search_resp.raise_for_status()
        pmids = search_resp.json().get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []

        fetch_resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={
                "db":      "pubmed",
                "id":      ",".join(pmids),
                "retmode": "xml",
                "rettype": "abstract",
            },
            headers={"User-Agent": "G2D-Diff-research/1.0 (academic use)"},
            timeout=20,
        )
        fetch_resp.raise_for_status()
        xml = fetch_resp.text

        articles = []
        for article_xml in re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", xml, re.DOTALL):
            title_m    = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", article_xml, re.DOTALL)
            abstract_m = re.search(r"<AbstractText[^>]*>(.*?)</AbstractText>", article_xml, re.DOTALL)
            year_m     = re.search(r"<PubDate>.*?<Year>(\d{4})</Year>", article_xml, re.DOTALL)
            pmid_m     = re.search(r"<PMID[^>]*>(\d+)</PMID>", article_xml)

            title    = re.sub(r"<[^>]+>", "", title_m.group(1)).strip()    if title_m    else ""
            abstract = re.sub(r"<[^>]+>", "", abstract_m.group(1)).strip() if abstract_m else ""
            year     = year_m.group(1)  if year_m  else ""
            pmid     = pmid_m.group(1)  if pmid_m  else ""

            # Try full text from PMC; fall back to abstract
            full_text = ""
            if pmid:
                log.debug(f"  [pubmed] fetching full text for pmid={pmid}...")
                full_text = _fetch_pmc_fulltext(pmid)

            articles.append({
                "title":     title,
                "abstract":  abstract,
                "full_text": full_text,   # empty string if not in PMC
                "year":      year,
                "pmid":      pmid,
                "source":    "pubmed",
            })
        return articles

    except Exception as e:
        log.debug(f"  [pubmed] search failed: {e}")
        return []


def _search_semantic_scholar(query: str, max_results: int = 2) -> list:
    """
    Search Semantic Scholar API (free, no key required).
    Fetches title, abstract, year, and open-access PDF URL when available.
    For open-access papers, retrieves full text via the tldr + abstract fields.

    Returns list of dicts: {title, abstract, full_text, year, paperId}.
    """
    try:
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query":  query,
                "fields": "title,year,abstract,tldr,openAccessPdf,externalIds",
                "limit":  max_results,
            },
            headers={"User-Agent": "G2D-Diff-research/1.0 (academic use)"},
            timeout=15,
        )
        resp.raise_for_status()
        papers = []
        for p in resp.json().get("data", []):
            abstract  = (p.get("abstract") or "")
            tldr      = (p.get("tldr") or {}).get("text", "")
            full_text = ""

            # Try to get more content via the paper details endpoint
            paper_id = p.get("paperId", "")
            if paper_id:
                try:
                    detail_resp = requests.get(
                        f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}",
                        params={"fields": "abstract,tldr,openAccessPdf"},
                        headers={"User-Agent": "G2D-Diff-research/1.0 (academic use)"},
                        timeout=10,
                    )
                    if detail_resp.status_code == 200:
                        detail = detail_resp.json()
                        abstract  = detail.get("abstract") or abstract
                        tldr      = (detail.get("tldr") or {}).get("text", "") or tldr
                        # Build full_text from abstract + tldr for richer context
                        parts = []
                        if abstract:
                            parts.append(f"Abstract: {abstract}")
                        if tldr:
                            parts.append(f"Summary: {tldr}")
                        full_text = "\n".join(parts)
                except Exception:
                    pass

            papers.append({
                "title":     p.get("title", ""),
                "abstract":  abstract,
                "full_text": full_text,
                "year":      str(p.get("year", "")),
                "paperId":   paper_id,
                "source":    "semantic_scholar",
            })
        return papers

    except Exception as e:
        log.debug(f"  [semantic_scholar] search failed: {e}")
        return []


def _format_paper(p: dict, max_body_chars: int = 3000) -> str:
    """
    Format a single paper dict into a text block for the ChemistryAgent prompt.
    Uses full_text when available, otherwise falls back to abstract.
    """
    title  = p.get("title", "untitled")
    year   = p.get("year", "")
    pmid   = p.get("pmid", "")
    source = p.get("source", "")
    ref    = f"PMID:{pmid}" if pmid else p.get("paperId", "")

    body = p.get("full_text", "").strip()
    if not body:
        body = p.get("abstract", "").strip()
        body_label = "Abstract"
    else:
        body_label = "Full text (binding/docking sections)"

    if len(body) > max_body_chars:
        body = body[:max_body_chars] + "... [truncated]"

    lines = [f"• {title} ({year}) [{ref}]"]
    if body:
        lines.append(f"  [{body_label}]: {body}")
    return "\n".join(lines)


def fetch_literature(
    genes: list,
    model: str = DEFAULT_MODEL,
    max_results_per_gene: int = 10,
) -> str:
    """
    Retrieve full-text or abstract content for BOTH top genes.

    For each gene:
      1. Search PubMed — tries PMC full text for each paper (binding/docking sections)
      2. Fall back to Semantic Scholar (abstract + tldr)
      3. Fall back to broader query if specific query returns nothing
      4. If all fail, note it — ChemistryAgent uses training knowledge for that gene

    Returns a structured text block injected into the ChemistryAgent prompt.
    No Anthropic API calls — zero impact on your rate limits.
    """
    if not genes:
        return ""

    all_sections = []

    for gene in genes[:2]:   # both top genes
        query = f"{gene} protein cancer binding site inhibitor structure docking"
        log.info(f"  [literature] searching for {gene}...")

        papers = _search_pubmed(query, max_results=max_results_per_gene)

        if not papers:
            papers = _search_semantic_scholar(query, max_results=max_results_per_gene)

        if not papers:
            fallback_q = f"{gene} cancer protein structure interaction"
            log.info(f"  [literature] retrying with broader query for {gene}...")
            papers = _search_pubmed(fallback_q, max_results=max_results_per_gene)
            if not papers:
                papers = _search_semantic_scholar(fallback_q, max_results=max_results_per_gene)

        if not papers:
            log.info(f"  [literature] no papers found for {gene} — training knowledge only")
            all_sections.append(
                f"=== {gene} literature ===\n"
                f"  No papers retrieved. Use training knowledge for {gene} NCI analysis."
            )
            continue

        # Count how many have full text vs abstract only
        n_fulltext = sum(1 for p in papers if p.get("full_text"))
        log.info(
            f"  [literature] {gene}: {len(papers)} paper(s), "
            f"{n_fulltext} with full text, {len(papers)-n_fulltext} abstract-only"
        )

        section_lines = [f"=== {gene} literature ==="]
        for p in papers:
            section_lines.append(_format_paper(p))

        all_sections.append("\n".join(section_lines))

    return "\n\n".join(all_sections) if all_sections else ""

def fetch_literature_with_metadata(
    genes: list,
    model: str = DEFAULT_MODEL,
    max_results_per_gene: int = 10,
) -> tuple:
    """
    Same as fetch_literature but also returns a structured metadata dict
    describing what was retrieved — for logging and evaluation tracking.

    Returns:
        (literature_text: str, metadata: dict)

    metadata structure:
        {
          "genes_searched": ["GENE1", "GENE2"],
          "per_gene": {
            "GENE1": {
              "n_papers": int,
              "source": "pubmed" | "semantic_scholar" | "none",
              "query_used": str,
              "papers": [
                {
                  "title": str,
                  "year": str,
                  "pmid": str,          # pubmed only
                  "paper_id": str,      # semantic scholar only
                  "source": str,
                  "content_type": "full_text" | "abstract" | "abstract+tldr",
                  "n_chars": int,
                }
              ]
            }
          },
          "total_papers": int,
          "total_fulltext": int,
          "total_abstract_only": int,
        }
    """
    if not genes:
        return "", {"genes_searched": [], "per_gene": {}, "total_papers": 0,
                    "total_fulltext": 0, "total_abstract_only": 0}

    all_sections = []
    per_gene_meta = {}

    for gene in genes[:2]:
        query = f"{gene} protein cancer binding site inhibitor structure docking"
        query_used = query
        source_used = "none"

        papers = _search_pubmed(query, max_results=max_results_per_gene)
        if papers:
            source_used = "pubmed"
        else:
            papers = _search_semantic_scholar(query, max_results=max_results_per_gene)
            if papers:
                source_used = "semantic_scholar"

        if not papers:
            fallback_q = f"{gene} cancer protein structure interaction"
            query_used = fallback_q
            papers = _search_pubmed(fallback_q, max_results=max_results_per_gene)
            if papers:
                source_used = "pubmed_fallback"
            else:
                papers = _search_semantic_scholar(fallback_q, max_results=max_results_per_gene)
                if papers:
                    source_used = "semantic_scholar_fallback"

        # Build per-paper metadata
        paper_meta = []
        for p in papers:
            has_full = bool(p.get("full_text", "").strip())
            has_abstract = bool(p.get("abstract", "").strip())
            has_tldr = "Summary:" in p.get("full_text", "")

            if has_full and p.get("source") == "pubmed":
                content_type = "full_text"
            elif has_full and has_tldr:
                content_type = "abstract+tldr"
            elif has_abstract:
                content_type = "abstract"
            else:
                content_type = "none"

            body = p.get("full_text", "").strip() or p.get("abstract", "").strip()

            paper_meta.append({
                "title":        p.get("title", ""),
                "year":         p.get("year", ""),
                "pmid":         p.get("pmid", ""),
                "paper_id":     p.get("paperId", ""),
                "source":       p.get("source", ""),
                "content_type": content_type,
                "n_chars":      len(body),
            })

        n_fulltext = sum(1 for pm in paper_meta if pm["content_type"] == "full_text")
        log.info(
            f"  [literature] {gene}: {len(papers)} paper(s) from {source_used}, "
            f"{n_fulltext} full-text, {len(papers)-n_fulltext} abstract/other"
        )

        per_gene_meta[gene] = {
            "n_papers":   len(papers),
            "source":     source_used,
            "query_used": query_used,
            "papers":     paper_meta,
        }

        if not papers:
            all_sections.append(
                f"=== {gene} literature ===\n"
                f"  No papers retrieved. Use training knowledge for {gene} NCI analysis."
            )
        else:
            section_lines = [f"=== {gene} literature ==="]
            for p in papers:
                section_lines.append(_format_paper(p))
            all_sections.append("\n".join(section_lines))

    literature_text = "\n\n".join(all_sections) if all_sections else ""

    total_papers    = sum(v["n_papers"] for v in per_gene_meta.values())
    total_fulltext  = sum(
        sum(1 for pm in v["papers"] if pm["content_type"] == "full_text")
        for v in per_gene_meta.values()
    )
    metadata = {
        "genes_searched":     list(per_gene_meta.keys()),
        "per_gene":           per_gene_meta,
        "total_papers":       total_papers,
        "total_fulltext":     total_fulltext,
        "total_abstract_only": total_papers - total_fulltext,
    }

    return literature_text, metadata


def _parse_json(text: str, fallback: dict) -> dict:
    """
    Robustly extract a JSON object from an LLM response.
    Handles markdown fences, leading prose, and truncated responses.
    """
    clean = text.strip()

    # Strip markdown fences — handle ```json, ``` and missing closing fence
    if clean.startswith("```"):
        lines = clean.split("\n")
        inner = lines[1:]
        if inner and inner[-1].strip() in ("```", "~~~"):
            inner = inner[:-1]
        clean = "\n".join(inner).strip()

    # Try parsing the whole cleaned text first
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Find outermost { ... } and walk backwards from last }
    # to handle truncated JSON by finding the last valid closed object
    s = clean.find("{")
    if s >= 0:
        e = len(clean)
        while e > s:
            e = clean.rfind("}", s, e)
            if e < 0:
                break
            try:
                return json.loads(clean[s:e + 1])
            except json.JSONDecodeError:
                pass

    log.warning(f"Could not parse JSON response: {text[:300]}")
    return fallback


# ---------------------------------------------------------------------------
# ChemistryAgent — literature NCI analysis (runs FIRST, sequential)
# ---------------------------------------------------------------------------

class ChemistryAgent:
    """
    Stage 1 — searches published literature for docking/AlphaFold studies
    on the target genes identified for this cell line, and characterises
    key non-covalent interactions (NCIs) in known hit molecules that could
    also be present in our candidate.

    NCIs analysed:
      (a) pi-pi stacking / pi-pi interactions (aromatic ring systems)
      (b) hydrogen bonding (OH, NH, SH, FH donors/acceptors)
      (c) halogen bonding (F, Cl, Br substituents)
      (d) electrostatic interactions (charged groups, salt bridges)
      (e) hydrophobic interactions (aliphatic/aromatic pockets)

    Output is a structured NCI report — NOT a score.
    The score comes from BiochemistryAgent in Stage 2.
    """
    NAME   = "chemistry"
    SYSTEM = (
        "You are a structural chemist and drug discovery scientist with deep expertise "
        "in non-covalent interactions (NCIs) and molecular docking. "
        "Draw on your knowledge of published docking studies, AlphaFold-based "
        "structural analyses, co-crystal structures, PubMed, ChEMBL, and PDB entries. "
        "You should prefer recent (2020-2025) data when available. "
        "Do NOT give a score. Return a structured NCI analysis "
        "report as JSON only — no other text or markdown."
    )

    @classmethod
    def _format_attention_context(cls, attention_summary: dict) -> str:
        """
        Format the full attention analysis output into a structured block
        for the prompt. This is the core of what makes ChemistryAgent
        grounded in model evidence rather than genotype guessing.

        Includes:
          - Top attended genes with their attention scores
          - Enriched NeST pathways with p-values and overlap genes
          - Pathway overlap genes (intersection of high-attention genes
            and statistically enriched pathways) — the most specific
            targets the model identified
        """
        if not attention_summary:
            return "No attention analysis available — using genotype only."

        lines = []

        # Top attended genes with scores
        top_detail = attention_summary.get("top_genes_detail", [])
        if top_detail:
            gene_lines = [
                f"  {g['gene']} (attention={g['attention']:.4f})"
                for g in top_detail[:12]
            ]
            lines.append("Top-attended genes (model attention scores):")
            lines.extend(gene_lines)
        else:
            genes = attention_summary.get("top_genes", [])[:12]
            lines.append(f"Top-attended genes: {', '.join(genes)}")

        lines.append("")

        # Enriched NeST pathways
        pathways = attention_summary.get("enriched_pathways", [])
        if pathways:
            lines.append("Statistically enriched NeST pathways (Fisher p < 0.05):")
            for p in pathways[:5]:
                overlap = ", ".join(p.get("overlap_genes", [])[:6])
                lines.append(
                    f"  {p['pathway']}"
                    f"  enrichment={p.get('enrichment','?')}x"
                    f"  p={p.get('p_value','?'):.2e}"
                    f"  overlap genes: [{overlap}]"
                )
        else:
            lines.append("No significantly enriched pathways (p < 0.05).")

        lines.append("")

        # Most important: intersection genes
        # These are genes that are BOTH highly attended AND in an enriched pathway
        top_gene_set  = set(attention_summary.get("top_genes", []))
        pathway_genes = set()
        for p in pathways:
            pathway_genes.update(p.get("overlap_genes", []))
        intersection = sorted(top_gene_set & pathway_genes)

        if intersection:
            lines.append(
                "Priority target genes — HIGH ATTENTION + enriched pathway "
                f"(use these first for docking search): {', '.join(intersection[:10])}"
            )
        else:
            lines.append(
                "No intersection between top-attended genes and enriched pathways. "
                "Use top-attended genes as primary targets."
            )

        return " ".join(lines)

    @classmethod
    def user_prompt(cls, smiles: str, cell_name: str, gene_ctx: dict,
                    desc: dict, attention_summary: dict,
                    literature_context: str = "") -> str:
        """
        Build the ChemistryAgent prompt with full attention analysis context.

        The attention_summary contains:
          - top_genes_detail: genes + their exact attention scores
          - enriched_pathways: NeST pathways enriched among top genes
          - The intersection of both = most specific model-identified targets

        These are passed to the LLM so it can focus its docking literature
        search on the genes the MODEL identified as important, not just the
        genes that happen to be mutated in the cell line.
        """
        attention_block    = cls._format_attention_context(attention_summary)
        literature_block   = (
            f"\n=== RETRIEVED LITERATURE (from web search) ===\n"
            f"{literature_context}\n"
            f"=== END LITERATURE ===\n\n"
            if literature_context else ""
        )

        # Fallback gene list for when attention is unavailable
        if attention_summary and attention_summary.get("top_genes"):
            # Prefer intersection genes, then top attended, then genotype
            top_gene_set  = set(attention_summary.get("top_genes", []))
            pathway_genes = set()
            for p in attention_summary.get("enriched_pathways", []):
                pathway_genes.update(p.get("overlap_genes", []))
            intersection = sorted(top_gene_set & pathway_genes)
            # Limit to 2 — each gene adds ~300 tokens of NCI JSON
            priority_genes = intersection[:2] if intersection else \
                             attention_summary.get("top_genes", [])[:2]
        else:
            priority_genes = gene_ctx.get("mutated_genes", [])[:2]

        priority_str = ", ".join(priority_genes) if priority_genes else "unknown"

        return f"""Cancer cell line: {cell_name}

GENOTYPE (raw data):
  Mutated genes:             {', '.join(gene_ctx.get('mutated_genes', [])[:10]) or 'none'}
  Amplified oncogenes:       {', '.join(gene_ctx.get('amplified_genes', [])) or 'none'}
  Deleted tumor suppressors: {', '.join(gene_ctx.get('deleted_genes', [])) or 'none'}

MODEL ATTENTION ANALYSIS (G2D-Diff condition encoder — what the model focused on):
{attention_block}

Priority target genes for docking literature search: {priority_str}

Candidate molecule SMILES: {smiles}
MW={desc.get('mw','?')}, LogP={desc.get('logp','?')}, AromaticRings={desc.get('arom_rings','?')},
HBD={desc.get('hbd','?')}, HBA={desc.get('hba','?')}, RotBonds={desc.get('rotbonds','?')}

The MODEL ATTENTION ANALYSIS above shows which genes the G2D-Diff diffusion model
weighted most heavily when generating molecules predicted to be sensitive for this
cell line. Focus your analysis on these model-identified targets —
especially the priority genes (intersection of high attention + enriched pathways).
{literature_block}
Analyse AT MOST 2 priority genes. For each gene where published docking data exists:

1. Identify known hit molecules or approved drugs that bind this target
2. Characterise their key binding NCIs:
   (a) pi-pi stacking — which aromatic ring systems, which residues
   (b) hydrogen bonds — donor/acceptor atoms (OH, NH, SH, FH), key residues
   (c) halogen bonds — F, Cl, Br interactions with backbone carbonyls or polar residues
   (d) electrostatic interactions — charged groups, salt bridges, ionic contacts
   (e) hydrophobic interactions — aliphatic/aromatic pocket contacts

3. Compare these NCI patterns to our candidate molecule's structural features.
   Does the candidate have ring systems for pi-pi interactions? H-bonding interactions with donors/acceptors?
   Halogens interactions? Charged groups? Hydrophobic regions?

4. Given the genotype context and model-identified pathways, reason about how
   this candidate could exploit the cell line's specific vulnerabilities:
   - Direct oncogene inhibition (does the candidate resemble known kinase
     inhibitors or oncogene-targeted drugs for the mutated/amplified genes?)
   - Synthetic lethality with deleted tumor suppressors (e.g. PARP inhibitors
     for BRCA-deleted cells, MDM2 inhibitors for TP53-deleted contexts)
   - Amplification-driven dependency (does a highly amplified gene create an
     addiction the candidate could exploit?)
   - Pathway bypass resistance (does the candidate target a node that prevents
     known resistance mechanisms for this genotype, e.g. downstream effectors
     or parallel pathway members?)
   - Known targeted therapies and kinase inhibitors for these genetic
     alterations — are there approved drugs or clinical candidates for these
     genes, and does the candidate share their pharmacophore features?

IMPORTANT: After your web searches are complete, output ONLY the following JSON object and nothing else — no prose, no markdown fences, no preamble. Keep all string values concise (max 20 words):
{{
  "target_genes_with_docking_data": ["<gene1>", "<gene2>"],
  "attention_grounded": <true if attention data was used, false if genotype only>,
  "nci_analysis": [
    {{
      "gene": "<gene name>",
      "attention_score": "<score from model attention or null>",
      "in_enriched_pathway": <true/false>,
      "known_hits": ["<drug/compound name>"],
      "key_ncis": {{
        "pi_pi": "<max 15 words or null>",
        "hydrogen_bonding": "<max 15 words or null>",
        "halogen_bonding": "<max 15 words or null>",
        "electrostatic": "<max 15 words or null>",
        "hydrophobic": "<max 15 words or null>"
      }},
      "candidate_nci_overlap": "<max 20 words>",
      "candidate_nci_gaps": "<max 20 words>"
    }}
  ],
  "overall_nci_similarity": "<high | moderate | low>",
  "structural_rationale": "<2 sentences max>",
  "genotype_exploitation": {{
    "mechanism": "<oncogene_inhibition | synthetic_lethality | amplification_dependency | pathway_dependency | unknown>",
    "key_gene": "<most relevant gene from the model-identified targets>",
    "known_therapy_analogy": "<approved drug or clinical candidate with similar target/pharmacophore, or null>",
    "rationale": "<one sentence: how the candidate could exploit this cell line's vulnerabilities>"
  }}
}}"""

    @classmethod
    def run(cls, smiles: str, cell_name: str, gene_ctx: dict, desc: dict,
            attention_summary: dict, model: str) -> dict:
        fb = {
            "target_genes_with_docking_data": [],
            "attention_grounded": False,
            "nci_analysis": [],
            "overall_nci_similarity": "unknown",
            "structural_rationale": "API error",
        }
       # try:
            # Step 1 (optional): fetch paper summaries with simple queries
            # This is a separate cheap call — only gene names go to web search,
            # not the full complex prompt.
        literature_context = ""
        if USE_WEB_SEARCH:
            if attention_summary and attention_summary.get("top_genes"):
                search_genes = attention_summary["top_genes"][:2]
            else:
                search_genes = gene_ctx.get("mutated_genes", [])[:2]
            try:
                literature_context = fetch_literature(search_genes, model,max_results_per_gene=10)
                if literature_context:
                    log.info("  [ChemistryAgent] literature retrieved, injecting into prompt")
                else:
                    log.info("  [ChemistryAgent] no literature retrieved")
            except Exception as lit_err:
                log.warning(f"  [ChemistryAgent] literature fetch failed: {lit_err}")

        # Cooldown after web search calls — let the rate limit window reset
        

        # Step 2: full NCI analysis with plain _call_api (no web search tool)
        # Literature context (if any) is injected into the prompt as text.
        raw = _call_api(
            cls.SYSTEM,
            cls.user_prompt(smiles, cell_name, gene_ctx, desc,
                            attention_summary, literature_context),
            model,
            max_tokens=1200,
        )
        return _parse_json(raw, fb)
        #except Exception as e:
        #    return {**fb, "structural_rationale": str(e)}


# ---------------------------------------------------------------------------
# BiochemistryAgent — final scorer (runs LAST, sequential after Stage 1)
# ---------------------------------------------------------------------------

class BiochemistryAgent:
    """
    Stage 2 — receives the ChemistryAgent NCI report plus molecular
    descriptors and the four specialist agent reports, and produces the
    final score for the latent optimization reward.

    This agent is the only one that emits a score. It reasons about:
      - How well the candidate's NCI profile matches known binders
        (from ChemistryAgent)
      - Whether the descriptors support the binding mode implied by
        the NCI analysis
      - Whether ADMET, alignment, and selectivity reports are consistent
        with the NCI evidence

    The score reflects: NCI compatibility × biochemical plausibility
    × drug-likeness, weighted by the quality of the NCI evidence.
    """
    NAME   = "biochemistry"
    SYSTEM = (
        "You are a biochemist and computational drug discovery scientist. "
        "You receive a structural non-covalent interaction (NCI) analysis report from a chemistry specialist "
        "and molecular descriptor data, and your task is to translate this into a final "
        "drug candidate score for optimization guidance. "
        "A high score means: strong NCI overlap with known binders, good descriptors "
        "supporting the binding mode, and no fatal ADMET issues. "
        "Return valid JSON only."
    )

    @classmethod
    def user_prompt(
        cls,
        smiles: str,
        desc: dict,
        pred_auc: float,
        qed: float,
        sas: float,
        validity: float,
        gene_ctx: dict,
        attention_summary: dict,
        chemistry_report: dict,
    ) -> str:
        nci_blocks = []
        for item in chemistry_report.get("nci_analysis", []):
            nci_blocks.append(
                f"  Gene: {item.get('gene','?')} "
                f"(attention={item.get('attention_score','?')} "
                f"in_enriched_pathway={item.get('in_enriched_pathway','?')})\n"
                f"  Known hits: {item.get('known_hits',[])}\n"
                f"  NCI overlap: {item.get('candidate_nci_overlap','?')}\n"
                f"  NCI gaps:    {item.get('candidate_nci_gaps','?')}"
            )
        nci_block = "\n".join(nci_blocks) if nci_blocks else "  No docking data found"

        top_genes = attention_summary.get("top_genes", [])[:12] if attention_summary else \
                    gene_ctx.get("mutated_genes", [])[:8]
        pathways  = attention_summary.get("enriched_pathways", []) if attention_summary else []
        pw_lines  = []
        for p in pathways[:4]:
            pw_lines.append(
                f"  {p['pathway']}  enrichment={p.get('enrichment','?')}x"
                f"  p={p.get('p_value',0):.2e}"
                f"  genes=[{', '.join(p.get('overlap_genes',[])[:5])}]"
            )
        pw_block = "\n".join(pw_lines) if pw_lines else "  none"

        gex      = chemistry_report.get("genotype_exploitation") or {}
        gex_mech = gex.get("mechanism", "?")
        gex_gene = gex.get("key_gene", "?")
        gex_anal = gex.get("known_therapy_analogy", "?")
        gex_rat  = gex.get("rationale", "?")
        pred_auc_clipped = float(np.clip(pred_auc, 0.0, 1.0))
        return (
            f"Score this candidate molecule for the G2D-Diff latent optimization reward.\n\n"
            f"=== G2D-DIFF MODEL OUTPUT (biology agent) ===\n"
            f"Cell line:  {gene_ctx.get('cell_name', 'unknown')}\n"
            f"Pred AUC:   {pred_auc_clipped:.4f}  [lower = more sensitive = better; 0-1]\n"
            f"QED:        {qed:.3f}        [drug-likeness 0-1; higher = better]\n"
            f"SAS:        {sas:.2f}        [synthetic accessibility 1-10; lower = easier]\n"
            f"Validity:   {validity:.1%}   [fraction of valid SMILES from this z region]\n\n"
            f"Molecular descriptors:\n"
            f"  SMILES: {smiles}\n"
            f"  MW={desc.get('mw','?')}, LogP={desc.get('logp','?')}, "
            f"HBD={desc.get('hbd','?')}, HBA={desc.get('hba','?')}, "
            f"TPSA={desc.get('tpsa','?')}, RotBonds={desc.get('rotbonds','?')}, "
            f"AromaticRings={desc.get('arom_rings','?')}\n\n"
            f"Cell line genotype:\n"
            f"  Mutated:   {', '.join(gene_ctx.get('mutated_genes',[])[:10]) or 'none'}\n"
            f"  Amplified: {', '.join(gene_ctx.get('amplified_genes',[]))   or 'none'}\n"
            f"  Deleted:   {', '.join(gene_ctx.get('deleted_genes',[]))     or 'none'}\n\n"
            f"Model attention — top-attended genes:\n"
            f"  {', '.join(top_genes)}\n\n"
            f"Model attention — enriched NeST pathways:\n"
            f"{pw_block}\n\n"
            f"=== CHEMISTRY AGENT (NCI literature analysis) ===\n"
            f"Attention-grounded: {chemistry_report.get('attention_grounded', False)}\n"
            f"Targets with docking data: {chemistry_report.get('target_genes_with_docking_data', [])}\n"
            f"Overall NCI similarity: {chemistry_report.get('overall_nci_similarity','?')}\n"
            f"Structural rationale: {chemistry_report.get('structural_rationale','?')}\n"
            f"Per-gene NCI breakdown:\n{nci_block}\n\n"
            f"Genotype exploitation (from ChemistryAgent):\n"
            f"  mechanism={gex_mech}\n"
            f"  key_gene={gex_gene}\n"
            f"  known_therapy_analogy={gex_anal}\n"
            f"  rationale={gex_rat}\n\n"
            f"=== SCORING TASK ===\n"
            f"Integrate the G2D-Diff model output with the NCI literature evidence.\n\n"
            f"Scoring guidance:\n"
            f"- Low pred AUC = high predicted sensitivity → strong positive signal\n"
            f" A high AUC (resistant) is a strong negative signal regardless of NCI quality.\n"
            f" Do NOT interpret a high AUC as near a threshold — it means the cell is NOT sensitive.\n"
            f"- High NCI overlap with known binders of model-identified targets → boost\n"
            f"- NCI gaps (missing critical interactions vs known hits) → penalise\n"
            f"- Good QED (>0.5) + low SAS (<4) → better candidate; poor → penalise\n"
            f"- High validity → z region decodes well; low → penalise\n"
            f"- If no docking data: rely on AUC + descriptors + pathway evidence\n\n"
            f"Drug-likeness and safety — flag any of the following from the SMILES and descriptors:\n"
            f"  Reactive groups: Michael acceptors, aldehydes, epoxides, acyl halides\n"
            f"  Toxicophores: PAINS substructures, mutagenic groups, hERG risk (high LogP + basic N)\n"
            f"  Metabolic: CYP liability (aromatic amines, quinones), short half-life concerns\n"
            f"  Permeability: Lipinski violations (MW>500, LogP>5, HBD>5, HBA>10), efflux risk\n"
            f"  → Fatal issues (toxicophore, Lipinski violation) should strongly penalise final_score\n"
            f"  Keep each admet_flags entry to 8 words or fewer\n\n"
            f"Genotype exploitation — strong genotype-mechanism match boosts confidence\n\n"
            f"Return ONLY valid JSON:\n"
            f"{{\n"
            f'  "final_score": <float 0-1>,\n'
            f'  "confidence": <float 0-1>,\n'
            f'  "nci_score": <float 0-1, NCI overlap contribution>,\n'
            f'  "descriptor_score": <float 0-1, QED+SAS+descriptor quality>,\n'
            f'  "admet_flags": [<max 8 words each, e.g. "MW 807 exceeds Lipinski limit">, ...],\n'
            f'  "summary": "<2-3 sentences integrating model evidence, NCI analysis, and ADMET>",\n'
            f'  "key_factors": ["<top 2-3 positive factors>"],\n'
            f'  "concerns": ["<top 1-2 concerns, empty list if none>"],\n'
            f'  "recommendation": "<pursue | deprioritize | investigate_further>"\n'
            f"}}\n"
        )


    @classmethod
    def run(
        cls,
        smiles: str,
        desc: dict,
        pred_auc: float,
        qed: float,
        sas: float,
        validity: float,
        gene_ctx: dict,
        attention_summary: dict,
        chemistry_report: dict,
        model: str,
    ) -> dict:
        fb = {
            "final_score": 0.5, "confidence": 0.3,
            "nci_score": 0.5, "descriptor_score": 0.5,
            "admet_flags": [],
            "summary": "BiochemistryAgent error",
            "key_factors": [], "concerns": ["API error"],
            "recommendation": "investigate_further",
        }
        #try:
        raw = _call_api(
            cls.SYSTEM,
            cls.user_prompt(
                smiles, desc, pred_auc, qed, sas, validity,
                gene_ctx, attention_summary, chemistry_report,
            ),
            model,
            max_tokens=1000,
        )
        r = _parse_json(raw, fb)
        r["final_score"]      = float(np.clip(r.get("final_score",      0.5), 0.0, 1.0))
        r["confidence"]       = float(np.clip(r.get("confidence",       0.3), 0.0, 1.0))
        r["nci_score"]        = float(np.clip(r.get("nci_score",        0.5), 0.0, 1.0))
        r["descriptor_score"] = float(np.clip(r.get("descriptor_score", 0.5), 0.0, 1.0))
        if not isinstance(r.get("admet_flags"), list):
            r["admet_flags"] = []
        return r
        #except Exception as e:
        #    return {**fb, "concerns": [str(e)]}


# ---------------------------------------------------------------------------
# Orchestrator: runs all agents and returns combined result
# ---------------------------------------------------------------------------

def score_molecule(
    smiles: str,
    cell_name: str,
    gene_ctx: dict,
    pred_auc: float = 0.5,
    qed: float = 0.0,
    sas: float = 10.0,
    validity: float = 1.0,
    model: str = DEFAULT_MODEL,
    attention_summary: dict = None,
) -> dict:
    """
    Run the two-agent pipeline for one molecule.

    Stage 1 — ChemistryAgent:
        Receives model-identified target genes (from attention analysis),
        their attention scores, enriched NeST pathways, and the candidate
        SMILES. Searches published docking/AlphaFold literature for those
        specific targets and reports NCIs of known hit molecules vs candidate.
        Does NOT score.

    Stage 2 — BiochemistryAgent:
        Receives the full G2D-Diff model output (pred_AUC, QED, SAS,
        validity, descriptors, attention-identified genes and pathways,
        genotype context) plus the ChemistryAgent NCI report.
        Produces final_score (0-1) for the latent optimization reward.

    Returns:
        final_score      float (0-1) — reward signal for optimizer
        confidence       float (0-1)
        nci_score        float (0-1) — NCI overlap contribution
        descriptor_score float (0-1) — QED/SAS/descriptor quality
        summary          str
        key_factors      list[str]
        concerns         list[str]
        recommendation   str
        chemistry        dict  (ChemistryAgent report)
        error            str or None
    """
    if not smiles or Chem.MolFromSmiles(smiles) is None:
        return {
            "final_score": 0.0, "confidence": 0.0,
            "nci_score": 0.0, "descriptor_score": 0.0,
            "summary": "Invalid SMILES",
            "key_factors": [], "concerns": ["invalid_smiles"],
            "recommendation": "deprioritize",
            "chemistry": {}, "error": "invalid_smiles",
        }

    # Embed cell_name into gene_ctx for the prompt
    gene_ctx = dict(gene_ctx)
    gene_ctx["cell_name"] = cell_name

    desc = get_mol_descriptors(smiles)

    # ── Stage 1: ChemistryAgent ─────────────────────────────────────────────
    # Full attention_summary passed — genes + scores + enriched pathways +
    # intersection genes. ChemistryAgent focuses literature search on what
    # the MODEL identified as important, not just mutated genes.
    log.debug("  [chemistry] running (attention_grounded=%s)...",
              attention_summary is not None)
    chemistry_report = ChemistryAgent.run(
        smiles, cell_name, gene_ctx, desc, attention_summary, model
    )
    log.debug("  [chemistry] done — NCI similarity: %s | grounded: %s",
              chemistry_report.get("overall_nci_similarity", "?"),
              chemistry_report.get("attention_grounded", False))

    # ── Stage 2: BiochemistryAgent ──────────────────────────────────────────
    # Receives the model biology output (AUC, QED, SAS, validity, descriptors,
    # attention genes + pathways, genotype) + ChemistryAgent NCI report.
    # Pause between agents to avoid consecutive-call rate limits
    time.sleep(5)
    log.debug("  [biochemistry] running...")
    biochem = BiochemistryAgent.run(
        smiles=smiles,
        desc=desc,
        pred_auc=pred_auc,
        qed=qed,
        sas=sas,
        validity=validity,
        gene_ctx=gene_ctx,
        attention_summary=attention_summary,
        chemistry_report=chemistry_report,
        model=model,
    )

    return {**biochem, "chemistry": chemistry_report, "error": None}


# ---------------------------------------------------------------------------
# Batch scorer with caching — drop-in for optimize_latent_batch()
# ---------------------------------------------------------------------------

class LLMScorer:
    """
    Multi-agent scorer for batches of molecules with caching.

    Parameters
    ----------
    cell2mut, cell2cna, cell2cnd : DataFrames with ccle_name as column
    gene_names    : list of 718 gene names from load_gene_names()
    top_k_genes   : how many top genes to include per cell in agent prompts
    score_every   : call agents every N steps (0 = end of run only)
    score_top_n   : only score the top N molecules by AUC per call
    model         : Anthropic model name
    save_dir      : if set, saves all results as JSONL per batch
    """

    def __init__(
        self,
        cell2mut: pd.DataFrame,
        cell2cna: pd.DataFrame,
        cell2cnd: pd.DataFrame,
        gene_names: list,
        top_k_genes: int = 15,
        score_every: int = 0,
        score_top_n: int = 3,
        model:       str = DEFAULT_MODEL,
        save_dir:    str = None,
    ):
        self.cell2mut    = cell2mut
        self.cell2cna    = cell2cna
        self.cell2cnd    = cell2cnd
        self.gene_names  = gene_names
        self.top_k_genes = top_k_genes
        self.score_every = score_every
        self.score_top_n = score_top_n
        self.model       = model
        self.save_dir    = save_dir
        self._cache: dict    = {}
        self._call_log: list = []

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            log.warning("ANTHROPIC_API_KEY not set — LLM scoring will be skipped.")

    def _should_score(self, step: int, n_steps: int) -> bool:
        if step == n_steps - 1:
            return True
        return self.score_every > 0 and step > 0 and step % self.score_every == 0

    def maybe_score(
        self,
        step: int,
        n_steps: int,
        batch_idx: int,
        cell_names: list,
        best_smiles: list,
        best_auc: list,
        batch: dict = None,            # needed for attention extraction
        attention_extractor = None,    # AttentionExtractor instance (optional)
    ) :
        """
        Score the top_n molecules if this step qualifies.
        Returns list of result dicts or None if skipped.
        Each result has a 'final_score' key suitable for use as reward.
        """
        if not self._should_score(step, n_steps):
            return None
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None

        pairs = [(s, a, c) for s, a, c in zip(best_smiles, best_auc, cell_names) if s]
        pairs.sort(key=lambda x: x[1])
        to_score = pairs[:self.score_top_n]

        # Extract attention once per batch if extractor and batch are provided
        attention_cache = {}
        if attention_extractor is not None and batch is not None:
            try:
                unique_cells = list(set(cell_names))
                for cell in unique_cells:
                    att_result = attention_extractor.extract(batch, cell_name=cell)
                    attention_cache[cell] = att_result.attention_summary
                    n_paths = len(att_result.enriched_pathways)
                    log.info(
                        f"  [attention] {cell}: {len(att_result.top_genes)} top genes, "
                        #f"{n_paths} enriched pathways"
                    )
            except Exception as e:
                log.warning(f"  [attention] extraction failed: {e} — using genotype only")

        results = []
        for smiles, auc, cell_name in to_score:
            key = hashlib.md5(f"{smiles}|{cell_name}".encode()).hexdigest()

            if key in self._cache:
                result = self._cache[key]
                log.info(f"  [cache] {smiles[:45]}... → final={result['final_score']:.3f}")
            else:
                gene_ctx = get_cell_gene_context(
                    cell_name, self.cell2mut, self.cell2cna, self.cell2cnd,
                    self.gene_names, self.top_k_genes,
                )
                att_summary = attention_cache.get(cell_name)
                if att_summary:
                    log.info(f"\n  [agents ×4 → coordinator | +attention] {cell_name} | {smiles[:50]}...")
                else:
                    log.info(f"\n  [agents ×4 → coordinator] {cell_name} | {smiles[:50]}...")
                # Recover QED/SAS/validity from best_props if available, else use defaults
                mol = Chem.MolFromSmiles(smiles)
                mol_qed  = float(QED.qed(mol)) if mol else 0.0
                mol_sas  = float(sascorer.calculateScore(mol)) if mol else 10.0
                result = score_molecule(
                    smiles, cell_name, gene_ctx,
                    pred_auc=auc,
                    qed=mol_qed,
                    sas=mol_sas,
                    validity=1.0,
                    model=self.model,
                    attention_summary=att_summary,
                )
                self._cache[key] = result

            self._log_result(result, step, batch_idx, cell_name, smiles, auc)
            entry = {"step": step, "batch_idx": batch_idx,
                     "cell_name": cell_name, "smiles": smiles, "pred_auc": auc, **result}
            results.append(entry)
            self._call_log.append(entry)

        if self.save_dir and results:
            path = os.path.join(self.save_dir, f"agent_scores_batch{batch_idx:03d}.jsonl")
            with open(path, "a") as f:
                for r in results:
                    safe = {k: (str(v) if isinstance(v, dict) else v) for k, v in r.items()}
                    f.write(json.dumps(safe) + "\n")

        return results

    def get_mean_final_score(self, results) -> float:
        """Mean final_score from maybe_score() results. Returns 0.5 if None."""
        if not results:
            return 0.5
        return float(np.mean([r["final_score"] for r in results]))

    def _log_result(self, r, step, batch_idx, cell_name, smiles, auc):
        ch  = r.get("chemistry", {})
        gex = ch.get("genotype_exploitation", {})
        log.info(f"  ── LLM scoring (batch {batch_idx}, step {step}) ──")
        log.info(f"     {cell_name} | {smiles[:55]}")
        log.info(f"     ── G2D-Diff model (biology agent) ──")
        log.info(f"     Pred AUC:    {auc:.3f}")
        log.info(f"     ── ChemistryAgent ──")
        log.info(f"     Attention-grounded: {ch.get('attention_grounded', False)}")
        log.info(f"     NCI similarity:     {ch.get('overall_nci_similarity','?')}")
        log.info(f"     Targets w/ data:    {ch.get('target_genes_with_docking_data',[])}")
        log.info(f"     NCI rationale:      {ch.get('structural_rationale','?')[:100]}")
        log.info(f"     Mechanism:          {gex.get('mechanism','?')}  "
                 f"key_gene={gex.get('key_gene','?')}")
        log.info(f"     Known analogy:      {gex.get('known_therapy_analogy','?')}")
        log.info(f"     ── BiochemistryAgent (final scorer) ──")
        log.info(f"     Final score:     {r.get('final_score','?'):.3f}  confidence={r.get('confidence','?'):.2f}")
        log.info(f"     NCI score:       {r.get('nci_score','?'):.3f}")
        log.info(f"     Descriptor score:{r.get('descriptor_score','?'):.3f}")
        log.info(f"     ADMET flags:     {r.get('admet_flags',[])}")
        log.info(f"     Summary:         {r.get('summary','')}")
        log.info(f"     Factors:         {r.get('key_factors',[])}")
        log.info(f"     Concerns:        {r.get('concerns',[])}")
        log.info(f"     Recommend:       {r.get('recommendation','?')}")

    def summary_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._call_log)


# ---------------------------------------------------------------------------
# Integration patch for latent_opt_g2d.py
# ---------------------------------------------------------------------------
# 1. Import:
#       from llm_scorer import LLMScorer, load_gene_names
#
# 2. parse_args():
#       parser.add_argument("--llm_scoring",   action="store_true")
#       parser.add_argument("--score_every",   type=int,   default=50)
#       parser.add_argument("--score_top_n",   type=int,   default=3)
#       parser.add_argument("--w_llm",         type=float, default=0.2)
#       parser.add_argument("--llm_score_dir", default="./llm_scores")
#
# 3. main(), after loading genotype data:
#       scorer = None
#       if args.llm_scoring:
#           gene_names = load_gene_names(args.mut_data)
#           scorer = LLMScorer(
#               cell2mut=cell2mut, cell2cna=cell2cna, cell2cnd=cell2cnd,
#               gene_names=gene_names,
#               score_every=args.score_every,
#               score_top_n=args.score_top_n,
#               save_dir=args.llm_score_dir,
#           )
#
# 4. Optionally create AttentionExtractor (needs attention_analysis.py):
#       from attention_analysis import AttentionExtractor
#       att_extractor = AttentionExtractor(diff_model, gene_names) if args.llm_scoring else None
#
# 5. optimize_latent_batch(), at the log_every block:
#       llm_results = scorer.maybe_score(
#           step=step, n_steps=n_steps, batch_idx=batch_idx,
#           cell_names=batch["cell_name"],
#           best_smiles=best_smiles,
#           best_auc=best_auc_vals,
#       ) if scorer else None
#
#       if llm_results:
#           llm_r = scorer.get_mean_final_score(llm_results)
#           llm_t = torch.tensor(llm_r, dtype=torch.float32, device=device)
#           loss  = loss - args.w_llm * llm_t
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Two-agent LLM scoring for G2D-Diff candidates")
    parser.add_argument("--smiles",    required=True, nargs="+")
    parser.add_argument("--cell_name", required=True)
    parser.add_argument("--mut_data",  required=True)
    parser.add_argument("--cna_data",  required=True)
    parser.add_argument("--cnd_data",  required=True)
    parser.add_argument("--pred_auc",  type=float, default=0.5,
                        help="Predicted AUC from the optimization loop (lower = better)")
    parser.add_argument("--validity",  type=float, default=1.0,
                        help="Fraction of valid SMILES from optimization loop")
    parser.add_argument("--top_k",     type=int,   default=15)
    parser.add_argument("--model",     default=DEFAULT_MODEL)
    parser.add_argument("--web_search", action="store_true", default=False,
                        help="Enable real-time web search in ChemistryAgent. "
                             "Requires a plan with web_search_20250305 support. "
                             "Disable if you get persistent 429 rate limit errors.")
    parser.add_argument("--save_dir",  default=None)

    # Optional: load diffusion model to run attention extraction
    # If provided, ChemistryAgent gets real attention scores + enriched pathways
    # instead of falling back to genotype only
    parser.add_argument("--diff_ckpt", default=None,
                        help="Path to diffusion model checkpoint (.ckpt). "
                             "If provided, attention analysis is run and "
                             "ChemistryAgent uses model-grounded gene targets.")
    parser.add_argument("--nest_adj",  default="./data/NeST_neighbor_adj.npy",
                    help="Path to NeST adjacency matrix for pathway enrichment")
    parser.add_argument("--nest_sets", default=None,
                        help="Path to named NeST pathway JSON (e.g. nest_pathways_table8.json). "
                            "If provided, enrichment uses biological pathway names instead of "
                            "anonymous connected components.")
    parser.add_argument("--cfgw",      type=float, default=7.0)
    parser.add_argument("--device",    default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S", level=logging.INFO)

    import torch
    device = args.device if torch.cuda.is_available() else "cpu"

    # Apply web search flag
    global USE_WEB_SEARCH
    USE_WEB_SEARCH = args.web_search
    if USE_WEB_SEARCH:
        log.info("Web search ENABLED for ChemistryAgent")
    else:
        log.info("Web search DISABLED (knowledge-only). Use --web_search to enable.")

    cell2mut   = pd.read_csv(args.mut_data, index_col=0).rename(columns={"index": "ccle_name"})
    cell2cna   = pd.read_csv(args.cna_data, index_col=0).rename(columns={"index": "ccle_name"})
    cell2cnd   = pd.read_csv(args.cnd_data, index_col=0).rename(columns={"index": "ccle_name"})
    gene_names = load_gene_names(args.mut_data)

    gene_ctx = get_cell_gene_context(
        args.cell_name, cell2mut, cell2cna, cell2cnd, gene_names, args.top_k
    )
    log.info(f"Cell:      {args.cell_name}")
    log.info(f"Mutated:   {gene_ctx['mutated_genes'][:8]}")
    log.info(f"Amplified: {gene_ctx['amplified_genes']}")
    log.info(f"Deleted:   {gene_ctx['deleted_genes']}")

    # ── Attention extraction (optional — requires diff_ckpt) ──────────────────
    attention_summary = None
    if args.diff_ckpt:
        try:
            from src.g2d_diff_diff import Diffusion
            from src.utils.g2d_diff_geno_dataset import GenoDataset, GenoCollator
            from torch.utils.data import DataLoader
            from attention_analysis import AttentionExtractor

            log.info(f"Loading diffusion model from {args.diff_ckpt} ...")
            diff_model = Diffusion(device=device, training=False, cfgw=args.cfgw).to(device).float()
            ckpt  = torch.load(args.diff_ckpt, map_location=device)
            state = ckpt.get("diffusion_state_dict", ckpt)
            diff_model.load_state_dict(state, strict=False)
            diff_model.eval()
            for p in diff_model.parameters():
                p.requires_grad_(False)

            extractor = AttentionExtractor(
                diff_model=diff_model,
                gene_names=gene_names,
                nest_adj_path=args.nest_adj,
                nest_sets_path=args.nest_sets,   # ← add this line
            )

            # Build a single-sample batch for this cell line
            input_df = pd.DataFrame(
                [(args.cell_name, 0)], columns=["ccle_name", "auc_label"]
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

            att_result = extractor.extract(batch, cell_name=args.cell_name)
            attention_summary = att_result.attention_summary

            log.info(
                f"Attention extracted: {len(att_result.top_genes)} top genes, "
                f"{len(att_result.enriched_pathways)} enriched pathways"
            )
            if att_result.top_genes:
                log.info(f"  Top genes: {att_result.top_genes[:8]}")
            if att_result.enriched_pathways:
                top_p = att_result.enriched_pathways[0]
                log.info(
                    f"  Top pathway: {top_p['pathway']}  "
                    f"(p={top_p['p_value']:.2e}, enrichment={top_p['enrichment']}x)"
                )

        except Exception as e:
            log.warning(f"Attention extraction failed: {e} — running without attention")
            attention_summary = None
    else:
        log.info(
            "No --diff_ckpt provided — ChemistryAgent will use genotype only "
            "(attention_grounded=false). Add --diff_ckpt for model-grounded gene targets."
        )

    # ── Score each molecule ───────────────────────────────────────────────────
    for smi in args.smiles:
        log.info(f"\nScoring: {smi}")
        log.info(
            f"Pipeline: ChemistryAgent (NCI analysis, attention_grounded="
            f"{attention_summary is not None}) → BiochemistryAgent (final score)"
        )
        mol = Chem.MolFromSmiles(smi)
        mol_qed = float(QED.qed(mol)) if mol else 0.0
        mol_sas = float(sascorer.calculateScore(mol)) if mol else 10.0
        result = score_molecule(
            smi, args.cell_name, gene_ctx,
            pred_auc=args.pred_auc,
            qed=mol_qed,
            sas=mol_sas,
            validity=args.validity,
            model=args.model,
            attention_summary=attention_summary,
        )
        print(json.dumps({
            "smiles":            smi,
            "pred_auc":          args.pred_auc,
            "qed":               mol_qed,
            "sas":               mol_sas,
            "attention_grounded": result.get("chemistry", {}).get("attention_grounded", False),
            "final_score":       result["final_score"],
            "nci_score":         result.get("nci_score"),
            "descriptor_score":  result.get("descriptor_score"),
            "confidence":        result["confidence"],
            "recommendation":    result["recommendation"],
            "summary":           result["summary"],
            "key_factors":       result["key_factors"],
            "concerns":          result["concerns"],
            "chemistry_report":  result.get("chemistry", {}),
        }, indent=2))


if __name__ == "__main__":
    main()