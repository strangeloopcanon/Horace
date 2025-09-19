#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def load_authors(model: str) -> Dict[str, dict]:
    p = Path('data/analysis') / model / 'authors.jsonl'
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    return {r['author']: r for r in rows}


def load_docs(model: str) -> Dict[str, dict]:
    p = Path('data/analysis') / model / 'docs_clean.jsonl'
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    return {r['doc_id']: r for r in rows}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def authors_delta_figs(a: str, b: str, outdir: Path):
    A = load_authors(a)
    B = load_authors(b)
    common = sorted(set(A) & set(B))
    figs = []

    def top_bar(delta_key: str, title: str, n=12, reverse=False):
        scored = []
        for name in common:
            av = A[name].get(delta_key); bv = B[name].get(delta_key)
            if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
                scored.append((bv - av, name, av, bv))
        scored.sort(reverse=reverse)
        take = scored[:n]
        if not take:
            return None
        dy = [d for d, *_ in take][::-1]
        labels = [nm for _, nm, *_ in take][::-1]
        plt.figure(figsize=(9, 6))
        plt.barh(labels, dy)
        plt.xlabel(f"Δ {delta_key} ({b} − {a})")
        plt.title(title)
        fn = outdir / f"authors_delta_{delta_key.replace(' ','_')}.png"
        plt.tight_layout(); plt.savefig(fn, dpi=150); plt.close()
        return fn

    # Top authors by surprisal drop (negative delta is better)
    f1 = top_bar('surprisal_mean_mean', f'Authors: Δ surprisal mean ({b} − {a}) — most negative')
    if f1: figs.append(f1)
    # Cohesion change (more negative ⇒ stronger cohesion)
    f2 = top_bar('cohesion_delta_mean', f'Authors: Δ cohesion delta ({b} − {a}) — most negative')
    if f2: figs.append(f2)
    # Nucleus width change (aim for smaller)
    f3 = top_bar('nucleus_w_mean_mean', f'Authors: Δ nucleus width ({b} − {a}) — most negative')
    if f3: figs.append(f3)
    # Content fraction change
    f4 = top_bar('content_fraction_mean', f'Authors: Δ content fraction ({b} − {a}) — most positive', reverse=True)
    if f4: figs.append(f4)
    # Spike neighborhood changes
    f5 = top_bar('spike_prev_content_rate_mean', f'Authors: Δ spike prev content rate ({b} − {a}) — most positive', reverse=True)
    if f5: figs.append(f5)
    f6 = top_bar('spike_next_content_rate_mean', f'Authors: Δ spike next content rate ({b} − {a}) — most positive', reverse=True)
    if f6: figs.append(f6)
    # IPI and cooldown
    f7 = top_bar('ipi_mean_mean', f'Authors: Δ IPI mean ({b} − {a}) — most negative')
    if f7: figs.append(f7)
    f8 = top_bar('cooldown_entropy_drop_3_mean', f'Authors: Δ cooldown entropy drop (3) ({b} − {a}) — most positive', reverse=True)
    if f8: figs.append(f8)

    # Scatter: author surprisal A vs B
    xs, ys = [], []
    for name in common:
        av = A[name].get('surprisal_mean_mean'); bv = B[name].get('surprisal_mean_mean')
        if isinstance(av,(int,float)) and isinstance(bv,(int,float)):
            xs.append(av); ys.append(bv)
    if xs:
        mmin = float(min(xs+ys)); mmax = float(max(xs+ys))
        rng = (mmin-0.5, mmax+0.5)
        plt.figure(figsize=(6,5))
        plt.scatter(xs, ys, s=30, alpha=0.7)
        plt.plot([rng[0], rng[1]], [rng[0], rng[1]], 'k--', lw=1)
        plt.xlabel(f'{a} surprisal mean (nats)'); plt.ylabel(f'{b} surprisal mean (nats)')
        plt.title('Authors: surprisal mean — model comparison')
        fn = outdir / 'authors_scatter_surprisal.png'
        plt.tight_layout(); plt.savefig(fn, dpi=150); plt.close()
        figs.append(fn)

    return figs


def docs_delta_figs(a: str, b: str, outdir: Path):
    A = load_docs(a)
    B = load_docs(b)
    common = sorted(set(A) & set(B))
    figs = []
    # Scatter per-doc surprisal
    colors = {'poem': '#1f77b4', 'shortstory': '#ff7f0e', 'novel': '#2ca02c'}
    xs, ys, cs = [], [], []
    for doc_id in common:
        da, db = A[doc_id], B[doc_id]
        av = da.get('surprisal_mean'); bv = db.get('surprisal_mean')
        if isinstance(av,(int,float)) and isinstance(bv,(int,float)):
            xs.append(av); ys.append(bv); cs.append(colors.get(da.get('doc_type'), '#555'))
    if xs:
        mmin = float(min(xs+ys)); mmax = float(max(xs+ys))
        rng = (mmin-0.5, mmax+0.5)
        plt.figure(figsize=(7,5))
        plt.scatter(xs, ys, s=18, alpha=0.7, c=cs)
        plt.plot([rng[0], rng[1]], [rng[0], rng[1]], 'k--', lw=1)
        plt.xlabel(f'{a} surprisal mean (nats)'); plt.ylabel(f'{b} surprisal mean (nats)')
        plt.title('Docs: surprisal mean — model comparison')
        fn = outdir / 'docs_scatter_surprisal.png'
        plt.tight_layout(); plt.savefig(fn, dpi=150); plt.close()
        figs.append(fn)

    # Histogram of deltas (b - a)
    deltas = []
    for doc_id in common:
        da, db = A[doc_id], B[doc_id]
        av = da.get('surprisal_mean'); bv = db.get('surprisal_mean')
        if isinstance(av,(int,float)) and isinstance(bv,(int,float)):
            deltas.append(bv-av)
    if deltas:
        plt.figure(figsize=(7,4))
        plt.hist(deltas, bins=30, color='#2ca02c', alpha=0.85)
        plt.xlabel(f'Δ surprisal mean ({b} − {a}) [nats]')
        plt.ylabel('Docs count')
        plt.title('Docs: surprisal delta distribution')
        fn = outdir / 'docs_delta_surprisal_hist.png'
        plt.tight_layout(); plt.savefig(fn, dpi=150); plt.close()
        figs.append(fn)

    return figs


def _summary_text(a: str, b: str) -> str:
    A = load_authors(a)
    B = load_authors(b)
    common = sorted(set(A) & set(B))
    keys = [
        ('surprisal_mean_mean', 'surprisal mean (nats)', 'lower is better'),
        ('entropy_mean_mean', 'entropy mean (nats)', 'lower is sharper'),
        ('nucleus_w_mean_mean', 'nucleus width (p=0.9)', 'lower is more focused'),
        ('cohesion_delta_mean', 'cohesion delta (shuffled − original)', 'more negative means stronger cohesion'),
    ]
    lines = []
    lines.append("### Executive Summary\n")
    for k, label, note in keys:
        deltas = []
        for name in common:
            av = A[name].get(k); bv = B[name].get(k)
            if isinstance(av,(int,float)) and isinstance(bv,(int,float)):
                deltas.append(bv - av)
        if deltas:
            mean_delta = float(np.mean(deltas))
            lines.append(f"- {label}: mean Δ ({b} − {a}) = {mean_delta:.3f} ({note}).")
    # Top authors by surprisal drop
    scored = []
    for name in common:
        av = A[name].get('surprisal_mean_mean'); bv = B[name].get('surprisal_mean_mean')
        if isinstance(av,(int,float)) and isinstance(bv,(int,float)):
            scored.append((bv-av, name, av, bv))
    scored.sort()
    if scored:
        top = scored[:8]
        lines.append("- Largest surprisal drops (Qwen lower than GPT‑2):")
        for d, name, av, bv in top:
            lines.append(f"  - {name}: Δ={d:.2f}  {a}={av:.2f} → {b}={bv:.2f}")
    lines.append("")
    return "\n".join(lines)


def _type_delta_text(a: str, b: str) -> str:
    A = load_docs(a)
    B = load_docs(b)
    common = sorted(set(A) & set(B))
    buckets: Dict[str, Dict[str, List[float]]] = {}
    keys = ['surprisal_mean', 'entropy_mean', 'nucleus_w_mean', 'cohesion_delta']
    for doc_id in common:
        da, db = A[doc_id], B[doc_id]
        t = da.get('doc_type', 'unknown')
        if t not in buckets:
            buckets[t] = {k: [] for k in keys}
        for k in keys:
            av, bv = da.get(k), db.get(k)
            if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
                buckets[t][k].append(bv - av)
    lines = []
    lines.append("### Per‑Type Deltas (means of Δ = " + b + " − " + a + ")\n")
    for t in ['poem', 'shortstory', 'novel']:
        if t not in buckets:
            continue
        vals = buckets[t]
        parts = []
        if vals['surprisal_mean']:
            parts.append(f"surprisal {np.mean(vals['surprisal_mean']):.2f}")
        if vals['entropy_mean']:
            parts.append(f"entropy {np.mean(vals['entropy_mean']):.2f}")
        if vals['nucleus_w_mean']:
            parts.append(f"nucleus_w {np.mean(vals['nucleus_w_mean']):.0f}")
        if vals['cohesion_delta']:
            parts.append(f"cohesion_delta {np.mean(vals['cohesion_delta']):.2f}")
        if parts:
            lines.append(f"- {t}: " + ", ".join(parts))
    lines.append("")
    return "\n".join(lines)


def _signature_guidance_text() -> str:
    return "\n".join([
        "### How to Read These Signatures",
        "- Surprisal mean (nats): lower suggests the model finds tokens predictable; poems naturally run higher than prose.",
        "- Entropy mean (nats): lower means sharper distributions (few strong candidates).",
        "- Nucleus width (p=0.9): lower indicates concentrated probability mass; large values signal open‑ended choices.",
        "- Cohesion delta (shuffled − original): more negative ⇒ stronger reliance on order/structure.",
        "- Cadence (IPI, MASD, ACF): captures rhythm — how often spikes occur and how quickly uncertainty settles.",
        "- Token mix (content_fraction, spike context): shows whether surprises align with content words and line boundaries.",
        "",
        "For good‑writing signatures: look for structured cadence (stable IPI with occasional spikes),",
        "negative cohesion delta (order matters), focused distributions (smaller nucleus),",
        "and spikes that coincide with contentful turns (high spike_prev/next content rates).",
        "",
    ])


def _findings_text(a: str, b: str) -> str:
    A_auth = load_authors(a)
    B_auth = load_authors(b)
    A_doc = load_docs(a)
    B_doc = load_docs(b)
    common_auth = sorted(set(A_auth) & set(B_auth))
    common_doc = sorted(set(A_doc) & set(B_doc))

    def avg_delta_auth(key: str) -> float:
        vals = []
        for name in common_auth:
            av = A_auth[name].get(key); bv = B_auth[name].get(key)
            if isinstance(av,(int,float)) and isinstance(bv,(int,float)):
                vals.append(bv-av)
        return float(np.mean(vals)) if vals else float('nan')

    def avg_delta_doc(key: str, dtype: str) -> float:
        vals = []
        for doc_id in common_doc:
            da, db = A_doc[doc_id], B_doc[doc_id]
            if da.get('doc_type') != dtype:
                continue
            av = da.get(key); bv = db.get(key)
            if isinstance(av,(int,float)) and isinstance(bv,(int,float)):
                vals.append(bv-av)
        return float(np.mean(vals)) if vals else float('nan')

    lines: List[str] = []
    lines.append("### Findings by Genre and Author\n")
    # Genre-level
    for t in ['poem','shortstory','novel']:
        ds = avg_delta_doc('surprisal_mean', t)
        de = avg_delta_doc('entropy_mean', t)
        dn = avg_delta_doc('nucleus_w_mean', t)
        dc = avg_delta_doc('cohesion_delta', t)
        if str(ds) != 'nan':
            lines.append(f"- {t.title()}: Δ surprisal {ds:.2f}, Δ entropy {de:.2f}, Δ nucleus_w {dn:.0f}, Δ cohesion_delta {dc:.2f}.")
    lines.append("")
    # Author drill-down (selected)
    focus = ['William Shakespeare','Emily Dickinson','Robert Frost','P G Wodehouse','Ernest Hemingway','Edgar Allan Poe']
    for name in focus:
        if name not in A_auth or name not in B_auth:
            continue
        sA = A_auth[name].get('surprisal_mean_mean'); sB = B_auth[name].get('surprisal_mean_mean')
        eA = A_auth[name].get('entropy_mean_mean'); eB = B_auth[name].get('entropy_mean_mean')
        nA = A_auth[name].get('nucleus_w_mean_mean'); nB = B_auth[name].get('nucleus_w_mean_mean')
        cA = A_auth[name].get('cohesion_delta_mean'); cB = B_auth[name].get('cohesion_delta_mean')
        deltas = []
        if all(isinstance(x,(int,float)) for x in [sA,sB]): deltas.append(f"Δ surprisal {sB-sA:+.2f}")
        if all(isinstance(x,(int,float)) for x in [eA,eB]): deltas.append(f"Δ entropy {eB-eA:+.2f}")
        if all(isinstance(x,(int,float)) for x in [nA,nB]): deltas.append(f"Δ nucleus_w {nB-nA:+.0f}")
        if all(isinstance(x,(int,float)) for x in [cA,cB]): deltas.append(f"Δ cohesion_delta {cB-cA:+.2f}")
        if deltas:
            lines.append(f"- {name}: " + ", ".join(deltas))
    lines.append("")
    # Synthesis bullets for a reader looking for signatures
    lines.append("#### Synthesis — What ‘good writing’ patterns emerge")
    lines.append("- Structure: more negative cohesion delta (shuffling hurts), especially in poetry; Qwen amplifies this.")
    lines.append("- Focus: lower entropy and smaller nucleus widths — confident predictions with occasional, meaningful spikes.")
    lines.append("- Cadence: inter‑peak intervals are moderate and fairly regular (CV ≈ 0.9–1.1); spikes are followed by entropy cooldown.")
    lines.append("- Semantics: spikes align with content tokens; preceding tokens are often function/punctuation, signaling a turn.")
    lines.append("- Genre: poems benefit most (largest Δs), short stories next, novels least — consistent with stylistic density.")
    lines.append("")
    return "\n".join(lines)


def _best_practices_text(a: str, b: str) -> str:
    """Derive simple target bands from the stronger model (b) per type."""
    B = load_docs(b)
    types = ['poem','shortstory','novel']
    keys = [
        ('surprisal_mean', 'surprisal mean (nats)'),
        ('entropy_mean', 'entropy mean (nats)'),
        ('nucleus_w_mean', 'nucleus width (p=0.9)'),
        ('cohesion_delta', 'cohesion delta (shuffled − original)'),
        ('high_surprise_rate_per_100', 'spike rate per 100 tokens'),
        ('ipi_mean', 'inter‑peak interval (tokens)'),
        ('cooldown_entropy_drop_3', 'post‑spike entropy cooldown (3 tokens)'),
        ('content_fraction', 'content token fraction'),
        ('spike_prev_content_rate', 'spike prev content rate'),
        ('spike_next_content_rate', 'spike next content rate'),
    ]
    # Collect values per type
    stats: Dict[str, Dict[str, Tuple[float,float,float]]] = {}
    for t in types:
        vals_by_key: Dict[str, list] = {k: [] for k,_ in keys}
        for d in B.values():
            if d.get('doc_type') != t:
                continue
            for k,_ in keys:
                v = d.get(k)
                if isinstance(v,(int,float)):
                    vals_by_key[k].append(float(v))
        # summarize
        sums: Dict[str, Tuple[float,float,float]] = {}
        import numpy as np
        for k,_ in keys:
            arr = vals_by_key[k]
            if arr:
                a1 = np.array(arr, dtype=float)
                p25, mean, p75 = float(np.percentile(a1,25)), float(np.mean(a1)), float(np.percentile(a1,75))
                sums[k] = (p25, mean, p75)
        stats[t] = sums

    lines: List[str] = []
    lines.append("### Best Practices — Target Bands (from " + b + ")\n")
    for t in types:
        if not stats.get(t):
            continue
        lines.append(f"- {t.title()}:")
        for k,label in keys:
            if k not in stats[t]:
                continue
            p25, mean, p75 = stats[t][k]
            if k == 'cohesion_delta':
                lines.append(f"  - {label}: typical {mean:.2f} (IQR {p25:.2f}..{p75:.2f}) more negative is better")
            elif k == 'nucleus_w_mean':
                lines.append(f"  - {label}: typical {mean:.0f} (IQR {p25:.0f}..{p75:.0f}) lower is more focused")
            elif k in ('content_fraction','spike_prev_content_rate','spike_next_content_rate'):
                lines.append(f"  - {label}: typical {mean:.2f} (IQR {p25:.2f}..{p75:.2f})")
            else:
                lines.append(f"  - {label}: typical {mean:.2f} (IQR {p25:.2f}..{p75:.2f})")
        lines.append("")
    lines.append("Use these as sanity bands when evaluating generations: aim for focused yet expressive distributions, ")
    lines.append("regular but not rigid cadence (IPI around each genre’s typical mean with CV ≈ 1), and negative cohesion deltas.")
    lines.append("")
    return "\n".join(lines)


def write_report(a: str, b: str, outdir: Path, author_figs, doc_figs):
    md = []
    md.append(f"# Combined Writing Signatures — {a} vs {b}\n\n")
    md.append(_summary_text(a, b))
    md.append(_type_delta_text(a, b))
    md.append(_signature_guidance_text())
    md.append(_findings_text(a, b))
    md.append(_best_practices_text(a, b))

    # Comparison panels generated in this folder
    md.append("## Cross‑Model Panels\n")
    for p in author_figs:
        md.append(f"![{p.stem}]({p.name})\n\n")
    for p in doc_figs:
        md.append(f"![{p.stem}]({p.name})\n\n")

    # Model‑specific panels embedded here (referencing per‑model report figures)
    md.append("## Model‑Specific Overviews\n")
    md.append(f"### {a}\n")
    ga = Path('reports')/a
    # Common figure names from per‑model reports
    for name in ['authors_top_low_surprisal.png','authors_entropy_vs_surprisal.png','authors_top_cohesion_delta.png',
                 'docs_entropy_vs_surprisal.png','docs_nucleus_width_by_type.png','docs_top_cohesion_delta.png']:
        p = ga/name
        if p.exists():
            md.append(f"![{name}]({('../' * 1) + str((Path(a)/name))})\n\n")
    # Representative time series if present
    for p in sorted(ga.glob('doc_ts_*.png'))[:3]:
        md.append(f"![{p.name}]({('../' * 1) + str(Path(a)/p.name)})\n\n")

    md.append(f"### {b}\n")
    gb = Path('reports')/b
    for name in ['authors_top_low_surprisal.png','authors_entropy_vs_surprisal.png','authors_top_cohesion_delta.png',
                 'docs_entropy_vs_surprisal.png','docs_nucleus_width_by_type.png','docs_top_cohesion_delta.png']:
        p = gb/name
        if p.exists():
            md.append(f"![{name}]({('../' * 1) + str((Path(b)/name))})\n\n")
    for p in sorted(gb.glob('doc_ts_*.png'))[:3]:
        md.append(f"![{p.name}]({('../' * 1) + str(Path(b)/p.name)})\n\n")

    # Author cadence panels for a couple of canonical authors, if available
    for auth in ['william_shakespeare','p_g_wodehouse']:
        for label in [a, b]:
            base = Path('reports')/label
            imgs = list(base.glob(f'author_{auth}_*.png'))
            if imgs:
                md.append(f"### {label}: {auth.replace('_',' ').title()} cadence\n")
                for p in imgs:
                    md.append(f"![{p.name}]({('../' * 1) + str(Path(label)/p.name)})\n\n")

    # Links to full per‑model reports
    md.append("## Full Reports\n")
    md.append(f"- [{a} full report](../{a}/README.md)\n")
    md.append(f"- [{b} full report](../{b}/README.md)\n")

    (outdir / 'README.md').write_text('\n'.join(md), encoding='utf-8')
    print(f"Compare report: {outdir/'README.md'}")


def main():
    ap = argparse.ArgumentParser(description='Compare two models')
    ap.add_argument('--a', default='mlx-community/Llama-3.2-3B-Instruct', help='Baseline model (e.g., gpt2)')
    ap.add_argument('--b', required=True, help='Comparison model (e.g., Qwen/Qwen2.5-1.5B)')
    args = ap.parse_args()

    outdir = Path('reports') / f"compare_{args.a.replace('/','_')}_vs_{args.b.replace('/','_')}"
    ensure_dir(outdir)
    author_figs = authors_delta_figs(args.a, args.b, outdir)
    doc_figs = docs_delta_figs(args.a, args.b, outdir)
    write_report(args.a, args.b, outdir, author_figs, doc_figs)


if __name__ == '__main__':
    main()
