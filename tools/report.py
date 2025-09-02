#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import math
import gzip
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as e:
    print("Please install matplotlib: pip install matplotlib")
    raise


def _find_signatures_dir(model: str) -> Path:
    """Best-effort: find a signatures_* directory under reports/ that matches the model.
    For model 'gpt2' → look for 'gpt2' in dirname; for 'qwen' models → look for 'qwen'.
    Pick the most recent by classification_report.json mtime if multiple.
    """
    root = Path('reports')
    want = 'gpt2' if 'gpt2' in model.lower() else ('qwen' if 'qwen' in model.lower() else None)
    cands = []
    for p in root.glob('signatures_*'):
        if not p.is_dir():
            continue
        name = p.name.lower()
        if want and want not in name:
            continue
        rep = p / 'classification_report.json'
        if rep.exists():
            cands.append((rep.stat().st_mtime, p))
    if not cands:
        return Path()
    cands.sort(key=lambda t: t[0], reverse=True)
    return cands[0][1]


def build_signatures_md(model: str, outdir: Path) -> str:
    sig_dir = _find_signatures_dir(model)
    if not sig_dir:
        return ''
    rep = sig_dir / 'classification_report.json'
    try:
        data = json.loads(rep.read_text(encoding='utf-8'))
    except Exception:
        return ''
    acc = data.get('acc_chunk_level')
    n_docs = data.get('n_docs')
    n_chunks = data.get('n_chunks')
    authors = data.get('authors') or []
    fig = data.get('confusion_figure')
    md = []
    md.append('## Signatures (Classification)\n')
    md.append(f"Directory: {sig_dir.name}\n\n")
    md.append(f"- Accuracy: {acc if acc is not None else 'N/A'}\n")
    md.append(f"- Documents: {n_docs}\n")
    md.append(f"- Chunks: {n_chunks}\n")
    md.append(f"- Authors: {len(authors)}\n")
    if fig and Path(fig).exists():
        rel = os.path.relpath(fig, outdir)
        md.append(f"\n![confusion_matrix]({rel})\n")
    return '\n'.join(md)


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def plot_authors(model: str, outdir: Path):
    base = Path('data/analysis') / model
    authors_path = base / 'authors.jsonl'
    authors = load_jsonl(authors_path)
    if not authors:
        print(f"No authors.jsonl at {authors_path}; run authors first.")
        return []

    figs = []

    # Top 10 lowest surprisal_mean_mean
    key = 'surprisal_mean_mean'
    have_key = [a for a in authors if a.get(key) is not None]
    have_key.sort(key=lambda r: r[key])
    top = have_key[:10]
    plt.figure(figsize=(8, 5))
    plt.barh([r['author'] for r in top][::-1], [r[key] for r in top][::-1])
    plt.xlabel('Surprisal mean (nats)')
    plt.title(f'Authors: Lowest surprisal ({model})')
    fn1 = outdir / 'authors_top_low_surprisal.png'
    plt.tight_layout(); plt.savefig(fn1, dpi=150); plt.close()
    figs.append(fn1)

    # Scatter: entropy_mean_mean vs surprisal_mean_mean, size by tokens_total
    xkey, ykey = 'entropy_mean_mean', 'surprisal_mean_mean'
    xs, ys, sizes, names = [], [], [], []
    for r in authors:
        if r.get(xkey) is None or r.get(ykey) is None:
            continue
        xs.append(r[xkey])
        ys.append(r[ykey])
        sizes.append(max(10.0, math.sqrt(r.get('tokens_total', 1))))
        names.append(r['author'])
    plt.figure(figsize=(6, 5))
    plt.scatter(xs, ys, s=sizes, alpha=0.6)
    plt.xlabel('Entropy mean (nats)')
    plt.ylabel('Surprisal mean (nats)')
    plt.title(f'Authors: Entropy vs Surprisal — {model} (size ~ tokens)')
    fn2 = outdir / 'authors_entropy_vs_surprisal.png'
    plt.tight_layout(); plt.savefig(fn2, dpi=150); plt.close()
    figs.append(fn2)

    # Bar: cohesion_delta_mean (if present)
    key_cd = 'cohesion_delta_mean'
    have_cd = [a for a in authors if a.get(key_cd) is not None]
    if have_cd:
        have_cd.sort(key=lambda r: r[key_cd], reverse=True)
        top_cd = have_cd[:10]
        plt.figure(figsize=(8, 5))
        plt.barh([r['author'] for r in top_cd][::-1], [r[key_cd] for r in top_cd][::-1])
        plt.xlabel('Cohesion delta (shuffled − original)')
        plt.title(f'Authors: Highest cohesion delta — {model}')
        fn3 = outdir / 'authors_top_cohesion_delta.png'
        plt.tight_layout(); plt.savefig(fn3, dpi=150); plt.close()
        figs.append(fn3)

    return figs


def plot_docs(model: str, outdir: Path):
    base = Path('data/analysis') / model
    docs_path = base / 'docs_clean.jsonl'
    docs = load_jsonl(docs_path)
    if not docs:
        print(f"No docs_clean.jsonl at {docs_path}; run docs first.")
        return []
    figs = []

    # Scatter: per-doc entropy vs surprisal, color by type
    colors = {'poem': '#1f77b4', 'shortstory': '#ff7f0e', 'novel': '#2ca02c'}
    plt.figure(figsize=(7, 5))
    for t, col in colors.items():
        pts = [r for r in docs if r.get('doc_type') == t and r.get('entropy_mean') is not None]
        if not pts:
            continue
        plt.scatter([r['entropy_mean'] for r in pts], [r['surprisal_mean'] for r in pts],
                    s=12, alpha=0.7, label=t, c=col)
    plt.xlabel('Entropy mean (nats)')
    plt.ylabel('Surprisal mean (nats)')
    plt.title(f'Docs: Entropy vs Surprisal — {model}')
    plt.legend()
    fn1 = outdir / 'docs_entropy_vs_surprisal.png'
    plt.tight_layout(); plt.savefig(fn1, dpi=150); plt.close()
    figs.append(fn1)

    # Boxplot: nucleus_w_mean by type
    groups = []
    labels = []
    for t in ['poem', 'shortstory', 'novel']:
        vals = [r['nucleus_w_mean'] for r in docs if r.get('doc_type') == t and r.get('nucleus_w_mean') is not None]
        if vals:
            groups.append(vals)
            labels.append(t)
    if groups:
        plt.figure(figsize=(6, 4))
        plt.boxplot(groups, labels=labels, showfliers=False)
        plt.ylabel('Nucleus width (p=0.9)')
        plt.title(f'Docs: Nucleus width by type — {model}')
        fn2 = outdir / 'docs_nucleus_width_by_type.png'
        plt.tight_layout(); plt.savefig(fn2, dpi=150); plt.close()
        figs.append(fn2)

    # Top docs by cohesion_delta if available
    if any('cohesion_delta' in r for r in docs):
        have = [r for r in docs if r.get('cohesion_delta') is not None]
        if have:
            have.sort(key=lambda r: r['cohesion_delta'], reverse=True)
            top = have[:12]
            names = [f"{r['author']} — {r['title']}" for r in top]
            vals = [r['cohesion_delta'] for r in top]
            plt.figure(figsize=(9, 6))
            plt.barh(names[::-1], vals[::-1])
            plt.xlabel('Cohesion delta (shuffled − original)')
            plt.title(f'Docs: Highest cohesion delta — {model}')
            fn3 = outdir / 'docs_top_cohesion_delta.png'
            plt.tight_layout(); plt.savefig(fn3, dpi=150); plt.close()
            figs.append(fn3)

    return figs


def _pick_representative_docs(model: str) -> List[Dict]:
    # pick one doc per type closest to median surprisal_mean
    base = Path('data/analysis') / model
    docs = load_jsonl(base / 'docs_clean.jsonl')
    reps = []
    for t in ['poem', 'shortstory', 'novel']:
        cand = [r for r in docs if r.get('doc_type') == t and r.get('surprisal_mean') is not None]
        if not cand:
            continue
        med = np.median([r['surprisal_mean'] for r in cand])
        cand.sort(key=lambda r: abs(r['surprisal_mean'] - med))
        reps.append(cand[0])
    return reps


def plot_doc_timeseries(model: str, outdir: Path):
    reps = _pick_representative_docs(model)
    if not reps:
        return []
    base = Path('data/analysis') / model / 'tokens'
    figs = []
    for r in reps:
        doc_id = r['doc_id']
        tok_path = base / f"{doc_id}.jsonl.gz"
        if not tok_path.exists():
            continue
        s = []
        with gzip.open(tok_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                s.append(-row.get('logp_true', 0.0))
                if len(s) >= 600:  # keep plot readable
                    break
        if len(s) < 10:
            continue
        s = np.array(s)
        mu = float(np.mean(s)); sd = float(np.std(s)); thr = mu + sd
        plt.figure(figsize=(9, 3))
        plt.plot(s, lw=0.8)
        plt.axhline(thr, color='red', linestyle='--', alpha=0.6, label='high-surprise threshold')
        plt.title(f"{r['author']} — {r['title']} ({r['doc_type']})")
        plt.xlabel('Token index (first ~600)')
        plt.ylabel('Surprisal (nats)')
        plt.legend()
        fn = outdir / f"doc_ts_{doc_id}.png"
        plt.tight_layout(); plt.savefig(fn, dpi=150); plt.close()
        figs.append(fn)
    return figs


def plot_author_cadence(model: str, outdir: Path, authors_want: List[str]):
    base = Path('data/analysis') / model
    docs = load_jsonl(base / 'docs_clean.jsonl')
    tok_dir = base / 'tokens'
    figs = []
    for author in authors_want:
        sel = [d for d in docs if d.get('author','').lower() == author.lower()]
        if not sel:
            continue
        # Gather surprisal sequences (first 1000 tokens per doc)
        series = []
        for d in sel:
            p = tok_dir / f"{d['doc_id']}.jsonl.gz"
            if not p.exists():
                continue
            s = []
            try:
                import gzip, json
                with gzip.open(p, 'rt', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        r = json.loads(line)
                        s.append(-r.get('logp_true', 0.0))
                        if i >= 999:
                            break
                if len(s) >= 20:
                    series.append(s)
            except Exception:
                pass
        if not series:
            continue
        # Panel 1: overlay density of surprisal
        import numpy as np
        plt.figure(figsize=(7, 4))
        for s in series[:6]:
            arr = np.array(s)
            plt.hist(arr, bins=40, density=True, alpha=0.3)
        plt.xlabel('Surprisal (nats)'); plt.ylabel('Density')
        plt.title(f'{author}: Surprisal distributions (first 1k tokens)')
        fn1 = outdir / f"author_{author.lower().replace(' ','_')}_surprisal_hist.png"
        plt.tight_layout(); plt.savefig(fn1, dpi=150); plt.close()
        figs.append(fn1)

        # Panel 2: inter-peak interval histogram (using mean+std threshold)
        ipis_all = []
        for s in series:
            arr = np.array(s)
            mu, sd = float(np.mean(arr)), float(np.std(arr))
            thr = mu + sd
            peaks = [i for i in range(1, len(arr)-1) if arr[i] >= thr and arr[i] > arr[i-1] and arr[i] >= arr[i+1]]
            if len(peaks) > 1:
                ipi = np.diff(np.array(peaks))
                ipis_all.extend(ipi.tolist())
        if ipis_all:
            plt.figure(figsize=(7, 4))
            plt.hist(ipis_all, bins=30, color='#9467bd', alpha=0.8)
            plt.xlabel('Inter-peak interval (tokens)'); plt.ylabel('Count')
            plt.title(f'{author}: High-surprise inter-peak intervals')
            fn2 = outdir / f"author_{author.lower().replace(' ','_')}_ipi_hist.png"
            plt.tight_layout(); plt.savefig(fn2, dpi=150); plt.close()
            figs.append(fn2)

        # Panel 3: small multiples time series for up to 6 docs
        k = min(6, len(series))
        plt.figure(figsize=(12, 6))
        for i in range(k):
            arr = np.array(series[i])
            mu, sd = float(np.mean(arr)), float(np.std(arr))
            thr = mu + sd
            ax = plt.subplot(2, 3, i+1)
            ax.plot(arr, lw=0.7)
            ax.axhline(thr, color='red', linestyle='--', alpha=0.5)
            ax.set_title(f'doc {i+1}')
        plt.suptitle(f'{author}: Surprisal time series (first 1k tokens)')
        fn3 = outdir / f"author_{author.lower().replace(' ','_')}_series.png"
        plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(fn3, dpi=150); plt.close()
        figs.append(fn3)
    return figs


def write_report(model: str, outdir: Path, author_figs: List[Path], doc_figs: List[Path], author_cadence_figs: List[Path]):
    md = []
    md.append(f"# Writing Signatures — {model}\n")
    md.append("\n")
    # Inject signatures classification summary if available
    sig_md = build_signatures_md(model, outdir)
    if sig_md:
        md.append(sig_md)
        md.append("\n")
    md.append("## Authors\n")
    for p in author_figs:
        rel = p.name
        md.append(f"![{p.stem}]({rel})\n\n")
    md.append("## Documents\n")
    for p in doc_figs:
        rel = p.name
        md.append(f"![{p.stem}]({rel})\n\n")
    if author_cadence_figs:
        md.append("## Author Cadence\n")
        for p in author_cadence_figs:
            rel = p.name
            md.append(f"![{p.stem}]({rel})\n\n")
    (outdir / 'README.md').write_text('\n'.join(md), encoding='utf-8')
    print(f"Report written to {outdir / 'README.md'}")


def main():
    ap = argparse.ArgumentParser(description='Generate author/document signature report')
    ap.add_argument('--model', default='gpt2')
    args = ap.parse_args()

    outdir = Path('reports') / args.model
    ensure_dir(outdir)
    author_figs = plot_authors(args.model, outdir)
    doc_figs = plot_docs(args.model, outdir)
    ts_figs = plot_doc_timeseries(args.model, outdir)
    # Author cadence for Shakespeare and PG Wodehouse if present
    want = ['William Shakespeare', 'P G Wodehouse']
    ac_figs = plot_author_cadence(args.model, outdir, want)
    write_report(args.model, outdir, author_figs, doc_figs + ts_figs, ac_figs)


if __name__ == '__main__':
    main()
