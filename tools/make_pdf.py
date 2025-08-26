#!/usr/bin/env python3
import argparse
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def build_pdf(model: str, out_path: Path):
    report_dir = Path('reports') / model
    imgs = sorted([p for p in report_dir.glob('*.png')])
    if not imgs:
        raise SystemExit(f"No PNGs found in {report_dir}. Run: python tools/report.py --model {model}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    title = f"Writing Signatures — {model}"
    with PdfPages(out_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.6, title, ha='center', va='center', fontsize=24)
        subtitle = "Per-token distributions, doc/author signatures, and cadence metrics"
        fig.text(0.5, 0.54, subtitle, ha='center', va='center', fontsize=12)
        fig.text(0.5, 0.3, f"Report directory: {report_dir}", ha='center', fontsize=9)
        pdf.savefig(fig); plt.close(fig)

        # One page per image with caption
        for img in imgs:
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0.05, 0.12, 0.9, 0.8])
            im = plt.imread(img)
            ax.imshow(im)
            ax.axis('off')
            cap = img.name.replace('_', ' ').replace('.png', '')
            fig.text(0.5, 0.05, cap, ha='center', fontsize=10)
            pdf.savefig(fig); plt.close(fig)
    print(f"PDF written to {out_path}")


def main():
    ap = argparse.ArgumentParser(description='Create a PDF with report figures')
    ap.add_argument('--model', default='gpt2')
    ap.add_argument('--out', default=None, help='Output PDF path (default: reports/<model>/report.pdf)')
    args = ap.parse_args()

    out = Path(args.out) if args.out else (Path('reports') / args.model / 'report.pdf')
    build_pdf(args.model, out)


if __name__ == '__main__':
    main()

