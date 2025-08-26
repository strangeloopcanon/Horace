#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def load_index(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def make_report(entries: List[Dict]) -> str:
    # Group by model -> preset
    groups: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for e in entries:
        model = e.get('model', 'model')
        preset = e.get('preset', 'default')
        groups[model][preset].append(e)

    lines: List[str] = []
    lines.append('# Generated Samples')
    lines.append('')
    total = sum(len(vv) for v in groups.values() for vv in v.values())
    lines.append(f'- Total samples: {total}')
    lines.append('')
    for model in sorted(groups.keys()):
        lines.append(f'## {model}')
        for preset in sorted(groups[model].keys()):
            lines.append(f'### Preset: {preset}')
            for e in groups[model][preset]:
                prompt = e.get('prompt', '').rstrip()
                base_path = e.get('paths', {}).get('baseline')
                fix_path = e.get('paths', {}).get('fixed')
                # Read texts; fallback to path links
                base_text = ''
                fix_text = ''
                try:
                    if base_path:
                        base_text = Path(base_path).read_text(encoding='utf-8').strip()
                except Exception:
                    pass
                try:
                    if fix_path:
                        fix_text = Path(fix_path).read_text(encoding='utf-8').strip()
                except Exception:
                    pass
                lines.append('')
                lines.append(f'#### Prompt')
                lines.append('')
                lines.append('> ' + prompt.replace('\n', '\n> '))
                lines.append('')
                if base_path or fix_path:
                    lines.append('#### Normal')
                    lines.append('')
                    if base_text:
                        lines.append('```')
                        lines.append(base_text)
                        lines.append('```')
                    else:
                        lines.append(f'(missing) {base_path}')
                    lines.append('')
                    vlist = e.get('paths', {}).get('fixed_variants')
                    if vlist:
                        lines.append('#### Fixed-Up Variants')
                        lines.append('')
                        for i, vp in enumerate(vlist, 1):
                            try:
                                vt = Path(vp).read_text(encoding='utf-8').strip()
                            except Exception:
                                vt = ''
                            lines.append(f'Variant {i}')
                            if vt:
                                lines.append('```')
                                lines.append(vt)
                                lines.append('```')
                            else:
                                lines.append(f'(missing) {vp}')
                            lines.append('')
                    else:
                        lines.append('#### Fixed-Up')
                        lines.append('')
                        if fix_text:
                            lines.append('```')
                            lines.append(fix_text)
                            lines.append('```')
                        else:
                            lines.append(f'(missing) {fix_path}')
                else:
                    # Sampler single-output entries
                    text_path = e.get('paths', {}).get('text')
                    text_content = ''
                    try:
                        if text_path:
                            text_content = Path(text_path).read_text(encoding='utf-8').strip()
                    except Exception:
                        pass
                    lines.append('#### Output')
                    lines.append('')
                    if text_content:
                        lines.append('```')
                        lines.append(text_content)
                        lines.append('```')
                    else:
                        lines.append(f'(missing) {text_path}')
                lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description='Aggregate generated samples into a single markdown page')
    ap.add_argument('--index', default='data/generated/index.jsonl')
    ap.add_argument('--out', default='reports/generated/README.md')
    ap.add_argument('--model', default=None, help='Optional filter by model id substring')
    ap.add_argument('--preset', default=None, help='Optional filter by preset')
    args = ap.parse_args()

    idx = load_index(Path(args.index))
    if args.model:
        idx = [e for e in idx if args.model.lower() in str(e.get('model','')).lower()]
    if args.preset:
        idx = [e for e in idx if str(e.get('preset','')) == args.preset]

    out_text = make_report(idx)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_text, encoding='utf-8')
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
