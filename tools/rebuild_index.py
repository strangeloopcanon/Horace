#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def slug_to_display(s: str) -> str:
    parts = [p for p in s.replace('-', '_').split('_') if p]
    return ' '.join(p[:1].upper() + p[1:] for p in parts)


def title_from_filename(stem: str) -> str:
    # Filenames are Title_Case or lowercase_with_underscores
    s = stem.replace('_', ' ').strip()
    # Normalize multiple spaces
    s = ' '.join(s.split())
    # Title-case for display consistency
    return s.title()


def gather() -> list:
    entries = []
    for doc_type in ['poem', 'shortstory', 'novel']:
        base = DATA / doc_type
        if not base.exists():
            continue
        for author_dir in sorted(base.iterdir()):
            if not author_dir.is_dir():
                continue
            author_slug = author_dir.name
            author = slug_to_display(author_slug)
            for fp in sorted(author_dir.glob('*.txt')):
                title = title_from_filename(fp.stem)
                rel = fp.relative_to(ROOT).as_posix()
                entries.append({
                    'type': doc_type,
                    'author': author,
                    'title': title,
                    'path': rel,
                })
    # Stable order: type, author, title
    entries.sort(key=lambda r: (r['type'], r['author'], r['title']))
    return entries


def main():
    out = DATA / 'index.json'
    entries = gather()
    out.write_text(json.dumps(entries, indent=2), encoding='utf-8')
    print(f"Wrote {len(entries)} entries to {out}")


if __name__ == '__main__':
    main()

