#!/usr/bin/env python3
"""
Local web UI for cadence-controlled generation with optional author-guided tuning.

Dependencies: gradio (see requirements.txt)

Usage:
  python tools/ui.py --host 127.0.0.1 --port 7860
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Optional

# Allow running both as a module (-m tools.ui) and as a script (python tools/ui.py)
if __package__ in (None, ""):
    # Add repository root to sys.path so that "tools.*" imports resolve
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import gradio as gr

from tools.analyze import pick_backend
from tools.gen_compare import run_compare  # HF generate() path with logits-processor
from tools.sampler import (
    PRESETS,
    CadenceSampler,
    _load_author_stats,
    _match_author,
    _adjust_config_from_author,
    _generate_baseline,
)


def _slugify(s: str) -> str:
    s = s.strip().lower().replace("\n", " ")
    s = " ".join(s.split())
    keep = []
    for ch in s:
        if ch.isalnum():
            keep.append(ch)
        elif ch in (" ", "-", "_"):
            keep.append("_")
    slug = "".join(keep).strip("_") or "prompt"
    return slug[:64]


def _safe_model_id(mid: str) -> str:
    return "_".join([p for p in mid.replace("/", "_").split("_") if p])


def _save_outputs(
    model: str,
    preset: str,
    prompt: str,
    fixed_text: str,
    base_text: Optional[str],
    author_seed: Optional[str],
    author_stats_model: Optional[str],
) -> Dict:
    model_dir = _safe_model_id(model)
    prompt_slug = _slugify(prompt)
    h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
    tag = (author_seed or "").strip().lower().replace(" ", "_")
    preset_leaf = f"sampler_{preset}" + (f"_author_{tag}" if tag else "")
    base_dir = Path("data/generated") / model_dir / preset_leaf / f"{prompt_slug}_{h}"
    base_dir.parent.mkdir(parents=True, exist_ok=True)
    base_dir.mkdir(parents=True, exist_ok=True)

    meta: Dict = {
        "model": model,
        "preset": f"sampler_{preset}",
        "prompt": prompt,
    }
    paths: Dict[str, str] = {}
    if base_text is not None:
        (base_dir / "baseline.txt").write_text(base_text, encoding="utf-8")
        (base_dir / "fixed.txt").write_text(fixed_text, encoding="utf-8")
        paths["baseline"] = str(base_dir / "baseline.txt")
        paths["fixed"] = str(base_dir / "fixed.txt")
    else:
        (base_dir / "output.txt").write_text(fixed_text, encoding="utf-8")
        paths["text"] = str(base_dir / "output.txt")
    if author_seed:
        meta["author_seed"] = author_seed
    if author_stats_model:
        meta["author_stats_model"] = author_stats_model
    meta["paths"] = paths
    (base_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    # append to index and refresh summary page
    idx = Path("data/generated/index.jsonl")
    idx.parent.mkdir(parents=True, exist_ok=True)
    with idx.open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    try:
        from tools.show_generated import load_index, make_report  # type: ignore

        entries = load_index(idx)
        report = make_report(entries)
        out_path = Path("reports/generated/README.md")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
    except Exception:
        pass
    return {"dir": str(base_dir), "meta": meta}


def _list_authors_for_model(stats_model: str) -> list[str]:
    rows = _load_author_stats(stats_model or "")
    authors = sorted({r.get("author", "") for r in rows if r.get("author")})
    return ["(none)"] + authors


def _refresh_authors(stats_model: str, model: str):
    # If no explicit stats model, use model
    sm = (stats_model or model or "").strip()
    if not sm:
        return gr.update(choices=["(none)"], value="(none)")
    try:
        choices = _list_authors_for_model(sm)
    except Exception:
        choices = ["(none)"]
    return gr.update(choices=choices, value=choices[0] if choices else "(none)")


def _preset_label(preset: str) -> str:
    m = {
        'sonnet': 'a Shakespearean sonnet',
        'couplets': 'a poem in rhymed couplets',
        'poetry_default': 'a poem',
        'freeverse': 'a free‑verse poem',
        'default': 'a poem',
        'imagist': 'an imagist poem',
        'prose': 'a short prose passage',
    }
    return m.get(preset, 'a poem')


def generate(
    model: str,
    backend: str,
    preset: str,
    prompt: str,
    max_new_tokens: int,
    seed: Optional[int],
    author_seed_dd: str,
    author_seed_custom: Optional[str],
    author_stats_model: Optional[str],
    engine: str,
    rhyme_enabled: bool,
    rhyme_scheme: Optional[str],
    rhyme_boost: Optional[float],
    line_target: Optional[str],
    also_baseline: bool,
    baseline_temp: float,
    baseline_top_p: float,
    baseline_top_k: Optional[int],
    save_outputs: bool,
    safe_mode: bool,
    clean_unicode: bool,
    style_in_prompt: bool,
):
    try:
        base_text: Optional[str] = None
        fixed_text: str = ""
        # Resolve stats model early so it's available for metadata regardless of engine
        base_for_stats = author_stats_model.strip() if (author_stats_model and author_stats_model.strip()) else model

        # Resolve final style seed: prefer custom text, else dropdown unless "(none)"
        final_author_seed = None
        if author_seed_custom and author_seed_custom.strip():
            final_author_seed = author_seed_custom.strip()
        elif author_seed_dd and author_seed_dd.strip() and author_seed_dd.strip() != "(none)":
            final_author_seed = author_seed_dd.strip()

        if engine == "hf_generate":
            # Map sampler presets to gen_compare presets
            preset_map = {
                'poetry_default': 'default',
                'freeverse': 'imagist',
                'sonnet': 'sonnet',
                'couplets': 'couplets',
                'dickinson': 'default',  # closest available
            }
            gc_preset = preset_map.get(preset, 'default')
            # Combine author seed with prompt for gentle framing (optional)
            use_prompt = prompt or "At dawn, the city leans into light:\n"
            if final_author_seed and style_in_prompt:
                style_label = _preset_label(preset)
                directive = f"Write {style_label} in the style of {final_author_seed}.\n\n"
                use_prompt = directive + use_prompt
            # run_compare returns baseline and fixed-up using HF generate() with a logits processor
            base_text_gc, fixed_text_gc = run_compare(
                model,
                use_prompt,
                max_new_tokens,
                seed,
                gc_preset,
                author_seed=final_author_seed,
                author_stats_model=base_for_stats,
            )
            fixed_text = fixed_text_gc
            if also_baseline:
                base_text = base_text_gc
        else:
            # Manual sampler (existing path)
            backend_obj = pick_backend(model, prefer_mlx=True, backend=backend)
            cfg = PRESETS[preset]()
            if rhyme_enabled:
                setattr(cfg, "rhyme_enabled", True)
            if rhyme_scheme:
                setattr(cfg, "rhyme_scheme", rhyme_scheme)
                setattr(cfg, "rhyme_enabled", True)
            if rhyme_boost is not None:
                setattr(cfg, "rhyme_boost", float(rhyme_boost))
            if line_target:
                try:
                    a, b = [int(x.strip()) for x in line_target.split(",")]
                    setattr(cfg, "line_tokens_target", (a, b))
                except Exception:
                    pass
            if final_author_seed:
                rows = _load_author_stats(base_for_stats)
                row = _match_author(rows, final_author_seed)
                if row:
                    cfg = _adjust_config_from_author(cfg, row)
            if safe_mode:
                try:
                    cfg.spike.content_boost = 0.0
                    cfg.spike.stop_punct_penalty = 0.0
                    if getattr(cfg.base, 'top_k', None) is not None:
                        cfg.base.top_k = min(int(cfg.base.top_k or 0) or 200, 200)
                    if getattr(cfg.spike, 'top_k', None) is not None:
                        cfg.spike.top_k = min(int(cfg.spike.top_k or 0) or 220, 220)
                    if getattr(cfg.cool, 'top_k', None) is not None:
                        cfg.cool.top_k = min(int(cfg.cool.top_k or 0) or 120, 120)
                except Exception:
                    pass
            sampler = CadenceSampler(backend_obj, cfg, seed=seed, debug=False)
            use_prompt = prompt or "Write a short poem about dawn in the city.\n"
            if final_author_seed and style_in_prompt:
                style_label = _preset_label(preset)
                directive = f"Write {style_label} in the style of {final_author_seed}.\n\n"
                use_prompt = directive + use_prompt
            fixed_text = sampler.generate(use_prompt, max_new_tokens=max_new_tokens)
            if also_baseline:
                base_text = _generate_baseline(
                    backend_obj,
                    use_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=baseline_temp,
                    top_p=baseline_top_p,
                    top_k=baseline_top_k,
                    seed=seed,
                )

        # Optional cleanup of bad replacement chars
        def _clean(s: Optional[str]) -> Optional[str]:
            if not s:
                return s
            if clean_unicode:
                try:
                    s = s.replace('\ufffd', '')
                    # Strip other non-printables except whitespace/newlines
                    s = ''.join(ch for ch in s if ch.isprintable() or ch in '\n\r\t ')
                except Exception:
                    pass
            return s

        base_text = _clean(base_text)
        fixed_text = _clean(fixed_text)

        saved = {}
        if save_outputs:
            saved = _save_outputs(
                model=model,
                preset=preset,
                prompt=prompt,
                fixed_text=fixed_text,
                base_text=base_text,
                author_seed=final_author_seed,
                author_stats_model=base_for_stats if final_author_seed else None,
            )

        status = "OK"
        if saved:
            status += f" — saved to {saved.get('dir')}"
        return base_text or "(baseline disabled)", fixed_text, status
    except Exception as e:
        return "", "", f"Error: {e}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Horace — Cadence Sampler UI") as demo:
        gr.Markdown("""
        # Horace — Cadence Sampler
        Generate baseline vs cadence‑controlled snippets. Optionally tune cadence using an author seed from your analysis stats.
        """)
        with gr.Row():
            model = gr.Textbox(value="Qwen/Qwen2.5-1.5B", label="Model (HF or MLX id)")
            backend = gr.Radio(choices=["auto", "mlx", "hf"], value="auto", label="Backend (manual engine)")
            preset = gr.Dropdown(choices=sorted(PRESETS.keys()), value="poetry_default", label="Preset")
            seed = gr.Number(value=None, label="Seed", precision=0)
            max_new = gr.Slider(minimum=16, maximum=512, value=120, step=1, label="Max new tokens")

        prompt = gr.Textbox(value="At dawn, the city leans into light:\n", lines=4, label="Prompt")

        with gr.Row():
            author_stats_model = gr.Textbox(value="", label="Stats from model (default: same as Model)")
            refresh_btn = gr.Button(value="↻ Refresh authors", scale=0)

        # Author seed dropdown and custom
        try:
            initial_authors = _list_authors_for_model("Qwen/Qwen2.5-1.5B")
        except Exception:
            initial_authors = ["(none)"]
        author_seed_dd = gr.Dropdown(choices=initial_authors, value=initial_authors[0], label="Style seed (author)")
        author_seed_custom = gr.Textbox(value="", label="Or custom author (overrides dropdown)")

        engine = gr.Radio(choices=["hf_generate", "manual"], value="hf_generate", label="Engine (recommended: hf_generate)")

        with gr.Accordion("Rhyme & Line (optional)", open=False):
            rhyme = gr.Checkbox(value=False, label="Enable rhyme nudging")
            rhyme_scheme = gr.Textbox(value="", label="Rhyme scheme (e.g., ABAB CDCD EFEF GG)")
            rhyme_boost = gr.Number(value=None, label="Rhyme boost (e.g., 0.8)")
            line_target = gr.Textbox(value="", label="Line token target (e.g., 8,12)")

        with gr.Accordion("Baseline settings", open=False):
            also_baseline = gr.Checkbox(value=True, label="Also generate baseline")
            bl_temp = gr.Slider(minimum=0.1, maximum=1.5, value=0.85, step=0.01, label="Baseline temperature")
            bl_top_p = gr.Slider(minimum=0.05, maximum=0.99, value=0.92, step=0.01, label="Baseline top_p")
            bl_top_k = gr.Number(value=None, label="Baseline top_k", precision=0)

        with gr.Row():
            save_outputs = gr.Checkbox(value=True, label="Save outputs to data/generated and update reports")
            safe_mode = gr.Checkbox(value=True, label="Safe mode (softer biases; avoid artifacts)")
            clean_unicode = gr.Checkbox(value=True, label="Clean replacement characters (�) from outputs")
            style_in_prompt = gr.Checkbox(value=True, label="Add style instruction to prompt")

        run_btn = gr.Button("Generate")
        status = gr.Markdown()
        with gr.Row():
            base_out = gr.Textbox(label="Normal (baseline)", lines=16)
            fixed_out = gr.Textbox(label="Fixed‑Up (cadence‑controlled)", lines=16)

        run_btn.click(
            fn=generate,
            inputs=[
                model,
                backend,
                preset,
                prompt,
                max_new,
                seed,
                author_seed_dd,
                author_seed_custom,
                author_stats_model,
                engine,
                rhyme,
                rhyme_scheme,
                rhyme_boost,
                line_target,
                also_baseline,
                bl_temp,
                bl_top_p,
                bl_top_k,
                save_outputs,
                safe_mode,
                clean_unicode,
                style_in_prompt,
            ],
            outputs=[base_out, fixed_out, status],
        )

        # Refresh authors list based on stats model or base model
        refresh_btn.click(
            fn=_refresh_authors,
            inputs=[author_stats_model, model],
            outputs=[author_seed_dd],
        )
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description="Horace Cadence Sampler UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
