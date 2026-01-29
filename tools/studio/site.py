from __future__ import annotations


STUDIO_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Horace Studio</title>
    <style>
      :root {
        --bg: #0b1020;
        --panel: #111a33;
        --muted: #92a0c8;
        --text: #e8eeff;
        --border: rgba(255, 255, 255, 0.10);
        --accent: #6ea8ff;
        --bad: #ff6e6e;
        --good: #7dffb2;
        --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, "Apple Color Emoji",
          "Segoe UI Emoji", "Segoe UI Symbol";
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: var(--sans);
        background: radial-gradient(1200px 800px at 30% 0%, rgba(110, 168, 255, 0.18), transparent 55%),
          radial-gradient(1000px 700px at 90% 10%, rgba(125, 255, 178, 0.10), transparent 55%),
          var(--bg);
        color: var(--text);
      }
      a { color: var(--accent); text-decoration: none; }
      .wrap { max-width: 1100px; margin: 0 auto; padding: 18px 16px 48px; }
      header { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }
      h1 { margin: 0; font-size: 20px; letter-spacing: 0.2px; }
      .sub { color: var(--muted); font-size: 13px; margin-top: 4px; }
      .grid { display: grid; grid-template-columns: 1.05fr 0.95fr; gap: 14px; margin-top: 14px; }
      @media (max-width: 980px) { .grid { grid-template-columns: 1fr; } }
      .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px;
      }
      label { display: block; font-size: 12px; color: var(--muted); margin: 10px 0 6px; }
      input, select, textarea {
        width: 100%;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.14);
        background: rgba(0,0,0,0.22);
        color: var(--text);
        padding: 10px 10px;
        font-size: 13px;
        outline: none;
      }
      textarea { min-height: 280px; resize: vertical; font-family: var(--sans); line-height: 1.35; }
      .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
      .row3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
      .btns { display: flex; gap: 10px; margin-top: 12px; flex-wrap: wrap; }
      button {
        border: 1px solid rgba(255,255,255,0.14);
        background: rgba(110,168,255,0.14);
        color: var(--text);
        padding: 10px 12px;
        border-radius: 12px;
        cursor: pointer;
        font-size: 13px;
      }
      button.secondary { background: rgba(255,255,255,0.06); }
      button:disabled { opacity: 0.55; cursor: not-allowed; }
      details { margin-top: 10px; }
      summary { cursor: pointer; color: var(--muted); font-size: 13px; }
      .status { margin-top: 10px; font-family: var(--mono); font-size: 12px; color: var(--muted); white-space: pre-wrap; }
      .pill {
        display: inline-block;
        font-family: var(--mono);
        font-size: 12px;
        padding: 4px 8px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.14);
        background: rgba(0,0,0,0.18);
        color: var(--muted);
      }
      .score {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        margin-top: 6px;
      }
      .score strong { font-size: 22px; letter-spacing: 0.2px; }
      .kv { margin-top: 10px; display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
      .kv div { padding: 10px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.10); background: rgba(0,0,0,0.16); }
      .kv .k { font-size: 11px; color: var(--muted); font-family: var(--mono); }
      .kv .v { font-size: 13px; margin-top: 6px; font-family: var(--mono); }
      canvas { width: 100%; height: 120px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.10); background: rgba(0,0,0,0.16); margin-top: 10px; }
      .list { margin-top: 10px; padding-left: 18px; }
      .list li { margin: 8px 0; color: var(--text); }
      .muted { color: var(--muted); }
      .box { margin-top: 10px; padding: 10px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.10); background: rgba(0,0,0,0.16); }
      pre { white-space: pre-wrap; word-break: break-word; margin: 0; font-family: var(--mono); font-size: 12px; color: var(--muted); }
    </style>
  </head>
  <body>
    <div class="wrap">
      <header>
        <div>
          <h1>Horace Studio</h1>
          <div class="sub">Paste writing → score, profile vs baselines, grounded suggestions, optional rewrites.</div>
        </div>
        <div class="pill">/analyze + /rewrite</div>
      </header>

      <div class="grid">
        <div class="card">
          <label for="text">Your text</label>
          <textarea id="text">At dawn, the city leans into light:
A gull lifts, then drops, then lifts again.</textarea>

          <div class="row">
            <div>
              <label>Type</label>
              <select id="doc_type">
                <option value="prose" selected>prose</option>
                <option value="poem">poem</option>
                <option value="shortstory">shortstory</option>
                <option value="novel">novel</option>
              </select>
            </div>
            <div>
              <label>Max input tokens (cap)</label>
              <input id="max_input_tokens" type="number" value="512" min="64" step="64" />
            </div>
          </div>

          <div class="row">
            <div>
              <label>Scoring model id</label>
              <input id="scoring_model_id" value="gpt2" />
            </div>
            <div>
              <label>Baseline model id (or baseline JSON path)</label>
              <input id="baseline_model" value="gpt2_gutenberg_512" />
            </div>
          </div>

          <details>
            <summary>Advanced</summary>
            <div class="row">
              <div>
                <label>Backend</label>
                <select id="backend">
                  <option value="auto" selected>auto</option>
                  <option value="hf">hf</option>
                  <option value="mlx">mlx</option>
                </select>
              </div>
              <div>
                <label class="muted">&nbsp;</label>
                <div class="box">
                  <label style="margin:0; display:flex; align-items:center; gap:8px;">
                    <input id="normalize_text" type="checkbox" checked />
                    <span class="muted">Normalize formatting (fix hard wraps)</span>
                  </label>
                </div>
                <div class="box" style="margin-top:10px;">
                  <label style="margin:0; display:flex; align-items:center; gap:8px;">
                    <input id="compute_cohesion" type="checkbox" />
                    <span class="muted">Compute cohesion (slower)</span>
                  </label>
                </div>
              </div>
            </div>

            <div class="row">
              <div>
                <label>Calibrator JSON path (optional)</label>
                <input id="calibrator_path" placeholder="reports/calibrators/calibrator.json" />
              </div>
            </div>

            <details>
              <summary>LLM Critique (optional, non-deterministic)</summary>
              <div class="box">
                <label style="margin:0; display:flex; align-items:center; gap:8px;">
                  <input id="use_llm_critic" type="checkbox" />
                  <span class="muted">Enable LLM critique</span>
                </label>
              </div>
              <div class="row">
                <div>
                  <label>Critic model id</label>
                  <input id="critic_model_id" placeholder="Qwen/Qwen2.5-0.5B-Instruct" />
                </div>
                <div>
                  <label>Critic max new tokens</label>
                  <input id="critic_max_new_tokens" type="number" value="450" min="64" step="16" />
                </div>
              </div>
              <div class="row3">
                <div>
                  <label>Temperature</label>
                  <input id="critic_temperature" type="number" value="0.7" step="0.05" min="0.1" max="1.5" />
                </div>
                <div>
                  <label>Top-p</label>
                  <input id="critic_top_p" type="number" value="0.95" step="0.01" min="0.05" max="0.99" />
                </div>
                <div>
                  <label>Seed (optional)</label>
                  <input id="critic_seed" type="number" placeholder="(blank)" />
                </div>
              </div>
            </details>

            <details>
              <summary>Rewrite + rerank (slow)</summary>
              <div class="row">
                <div>
                  <label>Rewrite model id</label>
                  <input id="rewrite_model_id" value="gpt2" />
                </div>
                <div>
                  <label>Candidates / keep top</label>
                  <div class="row">
                    <input id="n_candidates" type="number" value="4" min="1" max="8" />
                    <input id="keep_top" type="number" value="3" min="1" max="5" />
                  </div>
                </div>
              </div>
              <div class="row3">
                <div>
                  <label>Max new tokens</label>
                  <input id="max_new_tokens" type="number" value="240" min="32" step="16" />
                </div>
                <div>
                  <label>Temperature</label>
                  <input id="rewrite_temperature" type="number" value="0.8" step="0.05" min="0.1" max="1.5" />
                </div>
                <div>
                  <label>Top-p</label>
                  <input id="rewrite_top_p" type="number" value="0.92" step="0.01" min="0.05" max="0.99" />
                </div>
              </div>
              <div class="row">
                <div>
                  <label>Seed (optional)</label>
                  <input id="rewrite_seed" type="number" placeholder="7" />
                </div>
                <div class="muted" style="padding-top: 30px;">Tip: use an instruct model for rewrites.</div>
              </div>
            </details>
          </details>

          <div class="btns">
            <button id="analyze_btn">Analyze</button>
            <button id="rewrite_btn" class="secondary">Rewrite + rerank</button>
            <button id="copy_btn" class="secondary">Copy raw JSON</button>
          </div>

          <div id="status" class="status"></div>
        </div>

        <div class="card">
          <div class="score">
            <div><strong id="score">–</strong></div>
            <div class="pill" id="meta">ready</div>
          </div>
          <div class="kv" id="cats"></div>
          <canvas id="spark" width="900" height="240"></canvas>
          <div class="box">
            <div class="muted" style="font-size:12px;">Profile (rubric metrics)</div>
            <ul class="list" id="metrics"></ul>
          </div>
          <div class="box">
            <div class="muted" style="font-size:12px;">Suggestions</div>
            <ul class="list" id="suggestions"></ul>
          </div>
          <div class="box">
            <div class="muted" style="font-size:12px;">Spikes (high surprisal excerpts)</div>
            <ul class="list" id="spikes"></ul>
          </div>
          <details class="box">
            <summary>Rewrites (reranked)</summary>
            <pre id="rewrites">(run “Rewrite + rerank”)</pre>
          </details>
          <details class="box">
            <summary>LLM critique (optional)</summary>
            <pre id="llm"></pre>
          </details>
          <details class="box">
            <summary>Raw JSON</summary>
            <pre id="raw"></pre>
          </details>
        </div>
      </div>

      <!-- Cadence Match Section -->
      <div style="margin-top: 24px;">
        <div class="card">
          <div style="display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 10px;">
            <div>
              <div style="font-size: 16px; font-weight: 600;">Cadence Match</div>
              <div class="sub">Generate text that matches the rhythm of a reference passage.</div>
            </div>
            <div class="pill">/cadence-match</div>
          </div>
          <div class="row">
            <div>
              <label for="cm_prompt">Your prompt (starting text)</label>
              <textarea id="cm_prompt" style="min-height: 80px;">The morning light crept through the window</textarea>
            </div>
            <div>
              <label for="cm_reference">Reference text (cadence to match)</label>
              <textarea id="cm_reference" style="min-height: 80px;">At dawn, the city leans into light. A gull lifts, then drops, then lifts again. The harbor breathes salt and diesel.</textarea>
            </div>
          </div>
          <div class="row">
            <div>
              <label>Max new tokens</label>
              <input id="cm_max_new_tokens" type="number" value="200" min="32" step="16" />
            </div>
            <div>
              <label>Seed (optional)</label>
              <input id="cm_seed" type="number" value="7" />
            </div>
          </div>
          <div class="btns">
            <button id="cm_btn">Generate matching cadence</button>
          </div>
          <div id="cm_status" class="status"></div>
          <div class="box" style="margin-top: 14px;">
            <div class="muted" style="font-size: 12px;">Generated text</div>
            <pre id="cm_output" style="margin-top: 8px; white-space: pre-wrap; color: var(--text);">(click "Generate matching cadence")</pre>
          </div>
          <details class="box">
            <summary>Cadence Match raw JSON</summary>
            <pre id="cm_raw"></pre>
          </details>
        </div>
      </div>
    </div>

    <script>
      function el(id) { return document.getElementById(id); }
      function v(id) { return (el(id).value || "").toString(); }
      function vb(id) { return !!el(id).checked; }
      function vn(id, fallback) {
        const t = v(id).trim();
        if (!t) return fallback;
        const x = Number(t);
        return Number.isFinite(x) ? x : fallback;
      }
      function setStatus(s) { el("status").textContent = s; }
      function setMeta(s) { el("meta").textContent = s; }

      function drawSparkline(series, thr) {
        const c = el("spark");
        const ctx = c.getContext("2d");
        ctx.clearRect(0, 0, c.width, c.height);
        ctx.fillStyle = "rgba(0,0,0,0.0)";
        ctx.fillRect(0, 0, c.width, c.height);
        if (!series || series.length < 2) {
          ctx.fillStyle = "rgba(255,255,255,0.35)";
          ctx.font = "14px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, Courier New, monospace";
          ctx.fillText("Cadence plot (first window): no data", 16, 40);
          return;
        }
        const n = series.length;
        let mn = Infinity, mx = -Infinity;
        for (let i = 0; i < n; i++) { const x = series[i]; if (x < mn) mn = x; if (x > mx) mx = x; }
        if (!Number.isFinite(mn) || !Number.isFinite(mx) || mx - mn < 1e-8) { mn = 0.0; mx = 1.0; }
        const pad = 12;
        const W = c.width - 2*pad;
        const H = c.height - 2*pad;
        const xAt = (i) => pad + (i / (n - 1)) * W;
        const yAt = (x) => pad + (1 - ((x - mn) / (mx - mn))) * H;

        // axes-ish frame
        ctx.strokeStyle = "rgba(255,255,255,0.10)";
        ctx.lineWidth = 1;
        ctx.strokeRect(pad, pad, W, H);

        // threshold line
        if (thr && Number.isFinite(thr)) {
          const y = yAt(thr);
          ctx.strokeStyle = "rgba(255,110,110,0.55)";
          ctx.setLineDash([6, 6]);
          ctx.beginPath();
          ctx.moveTo(pad, y);
          ctx.lineTo(pad + W, y);
          ctx.stroke();
          ctx.setLineDash([]);
        }

        // series line
        ctx.strokeStyle = "rgba(110,168,255,0.95)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(xAt(0), yAt(series[0]));
        for (let i = 1; i < n; i++) ctx.lineTo(xAt(i), yAt(series[i]));
        ctx.stroke();
      }

      function renderAnalyze(out) {
        const score = out?.score?.overall_0_100;
        el("score").textContent = (typeof score === "number") ? score.toFixed(1) + "/100" : "–";
        const tokens = out?.analysis?.doc_metrics?.tokens_count || 0;
        const trunc = !!out?.analysis?.truncated;
        const stability = (tokens < 80) ? "low stability" : ((tokens < 200) ? "medium stability" : "high stability");
        const norm = out?.analysis?.text_normalization;
        const normApplied = !!norm?.enabled && ((norm?.replaced_single_newlines || 0) > 0 || (norm?.joined_hyphen_breaks || 0) > 0);
        const normTag = normApplied ? " · normalized" : "";
        setMeta(`tokens=${tokens}${trunc ? " (truncated)" : ""} · ${stability}${normTag}`);

        // categories
        const cats = out?.score?.categories || {};
        const catsEl = el("cats");
        catsEl.innerHTML = "";
        for (const [k, v] of Object.entries(cats)) {
          const div = document.createElement("div");
          const kk = document.createElement("div");
          kk.className = "k";
          kk.textContent = k;
          const vv = document.createElement("div");
          vv.className = "v";
          vv.textContent = (typeof v === "number") ? Math.round(v * 100) + "/100" : "–";
          div.appendChild(kk);
          div.appendChild(vv);
          catsEl.appendChild(div);
        }

        // profile (rubric metrics)
        const metrics = out?.score?.metrics || {};
        const mEl = el("metrics");
        mEl.innerHTML = "";
        for (const [k, ms] of Object.entries(metrics)) {
          const li = document.createElement("li");
          const val = (typeof ms?.value === "number") ? Number(ms.value).toFixed(4) : "–";
          const p = (typeof ms?.percentile === "number") ? Math.round(Number(ms.percentile)) + "th" : "N/A";
          const s = (typeof ms?.score_0_1 === "number") ? Math.round(Number(ms.score_0_1) * 100) + "/100" : "N/A";
          const mode = (ms?.mode || "").toString();
          li.innerHTML = `<span class="muted"><span style="font-family: var(--mono);">${k}</span></span><br/>value=${val}, pctl=${p}, metric_score=${s} <span class="muted">(${mode})</span>`;
          mEl.appendChild(li);
        }

        // suggestions
        const sug = out?.critique?.suggestions || [];
        const sugEl = el("suggestions");
        sugEl.innerHTML = "";
        for (const s of sug) {
          const li = document.createElement("li");
          li.innerHTML = `<strong>${(s.title || "").toString()}</strong> — ${(s.why || "").toString()}<br/><span class="muted">Try:</span> ${(s.what_to_try || "").toString()}`;
          sugEl.appendChild(li);
        }

        // spikes
        const spikes = out?.analysis?.spikes || [];
        const spEl = el("spikes");
        spEl.innerHTML = "";
        for (const s of spikes.slice(0, 10)) {
          const li = document.createElement("li");
          const ctx = (s.context || "").toString();
          const meta = `s=${Number(s.surprisal||0).toFixed(2)}, entropy=${Number(s.entropy||0).toFixed(2)}, line_pos=${(s.line_pos||"")}`;
          li.innerHTML = `<span class="muted">${meta}</span><br/>${ctx}`;
          spEl.appendChild(li);
        }

        // cadence plot
        const series = out?.analysis?.series?.surprisal || [];
        const thr = out?.analysis?.series?.threshold_surprisal;
        drawSparkline(series, thr);

        // llm critique raw
        const llm = out?.llm_critique;
        el("llm").textContent = llm ? JSON.stringify(llm, null, 2) : "(disabled / none)";

        // raw json
        el("raw").textContent = JSON.stringify(out, null, 2);
        window.__lastRaw = out;
      }

      async function postJson(path, body) {
        const res = await fetch(path, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const txt = await res.text();
        let obj = null;
        try { obj = JSON.parse(txt); } catch (e) {}
        if (!res.ok) {
          const msg = obj?.detail ? JSON.stringify(obj.detail) : txt;
          throw new Error(`${res.status} ${res.statusText}: ${msg}`);
        }
        return obj;
      }

      function buildAnalyzeReq() {
        const seedRaw = v("critic_seed").trim();
        const seed = seedRaw ? Number(seedRaw) : null;
        return {
          text: v("text"),
          doc_type: v("doc_type"),
          scoring_model_id: v("scoring_model_id"),
          baseline_model: v("baseline_model"),
          calibrator_path: v("calibrator_path"),
          backend: v("backend"),
          max_input_tokens: vn("max_input_tokens", 512),
          normalize_text: vb("normalize_text"),
          compute_cohesion: vb("compute_cohesion"),
          use_llm_critic: vb("use_llm_critic"),
          critic_model_id: v("critic_model_id"),
          critic_max_new_tokens: vn("critic_max_new_tokens", 450),
          critic_temperature: vn("critic_temperature", 0.7),
          critic_top_p: vn("critic_top_p", 0.95),
          critic_seed: (seed !== null && Number.isFinite(seed)) ? Math.trunc(seed) : null,
        };
      }

      function buildRewriteReq() {
        const seedRaw = v("rewrite_seed").trim();
        const seed = seedRaw ? Number(seedRaw) : null;
        return {
          text: v("text"),
          doc_type: v("doc_type"),
          rewrite_model_id: v("rewrite_model_id"),
          scoring_model_id: v("scoring_model_id"),
          baseline_model: v("baseline_model"),
          calibrator_path: v("calibrator_path"),
          n_candidates: vn("n_candidates", 4),
          keep_top: vn("keep_top", 3),
          backend: v("backend"),
          max_input_tokens: vn("max_input_tokens", 512),
          normalize_text: vb("normalize_text"),
          compute_cohesion: vb("compute_cohesion"),
          max_new_tokens: vn("max_new_tokens", 240),
          temperature: vn("rewrite_temperature", 0.8),
          top_p: vn("rewrite_top_p", 0.92),
          seed: (seed !== null && Number.isFinite(seed)) ? Math.trunc(seed) : null,
        };
      }

      async function runAnalyze() {
        el("analyze_btn").disabled = true;
        el("rewrite_btn").disabled = true;
        setStatus("Analyzing… (first call may download models)");
        try {
          const out = await postJson("/analyze", buildAnalyzeReq());
          renderAnalyze(out);
          setStatus("Done.");
        } catch (e) {
          setStatus("Error: " + (e?.message || e));
        } finally {
          el("analyze_btn").disabled = false;
          el("rewrite_btn").disabled = false;
        }
      }

      async function runRewrite() {
        el("analyze_btn").disabled = true;
        el("rewrite_btn").disabled = true;
        setStatus("Rewriting… (this can be slow; use a small instruct model for best results)");
        try {
          const out = await postJson("/rewrite", buildRewriteReq());
          // Render rewrite results
          const parts = [];
          parts.push("Original score: " + (out?.original?.score?.toFixed ? out.original.score.toFixed(1) : out?.original?.score));
          const rw = out?.rewrites || [];
          for (let i = 0; i < rw.length; i++) {
            parts.push("");
            const d = rw[i]?.delta?.overall_delta_0_100;
            const dStr = (typeof d === "number") ? ` (Δ ${d.toFixed(1)})` : "";
            parts.push(`Rewrite ${i+1} score: ${(rw[i]?.score?.toFixed ? rw[i].score.toFixed(1) : rw[i]?.score)}${dStr}`);
            const gains = rw[i]?.delta?.top_metric_gains || [];
            if (gains && gains.length) {
              const top = gains.slice(0, 3).map(g => `${g.metric} ${(Number(g.delta_score_0_1||0)*100).toFixed(0)}/100`).join(", ");
              parts.push(`Top gains: ${top}`);
            }
            parts.push(rw[i]?.text || "");
          }
          el("raw").textContent = JSON.stringify(out, null, 2);
          el("rewrites").textContent = parts.join("\\n");
          window.__lastRaw = out;
          setStatus("Done.");
        } catch (e) {
          setStatus("Error: " + (e?.message || e));
        } finally {
          el("analyze_btn").disabled = false;
          el("rewrite_btn").disabled = false;
        }
      }

      async function copyRaw() {
        const raw = window.__lastRaw ? JSON.stringify(window.__lastRaw, null, 2) : "";
        if (!raw) { setStatus("Nothing to copy yet."); return; }
        try {
          await navigator.clipboard.writeText(raw);
          setStatus("Copied raw JSON to clipboard.");
        } catch (e) {
          setStatus("Copy failed: " + (e?.message || e));
        }
      }

      el("analyze_btn").addEventListener("click", runAnalyze);
      el("rewrite_btn").addEventListener("click", runRewrite);
      el("copy_btn").addEventListener("click", copyRaw);
      drawSparkline([], null);

      // Cadence Match
      function setCmStatus(s) { el("cm_status").textContent = s; }

      function buildCadenceMatchReq() {
        const seedRaw = v("cm_seed").trim();
        const seed = seedRaw ? Number(seedRaw) : null;
        return {
          prompt: v("cm_prompt"),
          reference_text: v("cm_reference"),
          doc_type: v("doc_type"),
          model_id: "gpt2",
          max_new_tokens: vn("cm_max_new_tokens", 200),
          seed: (seed !== null && Number.isFinite(seed)) ? Math.trunc(seed) : null,
        };
      }

      async function runCadenceMatch() {
        el("cm_btn").disabled = true;
        setCmStatus("Generating… (first call may download models)");
        try {
          const out = await postJson("/cadence-match", buildCadenceMatchReq());
          el("cm_output").textContent = out?.generated_text || out?.text || "(no text returned)";
          el("cm_raw").textContent = JSON.stringify(out, null, 2);
          setCmStatus("Done.");
        } catch (e) {
          setCmStatus("Error: " + (e?.message || e));
        } finally {
          el("cm_btn").disabled = false;
        }
      }

      el("cm_btn").addEventListener("click", runCadenceMatch);
    </script>
  </body>
</html>
"""
