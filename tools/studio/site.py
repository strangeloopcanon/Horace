from __future__ import annotations


API_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Horace API</title>
    <style>
      :root {
        --bg: #0b0b0c;
        --fg: #f0ede6;
        --muted: rgba(240, 237, 230, 0.7);
        --card: rgba(240, 237, 230, 0.06);
        --border: rgba(240, 237, 230, 0.16);
      }
      body {
        margin: 0;
        background: var(--bg);
        color: var(--fg);
        font-family: ui-sans-serif, -apple-system, system-ui, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji",
          "Segoe UI Emoji";
        line-height: 1.5;
      }
      .wrap { max-width: 860px; margin: 0 auto; padding: 40px 20px 80px; }
      h1 { margin: 0 0 8px; font-size: 34px; letter-spacing: -0.02em; }
      .sub { margin: 0 0 22px; color: var(--muted); }
      .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 16px 18px;
        margin: 14px 0;
      }
      code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
      pre { margin: 0; white-space: pre-wrap; word-break: break-word; }
      a { color: var(--fg); }
      .muted { color: var(--muted); }
      .pill {
        display: inline-block;
        padding: 2px 8px;
        border: 1px solid var(--border);
        border-radius: 999px;
        font-size: 12px;
        margin-right: 8px;
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <h1>Horace API</h1>
      <p class="sub">
        So what: call Horace from anything (scripts, apps, notebooks) to score text and generate cadence-matched writing.
      </p>

      <div class="card">
        <div class="pill">OpenAPI</div>
        <span class="muted">Interactive docs:</span> <a href="/docs">/docs</a>
        <span class="muted"> • JSON schema:</span> <a href="/openapi.json">/openapi.json</a>
      </div>

      <div class="card">
        <div class="pill">Auth</div>
        <span class="muted">
          If the server is configured with <code>HORACE_API_KEY</code>, send either
          <code>Authorization: Bearer &lt;key&gt;</code> or <code>X-API-Key: &lt;key&gt;</code>.
        </span>
      </div>

      <div class="card">
        <div class="pill">POST</div><code>/analyze</code> <span class="muted">Score a passage (0–100) + breakdown.</span>
        <pre>
curl -s http://127.0.0.1:8000/analyze \\
  -H 'Content-Type: application/json' \\
  -d '{"text":"At dawn, the city leans into light.","doc_type":"prose","scoring_model_id":"gpt2","baseline_model":"gpt2_gutenberg_512"}'
        </pre>
      </div>

      <div class="card">
        <div class="pill">POST</div><code>/rewrite</code> <span class="muted">Generate candidates and rerank by score.</span>
      </div>

      <div class="card">
        <div class="pill">POST</div><code>/cadence-match</code> <span class="muted">(alias: <code>/write-like</code>) Match cadence from a reference passage.</span>
      </div>

      <div class="card">
        <div class="pill">POST</div><code>/patch/suggest</code> <span class="muted">Find “dead zones” (flat spans).</span><br/>
        <div class="pill">POST</div><code>/patch/span</code> <span class="muted">Patch a span (MeaningLock + diff).</span>
      </div>

      <p class="muted">
        Notes: the API is intentionally literal—Horace scores cadence, not “truth”. Use MeaningLock when you need meaning-preserving edits.
      </p>
    </div>
  </body>
</html>
"""

STUDIO_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Horace</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;1,400&family=Inter:wght@400;500;600;900&family=Playfair+Display:ital,wght@0,500;0,700;1,400&display=swap" rel="stylesheet">
    
    <style>
      /* ========== Theme Variables ========== */
      :root {
        /* Dark Theme (Obsidian) */
        --bg: #1c1b1a;
        --surface: #262422;
        --border: #45403b;
        --text: #e6e1d3;
        --text-muted: #9d968b;
        --accent: #c5a059;
        --accent-glow: rgba(197, 160, 89, 0.15);
        --chart-color: #c5a059;
        
        --font-display: "Playfair Display", serif;
        --font-body: "Crimson Pro", serif;
        --font-ui: "Inter", sans-serif;
        --ease: cubic-bezier(0.25, 1, 0.5, 1);
      }
      
      [data-theme="light"] {
        /* Light Theme (Paper/Swiss) */
        --bg: #faf8f5;
        --surface: #ffffff;
        --border: #e5e2dd;
        --text: #1a1a1a;
        --text-muted: #666666;
        --accent: #000000;
        --accent-glow: rgba(0, 0, 0, 0.05);
        --chart-color: #1a1a1a;
      }
      
      /* ========== Base ========== */
      * { box-sizing: border-box; margin: 0; padding: 0; }
      
      body {
        font-family: var(--font-body);
        background: var(--bg);
        color: var(--text);
        min-height: 100vh;
        line-height: 1.6;
        transition: background 0.4s ease, color 0.4s ease;
      }
      
      .container {
        max-width: 820px;
        margin: 0 auto;
        padding: 48px 24px 120px;
      }
      
      /* ========== Theme Toggle ========== */
      .theme-toggle {
        position: fixed;
        top: 20px;
        left: 20px;
        z-index: 1000;
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: var(--font-ui);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        background: var(--surface);
        border: 1px solid var(--border);
        padding: 8px 12px;
        cursor: pointer;
        transition: all 0.3s ease;
      }
      .theme-toggle:hover { color: var(--text); border-color: var(--text-muted); }
      .theme-toggle svg { width: 14px; height: 14px; }
      
      /* ========== Header ========== */
      header {
        text-align: center;
        margin-bottom: 48px;
        padding-bottom: 24px;
        border-bottom: 1px solid var(--border);
      }
      
      h1 {
        font-family: var(--font-display);
        font-size: 42px;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 8px;
      }
      
      [data-theme="light"] h1 {
        font-family: var(--font-ui);
        font-size: 64px;
        font-weight: 900;
        letter-spacing: -0.03em;
        text-transform: uppercase;
      }
      
	      .tagline {
	        font-family: var(--font-ui);
	        color: var(--text-muted);
	        font-size: 12px;
	        text-transform: uppercase;
	        letter-spacing: 0.15em;
	      }
	      
	      [data-theme="light"] .tagline { color: var(--accent); }

	      .exp-toggle {
	        margin-top: 14px;
	        display: inline-flex;
	        align-items: center;
	        gap: 10px;
	        font-family: var(--font-ui);
	        font-size: 11px;
	        text-transform: uppercase;
	        letter-spacing: 0.1em;
	        color: var(--text-muted);
	      }
	      .exp-toggle input {
	        width: 14px;
	        height: 14px;
	        accent-color: var(--accent);
	      }
	      .exp-note {
	        margin: -16px 0 34px;
	        color: var(--text-muted);
	        font-family: var(--font-ui);
	        font-size: 12px;
	        text-align: center;
	      }
      
      /* ========== Tabs ========== */
      .tabs {
        display: flex;
        justify-content: center;
        gap: 32px;
        margin-bottom: 40px;
        font-family: var(--font-ui);
      }
      
      .tab {
        background: none;
        border: none;
        color: var(--text-muted);
        font-size: 13px;
        font-weight: 500;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        cursor: pointer;
        padding: 8px 0;
        position: relative;
        transition: color 0.2s;
      }
      .tab:hover { color: var(--text); }
      .tab::after {
        content: "";
        position: absolute;
        bottom: 0; left: 0;
        width: 100%; height: 2px;
        background: var(--accent);
        transform: scaleX(0);
        transition: transform 0.3s var(--ease);
      }
      .tab.active { color: var(--accent); }
      .tab.active::after { transform: scaleX(1); }
      
      [data-theme="light"] .tab.active { color: var(--text); font-weight: 600; }
      
      /* ========== Panels ========== */
      .panel { display: none; }
      .panel.active { display: block; animation: fade-in 0.5s var(--ease); }
      
      @keyframes fade-in {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
      }
      
      /* ========== Form Elements ========== */
      .input-group {
        margin-bottom: 24px;
      }
      
      .input-label {
        display: block;
        font-family: var(--font-ui);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-bottom: 10px;
      }
      
      textarea {
        width: 100%;
        min-height: 200px;
        padding: 20px;
        background: var(--surface);
        border: 1px solid var(--border);
        color: var(--text);
        font-family: var(--font-body);
        font-size: 19px;
        line-height: 1.7;
        resize: vertical;
        outline: none;
        transition: border-color 0.2s, box-shadow 0.2s;
      }
      textarea:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 3px var(--accent-glow);
      }
      textarea::placeholder { color: var(--text-muted); opacity: 0.6; }
      
      [data-theme="light"] textarea {
        font-size: 22px;
        border-width: 2px;
      }
      
      input[type="number"], select {
        width: 100%;
        padding: 10px 12px;
        background: var(--surface);
        border: 1px solid var(--border);
        color: var(--text);
        font-family: var(--font-ui);
        font-size: 14px;
        outline: none;
      }
      input:focus, select:focus { border-color: var(--accent); }
      input[type="text"] {
        width: 100%;
        padding: 10px 12px;
        background: var(--surface);
        border: 1px solid var(--border);
        color: var(--text);
        font-family: var(--font-ui);
        font-size: 14px;
        outline: none;
      }
      .settings-full { grid-column: 1 / -1; }
      .settings-hint {
        margin-top: 6px;
        color: var(--text-muted);
        font-family: var(--font-ui);
        font-size: 11px;
      }
      
      /* ========== Buttons ========== */
      .btn {
        display: block;
        width: 100%;
        padding: 16px;
        background: var(--accent);
        border: none;
        color: var(--bg);
        font-family: var(--font-ui);
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        cursor: pointer;
        transition: all 0.2s;
      }
      .btn:hover { opacity: 0.9; transform: translateY(-1px); }
      .btn:disabled { opacity: 0.5; cursor: wait; transform: none; }

      .btn-secondary {
        margin-top: 12px;
        background: transparent;
        color: var(--text);
        border: 1px solid var(--border);
      }
      .btn-secondary:hover { border-color: var(--accent); color: var(--accent); }
      
      [data-theme="light"] .btn {
        background: var(--text);
        color: var(--bg);
      }
      
      /* ========== Results ========== */
      .result-area {
        margin-top: 48px;
        padding-top: 40px;
        border-top: 1px solid var(--border);
        display: none;
      }
      .result-area.visible { display: block; animation: fade-in 0.6s var(--ease); }
      
      /* Score Display */
      .score-row {
        display: flex;
        align-items: flex-start;
        gap: 32px;
        margin-bottom: 48px;
      }
      
      .score-number {
        font-family: var(--font-display);
        font-size: 72px;
        font-weight: 500;
        line-height: 1;
        color: var(--accent);
        flex-shrink: 0;
      }
      .score-number sup {
        font-size: 20px;
        color: var(--text-muted);
        font-family: var(--font-ui);
        vertical-align: top;
        margin-left: 4px;
      }
      
      [data-theme="light"] .score-number {
        font-family: var(--font-ui);
        font-weight: 900;
        font-size: 80px;
      }
      
      .score-summary {
        flex: 1;
        font-size: 17px;
        line-height: 1.6;
        color: var(--text);
        padding-top: 8px;
        border-left: 2px solid var(--border);
        padding-left: 24px;
      }
      
      [data-theme="light"] .score-summary {
        font-size: 18px;
        font-weight: 400;
      }
      
      /* Section Labels */
      .section-label {
        font-family: var(--font-ui);
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 12px;
      }
      .section-label::after {
        content: "";
        flex: 1;
        height: 1px;
        background: var(--border);
      }
      
      /* Text Display */
      .text-display {
        background: var(--surface);
        padding: 32px;
        font-size: 18px;
        line-height: 1.8;
        border: 1px solid var(--border);
        margin-bottom: 40px;
        white-space: pre-wrap;
        word-wrap: break-word;
      }
      
      [data-theme="light"] .text-display {
        border-width: 2px;
        font-size: 20px;
      }
      
      .spike {
        border-bottom: 2px solid var(--accent);
        background: var(--accent-glow);
        cursor: help;
        transition: background 0.2s;
      }
      .spike:hover { background: rgba(197, 160, 89, 0.3); }
      
      [data-theme="light"] .spike {
        background: #fef3c7;
        border-color: #f59e0b;
      }
      [data-theme="light"] .spike:hover { background: #fde68a; }
      
      /* Metrics */
      .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1px;
        background: var(--border);
        border: 1px solid var(--border);
        margin-bottom: 40px;
      }
      .metric {
        background: var(--bg);
        padding: 20px 12px;
        text-align: center;
      }
      .metric-value {
        font-family: var(--font-display);
        font-size: 28px;
        font-weight: 500;
        margin-bottom: 4px;
      }
      .metric-label {
        font-family: var(--font-ui);
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
      }
      
      [data-theme="light"] .metric-value {
        font-family: var(--font-ui);
        font-weight: 700;
      }
      
      /* Chart */
      canvas {
        width: 100%;
        height: 100px;
        margin-bottom: 40px;
      }

      /* Improvements + Dead Zones */
      .improvements-list, .deadzone-list { margin-bottom: 40px; }
      .improvement, .deadzone {
        border-left: 2px solid var(--border);
        padding-left: 20px;
        margin-bottom: 18px;
      }
      .improvement-title, .deadzone-title {
        font-family: var(--font-ui);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-bottom: 6px;
      }
      .improvement-body, .deadzone-body {
        font-size: 15px;
        color: var(--text);
        line-height: 1.6;
        white-space: pre-wrap;
        word-break: break-word;
      }
      .deadzone-meta {
        font-family: var(--font-ui);
        font-size: 11px;
        color: var(--text-muted);
        margin-top: 6px;
      }
      
      /* Suggestions */
      .suggestions-list { margin-bottom: 40px; }
      
      .suggestion {
        border-left: 2px solid var(--border);
        padding-left: 20px;
        margin-bottom: 24px;
        transition: border-color 0.2s;
      }
      .suggestion:hover { border-color: var(--accent); }
      
      .sugg-title {
        font-family: var(--font-display);
        font-size: 18px;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
      }
      .sugg-title:hover { color: var(--accent); }
      
      .sugg-body {
        display: none;
        font-size: 15px;
        color: var(--text-muted);
        line-height: 1.6;
      }
      .suggestion.open .sugg-body { display: block; }
      .suggestion.open .sugg-title { color: var(--accent); }
      
      [data-theme="light"] .sugg-title {
        font-family: var(--font-ui);
        font-weight: 600;
        font-size: 16px;
      }
      
      /* Rewrite Cards */
      .rewrite-card {
        background: var(--surface);
        border: 1px solid var(--border);
        padding: 24px;
        margin-bottom: 20px;
      }
      .rewrite-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid var(--border);
        font-family: var(--font-ui);
        font-size: 12px;
      }
      .rewrite-score { font-weight: 600; color: var(--accent); }
      .rewrite-text {
        font-size: 17px;
        line-height: 1.7;
        white-space: pre-wrap;
      }
      .copy-btn {
        background: none;
        border: 1px solid var(--border);
        color: var(--text-muted);
        padding: 4px 10px;
        font-family: var(--font-ui);
        font-size: 10px;
        text-transform: uppercase;
        cursor: pointer;
        transition: all 0.2s;
      }
      .copy-btn:hover { border-color: var(--accent); color: var(--accent); }
      
      /* Settings */
      .settings-toggle {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        width: 100%;
        padding: 12px;
        margin-top: 32px;
        background: none;
        border: none;
        color: var(--text-muted);
        font-family: var(--font-ui);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        cursor: pointer;
      }
      .settings-toggle:hover { color: var(--text); }
      
      .settings-panel {
        display: none;
        background: var(--surface);
        border: 1px solid var(--border);
        padding: 20px;
        margin-top: 16px;
      }
      .settings-panel.open { display: block; animation: fade-in 0.3s; }
      .settings-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
      
      /* Mobile */
      @media (max-width: 640px) {
        .theme-toggle { position: static; margin-bottom: 24px; }
        .container { padding: 24px 16px 80px; }
        h1 { font-size: 32px; }
        [data-theme="light"] h1 { font-size: 40px; }
        .score-row { flex-direction: column; gap: 20px; }
        .score-summary { border-left: none; padding-left: 0; border-top: 1px solid var(--border); padding-top: 16px; }
        .metrics-grid { grid-template-columns: 1fr 1fr; }
        .tabs { gap: 16px; }
      }
    </style>
  </head>
  <body>
    <button class="theme-toggle" id="theme-toggle" title="Switch theme">
      <svg id="theme-icon-dark" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="5"></circle>
        <line x1="12" y1="1" x2="12" y2="3"></line>
        <line x1="12" y1="21" x2="12" y2="23"></line>
        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
        <line x1="1" y1="12" x2="3" y2="12"></line>
        <line x1="21" y1="12" x2="23" y2="12"></line>
        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
      </svg>
      <svg id="theme-icon-light" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="display:none;">
        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
      </svg>
      <span id="theme-label">Light</span>
    </button>
    
    <div class="container">
	      <header>
	        <h1>Horace</h1>
	        <div class="tagline">The Rhythm of Rhetoric</div>
	        <label class="exp-toggle">
	          <input type="checkbox" id="exp-enabled" />
	          Experimental: Generate
	        </label>
	      </header>
	      
	      <div class="tabs">
	        <button class="tab active" data-tab="score">Analysis</button>
	        <button class="tab" data-tab="rewrite" style="display:none;">Rewrite</button>
	        <button class="tab" data-tab="match" style="display:none;">Match</button>
	      </div>

	      <div class="exp-note" id="exp-note">
	        Rewrite and Match are experimental. Enable “Experimental: Generate” to show them.
	      </div>
      
      <!-- Analysis Panel -->
      <div id="panel-score" class="panel active">
        <div class="input-group">
          <label class="input-label">Source Text</label>
          <textarea id="score-text" placeholder="Paste your text here...">At dawn, the city leans into light. A gull lifts, then drops, then lifts again. The harbor breathes salt and diesel.</textarea>
        </div>
        
        <button class="btn" id="score-btn">Analyze</button>
        <button class="btn btn-secondary" id="deadzone-btn">Find Dead Zones</button>
        
        <div id="score-result" class="result-area">
          <div class="score-row">
            <div class="score-number" id="score-value">--<sup>/100</sup></div>
            <div class="score-summary" id="score-summary"></div>
          </div>
          
          <div class="section-label">Text Analysis</div>
          <div class="text-display" id="text-highlighted"></div>
          
          <div class="section-label">Cadence</div>
          <canvas id="cadence-chart"></canvas>
          
          <div class="section-label">Metrics</div>
          <div class="metrics-grid" id="score-metrics"></div>

          <div class="section-label">Top Improvements</div>
          <div class="improvements-list" id="score-improvements"></div>
          
          <div class="section-label">Suggestions</div>
          <div class="suggestions-list" id="score-suggestions"></div>

          <div class="section-label">Dead Zones</div>
          <div class="deadzone-list" id="score-deadzones"></div>
        </div>
        
        <button class="settings-toggle" id="score-settings-toggle">Settings</button>
        <div class="settings-panel" id="score-settings">
          <div class="settings-grid">
            <div class="settings-full">
              <label class="input-label">Trained Scorer Model Path</label>
              <input type="text" id="scorer-model-path" value="" />
              <div class="settings-hint">Uses <code>primary_score</code> from the trained scorer.</div>
            </div>
            <div class="settings-full">
              <label class="input-label">Anti-Pattern Model Path (Optional)</label>
              <input type="text" id="antipattern-model-path" value="" placeholder="models/..." />
            </div>
            <div class="settings-full">
              <label class="input-label">
                <input type="checkbox" id="apply-antipattern-penalty" />
                Apply authenticity penalty to overall score
              </label>
              <div class="settings-hint">Off by default: shows quality and authenticity as separate scores.</div>
            </div>
            <div>
              <label class="input-label">Primary Score Mode</label>
              <select id="primary-score-mode">
                <option value="auto">Auto</option>
                <option value="rubric">Rubric (Cadence)</option>
                <option value="trained">Trained Scorer</option>
                <option value="blend">Blend</option>
              </select>
            </div>
            <div>
              <label class="input-label">Blend Weight (0–1)</label>
              <input type="number" id="primary-blend-weight" value="0.35" step="0.05" min="0" max="1" />
            </div>
            <div>
              <label class="input-label">Document Type</label>
              <select id="doc-type">
                <option value="prose">Prose</option>
                <option value="poem">Poetry</option>
              </select>
            </div>
            <div>
              <label class="input-label">Max Tokens</label>
              <input type="number" id="max-tokens" value="512" />
            </div>
            <div>
              <label class="input-label">Anti-Pattern Threshold</label>
              <input type="number" id="antipattern-threshold" value="0.85" step="0.01" min="0" max="1" />
            </div>
            <div>
              <label class="input-label">Anti-Pattern Weight</label>
              <input type="number" id="antipattern-weight" value="0.85" step="0.01" min="0" max="2" />
            </div>
            <div>
              <label class="input-label">Anti-Pattern Combiner</label>
              <select id="antipattern-combiner-mode">
                <option value="adaptive">Adaptive (Recommended)</option>
                <option value="legacy">Legacy Threshold</option>
              </select>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Rewrite Panel -->
      <div id="panel-rewrite" class="panel" style="display:none;">
        <div class="input-group">
          <label class="input-label">Text to Rewrite</label>
          <textarea id="rewrite-text" placeholder="Enter text to improve..."></textarea>
        </div>
        
        <button class="btn" id="rewrite-btn">Generate Rewrites</button>
        
        <div id="rewrite-result" class="result-area">
          <div class="section-label">Variations</div>
          <div id="rewrite-output"></div>
        </div>
        
        <button class="settings-toggle" id="rewrite-settings-toggle">Settings</button>
        <div class="settings-panel" id="rewrite-settings">
          <div class="settings-grid">
            <div>
              <label class="input-label">Candidates</label>
              <input type="number" id="rewrite-candidates" value="4" />
            </div>
            <div>
              <label class="input-label">Temperature</label>
              <input type="number" id="rewrite-temp" value="0.8" step="0.1" />
            </div>
          </div>
        </div>
      </div>
      
      <!-- Match Panel -->
      <div id="panel-match" class="panel" style="display:none;">
        <div class="input-group">
          <label class="input-label">Reference (Target Cadence)</label>
          <textarea id="match-reference" style="min-height:140px;">At dawn, the city leans into light. A gull lifts, then drops, then lifts again.</textarea>
        </div>
        
        <div class="input-group">
          <label class="input-label">Your Prompt</label>
          <textarea id="match-prompt" style="min-height:100px;">The morning light crept through the window</textarea>
        </div>
        
        <button class="btn" id="match-btn">Generate</button>
        
        <div id="match-result" class="result-area">
          <div class="section-label">Generated Text</div>
          <div class="text-display" id="match-output"></div>
          <div style="margin-top:-24px;text-align:right;">
            <button class="copy-btn" id="match-copy" disabled>Copy</button>
          </div>
        </div>
        
        <button class="settings-toggle" id="match-settings-toggle">Settings</button>
        <div class="settings-panel" id="match-settings">
          <div class="settings-grid">
            <div>
              <label class="input-label">Max Tokens</label>
              <input type="number" id="match-tokens" value="200" />
            </div>
            <div>
              <label class="input-label">Seed</label>
              <input type="number" id="match-seed" value="7" />
            </div>
          </div>
        </div>
      </div>
    </div>

    <script>
      const $ = id => document.getElementById(id);
      const $$ = sel => document.querySelectorAll(sel);
      const IS_MODAL_HOST = String(window.location.hostname || '').includes('.modal.run');
      const DEFAULT_SCORER_MODEL_PATH = '';
      const DEFAULT_ANTIPATTERN_MODEL_PATH = IS_MODAL_HOST
        ? '/vol/models/scorer_v5_authenticity_v1'
        : 'models/scorer_v5_authenticity_v1';
      const DEFAULT_ANTIPATTERN_COMBINER_MODE = 'adaptive';
      const DEFAULT_APPLY_ANTIPATTERN_PENALTY = false;
      const DEFAULT_PRIMARY_SCORE_MODE = 'rubric';
      const DEFAULT_PRIMARY_BLEND_WEIGHT = '0.35';

      function safeStorageGet(key, fallback = null) {
        try {
          const value = window.localStorage.getItem(key);
          return value === null ? fallback : value;
        } catch (e) {
          return fallback;
        }
      }

      function safeStorageSet(key, value) {
        try {
          window.localStorage.setItem(key, value);
        } catch (e) {}
      }

      function parseNumberInput(value, fallback) {
        const raw = String(value || '').trim();
        if (!raw) {
          return fallback;
        }
        const n = Number(raw);
        return Number.isFinite(n) ? n : fallback;
      }

      function looksLikeWrongAntipatternModel(path) {
        const s = String(path || '').toLowerCase();
        return s.includes('v5_antipattern') && !s.includes('authenticity');
      }

      function looksLikeWrongScorerModel(path) {
        const s = String(path || '').toLowerCase();
        return s.includes('antipattern') && !s.includes('authenticity');
      }

      function bindPersistedInput(id, storageKey, defaultValue) {
        const el = $(id);
        if (!el) return;
        const stored = safeStorageGet(storageKey, '');
        if (stored !== null && String(stored).trim() !== '') {
          el.value = String(stored);
        } else if (typeof defaultValue === 'string') {
          el.value = defaultValue;
        }
        el.addEventListener('change', () => safeStorageSet(storageKey, String(el.value || '').trim()));
        el.addEventListener('blur', () => safeStorageSet(storageKey, String(el.value || '').trim()));
      }

      function bindPersistedCheckbox(id, storageKey, defaultValue) {
        const el = $(id);
        if (!el) return;
        const stored = safeStorageGet(storageKey, null);
        if (stored === null) {
          el.checked = Boolean(defaultValue);
        } else {
          el.checked = String(stored) === '1';
        }
        const persist = () => safeStorageSet(storageKey, el.checked ? '1' : '0');
        el.addEventListener('change', persist);
        el.addEventListener('blur', persist);
      }

      bindPersistedInput('scorer-model-path', 'horace-scorer-model-path', DEFAULT_SCORER_MODEL_PATH);
      bindPersistedInput('antipattern-model-path', 'horace-antipattern-model-path', DEFAULT_ANTIPATTERN_MODEL_PATH);
      bindPersistedInput('antipattern-combiner-mode', 'horace-antipattern-combiner-mode', DEFAULT_ANTIPATTERN_COMBINER_MODE);
      bindPersistedCheckbox('apply-antipattern-penalty', 'horace-apply-antipattern-penalty', DEFAULT_APPLY_ANTIPATTERN_PENALTY);
      bindPersistedInput('primary-score-mode', 'horace-primary-score-mode', DEFAULT_PRIMARY_SCORE_MODE);
      bindPersistedInput('primary-blend-weight', 'horace-primary-blend-weight', DEFAULT_PRIMARY_BLEND_WEIGHT);

      function refreshBlendUi() {
        const mode = String(($('primary-score-mode').value || '')).trim().toLowerCase();
        $('primary-blend-weight').disabled = (mode !== 'blend');
      }
      $('primary-score-mode').addEventListener('change', refreshBlendUi);
      refreshBlendUi();
      
      // Theme Toggle
      const savedTheme = safeStorageGet('horace-theme', 'dark');
      if (savedTheme === 'light') document.body.setAttribute('data-theme', 'light');
      updateThemeUI();
      
      $('theme-toggle').addEventListener('click', () => {
        const isLight = document.body.getAttribute('data-theme') === 'light';
        document.body.setAttribute('data-theme', isLight ? '' : 'light');
        safeStorageSet('horace-theme', isLight ? 'dark' : 'light');
        updateThemeUI();
      });
      
	      function updateThemeUI() {
	        const isLight = document.body.getAttribute('data-theme') === 'light';
	        $('theme-icon-dark').style.display = isLight ? 'none' : 'block';
	        $('theme-icon-light').style.display = isLight ? 'block' : 'none';
	        $('theme-label').textContent = isLight ? 'Dark' : 'Light';
	      }

	      // Experimental toggle (gates generation features)
	      const EXP_KEY = 'horace-exp-generate';

	      function applyExperimentalUI(enabled) {
	        const rewriteTab = document.querySelector('.tab[data-tab="rewrite"]');
	        const matchTab = document.querySelector('.tab[data-tab="match"]');
	        const rewritePanel = $('panel-rewrite');
	        const matchPanel = $('panel-match');
	        const note = $('exp-note');

	        const showEl = (el, show) => {
	          if (!el) return;
	          el.style.display = show ? '' : 'none';
	        };

	        showEl(rewriteTab, enabled);
	        showEl(matchTab, enabled);
	        showEl(rewritePanel, enabled);
	        showEl(matchPanel, enabled);
	        showEl(note, !enabled);

	        safeStorageSet(EXP_KEY, enabled ? '1' : '0');

	        if (!enabled) {
	          // Force Analysis active when hiding tabs/panels.
	          $$('.tab').forEach(t => t.classList.remove('active'));
	          $$('.panel').forEach(p => p.classList.remove('active'));
	          const scoreTab = document.querySelector('.tab[data-tab="score"]');
	          if (scoreTab) scoreTab.classList.add('active');
	          const scorePanel = $('panel-score');
	          if (scorePanel) scorePanel.classList.add('active');
	        }
	      }

	      (function initExperimentalToggle() {
	        const expBox = $('exp-enabled');
	        if (!expBox) return;
	        const enabled = safeStorageGet(EXP_KEY, '0') === '1';
	        expBox.checked = enabled;
	        applyExperimentalUI(enabled);
	        expBox.addEventListener('change', () => applyExperimentalUI(expBox.checked));
	      })();
	      
	      // Tabs
	      $$('.tab').forEach(tab => {
	        tab.addEventListener('click', () => {
	          $$('.tab').forEach(t => t.classList.remove('active'));
          $$('.panel').forEach(p => p.classList.remove('active'));
          tab.classList.add('active');
          $('panel-' + tab.dataset.tab).classList.add('active');
        });
      });
      
      // Settings
      $$('.settings-toggle').forEach(toggle => {
        toggle.addEventListener('click', () => {
          toggle.nextElementSibling.classList.toggle('open');
        });
      });
      
      // API
      async function api(endpoint, body) {
        const res = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });
        const raw = await res.text();
        let parsed = null;
        try {
          parsed = raw ? JSON.parse(raw) : null;
        } catch (e) {
          parsed = null;
        }
        if (!res.ok) {
          let msg = raw || `HTTP ${res.status}`;
          if (parsed && typeof parsed === 'object') {
            if (typeof parsed.error === 'string' && parsed.error) {
              msg = parsed.error;
              if (typeof parsed.details === 'string' && parsed.details) msg += `: ${parsed.details}`;
            } else if (Array.isArray(parsed.detail) && parsed.detail.length) {
              const first = parsed.detail[0] || {};
              msg = String(first.msg || first.type || msg);
            } else if (typeof parsed.detail === 'string' && parsed.detail) {
              msg = parsed.detail;
            }
          }
          throw new Error(msg);
        }
        if (parsed && typeof parsed === 'object') return parsed;
        throw new Error('Invalid API response');
      }
      
      // Helpers
      function escapeHtml(str) {
        return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      }
      
      function highlightText(text, spikes) {
        if (!spikes || !spikes.length) return escapeHtml(text);
        const sorted = [...spikes].sort((a, b) => a.char_start - b.char_start);
        let result = '';
        let lastEnd = 0;
        
        for (const spike of sorted) {
          const start = spike.char_start;
          const end = spike.char_end;
          if (start < lastEnd) continue;
          
          result += escapeHtml(text.slice(lastEnd, start));
          const word = text.slice(start, end);
          result += `<span class="spike" title="Surprisal: ${spike.surprisal?.toFixed(1)}">${escapeHtml(word)}</span>`;
          lastEnd = end;
        }
        result += escapeHtml(text.slice(lastEnd));
        return result;
      }
      
      function drawCadence(series) {
        const canvas = $('cadence-chart');
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, rect.width, rect.height);
        
        if (!series || series.length < 2) return;
        
        const style = getComputedStyle(document.body);
        const chartColor = style.getPropertyValue('--chart-color').trim() || '#c5a059';
        
        let min = Math.min(...series), max = Math.max(...series);
        if (max - min < 0.1) { min = 0; max = 10; }
        
        const w = rect.width, h = rect.height;
        const pad = h * 0.1;
        const xAt = i => (i / (series.length - 1)) * w;
        const yAt = v => pad + (1 - (v - min) / (max - min)) * (h - 2 * pad);
        
        ctx.beginPath();
        ctx.moveTo(xAt(0), yAt(series[0]));
        for (let i = 1; i < series.length; i++) ctx.lineTo(xAt(i), yAt(series[i]));
        
        ctx.strokeStyle = chartColor;
        ctx.lineWidth = 2;
        ctx.stroke();
        
        ctx.lineTo(w, h);
        ctx.lineTo(0, h);
        ctx.closePath();
        const grad = ctx.createLinearGradient(0, 0, 0, h);
        grad.addColorStop(0, chartColor + '33');
        grad.addColorStop(1, chartColor + '00');
        ctx.fillStyle = grad;
        ctx.fill();
        
        canvas._lastSeries = series;
      }
      
      // Score Handler
      $('score-btn').addEventListener('click', async () => {
        const btn = $('score-btn');
        const deadBtn = $('deadzone-btn');
        const result = $('score-result');
        const origText = btn.innerText;
        
        btn.disabled = true;
        deadBtn.disabled = true;
        btn.innerText = 'Analyzing...';
        result.classList.remove('visible');
        
        try {
          const text = $('score-text').value;
          const scorerModelPath = String(($('scorer-model-path').value || '')).trim();
          const antipatternModelPath = String(($('antipattern-model-path').value || '')).trim();
          const antipatternCombinerMode = String(($('antipattern-combiner-mode').value || '')).trim().toLowerCase();
          const applyAntipatternPenalty = Boolean($('apply-antipattern-penalty')?.checked);
          const primaryMode = String(($('primary-score-mode').value || '')).trim();
          const primaryBlendWeight = parseNumberInput($('primary-blend-weight').value, 0.35);
          const data = await api('/analyze', {
            text,
            doc_type: $('doc-type').value,
            max_input_tokens: parseInt($('max-tokens').value) || 512,
            normalize_text: true,
            scorer_model_path: scorerModelPath,
            antipattern_model_path: antipatternModelPath,
            antipattern_penalty_threshold: parseNumberInput($('antipattern-threshold').value, 0.85),
            antipattern_penalty_weight: parseNumberInput($('antipattern-weight').value, 0.85),
            antipattern_combiner_mode: antipatternCombinerMode || 'adaptive',
            apply_antipattern_penalty: applyAntipatternPenalty,
            primary_score_mode: primaryMode || 'auto',
            primary_blend_weight: Number.isFinite(primaryBlendWeight) ? primaryBlendWeight : 0.35,
          });

        const quality = data.quality_score || {};
        const authenticity = data.authenticity_score || {};
        const primary = data.primary_score || {};
        const qualityScore = Number(
          quality.overall_0_100
          ?? primary.base_overall_0_100
          ?? primary.overall_0_100
          ?? data.score?.overall_0_100
          ?? 0
        );
        const qualitySource = String(quality.source || primary.source || '');
        const authProb = Number(authenticity.llm_likelihood_0_1 ?? primary.antipattern_prob_0_1 ?? 0);
        const authProbRaw = Number(authenticity.llm_likelihood_raw_0_1 ?? primary.antipattern_prob_raw_0_1 ?? 0);
        const authInverted = Boolean(authenticity.llm_likelihood_inverted ?? primary.antipattern_prob_inverted);
        const authMode = String(authenticity.combiner_mode || primary.antipattern_penalty_mode || '').trim();
        const authSoftThreshold = Number(authenticity.soft_threshold_0_1 ?? primary.antipattern_soft_threshold_0_1);
        const authHardThreshold = Number(authenticity.hard_threshold_0_1 ?? primary.antipattern_hard_threshold_0_1);
        const authCap = Number(authenticity.authenticity_cap_0_100 ?? primary.antipattern_authenticity_cap_0_100);
        const suggestedPenalty = Number(authenticity.suggested_penalty_0_100 ?? primary.antipattern_suggested_penalty_0_100 ?? 0);
        const combinedPreview = Number(authenticity.combined_preview_overall_0_100 ?? primary.combined_preview_overall_0_100 ?? qualityScore);
        const appliedPenalty = Boolean(authenticity.penalty_applied ?? primary.apply_antipattern_penalty ?? false);

        const detailLines = [];
        const authRiskLabel = (() => {
          if (!Number.isFinite(authProb)) return 'Unknown';
          if (authProb < 0.35) return 'Low';
          if (authProb < 0.65) return 'Moderate';
          return 'High';
        })();
        detailLines.push(`Quality score: ${qualityScore.toFixed(1)}/100.`);
        if (Number.isFinite(authProb) && authProb > 0) {
          detailLines.push(`Authenticity risk: ${authRiskLabel} (${(authProb * 100).toFixed(1)}% AI-like style).`);
        }
        if (appliedPenalty) {
          const applied = Math.round((Number(primary.antipattern_penalty_0_100 || 0)) * 10) / 10;
          detailLines.push(`Combined score mode is on. Authenticity penalty applied: -${applied} points.`);
        } else if (suggestedPenalty > 0.05) {
          const sp = Math.round(suggestedPenalty * 10) / 10;
          detailLines.push(`If you enable combined mode, this would become ${combinedPreview.toFixed(1)}/100 (about -${sp} points).`);
        }
        if (Number.isFinite(authSoftThreshold) && Number.isFinite(authHardThreshold)) {
          detailLines.push(`Risk thresholds: soft ${(authSoftThreshold * 100).toFixed(0)}%, hard ${(authHardThreshold * 100).toFixed(0)}%.`);
        }
        if (authMode) {
          detailLines.push(`Risk model: ${authMode}.`);
        }
        if (authInverted) {
          detailLines.push(`Note: authenticity score was polarity-corrected for this model.`);
        }
        if (Number.isFinite(authCap) && authCap > 0 && authCap < 100) {
          detailLines.push(`Authenticity cap at this risk: ${Math.round(authCap * 10) / 10}/100.`);
        }
          const antiError = String(data.antipattern_score_error || '').trim();
          const antiModelMissing = String($('antipattern-model-path').value || '').trim() === '';
          if (antiError) {
            detailLines.push(`Authenticity model error: ${antiError}`);
          } else if (antiModelMissing) {
            detailLines.push('Authenticity model is not configured.');
          }
          if (String(data.primary_score_warning || '').trim()) {
            detailLines.push(`Note: ${String(data.primary_score_warning)}`);
          }
          if (scorerModelPath && looksLikeWrongScorerModel(scorerModelPath)) {
            detailLines.push('Warning: scorer model path looks like an anti-pattern model. Move it to the authenticity field.');
          }
          if (antipatternModelPath && looksLikeWrongAntipatternModel(antipatternModelPath)) {
            detailLines.push('Warning: authenticity model path looks like a quality model. Use an authenticity-oriented checkpoint.');
          }
          result.classList.add('visible');
          $('score-value').innerHTML = Math.round(qualityScore || 0) + '<sup>/100</sup>';
          const summary = String(data.critique?.summary || '');
          const sourceLine = qualitySource ? `\n\nQuality source: ${qualitySource}` : '';
          const debugLine = detailLines.length ? `\n\n${detailLines.join('\n')}` : '';
          $('score-summary').innerText = summary + sourceLine + debugLine;
          $('text-highlighted').innerHTML = highlightText(text, data.analysis?.spikes || []);
          drawCadence(data.analysis?.series?.surprisal);
          
          const cats = data.score?.categories || {};
          $('score-metrics').innerHTML = Object.entries(cats).map(([k, v]) => `
            <div class="metric">
              <div class="metric-value">${Math.round(v * 100)}</div>
              <div class="metric-label">${k}</div>
            </div>
          `).join('');

          const imps = data.score?.top_improvements || [];
          $('score-improvements').innerHTML = imps.length ? imps.map(h => `
            <div class="improvement">
              <div class="improvement-title">${escapeHtml(h.category || '')} · ${escapeHtml(h.metric || '')}</div>
              <div class="improvement-body">Direction: ${escapeHtml(h.direction || '')}\nPotential gain: ${(Number(h.potential_gain || 0) * 100).toFixed(1)}</div>
            </div>
          `).join('') : '<p style="color:var(--text-muted);">No improvement hints.</p>';
          
          const suggs = data.critique?.suggestions || [];
          $('score-suggestions').innerHTML = suggs.length ? suggs.map(s => `
            <div class="suggestion">
              <div class="sugg-title" onclick="this.parentElement.classList.toggle('open')">
                <span>→ ${escapeHtml(s.title || '')}</span>
              </div>
              <div class="sugg-body">
                <p>${escapeHtml(s.what_to_try || '')}</p>
                ${s.why ? `<p style="margin-top:8px;opacity:0.7;font-style:italic;">${escapeHtml(s.why)}</p>` : ''}
              </div>
            </div>
          `).join('') : '<p style="color:var(--text-muted);">No suggestions.</p>';

          $('score-deadzones').innerHTML = '<p style="color:var(--text-muted);">Click “Find Dead Zones”.</p>';
          
        } catch (e) {
          alert(e.message);
        } finally {
          btn.disabled = false;
          deadBtn.disabled = false;
          btn.innerText = origText;
        }
      });

      // Dead Zone Handler
      $('deadzone-btn').addEventListener('click', async () => {
        const btn = $('deadzone-btn');
        const orig = btn.innerText;
        btn.disabled = true;
        btn.innerText = 'Finding...';
        try {
          const text = $('score-text').value;
          const data = await api('/patch/suggest', {
            text,
            doc_type: $('doc-type').value,
            max_input_tokens: parseInt($('max-tokens').value) || 512,
            normalize_text: true
          });
          const zones = data.dead_zones || [];
          const baseText = data.text || text;
          window._lastDeadZones = zones;
          window._lastDeadZoneText = baseText;
          window._lastPatchByZone = {};

          $('score-deadzones').innerHTML = zones.length ? zones.map((z, i) => `
            <div class="deadzone">
              <div class="deadzone-title" style="display:flex;align-items:center;justify-content:space-between;gap:10px;">
                <span>Severity ${(Number(z.severity || 0)).toFixed(2)}</span>
                <button class="copy-btn" onclick="patchDeadZone(${i})">Patch</button>
              </div>
              <div class="deadzone-body">${escapeHtml(z.excerpt || '')}</div>
              <div class="deadzone-meta">${escapeHtml((z.reasons || []).join(', '))}</div>
              <div id="patch-out-${i}" style="margin-top:10px;"></div>
            </div>
          `).join('') : '<p style="color:var(--text-muted);">No dead zones found.</p>';
        } catch (e) {
          alert(e.message);
        } finally {
          btn.disabled = false;
          btn.innerText = orig;
        }
      });

      // Patch dead zone (span patcher)
      window.patchDeadZone = async function(zoneIdx) {
        const zones = window._lastDeadZones || [];
        const z = zones[zoneIdx];
        const outEl = document.getElementById('patch-out-' + String(zoneIdx));
        if (!z || !outEl) return;
        outEl.innerHTML = '<p style="color:var(--text-muted);">Patching…</p>';
        try {
          const baseText = window._lastDeadZoneText || $('score-text').value;
          const data = await api('/patch/span', {
            text: baseText,
            doc_type: $('doc-type').value,
            start_char: Number(z.start_char || 0),
            end_char: Number(z.end_char || 0),
            normalize_text: true
          });
          const cands = data.candidates || [];
          window._lastPatchByZone[zoneIdx] = cands;
          if (!cands.length) {
            outEl.innerHTML = '<p style="color:var(--text-muted);">No patch candidates.</p>';
            return;
          }
          outEl.innerHTML = cands.slice(0, 3).map((c, i) => {
            const delta = (typeof c.primary_delta_0_100 === 'number' && Number.isFinite(c.primary_delta_0_100))
              ? (c.primary_delta_0_100 >= 0 ? '+' : '') + (Math.round(c.primary_delta_0_100 * 10) / 10)
              : '';
            const deltaLine = delta ? `Score delta: ${delta}` : '';
            return `
              <div class="card" style="margin:10px 0 0;padding:12px;">
                <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;">
                  <div style="font-family:var(--font-ui);font-size:12px;letter-spacing:0.06em;text-transform:uppercase;color:var(--text-muted);">
                    Patch ${i + 1}${deltaLine ? ` · ${escapeHtml(deltaLine)}` : ''}
                  </div>
                  <button class="copy-btn" onclick="applyPatch(${zoneIdx}, ${i})">Apply</button>
                </div>
                <pre style="margin-top:10px;white-space:pre-wrap;word-break:break-word;border:1px solid var(--border);border-radius:8px;padding:10px;background:rgba(0,0,0,0.12);">${escapeHtml(String(c.span_diff || '').slice(0, 5000))}</pre>
              </div>
            `;
          }).join('');
        } catch (e) {
          outEl.innerHTML = `<p style="color:var(--text-muted);">Patch failed: ${escapeHtml(String(e.message || e))}</p>`;
        }
      };

      window.applyPatch = function(zoneIdx, candIdx) {
        const zoneMap = window._lastPatchByZone || {};
        const cands = zoneMap[zoneIdx] || [];
        const cand = cands[candIdx];
        if (!cand || !cand.patched_text) return;
        $('score-text').value = String(cand.patched_text);
        // Clear old dead zones since offsets no longer match.
        $('score-deadzones').innerHTML = '<p style="color:var(--text-muted);">Click “Find Dead Zones”.</p>';
      };
      
      // Rewrite Handler
      $('rewrite-btn').addEventListener('click', async () => {
        const btn = $('rewrite-btn');
        const result = $('rewrite-result');
        const origText = btn.innerText;
        
        btn.disabled = true;
        btn.innerText = 'Generating...';
        result.classList.remove('visible');
        
        try {
          const data = await api('/rewrite', {
            text: $('rewrite-text').value,
            n_candidates: parseInt($('rewrite-candidates').value) || 4,
            temperature: parseFloat($('rewrite-temp').value) || 0.8,
            normalize_text: true
          });
          
          result.classList.add('visible');
          const rewrites = data.rewrites || [];
          
          $('rewrite-output').innerHTML = rewrites.length ? rewrites.map((r, i) => `
            <div class="rewrite-card">
              <div class="rewrite-header">
                <span>Version ${i + 1}</span>
                <span class="rewrite-score">Score: ${Math.round(r.score || 0)}</span>
              </div>
              <div class="rewrite-text">${escapeHtml(r.text || '')}</div>
              <div style="margin-top:12px;text-align:right;">
                <button class="copy-btn" onclick="navigator.clipboard.writeText(this.closest('.rewrite-card').querySelector('.rewrite-text').innerText)">Copy</button>
              </div>
            </div>
          `).join('') : '<p style="color:var(--text-muted);">No rewrites generated.</p>';
          
        } catch (e) {
          alert(e.message);
        } finally {
          btn.disabled = false;
          btn.innerText = origText;
        }
      });
      
      // Match Handler
      $('match-btn').addEventListener('click', async () => {
        const btn = $('match-btn');
        const result = $('match-result');
        const origText = btn.innerText;
        
        btn.disabled = true;
        btn.innerText = 'Generating...';
        result.classList.remove('visible');
        $('match-copy').disabled = true;
        $('match-output').innerText = '';
        
        try {
          const prompt = $('match-prompt').value || '';
          const data = await api('/cadence-match', {
            prompt,
            reference_text: $('match-reference').value,
            max_new_tokens: parseInt($('match-tokens').value) || 200,
            seed: parseInt($('match-seed').value) || 7
          });
          
          result.classList.add('visible');
          const continuation = data.generated_text || data.text || '';
          let out = '';
          if (prompt.trim() && continuation.trim()) {
            out = prompt.trimEnd() + '\\n\\n' + continuation.trimStart();
          } else {
            out = (prompt || continuation).trim();
          }
          $('match-output').innerText = out;
          $('match-copy').disabled = !out;
          
        } catch (e) {
          alert(e.message);
        } finally {
          btn.disabled = false;
          btn.innerText = origText;
        }
      });

      $('match-copy').addEventListener('click', async () => {
        const text = $('match-output').innerText || '';
        if (!text) return;
        try {
          await navigator.clipboard.writeText(text);
        } catch (e) {
          alert('Copy failed. Select and copy manually.');
        }
      });
      
      window.addEventListener('resize', () => {
        const c = $('cadence-chart');
        if (c._lastSeries) drawCadence(c._lastSeries);
      });
    </script>
  </body>
</html>
"""
