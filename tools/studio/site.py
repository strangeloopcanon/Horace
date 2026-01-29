from __future__ import annotations


STUDIO_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Horace</title>
    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=Inter:wght@400;500;600&family=Playfair+Display:ital,wght@0,600;0,700;1,400&display=swap" rel="stylesheet">
    
    <style>
      :root {
        /* Palette: The Obsidian Study */
        --bg: #1c1b1a;         /* Warm Obsidian */
        --surface: #262422;    /* Oiled Walnut */
        --surface-light: #2f2d2b;
        --border: #45403b;     /* Worn Iron */
        --text: #e6e1d3;       /* Bone/Vellum */
        --text-muted: #9d968b;
        
        --accent: #c5a059;     /* Antique Brass */
        --accent-glow: rgba(197, 160, 89, 0.2);
        --accent-hover: #d6b064;
        
        --success: #a8bba0;    /* Sage */
        --warning: #d4a373;    /* Terracotta */
        
        /* Typography */
        --font-display: "Playfair Display", serif;
        --font-body: "Cormorant Garamond", serif;
        --font-ui: "Inter", sans-serif;
        
        --ease: cubic-bezier(0.25, 1, 0.5, 1);
      }
      
      * { box-sizing: border-box; margin: 0; padding: 0; }
      
      body {
        font-family: var(--font-body);
        background-color: var(--bg);
        color: var(--text);
        min-height: 100vh;
        line-height: 1.6;
        overflow-y: scroll;
        /* Noise Texture */
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' opacity='0.03'/%3E%3C/svg%3E");
      }
      
      .container {
        max-width: 860px;
        margin: 0 auto;
        padding: 64px 24px 120px;
      }
      
      /* Typography & Header */
      header {
        text-align: center;
        margin-bottom: 56px;
        position: relative;
        padding-bottom: 24px;
        border-bottom: 1px solid var(--border);
      }
      header::after {
        content: "";
        position: absolute;
        bottom: -3px;
        left: 50%;
        transform: translateX(-50%);
        width: 40px;
        height: 5px;
        background: var(--bg);
        border-left: 1px solid var(--border);
        border-right: 1px solid var(--border);
      }
      
      h1 {
        font-family: var(--font-display);
        font-size: 48px;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: var(--text);
        margin-bottom: 8px;
      }
      .tagline {
        font-family: var(--font-ui);
        color: var(--accent);
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        font-weight: 500;
      }
      
      /* Tabs - Editorial Style */
      .tabs {
        display: flex;
        justify-content: center;
        gap: 40px;
        margin-bottom: 48px;
        font-family: var(--font-ui);
      }
      .tab {
        background: none;
        border: none;
        color: var(--text-muted);
        font-size: 14px;
        font-weight: 500;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        cursor: pointer;
        padding-bottom: 8px;
        position: relative;
        transition: color 0.3s ease;
      }
      .tab:hover { color: var(--text); }
      .tab::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 1px;
        background: var(--accent);
        transform: scaleX(0);
        transition: transform 0.3s var(--ease);
      }
      .tab.active { color: var(--accent); }
      .tab.active::after { transform: scaleX(1); }
      
      /* Panels */
      .panel { display: none; opacity: 0; transform: translateY(10px); transition: all 0.5s var(--ease); }
      .panel.active { display: block; opacity: 1; transform: translateY(0); }
      
      /* Form Elements */
      .input-wrapper {
        position: relative;
        margin-bottom: 32px;
      }
      .input-label {
        position: absolute;
        top: -10px;
        left: 16px;
        background: var(--bg);
        padding: 0 8px;
        font-family: var(--font-ui);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--accent);
        z-index: 10;
      }
      
      textarea {
        width: 100%;
        min-height: 240px;
        padding: 24px;
        background: transparent;
        border: 1px solid var(--border);
        color: var(--text);
        font-family: var(--font-body);
        font-size: 20px;
        line-height: 1.6;
        resize: vertical;
        outline: none;
        transition: all 0.3s ease;
        border-radius: 2px; /* Sharp corners */
      }
      textarea:focus {
        border-color: var(--accent);
        box-shadow: 0 0 20px var(--accent-glow);
      }
      textarea::placeholder {
        color: var(--text-muted);
        font-style: italic;
        opacity: 0.5;
      }
      
      input[type="number"], select {
        width: 100%;
        padding: 12px;
        background: transparent;
        border: 1px solid var(--border);
        color: var(--text);
        font-family: var(--font-ui);
        font-size: 14px;
        outline: none;
        border-radius: 2px;
      }
      input:focus, select:focus { border-color: var(--accent); }
      
      /* Primary Button - Gold Outline */
      .btn {
        display: block;
        width: 100%;
        padding: 18px;
        background: transparent;
        border: 1px solid var(--accent);
        color: var(--accent);
        font-family: var(--font-ui);
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        cursor: pointer;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
      }
      .btn:hover {
        background: var(--accent);
        color: var(--bg);
        box-shadow: 0 0 30px var(--accent-glow);
      }
      .btn:disabled { opacity: 0.5; cursor: wait; }
      
      /* Results Area */
      .result-area {
        margin-top: 64px;
        border-top: 1px solid var(--border);
        padding-top: 48px;
        display: none;
      }
      .result-area.visible { display: block; animation: fade-in 0.8s var(--ease); }
      
      @keyframes fade-in {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
      }
      
      /* Score Hero */
      .score-display {
        display: flex;
        align-items: center;
        gap: 48px;
        margin-bottom: 56px;
      }
      .score-big {
        font-family: var(--font-display);
        font-size: 80px;
        line-height: 1;
        color: var(--accent);
        font-weight: 400;
        position: relative;
      }
      .score-big::after {
        content: "/100";
        font-size: 20px;
        color: var(--text-muted);
        font-family: var(--font-ui);
        position: absolute;
        top: 8px;
        right: -40px;
      }
      .score-context {
        flex: 1;
        font-family: var(--font-body);
        font-size: 18px;
        color: var(--text);
        font-style: italic;
        border-left: 2px solid var(--border);
        padding-left: 24px;
      }
      
      /* Section Headers */
      .section-label {
        font-family: var(--font-ui);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-bottom: 24px;
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
      
      /* Text Display with Highlights */
      .text-display {
        background: var(--surface);
        padding: 40px;
        font-size: 19px;
        line-height: 1.8;
        border: 1px solid var(--border);
        position: relative;
        margin-bottom: 48px;
      }
      .text-display::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 4px;
        background: var(--border);
      }
      
      .spike {
        border-bottom: 2px solid var(--accent);
        background: rgba(197, 160, 89, 0.1);
        cursor: help;
        transition: background 0.2s;
      }
      .spike:hover { background: rgba(197, 160, 89, 0.3); color: #fff; }
      
      /* Charts & Metrics */
      .metrics-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1px;
        background: var(--border);
        border: 1px solid var(--border);
        margin-bottom: 48px;
      }
      .metric-cell {
        background: var(--bg);
        padding: 24px 16px;
        text-align: center;
      }
      .metric-val {
        font-family: var(--font-display);
        font-size: 32px;
        color: var(--text);
        margin-bottom: 8px;
      }
      .metric-name {
        font-family: var(--font-ui);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
      }
      
      canvas {
        width: 100%;
        height: 120px;
        margin-bottom: 48px;
        border-bottom: 1px solid var(--border);
      }
      
      /* Suggestions */
      .suggestion-group {
        border-left: 1px solid var(--border);
        margin-left: 20px;
        padding-left: 32px;
        position: relative;
      }
      .suggestion {
        margin-bottom: 32px;
      }
      .suggestion::before {
        content: "";
        position: absolute;
        left: -5px;
        width: 9px;
        height: 9px;
        background: var(--bg);
        border: 1px solid var(--accent);
        border-radius: 50%;
        margin-top: 6px;
      }
      .sugg-title {
        font-family: var(--font-display);
        font-size: 20px;
        color: var(--text);
        margin-bottom: 8px;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .sugg-title:hover { color: var(--accent); }
      .sugg-body {
        font-size: 16px;
        color: var(--text-muted);
        display: none;
        padding-top: 8px;
      }
      .suggestion.open .sugg-body { display: block; }
      .suggestion.open .sugg-title { color: var(--accent); }
      
      /* Settings */
      .settings-toggle {
        font-family: var(--font-ui);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        background: none;
        border: none;
        cursor: pointer;
        margin-top: 48px;
        display: flex;
        align-items: center;
        gap: 8px;
        width: 100%;
        justify-content: center;
      }
      .settings-toggle:hover { color: var(--accent); }
      .settings-content {
        display: none;
        background: var(--surface);
        padding: 24px;
        border: 1px solid var(--border);
        margin-top: 24px;
      }
      .settings-content.open { display: block; animation: fade-in 0.3s; }
      .settings-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
      
      /* Mobile */
      @media (max-width: 600px) {
        .score-display { flex-direction: column; text-align: center; gap: 24px; }
        .score-context { border-left: none; padding-left: 0; border-top: 1px solid var(--border); padding-top: 24px; }
        .metrics-row { grid-template-columns: 1fr 1fr; }
        .tabs { gap: 20px; }
        h1 { font-size: 36px; }
      }
      
      /* Helper classes */
      .copy-link {
        font-family: var(--font-ui);
        font-size: 11px;
        text-transform: uppercase;
        color: var(--accent);
        cursor: pointer;
        float: right;
        margin-bottom: 8px;
      }
      .copy-link:hover { text-decoration: underline; }
    </style>
  </head>
  <body>
    <div class="container">
      <header>
        <h1>Horace</h1>
        <div class="tagline">The Rhythm of Rhetoric</div>
      </header>
      
      <div class="tabs">
        <button class="tab active" data-tab="score">Analysis</button>
        <button class="tab" data-tab="rewrite">Rewrite</button>
        <button class="tab" data-tab="match">Cadence Match</button>
      </div>
      
      <!-- Score Panel -->
      <div id="panel-score" class="panel active">
        <div class="input-wrapper">
          <span class="input-label">Source Text</span>
          <textarea id="score-text" placeholder="Paste your draft here. Let us measure its pulse...">At dawn, the city leans into light. A gull lifts, then drops, then lifts again. The harbor breathes salt and diesel.</textarea>
        </div>
        
        <button class="btn" id="score-btn">Measure Cadence</button>
        
        <div id="score-result" class="result-area">
          <div class="score-display">
            <div class="score-big" id="score-value">--</div>
            <div class="score-context" id="score-summary"></div>
          </div>
          
          <div class="section-label">Cadence Topography</div>
          <div class="text-display" id="text-highlighted"></div>
          
          <div class="section-label">Rhythmic Flow</div>
          <canvas id="cadence-chart"></canvas>
          
          <div class="section-label">Metrics</div>
          <div class="metrics-row" id="score-metrics"></div>
          
          <div class="section-label">Editorial Notes</div>
          <div class="suggestion-group" id="score-suggestions"></div>
        </div>
        
        <button class="settings-toggle" id="score-settings-toggle">
          <span>Config &middot; Advanced</span>
        </button>
        <div class="settings-content" id="score-settings">
          <div class="settings-grid">
            <div>
              <label class="input-label" style="position:static;margin-bottom:8px;display:block;">Form</label>
              <select id="doc-type">
                <option value="prose">Prose</option>
                <option value="poem">Poetry</option>
              </select>
            </div>
            <div>
              <label class="input-label" style="position:static;margin-bottom:8px;display:block;">Token Limit</label>
              <input type="number" id="max-tokens" value="512" />
            </div>
          </div>
        </div>
      </div>
      
      <!-- Rewrite Panel -->
      <div id="panel-rewrite" class="panel">
        <div class="input-wrapper">
          <span class="input-label">Draft to Rewrite</span>
          <textarea id="rewrite-text" placeholder="Enter text to re-imagine..."></textarea>
        </div>
        
        <button class="btn" id="rewrite-btn">Generate Variations</button>
        
        <div id="rewrite-result" class="result-area">
          <div class="section-label">Variations</div>
          <div id="rewrite-output"></div>
        </div>
        
        <button class="settings-toggle" id="rewrite-settings-toggle"><span>Configuration</span></button>
        <div class="settings-content" id="rewrite-settings">
          <div class="settings-grid">
            <div>
              <label class="input-label" style="position:static;margin-bottom:8px;display:block;">Count</label>
              <input type="number" id="rewrite-candidates" value="4" />
            </div>
            <div>
              <label class="input-label" style="position:static;margin-bottom:8px;display:block;">Temperature</label>
              <input type="number" id="rewrite-temp" value="0.8" step="0.1" />
            </div>
          </div>
        </div>
      </div>
      
      <!-- Match Panel -->
      <div id="panel-match" class="panel">
        <div class="input-wrapper">
          <span class="input-label">Reference Cadence</span>
          <textarea id="match-reference" style="min-height:160px;">At dawn, the city leans into light. A gull lifts, then drops, then lifts again.</textarea>
        </div>
        
        <div class="input-wrapper">
          <span class="input-label">Starting Prompt</span>
          <textarea id="match-prompt" style="min-height:100px;">The morning light crept through the window</textarea>
        </div>
        
        <button class="btn" id="match-btn">Extrapolate Pattern</button>
        
        <div id="match-result" class="result-area">
          <span class="copy-link" onclick="copyText('match-output')">Copy to Clipboard</span>
          <div class="section-label">Generated Continuation</div>
          <div class="text-display" id="match-output"></div>
        </div>
        
        <button class="settings-toggle" id="match-settings-toggle"><span>Configuration</span></button>
        <div class="settings-content" id="match-settings">
          <div class="settings-grid">
            <div>
              <label class="input-label" style="position:static;margin-bottom:8px;display:block;">Length</label>
              <input type="number" id="match-tokens" value="200" />
            </div>
            <div>
              <label class="input-label" style="position:static;margin-bottom:8px;display:block;">Seed</label>
              <input type="number" id="match-seed" value="7" />
            </div>
          </div>
        </div>
      </div>
    </div>

    <script>
      const $ = id => document.getElementById(id);
      const $$ = sel => document.querySelectorAll(sel);
      
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
      
      // Copy
      async function copyText(id) {
        const text = $(id).textContent;
        await navigator.clipboard.writeText(text);
        alert('Copied to clipboard');
      }
      
      // API Wrapper
      async function api(endpoint, body) {
        const res = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });
        if (!res.ok) throw new Error(await res.text());
        return res.json();
      }
      
      // Logic
      function highlightText(text, spikes) {
        if (!spikes || !spikes.length) return text; // Basic escape handled by browser innerText usually, but let's be safe
        const safeText = text.replace(/&/g, '&amp;').replace(/</g, '&lt;');
        
        const sorted = [...spikes].sort((a, b) => a.char_start - b.char_start);
        let result = '';
        let lastEnd = 0;
        
        for (const spike of sorted) {
          const start = spike.char_start;
          const end = spike.char_end;
          if (start < lastEnd) continue;
          
          result += safeText.slice(lastEnd, start);
          const word = safeText.slice(start, end);
          result += `<span class="spike" title="Surprisal: ${spike.surprisal?.toFixed(1)}">${word}</span>`;
          lastEnd = end;
        }
        result += safeText.slice(lastEnd);
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
        
        const accent = getComputedStyle(document.body).getPropertyValue('--accent').trim();
        
        let min = Math.min(...series), max = Math.max(...series);
        if (max - min < 0.1) { min = 0; max = 10; }
        
        const w = rect.width;
        const h = rect.height;
        const xAt = i => (i / (series.length - 1)) * w;
        const yAt = v => h - ((v - min) / (max - min)) * (h * 0.8) - (h * 0.1); // add padding
        
        ctx.beginPath();
        ctx.moveTo(xAt(0), yAt(series[0]));
        for (let i = 1; i < series.length; i++) {
            // Smooth curve
            const x = xAt(i);
            const y = yAt(series[i]);
            ctx.lineTo(x, y);
        }
        
        ctx.strokeStyle = accent;
        ctx.lineWidth = 1.5;
        ctx.stroke();
        
        // Fill
        ctx.lineTo(w, h);
        ctx.lineTo(0, h);
        ctx.closePath();
        const grad = ctx.createLinearGradient(0, 0, 0, h);
        grad.addColorStop(0, accent + '22'); // 22 is hex alpha
        grad.addColorStop(1, accent + '00');
        ctx.fillStyle = grad;
        ctx.fill();
        
        canvas._lastSeries = series;
      }
      
      // Handlers
      $('score-btn').addEventListener('click', async () => {
        const btn = $('score-btn');
        const result = $('score-result');
        const origText = btn.innerText;
        
        btn.disabled = true;
        btn.innerText = 'Analyzing...';
        result.classList.remove('visible');
        
        try {
          const text = $('score-text').value;
          const data = await api('/analyze', {
            text,
            doc_type: $('doc-type').value,
            max_input_tokens: parseInt($('max-tokens').value) || 512,
            normalize_text: true
          });
          
          result.classList.add('visible');
          $('score-value').innerText = Math.round(data.score?.overall_0_100 || 0);
          $('score-summary').innerText = data.critique?.summary || 'Analysis complete.';
          $('text-highlighted').innerHTML = highlightText(text, data.analysis?.spikes || []);
          drawCadence(data.analysis?.series?.surprisal);
          
          const cats = data.score?.categories || {};
          $('score-metrics').innerHTML = Object.entries(cats).map(([k, v]) => `
            <div class="metric-cell">
              <div class="metric-val">${Math.round(v * 100)}</div>
              <div class="metric-name">${k}</div>
            </div>
          `).join('');
          
          const suggs = data.critique?.suggestions || [];
          $('score-suggestions').innerHTML = suggs.map(s => `
            <div class="suggestion">
              <div class="sugg-title" onclick="this.parentElement.classList.toggle('open')">
                <span>✦ ${s.title}</span>
              </div>
              <div class="sugg-body">
                <p><strong>Try:</strong> ${s.what_to_try}</p>
                <p style="margin-top:8px;font-style:italic;opacity:0.7;">${s.why || ''}</p>
              </div>
            </div>
          `).join('') || '<div style="color:var(--text-muted);font-style:italic;">No suggestions available.</div>';
          
        } catch (e) {
          alert(e.message);
        } finally {
          btn.disabled = false;
          btn.innerText = origText;
        }
      });
      
      $('rewrite-btn').addEventListener('click', async () => {
        const btn = $('rewrite-btn');
        const result = $('rewrite-result');
        btn.disabled = true;
        btn.innerText = 'Rewriting...';
        result.classList.remove('visible');
        
        try {
          const data = await api('/rewrite', {
            text: $('rewrite-text').value,
            n_candidates: parseInt($('rewrite-candidates').value) || 3,
            temperature: parseFloat($('rewrite-temp').value) || 0.8,
            normalize_text: true
          });
          
          result.classList.add('visible');
          $('rewrite-output').innerHTML = (data.rewrites || []).map((r, i) => `
            <div class="text-display" style="margin-bottom:24px;">
              <span class="copy-link" onclick="navigator.clipboard.writeText(this.nextElementSibling.innerText)">Copy</span>
              <div style="font-family:var(--font-body);">${r.text.replace(/</g, '&lt;')}</div>
              <div class="section-label" style="margin-top:16px;margin-bottom:0;">Score: ${Math.round(r.score)}</div>
            </div>
          `).join('');
          
        } catch (e) {
          alert(e.message);
        } finally {
          btn.disabled = false;
          btn.innerText = 'Generate Variations';
        }
      });
      
      $('match-btn').addEventListener('click', async () => {
        const btn = $('match-btn');
        const result = $('match-result');
        btn.disabled = true;
        btn.innerText = 'Processing...';
        result.classList.remove('visible');
        
        try {
          const data = await api('/cadence-match', {
            prompt: $('match-prompt').value,
            reference_text: $('match-reference').value,
            max_new_tokens: parseInt($('match-tokens').value) || 200,
            seed: parseInt($('match-seed').value) || 7
          });
          result.classList.add('visible');
          $('match-output').innerText = data.generated_text || data.text || '';
        } catch (e) {
          alert(e.message);
        } finally {
          btn.disabled = false;
          btn.innerText = 'Extrapolate Pattern';
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
