from __future__ import annotations


STUDIO_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Horace</title>
    <style>
      :root {
        --bg: #0a0f1a;
        --card: #111827;
        --border: rgba(255, 255, 255, 0.08);
        --text: #f3f4f6;
        --muted: #9ca3af;
        --accent: #6366f1;
        --accent-hover: #818cf8;
        --accent-dim: rgba(99, 102, 241, 0.15);
        --success: #10b981;
        --warning: #f59e0b;
        --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      }
      * { box-sizing: border-box; margin: 0; padding: 0; }
      body {
        font-family: var(--sans);
        background: var(--bg);
        color: var(--text);
        min-height: 100vh;
        line-height: 1.5;
      }
      .container {
        max-width: 900px;
        margin: 0 auto;
        padding: 32px 24px 64px;
      }
      header {
        text-align: center;
        margin-bottom: 32px;
      }
      h1 {
        font-size: 28px;
        font-weight: 600;
        letter-spacing: -0.5px;
      }
      .tagline {
        color: var(--muted);
        font-size: 15px;
        margin-top: 6px;
      }
      
      /* Tabs */
      .tabs {
        display: flex;
        gap: 4px;
        background: var(--card);
        padding: 4px;
        border-radius: 12px;
        margin-bottom: 24px;
        max-width: 400px;
        margin-left: auto;
        margin-right: auto;
      }
      .tab {
        flex: 1;
        padding: 10px 20px;
        border: none;
        background: transparent;
        color: var(--muted);
        font-size: 14px;
        font-weight: 500;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.15s;
      }
      .tab:hover { color: var(--text); }
      .tab.active {
        background: var(--accent);
        color: white;
      }
      
      /* Panels */
      .panel { display: none; }
      .panel.active { display: block; }
      
      /* Form elements */
      .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
      }
      label {
        display: block;
        font-size: 13px;
        color: var(--muted);
        margin-bottom: 8px;
      }
      textarea {
        width: 100%;
        min-height: 160px;
        padding: 16px;
        background: rgba(0,0,0,0.3);
        border: 1px solid var(--border);
        border-radius: 10px;
        color: var(--text);
        font-family: var(--sans);
        font-size: 15px;
        line-height: 1.7;
        resize: vertical;
        outline: none;
      }
      textarea:focus { border-color: var(--accent); }
      textarea::placeholder { color: var(--muted); opacity: 0.6; }
      input[type="number"], input[type="text"], select {
        width: 100%;
        padding: 10px 12px;
        background: rgba(0,0,0,0.3);
        border: 1px solid var(--border);
        border-radius: 8px;
        color: var(--text);
        font-size: 14px;
        outline: none;
      }
      input:focus, select:focus { border-color: var(--accent); }
      
      /* Buttons */
      .btn {
        display: block;
        width: 100%;
        padding: 14px 24px;
        background: var(--accent);
        color: white;
        border: none;
        border-radius: 10px;
        font-size: 15px;
        font-weight: 500;
        cursor: pointer;
        transition: background 0.15s;
      }
      .btn:hover { background: var(--accent-hover); }
      .btn:disabled { opacity: 0.5; cursor: not-allowed; }
      
      /* Results */
      .result {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 24px;
        margin-top: 20px;
        display: none;
      }
      .result.visible { display: block; }
      
      /* Score header */
      .score-header {
        display: flex;
        align-items: center;
        gap: 24px;
        margin-bottom: 24px;
      }
      .score-circle {
        width: 88px;
        height: 88px;
        border-radius: 50%;
        background: var(--accent-dim);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
      }
      .score-number {
        font-size: 32px;
        font-weight: 700;
        color: var(--accent);
      }
      .score-meta {
        flex: 1;
      }
      .score-label {
        font-size: 13px;
        color: var(--muted);
        margin-bottom: 4px;
      }
      .score-summary {
        font-size: 15px;
        color: var(--text);
        line-height: 1.6;
      }
      
      /* Highlighted text */
      .section-title {
        font-size: 12px;
        font-weight: 600;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .text-display {
        background: rgba(0,0,0,0.25);
        padding: 16px;
        border-radius: 10px;
        font-size: 15px;
        line-height: 1.8;
        margin-bottom: 24px;
        white-space: pre-wrap;
        word-wrap: break-word;
      }
      .spike {
        background: rgba(251, 191, 36, 0.2);
        border-bottom: 2px solid #fbbf24;
        padding: 2px 3px;
        border-radius: 3px;
        cursor: help;
      }
      .spike:hover {
        background: rgba(251, 191, 36, 0.35);
      }
      .legend-hint {
        font-size: 11px;
        color: var(--muted);
        font-weight: 400;
        text-transform: none;
        letter-spacing: 0;
      }
      
      /* Metrics */
      .metrics {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 24px;
      }
      @media (max-width: 600px) {
        .metrics { grid-template-columns: repeat(2, 1fr); }
      }
      .metric {
        background: rgba(0,0,0,0.25);
        padding: 16px;
        border-radius: 10px;
        text-align: center;
      }
      .metric-value {
        font-size: 24px;
        font-weight: 600;
      }
      .metric-label {
        font-size: 11px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.3px;
        margin-top: 4px;
      }
      
      /* Canvas */
      .chart-container {
        margin-bottom: 24px;
      }
      canvas {
        width: 100%;
        height: 80px;
        border-radius: 10px;
        background: rgba(0,0,0,0.25);
      }
      
      /* Suggestions */
      .suggestions-list {
        display: flex;
        flex-direction: column;
        gap: 8px;
      }
      .suggestion {
        background: rgba(0,0,0,0.25);
        border-radius: 10px;
        overflow: hidden;
        border-left: 3px solid var(--accent);
      }
      .suggestion-header {
        padding: 14px 16px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
      }
      .suggestion-header:hover {
        background: rgba(255,255,255,0.02);
      }
      .suggestion-title {
        font-weight: 500;
        font-size: 14px;
      }
      .suggestion-arrow {
        color: var(--muted);
        font-size: 18px;
        transition: transform 0.2s;
      }
      .suggestion.open .suggestion-arrow {
        transform: rotate(90deg);
      }
      .suggestion-body {
        display: none;
        padding: 0 16px 16px;
        font-size: 14px;
        color: var(--muted);
        line-height: 1.6;
      }
      .suggestion.open .suggestion-body {
        display: block;
      }
      .suggestion-why {
        font-size: 12px;
        color: var(--muted);
        opacity: 0.7;
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px solid var(--border);
        font-style: italic;
      }
      
      /* Generated text */
      .generated-text {
        font-size: 15px;
        line-height: 1.8;
        white-space: pre-wrap;
      }
      
      /* Rewrite output */
      .rewrites-grid {
        display: flex;
        flex-direction: column;
        gap: 16px;
      }
      .rewrite-item {
        background: rgba(0,0,0,0.25);
        border-radius: 10px;
        padding: 16px;
        border-left: 3px solid var(--accent);
      }
      .rewrite-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        padding-bottom: 12px;
        border-bottom: 1px solid var(--border);
      }
      .rewrite-rank {
        font-size: 13px;
        color: var(--muted);
      }
      .rewrite-score {
        font-size: 14px;
        color: var(--accent);
        font-weight: 600;
      }
      .rewrite-text {
        font-size: 15px;
        line-height: 1.7;
        white-space: pre-wrap;
      }
      
      /* Settings */
      .settings-toggle {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 12px 0;
        color: var(--muted);
        font-size: 13px;
        cursor: pointer;
        border: none;
        background: none;
        width: 100%;
        text-align: left;
      }
      .settings-toggle:hover { color: var(--text); }
      .settings-toggle svg { transition: transform 0.2s; }
      .settings-toggle.open svg { transform: rotate(90deg); }
      .settings-content {
        display: none;
        padding-top: 8px;
      }
      .settings-content.open { display: block; }
      .settings-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 12px;
      }
      
      /* Status */
      .status {
        text-align: center;
        padding: 12px;
        color: var(--muted);
        font-size: 14px;
      }
      
      /* Responsive */
      @media (max-width: 600px) {
        .container { padding: 20px 16px 48px; }
        h1 { font-size: 24px; }
        .tabs { max-width: 100%; }
        .score-header { flex-direction: column; text-align: center; gap: 16px; }
        .score-circle { width: 72px; height: 72px; }
        .score-number { font-size: 28px; }
      }
    </style>
  </head>
  <body>
    <div class="container">
      <header>
        <h1>Horace</h1>
        <p class="tagline">Measure the rhythm of your writing</p>
      </header>
      
      <div class="tabs">
        <button class="tab active" data-tab="score">Score</button>
        <button class="tab" data-tab="rewrite">Rewrite</button>
        <button class="tab" data-tab="match">Match Cadence</button>
      </div>
      
      <!-- Score Panel -->
      <div id="panel-score" class="panel active">
        <div class="card">
          <label>Your writing</label>
          <textarea id="score-text" placeholder="Paste your text here...">At dawn, the city leans into light. A gull lifts, then drops, then lifts again. The harbor breathes salt and diesel.</textarea>
        </div>
        
        <button class="btn" id="score-btn">Analyze</button>
        
        <div id="score-result" class="result">
          <div class="score-header">
            <div class="score-circle">
              <span class="score-number" id="score-value">--</span>
            </div>
            <div class="score-meta">
              <div class="score-label">out of 100</div>
              <div class="score-summary" id="score-summary">Analyzing your text...</div>
            </div>
          </div>
          
          <div class="section-title">
            Your text 
            <span class="legend-hint">— highlighted words have high surprise (spikes)</span>
          </div>
          <div class="text-display" id="text-highlighted"></div>
          
          <div class="section-title">Cadence curve</div>
          <div class="chart-container">
            <canvas id="cadence-chart" width="852" height="80"></canvas>
          </div>
          
          <div class="section-title">Category scores</div>
          <div class="metrics" id="score-metrics"></div>
          
          <div class="section-title">Suggestions</div>
          <div class="suggestions-list" id="score-suggestions"></div>
        </div>
        
        <div class="status" id="score-status"></div>
        
        <button class="settings-toggle" id="score-settings-toggle">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="9 18 15 12 9 6"></polyline>
          </svg>
          Advanced settings
        </button>
        <div class="settings-content" id="score-settings">
          <div class="settings-grid">
            <div>
              <label>Document type</label>
              <select id="doc-type">
                <option value="prose">Prose</option>
                <option value="poem">Poetry</option>
              </select>
            </div>
            <div>
              <label>Max tokens</label>
              <input type="number" id="max-tokens" value="512" />
            </div>
          </div>
        </div>
      </div>
      
      <!-- Rewrite Panel -->
      <div id="panel-rewrite" class="panel">
        <div class="card">
          <label>Text to improve</label>
          <textarea id="rewrite-text" placeholder="Paste text to rewrite with better cadence..."></textarea>
        </div>
        
        <button class="btn" id="rewrite-btn">Rewrite</button>
        
        <div id="rewrite-result" class="result">
          <div class="section-title">Rewritten versions (ranked by score)</div>
          <div class="rewrites-grid" id="rewrite-output"></div>
        </div>
        
        <div class="status" id="rewrite-status"></div>
        
        <button class="settings-toggle" id="rewrite-settings-toggle">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="9 18 15 12 9 6"></polyline>
          </svg>
          Advanced settings
        </button>
        <div class="settings-content" id="rewrite-settings">
          <div class="settings-grid">
            <div>
              <label>Candidates to generate</label>
              <input type="number" id="rewrite-candidates" value="4" />
            </div>
            <div>
              <label>Temperature</label>
              <input type="number" id="rewrite-temp" value="0.8" step="0.1" />
            </div>
          </div>
        </div>
      </div>
      
      <!-- Match Panel -->
      <div id="panel-match" class="panel">
        <div class="card">
          <label>Reference text (cadence to match)</label>
          <textarea id="match-reference" style="min-height:120px;">At dawn, the city leans into light. A gull lifts, then drops, then lifts again.</textarea>
        </div>
        
        <div class="card">
          <label>Your starting prompt</label>
          <textarea id="match-prompt" style="min-height:80px;">The morning light crept through the window</textarea>
        </div>
        
        <button class="btn" id="match-btn">Generate</button>
        
        <div id="match-result" class="result">
          <div class="section-title">Generated text</div>
          <div class="text-display generated-text" id="match-output"></div>
        </div>
        
        <div class="status" id="match-status"></div>
        
        <button class="settings-toggle" id="match-settings-toggle">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="9 18 15 12 9 6"></polyline>
          </svg>
          Advanced settings
        </button>
        <div class="settings-content" id="match-settings">
          <div class="settings-grid">
            <div>
              <label>Max tokens to generate</label>
              <input type="number" id="match-tokens" value="200" />
            </div>
            <div>
              <label>Seed</label>
              <input type="number" id="match-seed" value="7" />
            </div>
          </div>
        </div>
      </div>
    </div>

    <script>
      const $ = id => document.getElementById(id);
      const $$ = sel => document.querySelectorAll(sel);
      
      // Tab switching
      $$('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
          $$('.tab').forEach(t => t.classList.remove('active'));
          $$('.panel').forEach(p => p.classList.remove('active'));
          tab.classList.add('active');
          $('panel-' + tab.dataset.tab).classList.add('active');
        });
      });
      
      // Settings toggles
      $$('.settings-toggle').forEach(toggle => {
        toggle.addEventListener('click', () => {
          toggle.classList.toggle('open');
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
        if (!res.ok) throw new Error(await res.text());
        return res.json();
      }
      
      // Highlight text with spikes
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
          const title = `Surprisal: ${spike.surprisal?.toFixed(1) || '?'}`;
          result += `<span class="spike" title="${title}">${escapeHtml(word)}</span>`;
          lastEnd = end;
        }
        result += escapeHtml(text.slice(lastEnd));
        return result;
      }
      
      function escapeHtml(str) {
        return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      }
      
      // Cadence chart
      function drawCadence(series, threshold) {
        const canvas = $('cadence-chart');
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, rect.width, rect.height);
        
        if (!series || series.length < 2) return;
        
        let min = Math.min(...series), max = Math.max(...series);
        if (max - min < 0.1) { min = 0; max = 10; }
        
        const pad = 12, w = rect.width - pad*2, h = rect.height - pad*2;
        const xAt = i => pad + (i / (series.length - 1)) * w;
        const yAt = v => pad + (1 - (v - min) / (max - min)) * h;
        
        if (threshold) {
          ctx.strokeStyle = 'rgba(239,68,68,0.3)';
          ctx.setLineDash([4,4]);
          ctx.beginPath();
          ctx.moveTo(pad, yAt(threshold));
          ctx.lineTo(pad + w, yAt(threshold));
          ctx.stroke();
          ctx.setLineDash([]);
        }
        
        ctx.strokeStyle = '#6366f1';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(xAt(0), yAt(series[0]));
        for (let i = 1; i < series.length; i++) ctx.lineTo(xAt(i), yAt(series[i]));
        ctx.stroke();
      }
      
      // Suggestion toggle
      function setupSuggestionToggles() {
        $$('.suggestion-header').forEach(h => {
          h.addEventListener('click', () => {
            h.parentElement.classList.toggle('open');
          });
        });
      }
      
      // Score
      $('score-btn').addEventListener('click', async () => {
        const btn = $('score-btn');
        const status = $('score-status');
        btn.disabled = true;
        status.textContent = 'Analyzing...';
        $('score-result').classList.remove('visible');
        
        try {
          const text = $('score-text').value;
          const data = await api('/analyze', {
            text,
            doc_type: $('doc-type').value,
            max_input_tokens: parseInt($('max-tokens').value) || 512,
            normalize_text: true
          });
          
          // Score
          const score = Math.round(data.score?.overall_0_100 || 0);
          $('score-value').textContent = score;
          $('score-summary').textContent = data.critique?.summary || '';
          
          // Highlighted text
          const spikes = data.analysis?.spikes || [];
          $('text-highlighted').innerHTML = highlightText(text, spikes);
          
          // Metrics
          const cats = data.score?.categories || {};
          $('score-metrics').innerHTML = Object.entries(cats).map(([k, v]) => `
            <div class="metric">
              <div class="metric-value">${Math.round(v * 100)}</div>
              <div class="metric-label">${k}</div>
            </div>
          `).join('');
          
          // Cadence
          drawCadence(data.analysis?.series?.surprisal, data.analysis?.series?.threshold_surprisal);
          
          // Suggestions
          const suggestions = data.critique?.suggestions || [];
          if (suggestions.length) {
            $('score-suggestions').innerHTML = suggestions.slice(0, 4).map(s => `
              <div class="suggestion">
                <div class="suggestion-header">
                  <span class="suggestion-title">${s.title || ''}</span>
                  <span class="suggestion-arrow">›</span>
                </div>
                <div class="suggestion-body">
                  ${s.what_to_try || ''}
                  ${s.why ? `<div class="suggestion-why">${s.why}</div>` : ''}
                </div>
              </div>
            `).join('');
            setupSuggestionToggles();
          } else {
            $('score-suggestions').innerHTML = '<p style="color:var(--muted);font-size:14px;">No suggestions.</p>';
          }
          
          $('score-result').classList.add('visible');
          status.textContent = '';
        } catch (e) {
          status.textContent = 'Error: ' + e.message;
        } finally {
          btn.disabled = false;
        }
      });
      
      // Rewrite
      $('rewrite-btn').addEventListener('click', async () => {
        const btn = $('rewrite-btn');
        const status = $('rewrite-status');
        btn.disabled = true;
        status.textContent = 'Rewriting... (this may take a minute)';
        $('rewrite-result').classList.remove('visible');
        
        try {
          const data = await api('/rewrite', {
            text: $('rewrite-text').value,
            doc_type: $('doc-type').value,
            n_candidates: parseInt($('rewrite-candidates').value) || 4,
            keep_top: 3,
            temperature: parseFloat($('rewrite-temp').value) || 0.8,
            normalize_text: true
          });
          
          const rewrites = data.rewrites || [];
          $('rewrite-output').innerHTML = rewrites.length ? rewrites.map((r, i) => `
            <div class="rewrite-item">
              <div class="rewrite-header">
                <span class="rewrite-rank">#${i + 1}</span>
                <span class="rewrite-score">Score: ${Math.round(r.score || 0)}</span>
              </div>
              <div class="rewrite-text">${escapeHtml(r.text || '')}</div>
            </div>
          `).join('') : '<p style="color:var(--muted);font-size:14px;">No rewrites generated.</p>';
          
          $('rewrite-result').classList.add('visible');
          status.textContent = '';
        } catch (e) {
          status.textContent = 'Error: ' + e.message;
        } finally {
          btn.disabled = false;
        }
      });
      
      // Match
      $('match-btn').addEventListener('click', async () => {
        const btn = $('match-btn');
        const status = $('match-status');
        btn.disabled = true;
        status.textContent = 'Generating...';
        $('match-result').classList.remove('visible');
        
        try {
          const data = await api('/cadence-match', {
            prompt: $('match-prompt').value,
            reference_text: $('match-reference').value,
            max_new_tokens: parseInt($('match-tokens').value) || 200,
            seed: parseInt($('match-seed').value) || 7
          });
          
          $('match-output').textContent = data.generated_text || data.text || '(no text)';
          $('match-result').classList.add('visible');
          status.textContent = '';
        } catch (e) {
          status.textContent = 'Error: ' + e.message;
        } finally {
          btn.disabled = false;
        }
      });
      
      // Resize handler for canvas
      window.addEventListener('resize', () => {
        const canvas = $('cadence-chart');
        if (canvas._lastSeries) {
          drawCadence(canvas._lastSeries, canvas._lastThreshold);
        }
      });
    </script>
  </body>
</html>
"""
