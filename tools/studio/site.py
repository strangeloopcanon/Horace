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
        --success: #10b981;
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
        max-width: 720px;
        margin: 0 auto;
        padding: 32px 20px 64px;
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
      }
      .tab {
        flex: 1;
        padding: 10px 16px;
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
        border-radius: 16px;
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
        min-height: 180px;
        padding: 14px;
        background: rgba(0,0,0,0.3);
        border: 1px solid var(--border);
        border-radius: 12px;
        color: var(--text);
        font-family: var(--sans);
        font-size: 15px;
        line-height: 1.6;
        resize: vertical;
        outline: none;
      }
      textarea:focus {
        border-color: var(--accent);
      }
      textarea::placeholder {
        color: var(--muted);
        opacity: 0.6;
      }
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
      input:focus, select:focus {
        border-color: var(--accent);
      }
      
      /* Buttons */
      .btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        padding: 12px 24px;
        background: var(--accent);
        color: white;
        border: none;
        border-radius: 10px;
        font-size: 15px;
        font-weight: 500;
        cursor: pointer;
        transition: background 0.15s;
        width: 100%;
      }
      .btn:hover { background: var(--accent-hover); }
      .btn:disabled {
        opacity: 0.5;
        cursor: not-allowed;
      }
      
      /* Results */
      .result {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px;
        margin-top: 20px;
        display: none;
      }
      .result.visible { display: block; }
      .score-display {
        text-align: center;
        padding: 20px 0;
      }
      .score-number {
        font-size: 64px;
        font-weight: 700;
        color: var(--accent);
        line-height: 1;
      }
      .score-label {
        font-size: 14px;
        color: var(--muted);
        margin-top: 4px;
      }
      .score-summary {
        text-align: center;
        color: var(--muted);
        font-size: 15px;
        margin-top: 16px;
        padding: 0 20px;
      }
      .divider {
        height: 1px;
        background: var(--border);
        margin: 20px 0;
      }
      .metrics-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
      }
      .metric {
        background: rgba(0,0,0,0.2);
        padding: 14px;
        border-radius: 10px;
      }
      .metric-label {
        font-size: 12px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }
      .metric-value {
        font-size: 20px;
        font-weight: 600;
        margin-top: 4px;
      }
      
      /* Suggestions */
      .suggestions {
        margin-top: 20px;
      }
      .suggestions h3 {
        font-size: 14px;
        color: var(--muted);
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }
      .suggestion {
        padding: 14px;
        background: rgba(0,0,0,0.2);
        border-radius: 10px;
        margin-bottom: 10px;
      }
      .suggestion-title {
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 4px;
      }
      .suggestion-text {
        font-size: 14px;
        color: var(--muted);
      }
      
      /* Generated text */
      .generated-text {
        font-size: 15px;
        line-height: 1.7;
        white-space: pre-wrap;
      }
      
      /* Settings collapse */
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
      .settings-toggle svg {
        transition: transform 0.2s;
      }
      .settings-toggle.open svg {
        transform: rotate(90deg);
      }
      .settings-content {
        display: none;
        padding-top: 12px;
      }
      .settings-content.open {
        display: block;
      }
      .settings-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
      }
      .settings-grid > div {
        display: flex;
        flex-direction: column;
      }
      
      /* Status */
      .status {
        text-align: center;
        padding: 12px;
        color: var(--muted);
        font-size: 14px;
      }
      
      /* Cadence chart placeholder */
      .chart-placeholder {
        height: 100px;
        background: rgba(0,0,0,0.2);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--muted);
        font-size: 13px;
      }
      
      canvas {
        width: 100%;
        height: 100px;
        border-radius: 10px;
        background: rgba(0,0,0,0.2);
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
          <textarea id="score-text" placeholder="Paste your text here...">At dawn, the city leans into light.
A gull lifts, then drops, then lifts again.</textarea>
        </div>
        
        <button class="btn" id="score-btn">Analyze</button>
        
        <div id="score-result" class="result">
          <div class="score-display">
            <div class="score-number" id="score-value">--</div>
            <div class="score-label">out of 100</div>
          </div>
          <p class="score-summary" id="score-summary"></p>
          
          <div class="divider"></div>
          
          <canvas id="cadence-chart" width="680" height="100"></canvas>
          
          <div class="divider"></div>
          
          <div class="metrics-grid" id="score-metrics"></div>
          
          <div class="suggestions" id="score-suggestions"></div>
        </div>
        
        <div class="status" id="score-status"></div>
        
        <button class="settings-toggle" id="score-settings-toggle">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
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
                <option value="shortstory">Short story</option>
                <option value="novel">Novel excerpt</option>
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
          <textarea id="rewrite-text" placeholder="Paste text you want to rewrite with better cadence..."></textarea>
        </div>
        
        <button class="btn" id="rewrite-btn">Rewrite</button>
        
        <div id="rewrite-result" class="result">
          <h3 style="font-size: 14px; color: var(--muted); margin-bottom: 16px;">Rewritten versions (ranked by score)</h3>
          <div id="rewrite-output"></div>
        </div>
        
        <div class="status" id="rewrite-status"></div>
        
        <button class="settings-toggle" id="rewrite-settings-toggle">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="9 18 15 12 9 6"></polyline>
          </svg>
          Advanced settings
        </button>
        <div class="settings-content" id="rewrite-settings">
          <div class="settings-grid">
            <div>
              <label>Candidates</label>
              <input type="number" id="rewrite-candidates" value="4" min="1" max="8" />
            </div>
            <div>
              <label>Temperature</label>
              <input type="number" id="rewrite-temp" value="0.8" step="0.1" min="0.1" max="1.5" />
            </div>
          </div>
        </div>
      </div>
      
      <!-- Match Cadence Panel -->
      <div id="panel-match" class="panel">
        <div class="card">
          <label>Reference text (cadence to match)</label>
          <textarea id="match-reference" placeholder="Paste a passage whose rhythm you want to emulate...">At dawn, the city leans into light. A gull lifts, then drops, then lifts again. The harbor breathes salt and diesel.</textarea>
        </div>
        
        <div class="card">
          <label>Your starting prompt</label>
          <textarea id="match-prompt" style="min-height: 80px;" placeholder="Start of your new text...">The morning light crept through the window</textarea>
        </div>
        
        <button class="btn" id="match-btn">Generate</button>
        
        <div id="match-result" class="result">
          <label style="margin-bottom: 12px;">Generated text</label>
          <div class="generated-text" id="match-output"></div>
        </div>
        
        <div class="status" id="match-status"></div>
        
        <button class="settings-toggle" id="match-settings-toggle">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="9 18 15 12 9 6"></polyline>
          </svg>
          Advanced settings
        </button>
        <div class="settings-content" id="match-settings">
          <div class="settings-grid">
            <div>
              <label>Max new tokens</label>
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
      
      // API helper
      async function api(endpoint, body) {
        const res = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });
        if (!res.ok) {
          const err = await res.text();
          throw new Error(err);
        }
        return res.json();
      }
      
      // Draw cadence chart
      function drawCadence(series, threshold) {
        const canvas = $('cadence-chart');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (!series || series.length < 2) return;
        
        const n = series.length;
        let min = Math.min(...series);
        let max = Math.max(...series);
        if (max - min < 0.1) { min = 0; max = 10; }
        
        const pad = 10;
        const w = canvas.width - pad * 2;
        const h = canvas.height - pad * 2;
        const xAt = i => pad + (i / (n - 1)) * w;
        const yAt = v => pad + (1 - (v - min) / (max - min)) * h;
        
        // Threshold line
        if (threshold) {
          ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
          ctx.setLineDash([4, 4]);
          ctx.beginPath();
          ctx.moveTo(pad, yAt(threshold));
          ctx.lineTo(pad + w, yAt(threshold));
          ctx.stroke();
          ctx.setLineDash([]);
        }
        
        // Series
        ctx.strokeStyle = '#6366f1';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(xAt(0), yAt(series[0]));
        for (let i = 1; i < n; i++) ctx.lineTo(xAt(i), yAt(series[i]));
        ctx.stroke();
      }
      
      // Score
      $('score-btn').addEventListener('click', async () => {
        const btn = $('score-btn');
        const status = $('score-status');
        btn.disabled = true;
        status.textContent = 'Analyzing...';
        $('score-result').classList.remove('visible');
        
        try {
          const data = await api('/analyze', {
            text: $('score-text').value,
            doc_type: $('doc-type').value,
            max_input_tokens: parseInt($('max-tokens').value) || 512,
            normalize_text: true
          });
          
          // Score
          const score = Math.round(data.score?.overall_0_100 || 0);
          $('score-value').textContent = score;
          
          // Summary from critique
          const summary = data.critique?.summary || '';
          $('score-summary').textContent = summary;
          
          // Metrics
          const cats = data.score?.categories || {};
          const metricsHtml = Object.entries(cats).map(([k, v]) => `
            <div class="metric">
              <div class="metric-label">${k}</div>
              <div class="metric-value">${Math.round(v * 100)}</div>
            </div>
          `).join('');
          $('score-metrics').innerHTML = metricsHtml;
          
          // Suggestions
          const suggestions = data.critique?.suggestions || [];
          if (suggestions.length) {
            const sugHtml = suggestions.slice(0, 3).map(s => `
              <div class="suggestion">
                <div class="suggestion-title">${s.title || ''}</div>
                <div class="suggestion-text">${s.what_to_try || ''}</div>
              </div>
            `).join('');
            $('score-suggestions').innerHTML = '<h3>Suggestions</h3>' + sugHtml;
          } else {
            $('score-suggestions').innerHTML = '';
          }
          
          // Cadence chart
          const series = data.analysis?.series?.surprisal || [];
          const threshold = data.analysis?.series?.threshold_surprisal;
          drawCadence(series, threshold);
          
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
          const html = rewrites.map((r, i) => `
            <div class="suggestion" style="margin-bottom: 16px;">
              <div class="suggestion-title">Version ${i + 1} — Score: ${Math.round(r.score || 0)}</div>
              <div style="margin-top: 10px; white-space: pre-wrap; line-height: 1.6;">${r.text || ''}</div>
            </div>
          `).join('');
          
          $('rewrite-output').innerHTML = html || '<p style="color: var(--muted);">No rewrites generated.</p>';
          $('rewrite-result').classList.add('visible');
          status.textContent = '';
        } catch (e) {
          status.textContent = 'Error: ' + e.message;
        } finally {
          btn.disabled = false;
        }
      });
      
      // Match Cadence
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
          
          $('match-output').textContent = data.generated_text || data.text || '(no text generated)';
          $('match-result').classList.add('visible');
          status.textContent = '';
        } catch (e) {
          status.textContent = 'Error: ' + e.message;
        } finally {
          btn.disabled = false;
        }
      });
    </script>
  </body>
</html>
"""
