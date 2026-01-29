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
        max-width: 640px;
        margin: 0 auto;
        padding: 24px 16px 48px;
      }
      @media (max-width: 480px) {
        .container { padding: 16px 12px 32px; }
      }
      header {
        text-align: center;
        margin-bottom: 24px;
      }
      h1 {
        font-size: 24px;
        font-weight: 600;
        letter-spacing: -0.5px;
      }
      .tagline {
        color: var(--muted);
        font-size: 14px;
        margin-top: 4px;
      }
      
      /* Tabs */
      .tabs {
        display: flex;
        gap: 2px;
        background: var(--card);
        padding: 3px;
        border-radius: 10px;
        margin-bottom: 16px;
      }
      .tab {
        flex: 1;
        padding: 8px 12px;
        border: none;
        background: transparent;
        color: var(--muted);
        font-size: 13px;
        font-weight: 500;
        border-radius: 7px;
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
        padding: 16px;
        margin-bottom: 12px;
      }
      label {
        display: block;
        font-size: 12px;
        color: var(--muted);
        margin-bottom: 6px;
      }
      textarea {
        width: 100%;
        min-height: 140px;
        padding: 12px;
        background: rgba(0,0,0,0.3);
        border: 1px solid var(--border);
        border-radius: 10px;
        color: var(--text);
        font-family: var(--sans);
        font-size: 14px;
        line-height: 1.6;
        resize: vertical;
        outline: none;
      }
      textarea:focus { border-color: var(--accent); }
      textarea::placeholder { color: var(--muted); opacity: 0.6; }
      input[type="number"], input[type="text"], select {
        width: 100%;
        padding: 8px 10px;
        background: rgba(0,0,0,0.3);
        border: 1px solid var(--border);
        border-radius: 6px;
        color: var(--text);
        font-size: 13px;
        outline: none;
      }
      input:focus, select:focus { border-color: var(--accent); }
      
      /* Buttons */
      .btn {
        display: block;
        width: 100%;
        padding: 12px 20px;
        background: var(--accent);
        color: white;
        border: none;
        border-radius: 10px;
        font-size: 14px;
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
        padding: 16px;
        margin-top: 16px;
        display: none;
      }
      .result.visible { display: block; }
      
      /* Score header */
      .score-header {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 16px;
      }
      .score-circle {
        width: 72px;
        height: 72px;
        border-radius: 50%;
        background: var(--accent-dim);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
      }
      .score-number {
        font-size: 28px;
        font-weight: 700;
        color: var(--accent);
      }
      .score-summary {
        font-size: 14px;
        color: var(--muted);
        line-height: 1.5;
      }
      
      /* Highlighted text */
      .text-display {
        background: rgba(0,0,0,0.2);
        padding: 12px;
        border-radius: 8px;
        font-size: 14px;
        line-height: 1.7;
        margin-bottom: 16px;
        white-space: pre-wrap;
        word-wrap: break-word;
      }
      .spike {
        background: rgba(251, 191, 36, 0.25);
        border-bottom: 2px solid #fbbf24;
        padding: 1px 2px;
        border-radius: 2px;
        cursor: help;
      }
      .spike:hover {
        background: rgba(251, 191, 36, 0.4);
      }
      .text-legend {
        display: flex;
        gap: 16px;
        font-size: 11px;
        color: var(--muted);
        margin-bottom: 12px;
      }
      .legend-item {
        display: flex;
        align-items: center;
        gap: 4px;
      }
      .legend-dot {
        width: 12px;
        height: 12px;
        border-radius: 2px;
        background: rgba(251, 191, 36, 0.25);
        border-bottom: 2px solid #fbbf24;
      }
      
      /* Metrics */
      .metrics {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 16px;
      }
      .metric {
        flex: 1;
        min-width: 70px;
        background: rgba(0,0,0,0.2);
        padding: 10px;
        border-radius: 8px;
        text-align: center;
      }
      .metric-value {
        font-size: 18px;
        font-weight: 600;
      }
      .metric-label {
        font-size: 10px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.3px;
        margin-top: 2px;
      }
      
      /* Canvas */
      canvas {
        width: 100%;
        height: 60px;
        border-radius: 8px;
        background: rgba(0,0,0,0.2);
        margin-bottom: 16px;
      }
      
      /* Suggestions */
      .suggestions-title {
        font-size: 12px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 10px;
      }
      .suggestion {
        background: rgba(0,0,0,0.2);
        border-radius: 8px;
        margin-bottom: 8px;
        overflow: hidden;
      }
      .suggestion-header {
        padding: 10px 12px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }
      .suggestion-header:hover {
        background: rgba(255,255,255,0.03);
      }
      .suggestion-title {
        font-weight: 500;
        font-size: 13px;
      }
      .suggestion-arrow {
        color: var(--muted);
        transition: transform 0.2s;
      }
      .suggestion.open .suggestion-arrow {
        transform: rotate(90deg);
      }
      .suggestion-body {
        display: none;
        padding: 0 12px 12px;
        font-size: 13px;
        color: var(--muted);
        line-height: 1.5;
      }
      .suggestion.open .suggestion-body {
        display: block;
      }
      .suggestion-why {
        font-size: 11px;
        color: var(--muted);
        opacity: 0.7;
        margin-top: 8px;
        padding-top: 8px;
        border-top: 1px solid var(--border);
      }
      
      /* Generated text */
      .generated-text {
        font-size: 14px;
        line-height: 1.7;
        white-space: pre-wrap;
      }
      
      /* Rewrite output */
      .rewrite-item {
        background: rgba(0,0,0,0.2);
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
      }
      .rewrite-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
      }
      .rewrite-score {
        font-size: 12px;
        color: var(--accent);
        font-weight: 600;
      }
      .rewrite-text {
        font-size: 14px;
        line-height: 1.6;
        white-space: pre-wrap;
      }
      
      /* Settings */
      .settings-toggle {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 10px 0;
        color: var(--muted);
        font-size: 12px;
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
      .settings-row {
        display: flex;
        gap: 10px;
      }
      .settings-row > div { flex: 1; }
      
      /* Status */
      .status {
        text-align: center;
        padding: 10px;
        color: var(--muted);
        font-size: 13px;
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
        <button class="tab" data-tab="match">Match</button>
      </div>
      
      <!-- Score Panel -->
      <div id="panel-score" class="panel active">
        <div class="card">
          <textarea id="score-text" placeholder="Paste your text here...">At dawn, the city leans into light. A gull lifts, then drops, then lifts again. The harbor breathes salt and diesel.</textarea>
        </div>
        
        <button class="btn" id="score-btn">Analyze</button>
        
        <div id="score-result" class="result">
          <div class="score-header">
            <div class="score-circle">
              <span class="score-number" id="score-value">--</span>
            </div>
            <div class="score-summary" id="score-summary">Analyzing your text...</div>
          </div>
          
          <div class="text-legend">
            <div class="legend-item">
              <div class="legend-dot"></div>
              <span>High-impact words (spikes)</span>
            </div>
          </div>
          <div class="text-display" id="text-highlighted"></div>
          
          <canvas id="cadence-chart" width="608" height="60"></canvas>
          
          <div class="metrics" id="score-metrics"></div>
          
          <div id="score-suggestions"></div>
        </div>
        
        <div class="status" id="score-status"></div>
        
        <button class="settings-toggle" id="score-settings-toggle">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="9 18 15 12 9 6"></polyline>
          </svg>
          Settings
        </button>
        <div class="settings-content" id="score-settings">
          <div class="settings-row">
            <div>
              <label>Type</label>
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
          <textarea id="rewrite-text" placeholder="Paste text to rewrite with better cadence..."></textarea>
        </div>
        
        <button class="btn" id="rewrite-btn">Rewrite</button>
        
        <div id="rewrite-result" class="result">
          <div id="rewrite-output"></div>
        </div>
        
        <div class="status" id="rewrite-status"></div>
        
        <button class="settings-toggle" id="rewrite-settings-toggle">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="9 18 15 12 9 6"></polyline>
          </svg>
          Settings
        </button>
        <div class="settings-content" id="rewrite-settings">
          <div class="settings-row">
            <div>
              <label>Candidates</label>
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
          <label>Reference (cadence to match)</label>
          <textarea id="match-reference" style="min-height:100px;">At dawn, the city leans into light. A gull lifts, then drops, then lifts again.</textarea>
        </div>
        
        <div class="card">
          <label>Your starting prompt</label>
          <textarea id="match-prompt" style="min-height:60px;">The morning light crept through the window</textarea>
        </div>
        
        <button class="btn" id="match-btn">Generate</button>
        
        <div id="match-result" class="result">
          <div class="generated-text" id="match-output"></div>
        </div>
        
        <div class="status" id="match-status"></div>
        
        <button class="settings-toggle" id="match-settings-toggle">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="9 18 15 12 9 6"></polyline>
          </svg>
          Settings
        </button>
        <div class="settings-content" id="match-settings">
          <div class="settings-row">
            <div>
              <label>Max tokens</label>
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
        
        // Sort by char_start
        const sorted = [...spikes].sort((a, b) => a.char_start - b.char_start);
        let result = '';
        let lastEnd = 0;
        
        for (const spike of sorted) {
          const start = spike.char_start;
          const end = spike.char_end;
          if (start < lastEnd) continue; // overlapping
          
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
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (!series || series.length < 2) return;
        
        let min = Math.min(...series), max = Math.max(...series);
        if (max - min < 0.1) { min = 0; max = 10; }
        
        const pad = 8, w = canvas.width - pad*2, h = canvas.height - pad*2;
        const xAt = i => pad + (i / (series.length - 1)) * w;
        const yAt = v => pad + (1 - (v - min) / (max - min)) * h;
        
        if (threshold) {
          ctx.strokeStyle = 'rgba(239,68,68,0.3)';
          ctx.setLineDash([3,3]);
          ctx.beginPath();
          ctx.moveTo(pad, yAt(threshold));
          ctx.lineTo(pad + w, yAt(threshold));
          ctx.stroke();
          ctx.setLineDash([]);
        }
        
        ctx.strokeStyle = '#6366f1';
        ctx.lineWidth = 1.5;
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
            $('score-suggestions').innerHTML = `
              <div class="suggestions-title">Suggestions</div>
              ${suggestions.slice(0, 4).map(s => `
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
              `).join('')}
            `;
            setupSuggestionToggles();
          } else {
            $('score-suggestions').innerHTML = '';
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
        status.textContent = 'Rewriting... (may take a minute)';
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
                <span style="font-size:12px;color:var(--muted);">Version ${i + 1}</span>
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
    </script>
  </body>
</html>
"""
