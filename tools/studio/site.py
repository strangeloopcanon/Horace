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
        --text-secondary: #9ca3af;
        --accent: #6366f1; /* Indigo - UI state */
        --accent-hover: #818cf8;
        --score: #10b981; /* Emerald - Data/Score */
        --score-dim: rgba(16, 185, 129, 0.15);
        --warning: #f59e0b;
        --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        --ease: cubic-bezier(0.23, 1, 0.32, 1);
      }
      * { box-sizing: border-box; margin: 0; padding: 0; }
      body {
        font-family: var(--sans);
        background: var(--bg);
        color: var(--text);
        min-height: 100vh;
        line-height: 1.5;
        overflow-y: scroll;
      }
      .container {
        max-width: 900px;
        margin: 0 auto;
        padding: 48px 24px 96px;
      }
      
      /* Typography */
      header {
        text-align: center;
        margin-bottom: 40px;
        animation: fade-in 0.6s var(--ease);
      }
      h1 {
        font-size: 32px;
        font-weight: 700;
        letter-spacing: -0.02em;
        background: linear-gradient(to right, #fff, #9ca3af);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
      }
      .tagline {
        color: var(--text-secondary);
        font-size: 16px;
        margin-top: 8px;
      }
      
      /* Animations */
      @keyframes fade-in {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
      }
      .fade-in { animation: fade-in 0.4s var(--ease) forwards; }
      
      /* Tabs */
      .tabs {
        display: flex;
        gap: 4px;
        background: rgba(255,255,255,0.03);
        padding: 4px;
        border-radius: 12px;
        margin-bottom: 32px;
        max-width: 420px;
        margin-left: auto;
        margin-right: auto;
        border: 1px solid var(--border);
      }
      .tab {
        flex: 1;
        padding: 10px 16px;
        border: none;
        background: transparent;
        color: var(--text-secondary);
        font-size: 14px;
        font-weight: 500;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s var(--ease);
      }
      .tab:hover { color: var(--text); background: rgba(255,255,255,0.05); }
      .tab.active {
        background: var(--accent);
        color: white;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
      }
      
      /* Panels */
      .panel { display: none; }
      .panel.active { display: block; animation: fade-in 0.3s var(--ease); }
      
      /* Cards & Inputs */
      .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
      }
      label {
        display: block;
        font-size: 13px;
        font-weight: 500;
        color: var(--text-secondary);
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
      }
      textarea {
        width: 100%;
        min-height: 180px;
        padding: 16px;
        background: rgba(0,0,0,0.2);
        border: 1px solid var(--border);
        border-radius: 12px;
        color: var(--text);
        font-family: var(--sans);
        font-size: 16px;
        line-height: 1.7;
        resize: vertical;
        outline: none;
        transition: border-color 0.2s;
      }
      textarea:focus { border-color: var(--accent); background: rgba(0,0,0,0.3); }
      textarea::placeholder { color: var(--text-secondary); opacity: 0.5; }
      
      input[type="number"], input[type="text"], select {
        width: 100%;
        padding: 12px;
        background: rgba(0,0,0,0.2);
        border: 1px solid var(--border);
        border-radius: 10px;
        color: var(--text);
        font-size: 14px;
        outline: none;
        transition: border-color 0.2s;
      }
      input:focus, select:focus { border-color: var(--accent); }
      
      /* Primary Action */
      .btn {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        padding: 16px;
        background: var(--accent);
        color: white;
        border: none;
        border-radius: 12px;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s var(--ease);
        box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.2);
      }
      .btn:hover { background: var(--accent-hover); transform: translateY(-1px); }
      .btn:active { transform: translateY(0); }
      .btn:disabled { opacity: 0.7; cursor: wait; transform: none; }
      
      /* Loading Skeleton */
      .skeleton {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
      }
      @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
      }
      
      /* Results Area */
      .result-area {
        margin-top: 40px;
        border-top: 1px solid var(--border);
        padding-top: 40px;
        display: none;
      }
      .result-area.visible { display: block; animation: fade-in 0.5s var(--ease); }
      
      /* Score Hero */
      .score-hero {
        display: flex;
        align-items: center;
        gap: 32px;
        margin-bottom: 40px;
        background: rgba(255,255,255,0.02);
        padding: 24px;
        border-radius: 20px;
        border: 1px solid var(--border);
      }
      .score-ring {
        position: relative;
        width: 100px;
        height: 100px;
        flex-shrink: 0;
      }
      .score-ring svg { transform: rotate(-90deg); width: 100%; height: 100%; }
      .score-ring circle {
        fill: none;
        stroke-width: 8;
        stroke-linecap: round;
      }
      .score-ring-bg { stroke: rgba(255,255,255,0.05); }
      .score-ring-val {
        stroke: var(--score);
        stroke-dasharray: 251; /* 2 * pi * 40 */
        stroke-dashoffset: 251;
        transition: stroke-dashoffset 1s var(--ease);
      }
      .score-value-text {
        position: absolute;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        font-size: 28px;
        font-weight: 700;
        color: var(--score);
      }
      .score-details { flex: 1; }
      .score-main-label { font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-secondary); margin-bottom: 8px; }
      .score-desc { font-size: 16px; line-height: 1.6; color: var(--text); }
      
      /* Section Headers */
      .section-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
        margin-top: 32px;
      }
      .section-title {
        font-size: 14px;
        font-weight: 700;
        color: var(--text);
        text-transform: uppercase;
        letter-spacing: 0.05em;
      }
      .copy-btn {
        background: transparent;
        border: 1px solid var(--border);
        color: var(--text-secondary);
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 12px;
        cursor: pointer;
        transition: all 0.2s;
      }
      .copy-btn:hover { border-color: var(--text-secondary); color: var(--text); }
      
      /* Text Display */
      .text-display {
        background: #0f1520;
        padding: 24px;
        border-radius: 12px;
        font-size: 16px;
        line-height: 1.8;
        white-space: pre-wrap;
        border: 1px solid var(--border);
      }
      .spike {
        background: rgba(251, 191, 36, 0.15);
        border-bottom: 2px solid rgba(251, 191, 36, 0.8);
        padding: 0 2px;
        border-radius: 2px;
        cursor: help;
        transition: background 0.2s;
      }
      .spike:hover { background: rgba(251, 191, 36, 0.3); }
      
      /* Metrics Grid */
      .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
      }
      .metric-card {
        background: rgba(255,255,255,0.03);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid transparent;
        transition: border-color 0.2s;
      }
      .metric-card:hover { border-color: var(--border); }
      .metric-val { font-size: 24px; font-weight: 700; color: var(--text); margin-bottom: 4px; }
      .metric-lbl { font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; }
      
      /* Chart */
      .chart-wrapper {
        background: rgba(255,255,255,0.02);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid var(--border);
      }
      canvas { width: 100%; height: 100px; display: block; }
      
      /* Suggestions */
      .suggestion {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        margin-bottom: 12px;
        overflow: hidden;
        border-left: 4px solid var(--accent);
        transition: background 0.2s;
      }
      .suggestion:hover { background: rgba(255,255,255,0.05); }
      .sugg-header {
        padding: 16px 20px;
        cursor: pointer;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-weight: 500;
        font-size: 15px;
      }
      .sugg-arrow { color: var(--text-secondary); transition: transform 0.3s var(--ease); }
      .suggestion.open .sugg-arrow { transform: rotate(180deg); }
      .sugg-body {
        display: none;
        padding: 0 20px 20px;
        font-size: 14px;
        color: var(--text-secondary);
        line-height: 1.6;
        animation: fade-in 0.3s;
      }
      .suggestion.open .sugg-body { display: block; }
      .sugg-why {
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px solid var(--border);
        font-size: 13px;
        font-style: italic;
        color: var(--text-secondary);
      }
      
      /* Settings */
      .settings-toggle {
        margin-top: 32px;
        display: flex;
        align-items: center;
        gap: 8px;
        color: var(--text-secondary);
        font-size: 13px;
        cursor: pointer;
        background: none;
        border: none;
        padding: 8px 0;
      }
      .settings-toggle:hover { color: var(--text); }
      .settings-toggle svg { transition: transform 0.2s; }
      .settings-toggle.open svg { transform: rotate(90deg); }
      .settings-panel { display: none; margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border); }
      .settings-panel.open { display: block; animation: fade-in 0.3s; }
      .settings-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
      
      /* Rewrites */
      .rewrite-card {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        border: 1px solid var(--border);
      }
      .rewrite-meta { display: flex; justify-content: space-between; margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid var(--border); font-size: 13px; }
      .rewrite-score { font-weight: 600; color: var(--score); }
      
      /* Mobile */
      @media (max-width: 640px) {
        .container { padding: 24px 16px 80px; }
        h1 { font-size: 26px; }
        .score-hero { flex-direction: column; text-align: center; gap: 16px; }
        .metrics-grid { grid-template-columns: 1fr 1fr; }
        .settings-grid { grid-template-columns: 1fr; }
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
          <textarea id="score-text" placeholder="Paste your text here (prose or poetry)...">At dawn, the city leans into light. A gull lifts, then drops, then lifts again. The harbor breathes salt and diesel.</textarea>
        </div>
        
        <button class="btn" id="score-btn">
          <span>Analyze Text</span>
        </button>
        
        <div id="score-result" class="result-area">
          <div class="score-hero">
            <div class="score-ring">
              <svg viewBox="0 0 100 100">
                <circle class="score-ring-bg" cx="50" cy="50" r="40"></circle>
                <circle class="score-ring-val" cx="50" cy="50" r="40" id="score-ring-val"></circle>
              </svg>
              <div class="score-value-text" id="score-value">--</div>
            </div>
            <div class="score-details">
              <div class="score-main-label">Analysis Complete</div>
              <div class="score-desc" id="score-summary"></div>
            </div>
          </div>
          
          <div class="section-header">
            <div class="section-title">Spike Analysis</div>
            <div style="font-size:12px; color:var(--text-secondary);">High surprisal words highlighted</div>
          </div>
          <div class="text-display" id="text-highlighted"></div>
          
          <div class="section-header">
            <div class="section-title">Cadence Curve</div>
          </div>
          <div class="chart-wrapper">
            <canvas id="cadence-chart"></canvas>
          </div>
          
          <div class="section-header">
            <div class="section-title">Metrics</div>
          </div>
          <div class="metrics-grid" id="score-metrics"></div>
          
          <div class="section-header">
            <div class="section-title">Suggestions</div>
          </div>
          <div id="score-suggestions"></div>
        </div>
        
        <button class="settings-toggle" id="score-settings-toggle">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="9 18 15 12 9 6"></polyline>
          </svg>
          Advanced settings
        </button>
        <div class="settings-panel" id="score-settings">
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
        
        <button class="btn" id="rewrite-btn">Generate Rewrites</button>
        
        <div id="rewrite-result" class="result-area">
          <div class="section-header">
            <div class="section-title">Rewritten Versions</div>
          </div>
          <div id="rewrite-output"></div>
        </div>
        
        <button class="settings-toggle" id="rewrite-settings-toggle">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="9 18 15 12 9 6"></polyline>
          </svg>
          Configuration
        </button>
        <div class="settings-panel" id="rewrite-settings">
          <div class="settings-grid">
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
          <label>Reference (Target Cadence)</label>
          <textarea id="match-reference" style="min-height:120px;">At dawn, the city leans into light. A gull lifts, then drops, then lifts again.</textarea>
        </div>
        
        <div class="card">
          <label>Your Starting Prompt</label>
          <textarea id="match-prompt" style="min-height:80px;">The morning light crept through the window</textarea>
        </div>
        
        <button class="btn" id="match-btn">Generate Match</button>
        
        <div id="match-result" class="result-area">
          <div class="section-header">
            <div class="section-title">Generated Text</div>
            <button class="copy-btn" onclick="copyText('match-output')">Copy</button>
          </div>
          <div class="text-display" id="match-output"></div>
        </div>
        
        <button class="settings-toggle" id="match-settings-toggle">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="9 18 15 12 9 6"></polyline>
          </svg>
          Advanced settings
        </button>
        <div class="settings-panel" id="match-settings">
          <div class="settings-grid">
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
          toggle.classList.toggle('open');
          toggle.nextElementSibling.classList.toggle('open');
        });
      });
      
      // Copy
      async function copyText(id) {
        const text = $(id).textContent;
        await navigator.clipboard.writeText(text);
        const btn = document.querySelector(`button[onclick="copyText('${id}')"]`);
        const original = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => btn.textContent = original, 2000);
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
      
      // Helpers
      const escapeHtml = str => str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      
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
        
        // Gradient
        const grad = ctx.createLinearGradient(0, 0, 0, rect.height);
        grad.addColorStop(0, 'rgba(99, 102, 241, 0.4)');
        grad.addColorStop(1, 'rgba(99, 102, 241, 0.0)');
        
        let min = Math.min(...series), max = Math.max(...series);
        if (max - min < 0.1) { min = 0; max = 10; }
        const range = max - min;
        
        const pad = 0;
        const w = rect.width;
        const h = rect.height;
        const xAt = i => (i / (series.length - 1)) * w;
        const yAt = v => h - ((v - min) / range) * h;
        
        // Path
        ctx.beginPath();
        ctx.moveTo(xAt(0), yAt(series[0]));
        for (let i = 1; i < series.length; i++) ctx.lineTo(xAt(i), yAt(series[i]));
        
        // Stroke
        ctx.strokeStyle = '#6366f1';
        ctx.lineWidth = 2;
        ctx.lineJoin = 'round';
        ctx.stroke();
        
        // Fill
        ctx.lineTo(w, h);
        ctx.lineTo(0, h);
        ctx.closePath();
        ctx.fillStyle = grad;
        ctx.fill();
        
        canvas._lastSeries = series;
      }
      
      function setScore(score) {
        const circle = $('score-ring-val');
        const radius = 40;
        const circumference = 2 * Math.PI * radius;
        const offset = circumference - (score / 100) * circumference;
        circle.style.strokeDashoffset = offset;
        $('score-value').textContent = Math.round(score);
        
        // Color transition based on score
        const color = score > 80 ? '#10b981' : score > 60 ? '#6366f1' : '#f59e0b';
        circle.style.stroke = color;
        $('score-value').style.color = color;
      }
      
      // Handlers
      $('score-btn').addEventListener('click', async () => {
        const btn = $('score-btn');
        const result = $('score-result');
        const originalText = btn.innerHTML;
        
        btn.disabled = true;
        btn.innerHTML = '<div class="skeleton" style="width:20px;height:20px;border-radius:50%;"></div>Analyzing...';
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
          
          // Data population
          setScore(data.score?.overall_0_100 || 0);
          $('score-summary').textContent = data.critique?.summary || 'Analysis complete.';
          $('text-highlighted').innerHTML = highlightText(text, data.analysis?.spikes || []);
          drawCadence(data.analysis?.series?.surprisal);
          
          // Metrics
          const cats = data.score?.categories || {};
          $('score-metrics').innerHTML = Object.entries(cats).map(([k, v]) => `
            <div class="metric-card">
              <div class="metric-val">${Math.round(v * 100)}</div>
              <div class="metric-lbl">${k}</div>
            </div>
          `).join('');
          
          // Suggestions
          const suggs = data.critique?.suggestions || [];
          $('score-suggestions').innerHTML = suggs.length ? suggs.slice(0, 5).map(s => `
            <div class="suggestion">
              <div class="sugg-header" onclick="this.parentElement.classList.toggle('open')">
                <span>${s.title}</span>
                <span class="sugg-arrow">▼</span>
              </div>
              <div class="sugg-body">
                <div>${s.what_to_try}</div>
                ${s.why ? `<div class="sugg-why">${s.why}</div>` : ''}
              </div>
            </div>
          `).join('') : '<p style="color:var(--text-secondary);font-style:italic;">No specific suggestions found.</p>';
          
        } catch (e) {
          alert('Error: ' + e.message);
        } finally {
          btn.disabled = false;
          btn.innerHTML = originalText;
        }
      });
      
      $('rewrite-btn').addEventListener('click', async () => {
        const btn = $('rewrite-btn');
        const result = $('rewrite-result');
        const originalText = btn.innerHTML;
        
        btn.disabled = true;
        btn.innerHTML = 'Rewriting...';
        result.classList.remove('visible');
        
        try {
          const data = await api('/rewrite', {
            text: $('rewrite-text').value,
            doc_type: $('doc-type').value,
            n_candidates: parseInt($('rewrite-candidates').value) || 4,
            keep_top: 3,
            temperature: parseFloat($('rewrite-temp').value) || 0.8,
            normalize_text: true
          });
          
          result.classList.add('visible');
          const rewrites = data.rewrites || [];
          
          $('rewrite-output').innerHTML = rewrites.length ? rewrites.map((r, i) => `
            <div class="rewrite-card">
              <div class="rewrite-meta">
                <span>Option ${i + 1}</span>
                <span class="rewrite-score">Score: ${Math.round(r.score)}</span>
              </div>
              <div style="white-space:pre-wrap;line-height:1.7;">${escapeHtml(r.text)}</div>
              <div style="margin-top:12px;text-align:right;">
                <button class="copy-btn" onclick="navigator.clipboard.writeText(this.parentElement.previousElementSibling.textContent).then(()=>alert('Copied!'))">Copy</button>
              </div>
            </div>
          `).join('') : '<p style="color:var(--text-secondary);">No rewrites generated.</p>';
          
        } catch (e) {
          alert(e.message);
        } finally {
          btn.disabled = false;
          btn.innerHTML = originalText;
        }
      });
      
      $('match-btn').addEventListener('click', async () => {
        const btn = $('match-btn');
        const result = $('match-result');
        const originalText = btn.innerHTML;
        
        btn.disabled = true;
        btn.innerHTML = 'Generating...';
        result.classList.remove('visible');
        
        try {
          const data = await api('/cadence-match', {
            prompt: $('match-prompt').value,
            reference_text: $('match-reference').value,
            max_new_tokens: parseInt($('match-tokens').value) || 200,
            seed: parseInt($('match-seed').value) || 7
          });
          
          result.classList.add('visible');
          $('match-output').textContent = data.generated_text || data.text || '';
          
        } catch (e) {
          alert(e.message);
        } finally {
          btn.disabled = false;
          btn.innerHTML = originalText;
        }
      });
      
      window.addEventListener('resize', () => {
        const c = $('cadence-chart');
        if(c && c._lastSeries) drawCadence(c._lastSeries);
      });
    </script>
  </body>
</html>
"""
