
# Common JS to reuse across templates (injected via f-string)
COMMON_JS = """
<script>
  const $ = id => document.getElementById(id);
  const $$ = sel => document.querySelectorAll(sel);
  
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

  // Basic interaction logic for the Analyze button
  const btn = $('analyze-btn');
  if (btn) {
    btn.addEventListener('click', async () => {
        const originalText = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = 'Processing...';
        
        // Hide previous results if any
        const results = document.getElementById('results-area');
        if(results) results.style.opacity = '0.5';

        try {
            const text = $('input-text').value;
            const data = await api('/analyze', {
                text,
                doc_type: 'prose',
                max_input_tokens: 512,
                normalize_text: true
            });

            // Populate Score
            const scoreEl = $('score-display');
            if(scoreEl) scoreEl.innerText = Math.round(data.score?.overall_0_100 || 0);
            
            // Populate Summary
            const sumEl = $('summary-display');
            if(sumEl) sumEl.innerText = data.critique?.summary || '';

            // Populate Chart (Simple render)
            const canvas = $('cadence-chart');
            if(canvas && data.analysis?.series?.surprisal) {
                drawCadence(canvas, data.analysis.series.surprisal);
            }
            
            // Populate Highlights (Basic)
            const hlEl = $('highlight-display');
            if(hlEl) {
                hlEl.innerHTML = highlightText(text, data.analysis?.spikes || []);
            }

            if(results) {
                results.style.opacity = '1';
                results.style.display = 'block'; // Ensure visible
                // Scroll to results if needed
                results.scrollIntoView({behavior: 'smooth', block: 'start'});
            }

        } catch(e) {
            alert(e.message);
        } finally {
            btn.disabled = false;
            btn.innerHTML = originalText;
        }
    });
  }

  function highlightText(text, spikes) {
    if (!spikes || !spikes.length) return text;
    const sorted = [...spikes].sort((a, b) => a.char_start - b.char_start);
    let res = '';
    let last = 0;
    for (const s of sorted) {
        if (s.char_start < last) continue;
        res += text.slice(last, s.char_start);
        res += `<span class="spike">${text.slice(s.char_start, s.char_end)}</span>`;
        last = s.char_end;
    }
    res += text.slice(last);
    return res;
  }

  function drawCadence(canvas, series) {
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const w = rect.width, h = rect.height;
    
    ctx.clearRect(0, 0, w, h);
    ctx.beginPath();
    
    let min = Math.min(...series), max = Math.max(...series);
    if(max-min<0.1) {min=0;max=10;}
    
    const xAt = i => (i/(series.length-1)) * w;
    const yAt = v => h - ((v-min)/(max-min))*h;
    
    ctx.moveTo(xAt(0), yAt(series[0]));
    for(let i=1; i<series.length; i++) ctx.lineTo(xAt(i), yAt(series[i]));
    
    // Style depends on CSS var or hardcoded per theme
    const style = getComputedStyle(document.body);
    ctx.strokeStyle = style.getPropertyValue('--chart-color') || '#000';
    ctx.lineWidth = 2;
    ctx.stroke();
  }
</script>
"""

# 1. THE CONSOLE (Teenage Engineering / Ableton)
HTML_CONSOLE = f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Horace / Console</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;700&family=Inter:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #111;
            --panel: #1a1a1a;
            --border: #333;
            --text: #aaa;
            --text-bright: #fff;
            --accent: #ff3e00; /* Orange/Red technical */
            --chart-color: #ff3e00;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif;
            margin: 0; height: 100vh; overflow: hidden;
            display: grid; grid-template-columns: 300px 1fr 300px;
        }}
        /* Left Sidebar: Controls */
        aside {{
            border-right: 1px solid var(--border);
            padding: 20px;
            display: flex; flex-direction: column; gap: 20px;
        }}
        .brand {{ font-family: 'Roboto Mono'; font-weight: 700; color: var(--text-bright); font-size: 18px; letter-spacing: -1px; }}
        .control-group label {{ display: block; font-family: 'Roboto Mono'; font-size: 10px; text-transform: uppercase; margin-bottom: 8px; }}
        .knob {{ width: 40px; height: 40px; border-radius: 50%; border: 2px solid var(--border); display:inline-block; margin-right: 10px; }}
        
        /* Middle: Editor */
        main {{
            display: flex; flex-direction: column;
            border-right: 1px solid var(--border);
        }}
        .toolbar {{ height: 50px; border-bottom: 1px solid var(--border); display: flex; align-items: center; padding: 0 20px; font-family: 'Roboto Mono'; font-size: 12px; }}
        textarea {{
            flex: 1; background: var(--bg); border: none; padding: 40px; color: var(--text-bright);
            font-family: 'Roboto Mono', monospace; font-size: 16px; line-height: 1.8; outline: none; resize: none;
        }}
        
        /* Right: Analysis */
        .readout {{
            padding: 20px;
            display: flex; flex-direction: column; gap: 20px;
            overflow-y: auto;
        }}
        .meter {{ background: var(--panel); border: 1px solid var(--border); padding: 10px; border-radius: 4px; }}
        .meter h4 {{ margin: 0 0 10px 0; font-family: 'Roboto Mono'; font-size: 10px; text-transform: uppercase; }}
        .score-big {{ font-size: 48px; font-weight: 700; color: var(--text-bright); font-family: 'Roboto Mono'; line-height: 1; }}
        canvas {{ width: 100%; height: 60px; background: #000; }}
        
        button {{
            background: var(--text-bright); color: var(--bg); border: none;
            font-family: 'Roboto Mono'; text-transform: uppercase; font-weight: 700;
            padding: 12px; cursor: pointer; margin-top: auto;
        }}
        button:hover {{ background: var(--accent); color: #fff; }}

        .spike {{ color: var(--accent); border-bottom: 1px dotted var(--accent); }}
        
        @media (max-width: 900px) {{
            body {{ grid-template-columns: 1fr; overflow-y: scroll; height: auto; }}
            aside, main, .readout {{ border-right: none; border-bottom: 1px solid var(--border); height: auto; }}
            textarea {{ min-height: 300px; }}
        }}
    </style>
</head>
<body>
    <aside>
        <div class="brand">HORACE // STUDIO</div>
        <div class="control-group">
            <label>Mode</label>
            <select style="background:var(--bg); color:var(--text); border:1px solid var(--border); padding:8px; width:100%;">
                <option>Prose Analysis</option>
                <option>Poetry Scansion</option>
            </select>
        </div>
        <div class="control-group">
            <label>Parameters</label>
            <div style="display:flex; align-items:center;">
                <div class="knob"></div>
                <div class="knob"></div>
                <div class="knob"></div>
            </div>
        </div>
        <button id="analyze-btn">RUN DIAGNOSTICS</button>
    </aside>
    
    <main>
        <div class="toolbar">
            <span>UNTITLED_DRAFT_01.TXT</span>
            <span style="margin-left:auto;">Ln 1, Col 1</span>
        </div>
        <textarea id="input-text">At dawn, the city leans into light. A gull lifts, then drops, then lifts again. The harbor breathes salt and diesel.</textarea>
    </main>
    
    <div class="readout" id="results-area">
        <div class="meter">
            <h4>Global Score</h4>
            <div class="score-big" id="score-display">--</div>
        </div>
        <div class="meter">
            <h4>Rhythm Topology</h4>
            <canvas id="cadence-chart"></canvas>
        </div>
        <div class="meter">
            <h4>System Log</h4>
            <div id="summary-display" style="font-size:12px; line-height:1.5;">Ready to analyze.</div>
        </div>
        <div class="meter">
            <h4>Highlights</h4>
            <div id="highlight-display" style="font-size:12px; font-family:'Roboto Mono';"></div>
        </div>
    </div>
    {COMMON_JS}
</body>
</html>
"""

# 2. THE SWISS GALLEY (International Typographic Style)
HTML_SWISS = f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Horace / Swiss</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #fff;
            --text: #000;
            --line: #000;
            --accent: #ff4400; /* International Orange */
            --chart-color: #000;
        }}
        body {{
            background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif;
            margin: 0; padding: 40px;
        }}
        h1 {{
            font-size: 120px; line-height: 0.8; letter-spacing: -6px; margin: 0 0 40px 0;
            font-weight: 900;
        }}
        .grid {{
            display: grid; grid-template-columns: 2fr 1fr; gap: 40px;
            border-top: 4px solid var(--line);
            padding-top: 20px;
        }}
        textarea {{
            width: 100%; min-height: 400px; border: none; 
            font-size: 32px; font-weight: 700; line-height: 1.2; letter-spacing: -0.5px;
            outline: none; resize: vertical; font-family: 'Inter', sans-serif;
        }}
        textarea::placeholder {{ color: #ddd; }}
        
        .meta {{ border-left: 1px solid #ccc; padding-left: 40px; display: flex; flex-direction: column; gap: 40px; }}
        
        .label {{ font-size: 12px; font-weight: 700; text-transform: uppercase; margin-bottom: 10px; display:block; }}
        
        .score-wrap {{ font-size: 80px; font-weight: 900; letter-spacing: -4px; line-height: 1; }}
        
        button {{
            background: var(--text); color: var(--bg); border: none;
            padding: 20px 40px; font-size: 16px; font-weight: 700; cursor: pointer;
            width: 100%; text-align: left;
        }}
        button:hover {{ background: var(--accent); }}
        
        .spike {{ background: var(--accent); color: white; padding: 0 4px; }}
        
        canvas {{ width: 100%; height: 100px; border: 1px solid #000; }}
        
        @media (max-width: 800px) {{
            h1 {{ font-size: 60px; letter-spacing: -3px; }}
            .grid {{ grid-template-columns: 1fr; }}
            .meta {{ border-left: none; padding-left: 0; border-top: 1px solid #ccc; padding-top: 40px; }}
        }}
    </style>
</head>
<body>
    <h1>HORACE.</h1>
    
    <div class="grid">
        <div class="col-main">
            <span class="label">Input Source</span>
            <textarea id="input-text">At dawn, the city leans into light. A gull lifts, then drops, then lifts again. The harbor breathes salt and diesel.</textarea>
            
            <div id="results-area" style="margin-top: 60px; display:none;">
                <span class="label">Analysis Result</span>
                <div id="highlight-display" style="font-size: 24px; line-height: 1.4;"></div>
            </div>
        </div>
        
        <div class="col-meta meta">
            <div>
                <button id="analyze-btn">Analyze Text &rarr;</button>
            </div>
            
            <div>
                <span class="label">Rhythmic Score</span>
                <div class="score-wrap" id="score-display">00</div>
            </div>
            
            <div>
                <span class="label">Cadence</span>
                <canvas id="cadence-chart"></canvas>
            </div>
            
            <div>
                <span class="label">Critique</span>
                <div id="summary-display" style="font-size: 14px; line-height: 1.4;">Awaiting input.</div>
            </div>
        </div>
    </div>
    {COMMON_JS}
</body>
</html>
"""

# 3. THE AUGMENTED PAGE (Marginalia / Modern Paper)
HTML_PAPER = f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Horace / Paper</title>
    <link href="https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;1,400&family=Lato:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #fdfbf7;
            --text: #2c2c2c;
            --ink: #2c2c2c;
            --highlight: #fef08a; /* Yellow highlighter */
            --margin-text: #666;
            --chart-color: #555;
        }}
        body {{
            background: var(--bg); color: var(--text); 
            font-family: 'Crimson Pro', serif;
            margin: 0; min-height: 100vh;
        }}
        
        .layout {{
            max-width: 1200px; margin: 0 auto;
            display: grid; grid-template-columns: 1fr 680px 1fr;
            padding: 60px 20px;
        }}
        
        /* Main Text Column */
        .page {{
            grid-column: 2;
        }}
        
        h1 {{ font-family: 'Crimson Pro'; font-weight: 400; font-style: italic; font-size: 36px; text-align: center; margin-bottom: 60px; }}
        
        textarea {{
            width: 100%; min-height: 300px; border: none; background: transparent;
            font-family: 'Crimson Pro', serif; font-size: 22px; line-height: 1.6;
            color: var(--ink); outline: none; resize: vertical;
        }}
        
        .controls {{ text-align: center; margin: 40px 0; padding: 20px 0; border-top: 1px solid #e5e5e5; border-bottom: 1px solid #e5e5e5; }}
        button {{
            background: transparent; border: 1px solid var(--ink); color: var(--ink);
            padding: 10px 24px; font-family: 'Lato', sans-serif; text-transform: uppercase; letter-spacing: 1px;
            font-size: 11px; cursor: pointer; transition: all 0.2s;
        }}
        button:hover {{ background: var(--ink); color: var(--bg); }}
        
        /* Margins/Sidenotes */
        .margin-right {{
            grid-column: 3;
            padding-left: 40px;
            font-family: 'Lato', sans-serif; font-size: 13px; color: var(--margin-text);
        }}
        
        .score-card {{
            background: white; padding: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            border-radius: 4px; margin-bottom: 20px;
            display: none; /* hidden until result */
        }}
        
        .big-score {{ font-family: 'Crimson Pro'; font-size: 48px; display: block; margin-bottom: 10px; }}
        
        canvas {{ width: 100%; height: 60px; margin-top: 10px; }}
        
        .spike {{ background: linear-gradient(to bottom, transparent 60%, var(--highlight) 0); padding: 0 2px; }}
        
        @media (max-width: 1000px) {{
            .layout {{ grid-template-columns: 1fr; max-width: 600px; }}
            .page {{ grid-column: 1; }}
            .margin-right {{ grid-column: 1; padding-left: 0; margin-top: 40px; }}
        }}
    </style>
</head>
<body>
    <div class="layout">
        <div class="page">
            <h1>Horace</h1>
            
            <textarea id="input-text" placeholder="Start writing...">At dawn, the city leans into light. A gull lifts, then drops, then lifts again. The harbor breathes salt and diesel.</textarea>
            
            <div id="highlight-display" style="font-size: 22px; line-height: 1.6; display:none; margin-bottom: 40px;"></div>
            
            <div class="controls">
                <button id="analyze-btn">Analyze Draft</button>
            </div>
        </div>
        
        <div class="margin-right" id="results-area">
            <div class="score-card" style="display:block; opacity:0.5;">
                <span style="text-transform:uppercase; letter-spacing:1px; font-size:10px;">Analysis</span>
                <div style="margin-top:10px;">
                    <span class="big-score" id="score-display">--</span>
                    <canvas id="cadence-chart"></canvas>
                </div>
                <p id="summary-display" style="margin-top:20px; line-height:1.5;">
                    Results will appear here in the margin, preserving your reading flow.
                </p>
            </div>
        </div>
    </div>
    
    {COMMON_JS}
    <script>
        // Custom logic for Paper view to swap textarea with highlight div
        const superBtn = $('analyze-btn');
        superBtn.addEventListener('click', () => {{
            setTimeout(() => {{
                $('input-text').style.display = 'none';
                $('highlight-display').style.display = 'block';
                $('results-area').querySelector('.score-card').style.opacity = '1';
            }}, 500); // fake wait for api
        }});
        
        // Reset if they click the text
        $('highlight-display').addEventListener('click', () => {{
            $('highlight-display').style.display = 'none';
            $('input-text').style.display = 'block';
            $('input-text').focus();
        }});
    </script>
</body>
</html>
"""

# 4. INDEX PAGE
HTML_INDEX = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Horace / Design Prototypes</title>
    <style>
        body { background: #eee; font-family: sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; }
        .menu { background: white; padding: 40px; border-radius: 8px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); max-width: 400px; width: 100%; }
        h1 { margin-top: 0; font-size: 20px; margin-bottom: 20px; }
        a { display: block; padding: 15px; background: #f5f5f5; margin-bottom: 10px; text-decoration: none; color: #333; border-radius: 4px; border: 1px solid #ddd; transition: all 0.2s; }
        a:hover { background: #333; color: white; border-color: #333; }
        small { color: #666; display: block; margin-top: 4px; font-size: 12px; }
    </style>
</head>
<body>
    <div class="menu">
        <h1>Select a Design Concept</h1>
        <a href="/console">
            <strong>Concept 1: The Console</strong>
            <small>Dark, technical, modular (Teenage Engineering)</small>
        </a>
        <a href="/swiss">
            <strong>Concept 2: Swiss Galley</strong>
            <small>Stark, typographic, bold (Helvetica)</small>
        </a>
        <a href="/paper">
            <strong>Concept 3: Augmented Paper</strong>
            <small>Flow-focused, marginalia, classic (Editorial)</small>
        </a>
        <a href="/main">
            <strong>Current: Obsidian Study</strong>
            <small>The active redesign</small>
        </a>
    </div>
</body>
</html>
"""
