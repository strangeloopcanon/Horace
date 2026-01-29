# Deploy Horace for External Use

> **Status**: DEPLOYED
> **Live URL**: https://marquisdepolis--horace-studio-fastapi-app.modal.run

Horace Studio is deployed and publicly accessible. This document captures the deployment steps and configuration.

## What Users Get

- `/analyze` — Score text (0-100) with detailed breakdown and suggestions
- `/rewrite` — Generate and rerank rewrites by Horace metrics
- `/cadence-match` — Generate text matching a reference passage's rhythm

The embedded HTML frontend provides a dark-themed UI with score visualization, metrics display, and interactive controls.

## Architecture

```
Browser → Modal (HTTPS) → GPU Container (GPT-2) → Score + Metrics
                       ↘ Baselines (Gutenberg distributions)
```

## Deployment Steps

### 1. Build Baseline Files

The scoring system compares text against "top literature" distributions. These must exist before deployment.

```bash
make setup
make build-baseline-web MODEL=gpt2
```

Creates `data/baselines/gpt2_gutenberg_512_docs.json`.

### 2. Deploy to Modal

```bash
make setup-modal
make modal-token          # Authenticate (one-time)
.venv/bin/modal deploy deploy/modal/horace_studio.py
```

Modal returns a public URL like `https://horace-studio--fastapi-app.modal.run/`

### 3. Smoke Test

```bash
curl -s https://YOUR_MODAL_URL/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text":"At dawn, the city leans into light.","doc_type":"prose"}'
```

Visit `https://YOUR_MODAL_URL/` in a browser to use the web UI.

## Cost Estimate

Modal pricing (pay-per-use):
- GPU (T4): ~$1/hour of actual compute
- Cold start: ~10-30 seconds on first request, then containers stay warm

For moderate traffic (50-100 analyses/day), expect $5-20/month.

## Optional Enhancements

### Add Rate Limiting

Add to `deploy/modal/horace_studio.py`:

```python
from fastapi import Request
from collections import defaultdict
import time

rate_limits = defaultdict(list)
RATE_LIMIT = 20  # requests per minute

@web.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    ip = request.client.host
    now = time.time()
    rate_limits[ip] = [t for t in rate_limits[ip] if now - t < 60]
    if len(rate_limits[ip]) >= RATE_LIMIT:
        return JSONResponse({"error": "Rate limit exceeded"}, status_code=429)
    rate_limits[ip].append(now)
    return await call_next(request)
```

### Add Custom Domain

1. In Modal dashboard, go to app settings
2. Add custom domain (e.g., `horace.yourdomain.com`)
3. Configure DNS CNAME to point to Modal's domain

### Add Cadence Match Tab to Frontend

The current embedded HTML lacks the Cadence Match feature. Update `tools/studio/site.py` to add a second tab with prompt + reference text inputs.

## Files Involved

| File | Purpose |
|------|---------|
| `data/baselines/` | Generated baseline JSON |
| `deploy/modal/horace_studio.py` | Modal deployment (ready as-is) |
| `tools/studio/site.py` | Embedded HTML frontend |
