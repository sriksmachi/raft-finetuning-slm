# Presentation

Reveal.js deck summarizing the RAFT fine-tuning solution.

- **Edit content:** [slides.md](slides.md) — slides are separated by `---` on its own line.
- **View locally:** must be served over HTTP (browsers block `fetch` on `file://`).

```powershell
cd presentation
python -m http.server 8000
# then open http://localhost:8000
```

No build step, no install — reveal.js is loaded from CDN.
