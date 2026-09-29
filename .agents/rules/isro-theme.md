# ISRO PS 26172 — Project Rules

## Theme & Branding
- Use official ISRO brand colors: **Metallic Orange (#F47216)** and **Marine Blue (#0E88D3)**.
- Dark background palette: deep navy (`#04080f`) with subtle blue radial glows.
- Never use generic cyan/teal. Use ISRO Marine Blue for all accent elements.
- Orange is used for active/streaming states and the ISRO emblem.
- Green (#22c55e) only for "passes constraint" badges. Amber for warnings. Red for errors.

## Typography
- Sans-serif body: Inter. Monospace data: JetBrains Mono.
- All telemetry labels are uppercase with letter-spacing.

## Dashboard Design
- Keep the interface minimal and mission-control focused.
- No unnecessary modals, popups, or configuration panels visible by default.
- Every metric displayed must map to an ISRO PS 26172 constraint (SRAM < 256 KB, CPU < 10%, Latency < 15 ms, 0 syllable loss).
- The oscilloscope waveform color is ISRO Marine Blue (#0E88D3).

## Code Style
- Python: asyncio-based, PEP 8 compliant, descriptive docstrings.
- HTML/JS: Vanilla only, no frameworks. Inline CSS in single-file dashboards.
- All WebSocket URLs must be dynamically resolved (support localhost, cloud deploy, Vercel static).
