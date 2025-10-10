# Scaler Wizard
Scaler Wizard is the implementation hub for Daniel Ramos' Multi-Vibe Code In Chain initiative to extend TransformerLab with accessible model scaling workflows.

## Current Focus
- Expand Grok's initial plan into actionable architecture and backlog.
- Capture collaboration rituals for the AI partner swarm.
- Stage documentation that will guide the upcoming prototype sprint.

## Repository Map
- `docs/process/multi_vibe_chain.md` — collaboration paradigm and operating loop.
- `docs/architecture/scaler_wizard_overview.md` — expanded system blueprint and milestones.
- `docs/architecture/safety.md` — circuit breaker and telemetry guardrails.
- `docs/backlog/roadmap.md` — prioritized backlog for upcoming sprints.
- `specs/` — plugin interface, CLI prototype, and scaling engine contracts.
- `specs/cockpit_schema.json` — canonical payload for cockpit dashboards.
- `components/` — code scaffolding for scaling engine, evaluation, distillation, templates, safety, telemetry, caching, validation, and cognitive-load throttling.
- `tools/rollback.py` — script to restore the last safe checkpoint when the cockpit issues a rollback.
- `research/` — surveys comparing context extension and PEFT techniques.
- `ui/` — placeholder for cockpit wireframes and React components.
- `ui/services/voice_pipeline.js` — Web Voice queue used to narrate cockpit alerts.
- `benchmarks/` — datasets and results folders for evaluation.
- `community/` — future home for shared patterns and contributions.

As new partner insights arrive, Codex (this AI) will translate them into code modules, configs, and further documentation inside this structure.
