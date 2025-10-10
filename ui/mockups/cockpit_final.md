# Cockpit Final Mockup Notes

## Layout Highlights
- **Primary Header**: displays run name, phase counter, and contract version badge.
- **Telemetry Strip**: sparkline of perplexity + accuracy with status chips (`continue|warn|stop`).
- **Rollback Control**: dominant red button centred beneath telemetry strip reading `ROLLBACK TO SAFE PHASE`. Includes confirmation modal reminding Daniel of the stored checkpoint path.
- **Approve / Abort**: secondary buttons flanking the rollback control using neutral and amber colours respectively.
- **Alert Subtitle**: mirrors the canonical sentence used for voice narration; suppressed when voice channel is active to respect the cognitive throttle.
- **Schema Badge**: small chip showing the currently loaded cockpit schema version (from `specs/cockpit_schema.json`).

## Cognitive-Load Throttle Integration
- Voice alerts take precedence. When active, visual alerts dim and telegram feed pauses.
- Once voice completes, visual banner resumes with same message and timestamp.
- Telegram log remains accessible via collapsible panel for historical review.

## Pending Assets
- Voiceover persona selection (TBD).
- Contract modal flow wireframes showing review → accept → summary.
- Animated transition demonstrating rollback confirmation.
- Visual treatment for profile recommendations surfaced from the hardware-aware advisor.
