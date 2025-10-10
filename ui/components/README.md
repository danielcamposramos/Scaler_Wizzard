# UI Components

React component stubs and shared hooks will live here once the TransformerLab integration begins. Track design decisions alongside implementation notes to keep the swarm aligned.

- Plan to expose a `RollbackButton` component matching `ui/mockups/cockpit_final.md`.
- Integrate with the cognitive-load throttle from `components/ui/cognitive_throttle.py` to arbitrate voice/visual/telegram alerts.
- Consume the voice queue contract from `ui/services/voice_pipeline.js` so narration status stays in sync with the throttle.
