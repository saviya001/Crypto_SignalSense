# Reflection & Self-Critique Verification Checklist

## Pre-Signal Reflection Checklist
Before outputting any final signal, the SignalAgent must perform a Reflection pass verifying:
1. **Contradiction Check:** Does the technical indicator trend match the proposed signal direction?
2. **Risk-to-Reward Verification:** Is $(TP - Entry) / (Entry - SL) \ge 2.0$?
3. **Sentiment Alignment:** Does extreme negative news conflict with a Buy signal?
4. **RAG Grounding:** Does the rationale quote or reference at least one verified strategy document from the knowledge base?

If any check fails, flag the issue in `reflection_notes` and adjust confidence or action to HOLD.
