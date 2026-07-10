# Vex two-way sync

This folder is materialized into the Vex monorepo (`vex/home` under `jj/`) and
kept in sync with this GitHub repository in both directions:

- Commits landed on the Vex trunk that touch `jj/` are projected back here as
  regular git commits (authors and messages preserved).
- Commits pushed here are woven into the monorepo incrementally by the next
  materialization run.

This file was added from the Vex side as the first live round-trip test of the
push projection (roadmap/037 Stage 6).

Validated end to end on 2026-07-07: this line was committed to the jj/ subfolder of vex/main and carried to VEXscm/jj by the scheduled two-way sync, with no manual trigger.

That validation predates the canonical address change. `vex/main` remains a
permanent compatibility alias for the same aggregate repository now advertised
as `vex/home`.

Hook validation 14:19:21Z: this commit should reach VEXscm/jj within seconds via the event-driven sync hook.
