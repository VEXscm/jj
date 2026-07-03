# Vex two-way sync

This folder is materialized into the Vex monorepo (`vex/main` under `jj/`) and
kept in sync with this GitHub repository in both directions:

- Commits landed on the Vex trunk that touch `jj/` are projected back here as
  regular git commits (authors and messages preserved).
- Commits pushed here are woven into the monorepo incrementally by the next
  materialization run.

This file was added from the Vex side as the first live round-trip test of the
push projection (roadmap/037 Stage 6).
