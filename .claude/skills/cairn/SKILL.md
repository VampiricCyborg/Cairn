---
description: Explains Cairn, the project's session-derived knowledge store, and how to use `cairn review` and `cairn context` alongside it.
---

# Cairn

This repository uses Cairn, a git-native memory layer: a `SessionEnd` hook
enqueues this session for capture, and the next session's `SessionStart`
hook injects whatever knowledge has been approved since. You do not need to
run anything yourself for that half of the loop.

## What `.cairn/` holds

- `.cairn/entries/{strategy,gotcha,fact}/` — approved, human-reviewed
  lessons about this repository. Committed to git, safe to read directly.
- `.cairn/staging/` — candidates awaiting a human decision. Not yet trusted;
  do not treat anything here as established fact.
- `.cairn/CONTEXT.md` — the rendered, token-budgeted context block built
  from approved entries. Regenerated automatically, not hand-edited.

See `SPEC.md` at the repository root for the full store format and entry
lifecycle.

## What you should do

- Read `.cairn/CONTEXT.md` (or the context injected at session start) before
  assuming you're rediscovering something for the first time — a past
  session's gotcha or strategy may already answer the question.
- You do not write to `.cairn/staging/` or `.cairn/entries/` directly. A
  separate reflect worker proposes candidates from session transcripts;
  approval happens through `cairn review`, run by a human.
- If you notice a candidate in `.cairn/staging/` that looks wrong while
  working, say so — the reviewer decides, but a heads-up before review saves
  a mistaken entry from ever being approved.

## Commands a human on this project might run

- `cairn review` — walk staged candidates one at a time: approve, edit,
  merge into an existing entry, reject with a reason, or skip.
- `cairn context --scope "<glob>"` — preview what would be injected for a
  given area of the repo.
- `cairn doctor` — check that the store, hooks, and provider are all wired
  up correctly.
