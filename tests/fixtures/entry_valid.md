---
id: gotcha-7b21c4
type: gotcha
title: Alembic migrations must run before the test fixtures import
status: approved
spec_version: 0.1.0
scope:
  - "tests/**"
  - "alembic/**"
tags: [testing, database, ci]
confidence: high
evidence:
  harness: claude-code
  session_id: 0f3c1a9e-6b2d-4f77-9a10-2d4c8e51b3aa
  captured_at: 2026-09-12T11:04:02Z
  artifacts:
    - tests/conftest.py
    - alembic/env.py
  commit: 9d3f1ab
  excerpt_sha256: 4f1a...c9e2
created: 2026-09-12T11:06:44Z
updated: 2026-09-12T11:06:44Z
supersedes: []
review:
  approved_by: madhav
  approved_at: 2026-09-12T18:20:10Z
usage:
  injected: 12
  marked_useful: 3
  marked_stale: 0
---

## What happens

`pytest` collects `tests/conftest.py` before the Alembic upgrade runs, so the
fixture import sees the pre-migration schema and fails with a bare
`UndefinedColumn` that names no table.

## What to do

Run `alembic upgrade head` in the session fixture, not in the CI step. The
ordering is enforced in `tests/conftest.py::_migrate`.

## Why the obvious fix does not work

Moving the upgrade into the CI workflow fixes CI and leaves local runs broken,
which is how this was mis-diagnosed twice.
