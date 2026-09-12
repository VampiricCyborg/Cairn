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
---

## What happens

Same as the valid fixture, but the required `confidence` field has been
dropped from the frontmatter to exercise the negative path.
