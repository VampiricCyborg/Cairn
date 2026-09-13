# Gold fixtures for `cairn eval`

Each `session_trace_N.json` is a synthetic, normalized `SessionTrace`. Its
`gold_N.json` lists the entries a senior engineer would have written after
reading that session, as Entry-shaped dicts (`title`, `type`, `scope`, `tags`,
`confidence`, `body`). SPEC.md's seven quality criteria are the bar each gold
entry was written to meet. `cairn eval` pairs files by the `N` suffix.

The three fixtures come from three unrelated projects, so matches can't come
from shared vocabulary:

| N | Project | Kind | What the session shows |
|---|---|---|---|
| 1 | `ledgerline` (Python, pytest, SQLAlchemy) | gotcha | Setting `APP_ENV` in a fixture is too late because `app/db.py` builds its engine at import time, during collection. The error is a misleading Postgres auth failure, and CI hides it. |
| 2 | `tidepool` (Python, pytest-xdist) | strategy | Stream `tests/fixtures/large/*.ndjson.gz` through `iter_ndjson` instead of loading it into memory. Slicing the file is the tempting fix, and it loses the coverage the test exists for. |
| 3 | `harbor` (pnpm/TypeScript, vitest) | fact | The integration-test Redis is on host port 6380. |

## Fixture 3 is a precision trap

Trace 3 has two salient spans, and only one of them is a lesson:

- Turns 1-3: `ECONNREFUSED 127.0.0.1:6379`, which leads to the 6380 fact. This
  span has a gold entry.
- Turns 6-8: `ECONNREFUSED 127.0.0.1:6380`. This one is operator error. The
  agent ran `docker compose ... down -v` itself and never brought the stack
  back up. It looks like an infrastructure problem, but there is nothing to
  learn from it, so it has **no** gold entry. Between the spans, turns 4-5 are
  an uneventful stretch.

So `gold_3.json` has fewer entries than the trace has salient spans. A
candidate proposed for the red herring doesn't match any gold entry, which
means it counts against precision. It doesn't affect recall, because recall
only counts gold entries.
