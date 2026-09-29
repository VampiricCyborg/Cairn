"""`python -m cairn` entrypoint.

Claude Code's SessionStart hook is registered as
`<sys.executable> -m cairn context --hook ...` rather than as a bare `cairn`,
because the `cairn` console script only exists on PATH after a
`uv tool install` / `pipx install`, while the interpreter that ran
`cairn install claude-code` is an absolute path that always resolves. This
module is what makes the `-m` form work.
"""

from cairn.cli import app

app()
