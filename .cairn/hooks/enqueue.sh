#!/usr/bin/env bash
# .cairn/hooks/enqueue.sh — Claude Code SessionEnd capture hook.
#
# Reads the SessionEnd event JSON from stdin (session_id, transcript_path,
# cwd) and writes a small job record to .cairn/queue/<session_id>.json for
# the SessionStart sweep to pick up later. Per the README's hook-safety
# principle, this must never be able to block or fail a session end: no
# `set -e`, stdin is read with a builtin (not external `cat`) so the very
# first check below needs nothing but bash itself, and every failure path
# logs to stderr and falls through to the same unconditional `exit 0` at
# the bottom rather than propagating a non-zero exit.

payload=""
# `|| [ -n "$line" ]` matters: `read` fails at EOF, so a plain `while read`
# silently drops the final line when stdin has no trailing newline -- which
# is exactly how a single-line JSON payload with no trailing `\n` arrives.
while IFS= read -r line || [ -n "$line" ]; do
  payload+="$line"$'\n'
done

if ! command -v jq >/dev/null 2>&1; then
  echo "cairn enqueue.sh: jq not found; skipping capture" >&2
  exit 0
fi

session_id="$(printf '%s' "$payload" | jq -r '.session_id // empty' 2>/dev/null)"
transcript="$(printf '%s' "$payload" | jq -r '.transcript_path // empty' 2>/dev/null)"
cwd="$(printf '%s' "$payload" | jq -r '.cwd // empty' 2>/dev/null)"

if [ -z "$session_id" ] || [ -z "$cwd" ]; then
  echo "cairn enqueue.sh: malformed or incomplete event JSON; skipping capture" >&2
  exit 0
fi

queue_dir="$cwd/.cairn/queue"
if ! mkdir -p "$queue_dir" 2>/dev/null; then
  echo "cairn enqueue.sh: could not create $queue_dir; skipping capture" >&2
  exit 0
fi

enqueued_at="$(date -u +%FT%TZ 2>/dev/null || printf '')"
job="$(jq -nc \
  --arg s "$session_id" \
  --arg t "$transcript" \
  --arg h "claude-code" \
  --arg at "$enqueued_at" \
  '{session_id: $s, transcript_path: $t, harness: $h, enqueued_at: $at}' 2>/dev/null)"

if [ -z "$job" ]; then
  echo "cairn enqueue.sh: could not build job record; skipping capture" >&2
  exit 0
fi

tmp_file="$queue_dir/.$session_id.json.tmp.$$"
if printf '%s\n' "$job" > "$tmp_file" 2>/dev/null; then
  mv -f "$tmp_file" "$queue_dir/$session_id.json" 2>/dev/null || rm -f "$tmp_file" 2>/dev/null
else
  rm -f "$tmp_file" 2>/dev/null
fi

exit 0
