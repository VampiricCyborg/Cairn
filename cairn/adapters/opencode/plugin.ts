// opencode capture adapter: enqueue a Cairn job when a session goes idle.
//
// Mirrors the job-record shape Claude Code's `enqueue.sh` writes to
// `.cairn/queue/<session_id>.json` (session_id, transcript_path, harness,
// enqueued_at) so cairn's SessionStart queue sweep
// (`cairn.cli._sweep_queue` / `cairn.core.normalizer.normalize`) needs no
// changes to pick this up. Per the README's hook-safety principle for
// capture adapters, this must never throw or block the harness: the only
// work here is a fetch of already-computed session data plus two file
// writes, and every failure path is caught and logged, never re-thrown.
//
// Confirmed against opencode's published docs and `@opencode-ai/sdk` /
// `@opencode-ai/plugin` type declarations (checked 2026-09-14, package
// version 1.18.30) -- NOT assumed from Claude Code's shape:
//   - the session-idle event is `{ type: "session.idle", properties: {
//     sessionID } }` (opencode.ai/docs/plugins; `EventSessionIdle`).
//   - unlike Claude Code, opencode does not expose a single flat
//     transcript file a plugin can hand off by path -- session messages
//     live behind the SDK (`client.session.messages`) or in opencode's own
//     per-project storage under `~/.local/share/opencode/`, whose on-disk
//     layout is an internal, migration-prone implementation detail (see
//     `packages/opencode/src/storage/storage.ts` upstream). Reconstructing
//     that path ourselves would be exactly the kind of adapter-surface
//     guess the README warns about, so instead this plugin calls the
//     documented `client.session.messages({ path: { id } })` method and
//     writes its response verbatim to a side-car JSON file. That response
//     shape -- `Array<{ info: Message, parts: Part[] }>` -- is what
//     `cairn.core.normalizer.normalize_opencode_transcript` parses.
//
// If a future opencode release renames `session.idle`, changes its
// event payload, or changes `client.session.messages`' response shape,
// this plugin fails closed (catches, logs, does not enqueue) rather than
// writing a malformed job -- and `cairn doctor`'s opencode row is the
// intended tripwire for "the surface moved" (see
// `cairn.cli._check_opencode_version`), per the README's adapter-surface
// version-check policy.

import { mkdir, rename, writeFile } from "node:fs/promises"
import path from "node:path"

import type { Plugin } from "@opencode-ai/plugin"

export const CairnPlugin: Plugin = async ({ client, directory, worktree }) => {
  return {
    event: async ({ event }) => {
      if (event.type !== "session.idle") return

      const sessionId = event.properties.sessionID
      const root = worktree || directory

      try {
        const response = await client.session.messages({ path: { id: sessionId } })
        const messages = response.data
        if (!messages) {
          console.error("cairn: session.messages returned no data for", sessionId, response.error)
          return
        }

        const queueDir = path.join(root, ".cairn", "queue")
        await mkdir(queueDir, { recursive: true })

        const transcriptPath = path.join(queueDir, `${sessionId}.transcript.json`)
        await writeFile(transcriptPath, JSON.stringify(messages), "utf-8")

        const job = {
          session_id: sessionId,
          transcript_path: transcriptPath,
          harness: "opencode",
          enqueued_at: new Date().toISOString(),
        }
        const jobPath = path.join(queueDir, `${sessionId}.json`)
        const tmpPath = path.join(queueDir, `.${sessionId}.json.tmp`)
        await writeFile(tmpPath, JSON.stringify(job), "utf-8")
        await rename(tmpPath, jobPath)
      } catch (error) {
        console.error("cairn: failed to enqueue opencode session", sessionId, error)
      }
    },
  }
}

export default CairnPlugin
