#!/usr/bin/env bash
# GENERATED — canonical source: PyAutoMind/policy/end_at_deliverable_hook.sh
# Installed into every checked-out repo as .claude/hooks/end-at-deliverable.sh by
# `python3 PyAutoMind/scripts/repos_sync.py --write`, and drift-checked by
# `--check`. Edit the canonical file, never a copy.
#
# ---------------------------------------------------------------------------
# Sessions end at their deliverable (PyAutoMind/policy/end_at_deliverable.md).
#
# A PreToolUse guard on the tools that outlive the turn. It exists because the
# prose rule was already written and was still broken twice: five batch members
# armed hourly check-ins on 2026-08-31 (fixed for batch members only), and on
# 2026-09-02/03 a mobile `/prm` re-armed a 60-minute `send_later` hourly from
# 02:39 to 12:11 UTC with no task active, leaving twenty fired one-shots and a
# drained usage window. A rule a session can talk itself past is not a rule, so
# this one is enforced by the harness.
#
# Registered under `hooks.PreToolUse` with the matcher
#   ^(send_later|subscribe_pr_activity|ScheduleWakeup|CronCreate|RemoteTrigger|mcp__.*(send_later|subscribe_pr_activity).*)$
#
# Allowed through:
#   * `RemoteTrigger` with action list / get / list_runs / get_run_log — reading
#     what already exists never outlives the turn;
#   * anything at all when PYAUTO_ALLOW_TIMERS=1 — the human-authorised escape
#     for a routine the human actually asked for.
#
# Everything else exits 2 (the harness treats stderr as the reason and blocks
# the call). An unreadable payload also exits 2: this fails closed, because the
# failure it guards against is silent and costs a night of usage.
set -uo pipefail

# Human-authorised routine: the one way past this guard, and it has to be set
# deliberately in the environment.
[ "${PYAUTO_ALLOW_TIMERS:-}" = "1" ] && exit 0

payload="$(cat)"

# python3 rather than jq: python3 is present everywhere this hook is installed
# (it is what the SessionStart hook guarantees), jq is not. The payload rides in
# the environment rather than on stdin so the heredoc keeps its own stdin.
PYAUTO_HOOK_PAYLOAD="$payload" python3 <<'PY'
import json
import os
import sys

# Read-only RemoteTrigger actions: they inspect what exists and schedule nothing.
READ_ONLY_REMOTE_TRIGGER = {"list", "get", "list_runs", "get_run_log"}

REASON = (
    "policy end_at_deliverable: sessions end at their deliverable "
    "— {tool} would outlive the turn.\n"
    "Report and stop; the human re-runs /prm. Set PYAUTO_ALLOW_TIMERS=1 only "
    "for a routine the human asked for.\n"
)


def block(tool):
    sys.stderr.write(REASON.format(tool=tool))
    raise SystemExit(2)


try:
    event = json.loads(os.environ.get("PYAUTO_HOOK_PAYLOAD", ""))
    if not isinstance(event, dict):
        raise ValueError("PreToolUse payload is not a JSON object")
    tool = event.get("tool_name")
    if not isinstance(tool, str) or not tool:
        raise ValueError("PreToolUse payload carries no tool_name")
    tool_input = event.get("tool_input")
    if not isinstance(tool_input, dict):
        tool_input = {}
except Exception:
    # Fail closed. A payload this hook cannot read is a call it cannot clear.
    block("an unreadable tool call")

if "RemoteTrigger" in tool:
    action = tool_input.get("action")
    if isinstance(action, str) and action in READ_ONLY_REMOTE_TRIGGER:
        raise SystemExit(0)

block(tool)
PY
