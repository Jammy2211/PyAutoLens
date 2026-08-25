#!/usr/bin/env bash
# GENERATED — canonical source: PyAutoMind/policy/session_start_hook.sh
# Installed into every checked-out repo as .claude/hooks/session-start.sh by
# `python3 PyAutoMind/scripts/repos_sync.py --write`, and drift-checked by
# `--check`. Edit the canonical file, never a copy.
#
# ---------------------------------------------------------------------------
# Python 3.12 is the default in Claude Code web/mobile sessions.
#
# The remote container ships /usr/local/bin/python{,3} -> /usr/bin/python3.11,
# a pip whose shebang is #!/usr/bin/python3, and a uv-managed tool set (pytest,
# ruff, black, mypy, pyright, flake8, poetry) every one of which was built on
# 3.11 — one minor version below the floor the organism set for itself in the
# Python 3.12 floor campaign, and below every CI leg it runs. The image's own
# `use-python 3.12` does not fix it: it moves the update-alternatives links
# under /usr/bin, which /usr/local/bin/python{,3} then shadow.
#
# This hook makes a session 3.12 on three surfaces:
#
#   1. a 3.12 virtualenv first on PATH — python, python3, pip, pytest;
#   2. the /usr/local/bin/python{,3} symlinks repointed at 3.12, so anything
#      resolving PATH without this session's env (a subprocess with a scrubbed
#      environment, a `#!/usr/bin/env python3` script) also gets 3.12;
#   3. the uv-managed tools rebuilt on 3.12 — mypy and flake8 read the
#      interpreter's version, so on 3.11 they judged code against 3.11 rules.
#
# What it deliberately does NOT touch: the update-alternatives links under
# /usr/bin. Scripts with a literal `#!/usr/bin/python3` shebang follow those,
# and some of the image's own tools (conan) are installed for 3.11 only — a
# flip there breaks them for no gain the three surfaces above don't already
# give.
#
# Remote-only (a local checkout keeps whatever the developer's shell provides),
# idempotent (everything is skipped once it already reads 3.12, so the second
# repo's copy in the same session costs ~0.2s), and non-blocking: every step
# degrades to a logged warning rather than failing the session start.
#
# Per-repo dependencies: a repo that needs more than pytest + PyYAML declares it
# in .claude/session-python.txt — one pip argument per line (`-e .`, a package
# spec, `-r requirements.txt`; `#` comments ignored). Installed additively into
# the shared venv, so the file stays out of this generated hook and the hook
# stays byte-identical in every repo.
set -euo pipefail

[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0

VENV="${PYAUTO_SESSION_VENV:-$HOME/.pyauto/session-py312}"
BASE_DEPS=(pytest PyYAML)
EXTRAS_FILE="${CLAUDE_PROJECT_DIR:-$PWD}/.claude/session-python.txt"

# stderr, not stdout: a SessionStart hook's stdout is fed to the agent as
# session context.
log() { printf '[session-start] %s\n' "$*" >&2; }

is_py312() {
    [ -x "$1" ] && "$1" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))' >/dev/null 2>&1
}

find_base_python() {
    local candidate
    for candidate in /usr/bin/python3.12 /usr/local/bin/python3.12 \
                     "$(command -v python3.12 2>/dev/null || true)"; do
        if [ -n "$candidate" ] && is_py312 "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    # No system 3.12 (a future base image could drop it) — uv can fetch one.
    if command -v uv >/dev/null 2>&1; then
        log "no system python3.12; asking uv to install one"
        uv python install 3.12 >&2 || return 1
        candidate="$(uv python find 3.12 2>/dev/null || true)"
        if [ -n "$candidate" ] && is_py312 "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    fi
    return 1
}

venv_ready() {
    is_py312 "$VENV/bin/python" \
        && [ -x "$VENV/bin/pip" ] \
        && "$VENV/bin/python" -c 'import pytest, yaml' >/dev/null 2>&1
}

# 1. The interpreter a session types: python, python3, pip, pytest.
ensure_venv() {
    local base_python
    if venv_ready; then
        log "reusing $VENV ($("$VENV/bin/python" -V 2>&1))"
        return 0
    fi
    if ! base_python="$(find_base_python)"; then
        log "WARNING: no Python 3.12 in this container; PATH left unchanged"
        return 1
    fi
    log "building $VENV on $base_python"
    rm -rf "$VENV"
    mkdir -p "$(dirname "$VENV")"
    if command -v uv >/dev/null 2>&1; then
        # --seed puts pip inside the venv too, so `pip install` targets 3.12
        # rather than falling through to the container's 3.11 /usr/bin/pip.
        uv venv --seed --python "$base_python" "$VENV" >&2
        uv pip install --python "$VENV/bin/python" --quiet "${BASE_DEPS[@]}" >&2
    else
        "$base_python" -m venv "$VENV" >&2
        "$VENV/bin/python" -m pip install --quiet --upgrade pip >&2
        "$VENV/bin/python" -m pip install --quiet "${BASE_DEPS[@]}" >&2
    fi
    venv_ready || { log "WARNING: could not build a 3.12 venv at $VENV"; return 1; }
}

# This repo's own dependencies, if it declares any. Additive and marked, so a
# session holding several repos installs each repo's set exactly once.
ensure_repo_extras() {
    [ -r "$EXTRAS_FILE" ] || return 0
    local marker args=()
    marker="$VENV/.extras-$(cksum <"$EXTRAS_FILE" | tr -d ' /')"
    [ -e "$marker" ] && return 0
    while IFS= read -r line; do
        line="${line%%#*}"
        line="$(printf '%s' "$line" | tr -d '\r' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
        [ -n "$line" ] && args+=("$line")
    done <"$EXTRAS_FILE"
    [ ${#args[@]} -gt 0 ] || return 0
    log "installing this repo's declared deps: ${args[*]}"
    if (cd "$(dirname "$(dirname "$EXTRAS_FILE")")" && "$VENV/bin/python" -m pip install --quiet "${args[@]}" >&2); then
        : >"$marker"
    else
        log "WARNING: $EXTRAS_FILE install failed; continuing without it"
    fi
}

# 2. What PATH means without this session's env file.
point_system_default() {
    local base_python="$1" link
    for link in /usr/local/bin/python /usr/local/bin/python3; do
        is_py312 "$link" && continue
        [ -w "$(dirname "$link")" ] || { log "WARNING: cannot rewrite $link (not writable)"; continue; }
        ln -sfn "$base_python" "$link"
    done
    is_py312 /usr/local/bin/python3 \
        && log "/usr/local/bin/python{,3} -> $base_python" \
        || log "WARNING: /usr/local/bin/python3 is still $(/usr/local/bin/python3 -V 2>&1)"
}

# 3. The uv-managed tools — rebuilt on 3.12 only where they are not already.
#
# Pinned to the version already installed: this hook changes the INTERPRETER a
# tool runs on, and nothing else. Unpinned, `uv tool install --force` fetches
# the latest release, which quietly moved mypy across a major version (1.19 ->
# 2.3) the first time this ran — a lint-result change nobody asked for, riding
# in on a Python upgrade. Falls back to unpinned only if the pin cannot be
# resolved for 3.12.
retool_uv_tools() {
    command -v uv >/dev/null 2>&1 || return 0
    local tools_dir tool name version spec
    tools_dir="$(uv tool dir 2>/dev/null || echo "$HOME/.local/share/uv/tools")"
    [ -d "$tools_dir" ] || return 0
    for tool in "$tools_dir"/*/; do
        name="$(basename "$tool")"
        is_py312 "$tool/bin/python" && continue
        version="$(uv tool list 2>/dev/null | awk -v n="$name" '$1 == n {print substr($2, 2); exit}')"
        spec="$name"
        [ -n "$version" ] && spec="$name==$version"
        if uv tool install --python 3.12 --force "$spec" >/dev/null 2>&1 \
           || uv tool install --python 3.12 --force "$name" >/dev/null 2>&1; then
            log "rebuilt ${spec} on 3.12"
        else
            log "WARNING: could not rebuild $name on 3.12; it stays on $("$tool/bin/python" -V 2>&1)"
        fi
    done
}

if ensure_venv; then
    ensure_repo_extras
    point_system_default "$(readlink -f "$VENV/bin/python")"
    retool_uv_tools
    # Every repo in the session registers this hook, so the second copy must not
    # prepend the venv a second time.
    if [ -n "${CLAUDE_ENV_FILE:-}" ] && ! grep -qs 'PYAUTO_SESSION_PY312=' "$CLAUDE_ENV_FILE"; then
        {
            echo "export PYAUTO_SESSION_PY312=\"$VENV\""
            echo "export VIRTUAL_ENV=\"$VENV\""
            echo "export PATH=\"$VENV/bin:\$PATH\""
        } >>"$CLAUDE_ENV_FILE"
    fi
    log "default python is now $("$VENV/bin/python" -V 2>&1) ($VENV/bin)"
fi
