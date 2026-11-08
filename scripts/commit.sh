#!/bin/bash
#
# Interactive commit helper.
#
# Builds a commit message from the *staged* changes only, then lets you
# commit, commit and push, edit the message, or cancel.
#
#   - Nothing staged? Offers to stage modified tracked files (`git add -u`).
#     Untracked files are never added automatically.
#   - Cancelling only unstages what this script staged.
#   - Nothing to commit but unpushed commits? Offers to push them.
#

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { echo -e "${BLUE}$*${NC}"; }
ok()   { echo -e "${GREEN}✓ $*${NC}"; }
warn() { echo -e "${YELLOW}$*${NC}"; }
die()  { echo -e "${RED}Error: $*${NC}" >&2; exit 1; }

git rev-parse --is-inside-work-tree > /dev/null 2>&1 || die "not in a git repository"

BRANCH=$(git branch --show-current)
[ -n "$BRANCH" ] || die "detached HEAD; check out a branch first"

push_branch() {
    if git rev-parse --abbrev-ref --symbolic-full-name '@{u}' > /dev/null 2>&1; then
        git push
    else
        git push -u origin "$BRANCH"
    fi
    ok "Pushed $BRANCH"
}

unpushed_count() {
    if git rev-parse --verify --quiet "origin/$BRANCH" > /dev/null; then
        git rev-list --count "origin/$BRANCH..HEAD"
    else
        git rev-list --count HEAD
    fi
}

# --- Make sure something is staged -------------------------------------------

STAGED_BY_SCRIPT=0
if git diff --cached --quiet; then
    if ! git diff --quiet; then
        warn "Nothing staged. Modified tracked files:"
        git diff --name-status
        echo
        read -r -p "Stage these with 'git add -u'? [y/N]: " STAGE
        if [[ "$STAGE" =~ ^[Yy]$ ]]; then
            git add -u
            STAGED_BY_SCRIPT=1
        fi
    fi
fi

if git diff --cached --quiet; then
    UNPUSHED=$(unpushed_count)
    if [ "$UNPUSHED" -gt 0 ]; then
        warn "Nothing to commit, but $UNPUSHED unpushed commit(s) on $BRANCH."
        read -r -p "Push now? [y/N]: " PUSH
        [[ "$PUSH" =~ ^[Yy]$ ]] && push_branch
    else
        warn "Nothing to commit."
    fi
    exit 0
fi

# --- Analyze staged changes ---------------------------------------------------

info "═══════════════════════════════════════"
info "  Staged changes on $BRANCH"
info "═══════════════════════════════════════"
git diff --cached --name-status
echo

CHANGES=$(git diff --cached --name-status)

# Lines added in a given path (without the leading '+').
added_in() {
    git diff --cached -- "$@" | grep '^+[^+]' | cut -c2- || true
}
# True if any of the staged paths match the pattern.
touched() {
    echo "$CHANGES" | cut -f2- | grep -qE "$1"
}
# True if a path with this status (A, M, D, ...) matches the pattern.
touched_with() {
    echo "$CHANGES" | awk -F'\t' -v s="$1" -v p="$2" \
        'substr($1, 1, 1) == s && $NF ~ p { found = 1 } END { exit !found }'
}

TITLE_PARTS=()
BULLETS=()

if touched 'src/.*\.py$'; then
    TITLE_PARTS+=("update library code")
fi

if touched 'pyproject\.toml$'; then
    PYPROJECT_ADDED=$(added_in pyproject.toml)
    NEW_VERSION=$(echo "$PYPROJECT_ADDED" | sed -nE 's/^version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' | head -1)
    if [ -n "$NEW_VERSION" ]; then
        TITLE_PARTS=("release v$NEW_VERSION" ${TITLE_PARTS[@]+"${TITLE_PARTS[@]}"})
        BULLETS+=("Bump version to $NEW_VERSION")
    fi
    BULLETS+=("Update pyproject.toml")
fi
touched 'poetry\.lock$' && BULLETS+=("Update poetry.lock")

if touched '^tests/'; then
    if touched_with A '^tests/'; then
        BULLETS+=("Add tests")
    else
        BULLETS+=("Update tests")
    fi
fi

touched '(^|/)README' && BULLETS+=("Update README")
touched '(^|/)CHANGELOG' && BULLETS+=("Update CHANGELOG")
touched '^\.github/' && BULLETS+=("Update CI workflows")
touched '(^|/)Makefile$' && BULLETS+=("Update Makefile")

if touched '^scripts/'; then
    while IFS= read -r path; do
        BULLETS+=("Add $(basename "$path")")
    done < <(echo "$CHANGES" | awk -F'\t' '$1 == "A" && $2 ~ /^scripts\// {print $2}')
    touched_with M '^scripts/' && BULLETS+=("Update scripts")
fi

# --- Build message ------------------------------------------------------------

FILE_COUNT=$(echo "$CHANGES" | grep -c . || true)
if [ ${#TITLE_PARTS[@]} -eq 0 ]; then
    if [ "$FILE_COUNT" -eq 1 ]; then
        TITLE="Update $(echo "$CHANGES" | cut -f2- | tail -1)"
    else
        TITLE="Update $FILE_COUNT files"
    fi
else
    TITLE=""
    for part in "${TITLE_PARTS[@]}"; do
        TITLE="${TITLE:+$TITLE, }$part"
    done
fi
TITLE="$(echo "${TITLE:0:1}" | tr '[:lower:]' '[:upper:]')${TITLE:1}"

COMMIT_MSG="$TITLE"
if [ ${#BULLETS[@]} -gt 0 ]; then
    COMMIT_MSG+=$'\n'
    for bullet in "${BULLETS[@]}"; do
        COMMIT_MSG+=$'\n'"- $bullet"
    done
fi

echo -e "${GREEN}Generated commit message:${NC}"
echo
echo -e "${YELLOW}${COMMIT_MSG}${NC}"
echo

# --- Act ----------------------------------------------------------------------

cancel() {
    warn "Cancelled"
    if [ "$STAGED_BY_SCRIPT" -eq 1 ]; then
        git reset -q
    fi
    exit 0
}

echo "1) Commit"
echo "2) Commit and push"
echo "3) Edit message, then commit"
echo "4) Cancel"
echo
read -r -p "Select [1-4]: " ACTION

case "$ACTION" in
    1)
        git commit -m "$COMMIT_MSG"
        ok "Committed"
        ;;
    2)
        git commit -m "$COMMIT_MSG"
        ok "Committed"
        push_branch
        ;;
    3)
        MSG_FILE=$(mktemp "${TMPDIR:-/tmp}/commit_msg.XXXXXX")
        trap 'rm -f "$MSG_FILE"' EXIT
        printf '%s\n' "$COMMIT_MSG" > "$MSG_FILE"
        "${EDITOR:-vi}" "$MSG_FILE"
        if ! grep -q '[^[:space:]]' "$MSG_FILE"; then
            cancel
        fi
        git commit -F "$MSG_FILE"
        ok "Committed"
        read -r -p "Push? [y/N]: " PUSH
        [[ "$PUSH" =~ ^[Yy]$ ]] && push_branch
        ;;
    *)
        cancel
        ;;
esac

git log --oneline -1
