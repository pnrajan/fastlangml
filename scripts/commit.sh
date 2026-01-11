#!/bin/bash
#
# Smart commit: analyzes changes vs origin and generates informative commit message
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if we're in a git repo
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo -e "${RED}Error: Not in a git repository${NC}"
    exit 1
fi

BRANCH=$(git branch --show-current)
ORIGIN="origin/$BRANCH"

# Check if origin branch exists
if ! git rev-parse --verify "$ORIGIN" > /dev/null 2>&1; then
    echo -e "${YELLOW}No remote tracking branch. Comparing against HEAD~1.${NC}"
    ORIGIN="HEAD~1"
fi

# Stage all changes
git add -A

# Check for changes
if git diff --cached --quiet && git diff "$ORIGIN"..HEAD --quiet 2>/dev/null; then
    echo -e "${YELLOW}No changes to commit${NC}"
    exit 0
fi

echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo -e "${BLUE}  Comparing $BRANCH vs $ORIGIN${NC}"
echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo

# Show what's different from origin
echo -e "${GREEN}Changed files:${NC}"
git diff --name-status "$ORIGIN" 2>/dev/null || git diff --cached --name-status
echo

# Collect all changes for analysis
CHANGES=$(git diff --name-status "$ORIGIN" 2>/dev/null || git diff --cached --name-status)
DIFF=$(git diff "$ORIGIN" 2>/dev/null || git diff --cached)

# Arrays for title keywords and bullet points
TITLE_PARTS=()
BULLETS=()

# Analyze result.py changes
if echo "$CHANGES" | grep -q "result.py"; then
    if echo "$DIFF" | grep -q "__str__\|__eq__"; then
        TITLE_PARTS+=("simplified API")
        BULLETS+=("Add __str__ and __eq__ to DetectionResult for cleaner usage")
    fi
fi

# Analyze store changes
if echo "$CHANGES" | grep -q "store.py\|disk_store\|redis_store"; then
    TITLE_PARTS+=("context persistence")
    if echo "$CHANGES" | grep -q "disk_store"; then
        BULLETS+=("Add DiskContextStore for file-based session storage")
    fi
    if echo "$CHANGES" | grep -q "redis_store"; then
        BULLETS+=("Add RedisContextStore for production deployments")
    fi
fi

# Analyze pyproject.toml
if echo "$CHANGES" | grep -q "pyproject.toml"; then
    if echo "$DIFF" | grep -q "twine"; then
        BULLETS+=("Add twine/build for PyPI publishing")
    fi
    if echo "$DIFF" | grep -q "diskcache\|redis"; then
        BULLETS+=("Add optional storage dependencies (diskcache, redis)")
    fi
    if echo "$DIFF" | grep -qE '^\+version\s*=\s*"[0-9]'; then
        VERSION=$(echo "$DIFF" | grep -oE '^\+version\s*=\s*"[^"]+"' | grep -oE '"[^"]+"' | tr -d '"')
        if [ -n "$VERSION" ]; then
            TITLE_PARTS=("v$VERSION" "${TITLE_PARTS[@]}")
        fi
    fi
fi

# Analyze Makefile
if echo "$CHANGES" | grep -q "Makefile"; then
    if echo "$DIFF" | grep -q "commit"; then
        BULLETS+=("Add commit helper to Makefile")
    fi
    if echo "$DIFF" | grep -q "twine"; then
        BULLETS+=("Switch publish target to use twine")
    fi
    if echo "$DIFF" | grep -q "poetry run"; then
        BULLETS+=("Update Makefile to use Poetry commands")
    fi
fi

# Analyze scripts
if echo "$CHANGES" | grep -q "^A.*scripts/"; then
    SCRIPT_NAME=$(echo "$CHANGES" | grep "^A.*scripts/" | head -1 | cut -f2 | xargs basename)
    BULLETS+=("Add $SCRIPT_NAME for interactive commits")
fi

# Analyze README
if echo "$CHANGES" | grep -q "README"; then
    if echo "$DIFF" | grep -q "Quick Start\|Usage"; then
        BULLETS+=("Update README with simplified API examples")
    else
        BULLETS+=("Update README documentation")
    fi
fi

# Analyze CHANGELOG
if echo "$CHANGES" | grep -q "CHANGELOG"; then
    BULLETS+=("Update CHANGELOG")
fi

# Analyze tests
if echo "$CHANGES" | grep -qE "test.*\.py"; then
    TEST_COUNT=$(echo "$CHANGES" | grep -cE "test.*\.py" || echo "0")
    if echo "$CHANGES" | grep -qE "^A.*test"; then
        BULLETS+=("Add new tests")
    elif [ "$TEST_COUNT" -gt 0 ]; then
        BULLETS+=("Update tests")
    fi
fi

# Build the title
if [ ${#TITLE_PARTS[@]} -eq 0 ]; then
    # Fallback title
    FILE_COUNT=$(echo "$CHANGES" | wc -l | tr -d ' ')
    TITLE="Update $FILE_COUNT files"
else
    TITLE=$(IFS=" and "; echo "${TITLE_PARTS[*]}")
    # Capitalize first letter
    TITLE="$(echo "${TITLE:0:1}" | tr '[:lower:]' '[:upper:]')${TITLE:1}"
fi

# Build full commit message
COMMIT_MSG="$TITLE"
if [ ${#BULLETS[@]} -gt 0 ]; then
    COMMIT_MSG="$TITLE

"
    for bullet in "${BULLETS[@]}"; do
        COMMIT_MSG="$COMMIT_MSG- $bullet
"
    done
fi

# Show generated message
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}Generated commit message:${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo
echo -e "${YELLOW}$COMMIT_MSG${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo

# Options
echo "1) Commit"
echo "2) Commit and push"
echo "3) Edit message"
echo "4) Cancel"
echo
read -p "Select [1-4]: " ACTION

case $ACTION in
    1)
        git commit -m "$COMMIT_MSG"
        echo -e "${GREEN}✓ Committed${NC}"
        git log --oneline -1
        ;;
    2)
        git commit -m "$COMMIT_MSG"
        git push origin "$BRANCH"
        echo -e "${GREEN}✓ Committed and pushed${NC}"
        git log --oneline -1
        ;;
    3)
        echo "$COMMIT_MSG" > /tmp/commit_msg.txt
        ${EDITOR:-vim} /tmp/commit_msg.txt
        git commit -F /tmp/commit_msg.txt
        rm /tmp/commit_msg.txt
        echo -e "${GREEN}✓ Committed${NC}"
        git log --oneline -1
        read -p "Push? [y/N]: " PUSH
        if [[ "$PUSH" =~ ^[Yy]$ ]]; then
            git push origin "$BRANCH"
            echo -e "${GREEN}✓ Pushed${NC}"
        fi
        ;;
    *)
        echo -e "${YELLOW}Cancelled${NC}"
        git reset HEAD > /dev/null 2>&1 || true
        exit 0
        ;;
esac
