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
    echo -e "${YELLOW}No remote tracking branch. Comparing against HEAD.${NC}"
    ORIGIN="HEAD"
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

# Analyze changes to build commit message
MSG_PARTS=()

# Check for specific patterns in the diff
if echo "$CHANGES" | grep -q "result.py"; then
    if echo "$DIFF" | grep -q "__str__\|__eq__"; then
        MSG_PARTS+=("simplified API with __str__/__eq__")
    fi
fi

if echo "$CHANGES" | grep -q "README"; then
    MSG_PARTS+=("updated README")
fi

if echo "$CHANGES" | grep -q "CHANGELOG"; then
    MSG_PARTS+=("updated CHANGELOG")
fi

if echo "$CHANGES" | grep -q "pyproject.toml\|poetry.lock"; then
    if echo "$DIFF" | grep -q "twine\|build"; then
        MSG_PARTS+=("added twine/build for publishing")
    elif echo "$DIFF" | grep -q "diskcache\|redis"; then
        MSG_PARTS+=("added storage dependencies")
    elif echo "$DIFF" | grep -q "version"; then
        MSG_PARTS+=("version bump")
    else
        MSG_PARTS+=("updated dependencies")
    fi
fi

if echo "$CHANGES" | grep -q "Makefile"; then
    if echo "$DIFF" | grep -q "commit"; then
        MSG_PARTS+=("added commit helper")
    elif echo "$DIFF" | grep -q "twine"; then
        MSG_PARTS+=("switched to twine for publishing")
    else
        MSG_PARTS+=("updated Makefile")
    fi
fi

if echo "$CHANGES" | grep -q "scripts/"; then
    NEW_SCRIPTS=$(echo "$CHANGES" | grep "^A.*scripts/" | cut -f2 | xargs -I{} basename {} 2>/dev/null | tr '\n' ', ' | sed 's/,$//')
    if [ -n "$NEW_SCRIPTS" ]; then
        MSG_PARTS+=("added $NEW_SCRIPTS script")
    fi
fi

if echo "$CHANGES" | grep -q "store.py\|disk_store\|redis_store"; then
    MSG_PARTS+=("added context persistence stores")
fi

if echo "$CHANGES" | grep -qE "^A.*test"; then
    MSG_PARTS+=("added tests")
fi

if echo "$CHANGES" | grep -qE "^M.*test"; then
    MSG_PARTS+=("updated tests")
fi

# Build the message
if [ ${#MSG_PARTS[@]} -eq 0 ]; then
    # Fallback: count files
    FILE_COUNT=$(echo "$CHANGES" | wc -l | tr -d ' ')
    COMMIT_MSG="Update $FILE_COUNT files"
else
    # Join parts with comma
    COMMIT_MSG=$(IFS=", "; echo "${MSG_PARTS[*]}")
    # Capitalize first letter
    COMMIT_MSG="$(echo "${COMMIT_MSG:0:1}" | tr '[:lower:]' '[:upper:]')${COMMIT_MSG:1}"
fi

# Show generated message
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}Generated commit message:${NC}"
echo
echo -e "  ${YELLOW}$COMMIT_MSG${NC}"
echo
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
        ;;
    2)
        git commit -m "$COMMIT_MSG"
        git push origin "$BRANCH"
        echo -e "${GREEN}✓ Committed and pushed${NC}"
        ;;
    3)
        read -p "Enter message: " CUSTOM_MSG
        if [ -n "$CUSTOM_MSG" ]; then
            git commit -m "$CUSTOM_MSG"
            echo -e "${GREEN}✓ Committed${NC}"
            read -p "Push? [y/N]: " PUSH
            if [[ "$PUSH" =~ ^[Yy]$ ]]; then
                git push origin "$BRANCH"
                echo -e "${GREEN}✓ Pushed${NC}"
            fi
        else
            echo -e "${RED}No message provided${NC}"
            exit 1
        fi
        ;;
    *)
        echo -e "${YELLOW}Cancelled${NC}"
        git reset HEAD > /dev/null 2>&1 || true
        exit 0
        ;;
esac
