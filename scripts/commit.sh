#!/bin/bash
#
# Smart git commit script
# Analyzes changes and generates an informed commit message for review
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Smart Commit Helper${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# Check if we're in a git repo
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo -e "${RED}Error: Not in a git repository${NC}"
    exit 1
fi

# Check for changes
if git diff --quiet && git diff --cached --quiet; then
    echo -e "${YELLOW}No changes to commit${NC}"
    exit 0
fi

# Stage all changes if nothing is staged
if git diff --cached --quiet; then
    echo -e "${YELLOW}No staged changes. Staging all changes...${NC}"
    git add -A
fi

# Show current status
echo -e "${GREEN}=== Files Changed ===${NC}"
git diff --cached --name-status
echo

# Get list of changed files
STAGED_FILES=$(git diff --cached --name-only)

# Show recent commits for context
echo -e "${GREEN}=== Recent Commits (for style reference) ===${NC}"
git log --oneline -5
echo

# Analyze the changes to generate commit message
echo -e "${CYAN}Analyzing changes...${NC}"
echo

# Get detailed diff for analysis
DIFF_STAT=$(git diff --cached --stat)
DIFF_CONTENT=$(git diff --cached --no-color | head -500)

# Determine commit type based on files changed
COMMIT_TYPE="feat"
if echo "$STAGED_FILES" | grep -qE "^tests?/|_test\.py$|test_.*\.py$"; then
    COMMIT_TYPE="test"
elif echo "$STAGED_FILES" | grep -qE "README|CHANGELOG|docs/|\.md$"; then
    COMMIT_TYPE="docs"
elif echo "$STAGED_FILES" | grep -qE "pyproject\.toml|Makefile|setup\.|requirements|poetry\.lock"; then
    COMMIT_TYPE="build"
elif echo "$STAGED_FILES" | grep -qE "\.github/|ci/|\.yml$"; then
    COMMIT_TYPE="ci"
fi

# Build a summary of changes
SUMMARY_PARTS=()

# Check for new files
NEW_FILES=$(git diff --cached --name-status | grep "^A" | cut -f2)
if [ -n "$NEW_FILES" ]; then
    NEW_COUNT=$(echo "$NEW_FILES" | wc -l | tr -d ' ')
    if [ "$NEW_COUNT" -eq 1 ]; then
        SUMMARY_PARTS+=("Add $(basename "$NEW_FILES")")
    else
        SUMMARY_PARTS+=("Add $NEW_COUNT new files")
    fi
fi

# Check for modified files and analyze content
MODIFIED_FILES=$(git diff --cached --name-status | grep "^M" | cut -f2)

# Analyze specific file changes
for file in $MODIFIED_FILES; do
    case "$file" in
        *result.py)
            if git diff --cached "$file" | grep -q "__str__\|__eq__"; then
                SUMMARY_PARTS+=("Add simplified API (__str__, __eq__)")
            fi
            ;;
        *README*)
            SUMMARY_PARTS+=("Update README")
            ;;
        *CHANGELOG*)
            SUMMARY_PARTS+=("Update CHANGELOG")
            ;;
        *pyproject.toml)
            if git diff --cached "$file" | grep -q "dependencies"; then
                SUMMARY_PARTS+=("Update dependencies")
            elif git diff --cached "$file" | grep -q "version"; then
                SUMMARY_PARTS+=("Bump version")
            fi
            ;;
        *Makefile)
            SUMMARY_PARTS+=("Update Makefile")
            ;;
        *.lock)
            # Skip lock files in summary, they're implied
            ;;
    esac
done

# Build the commit message
if [ ${#SUMMARY_PARTS[@]} -eq 0 ]; then
    # Fallback: use file names
    FILE_COUNT=$(echo "$STAGED_FILES" | wc -l | tr -d ' ')
    SHORT_DESC="Update $FILE_COUNT files"
else
    # Join summary parts
    SHORT_DESC=$(IFS=", "; echo "${SUMMARY_PARTS[*]}")
fi

# Truncate if too long
if [ ${#SHORT_DESC} -gt 72 ]; then
    SHORT_DESC="${SHORT_DESC:0:69}..."
fi

# Build detailed body from diff stats
BODY=""
if [ -n "$DIFF_STAT" ]; then
    BODY="Changes:\n"
    # List each file with its change type
    while IFS= read -r line; do
        STATUS=$(echo "$line" | cut -f1)
        FILE=$(echo "$line" | cut -f2)
        case $STATUS in
            A) BODY="$BODY- Add $FILE\n" ;;
            M) BODY="$BODY- Update $FILE\n" ;;
            D) BODY="$BODY- Remove $FILE\n" ;;
            R*) BODY="$BODY- Rename $FILE\n" ;;
        esac
    done <<< "$(git diff --cached --name-status)"
fi

# Build full commit message
COMMIT_MSG="$COMMIT_TYPE: $SHORT_DESC"
if [ -n "$BODY" ]; then
    COMMIT_MSG="$COMMIT_MSG\n\n$BODY"
fi

# Show the generated commit message
echo -e "${GREEN}=== Generated Commit Message ===${NC}"
echo -e "${CYAN}$COMMIT_MSG${NC}"
echo -e "${GREEN}=================================${NC}"
echo

# Ask for confirmation
echo -e "${YELLOW}What would you like to do?${NC}"
echo "  1) Approve and commit"
echo "  2) Approve, commit and push"
echo "  3) Edit message in editor"
echo "  4) Cancel"
echo
read -p "Select [1-4]: " ACTION

case $ACTION in
    1)
        # Commit only
        echo -e "$COMMIT_MSG" | git commit -F -
        echo
        echo -e "${GREEN}✓ Committed successfully!${NC}"
        echo
        git log --oneline -1
        ;;
    2)
        # Commit and push
        echo -e "$COMMIT_MSG" | git commit -F -

        BRANCH=$(git branch --show-current)
        echo
        echo -e "${BLUE}Pushing to origin/$BRANCH...${NC}"
        git push origin "$BRANCH"

        echo
        echo -e "${GREEN}✓ Committed and pushed successfully!${NC}"
        echo
        git log --oneline -1
        ;;
    3)
        # Edit in editor
        echo -e "$COMMIT_MSG" > /tmp/commit_msg.txt
        ${EDITOR:-vim} /tmp/commit_msg.txt
        git commit -F /tmp/commit_msg.txt
        rm /tmp/commit_msg.txt

        echo
        echo -e "${GREEN}✓ Committed successfully!${NC}"
        echo
        git log --oneline -1

        read -p "Push to remote? [y/N]: " PUSH_CONFIRM
        if [[ "$PUSH_CONFIRM" =~ ^[Yy]$ ]]; then
            BRANCH=$(git branch --show-current)
            git push origin "$BRANCH"
            echo -e "${GREEN}✓ Pushed successfully!${NC}"
        fi
        ;;
    4|*)
        echo -e "${YELLOW}Cancelled${NC}"
        exit 0
        ;;
esac
