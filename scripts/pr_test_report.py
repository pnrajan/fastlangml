"""Turn a pytest JUnit XML report into a Markdown list of every test case.

Used by .github/workflows/pr-tests.yml to post test results on pull requests.

    python scripts/pr_test_report.py report.xml > report.md
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

# Hidden marker so the workflow can find and update its own comment.
MARKER = "<!-- fastlangml-test-report -->"

ICONS = {
    "passed": "✅",
    "failed": "❌",
    "error": "💥",
    "skipped": "⏭️",
    "xfailed": "⚠️",
}


def outcome(case: ET.Element) -> tuple[str, str]:
    """Return (status, message) for one <testcase>."""
    for tag in ("failure", "error"):
        node = case.find(tag)
        if node is not None:
            status = "failed" if tag == "failure" else "error"
            return status, (node.get("message") or "").strip()
    node = case.find("skipped")
    if node is not None:
        message = (node.get("message") or "").strip()
        if node.get("type") == "pytest.xfail":
            return "xfailed", message
        return "skipped", message
    return "passed", ""


def short(text: str, limit: int = 160) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render(xml_path: str) -> tuple[str, bool]:
    root = ET.parse(xml_path).getroot()
    cases = root.iter("testcase")

    by_file: dict[str, list[tuple[str, str, str, float]]] = defaultdict(list)
    totals: dict[str, int] = defaultdict(int)
    for case in cases:
        classname = case.get("classname", "")
        parts = classname.split(".")
        # "tests.test_cache.TestCache" -> file "tests/test_cache.py", class "TestCache"
        file_parts = [p for p in parts if not p[:1].isupper()]
        cls = ".".join(p for p in parts if p[:1].isupper())
        file = "/".join(file_parts) + ".py"
        name = f"{cls}::{case.get('name')}" if cls else case.get("name", "")
        status, message = outcome(case)
        totals[status] += 1
        by_file[file].append((name, status, message, float(case.get("time", 0))))

    total = sum(totals.values())
    ok = totals["failed"] == 0 and totals["error"] == 0
    summary = " · ".join(
        f"{ICONS[s]} {totals[s]} {s}"
        for s in ("passed", "failed", "error", "skipped", "xfailed")
        if totals[s]
    )

    lines = [
        MARKER,
        f"## {'✅' if ok else '❌'} Test report: {total} test cases",
        "",
        summary,
        "",
    ]

    problems = [
        (file, name, status, message)
        for file, rows in by_file.items()
        for name, status, message, _ in rows
        if status in ("failed", "error")
    ]
    if problems:
        lines += ["### Failures", "", "| Test | Message |", "|---|---|"]
        lines += [f"| `{f}::{n}` | {short(m)} |" for f, n, _, m in problems]
        lines.append("")

    lines += ["### All test cases", ""]
    for file in sorted(by_file):
        rows = by_file[file]
        bad = any(s in ("failed", "error") for _, s, _, _ in rows)
        counts = defaultdict(int)
        for _, s, _, _ in rows:
            counts[s] += 1
        badge = " ".join(f"{ICONS[s]}{counts[s]}" for s in ICONS if counts[s])
        lines.append(f"<details{' open' if bad else ''}>")
        lines.append(f"<summary><code>{file}</code> — {len(rows)} tests {badge}</summary>")
        lines.append("")
        lines.append("| | Test case | Note |")
        lines.append("|---|---|---|")
        for name, status, message, _ in rows:
            note = short(message) if status != "passed" else ""
            lines.append(f"| {ICONS[status]} | `{name}` | {note} |")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines), ok


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    report, _ = render(sys.argv[1])
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
