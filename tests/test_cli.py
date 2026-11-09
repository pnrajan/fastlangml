"""CLI argument handling: subcommand routing and global output options.

These check which command ran and how output is formatted, not which
language was detected, so they pass with any set of installed backends.
"""

from __future__ import annotations

import json

import pytest

from fastlangml.cli.main import _insert_default_command, main

COMMANDS = {"detect", "batch", "bench", "backends"}


def run(capsys, *argv: str) -> str:
    assert main(list(argv)) == 0
    return capsys.readouterr().out


class TestDefaultCommand:
    @pytest.mark.parametrize(
        ("argv", "expected"),
        [
            (["Bonjour"], ["detect", "Bonjour"]),
            (["--pretty", "Bonjour"], ["--pretty", "detect", "Bonjour"]),
            (["-f", "table", "Bonjour"], ["-f", "table", "detect", "Bonjour"]),
            (["--format=table", "Bonjour"], ["--format=table", "detect", "Bonjour"]),
            (["--", "--pretty"], ["detect", "--", "--pretty"]),
            (["-"], ["detect", "-"]),
            (["backends"], ["backends"]),
            (["--pretty", "backends"], ["--pretty", "backends"]),
            (["-f", "json", "batch", "in.txt"], ["-f", "json", "batch", "in.txt"]),
            ([], []),
            (["--help"], ["--help"]),
        ],
    )
    def test_detect_inserted_only_when_no_command(self, argv, expected):
        assert _insert_default_command(argv, COMMANDS) == expected


class TestGlobalOptions:
    def test_global_pretty_before_subcommand_runs_that_subcommand(self, capsys):
        out = run(capsys, "--pretty", "backends")

        data = json.loads(out)
        assert "available" in data  # backends output, not a detection of "backends"
        assert "\n  " in out  # indented

    @pytest.mark.parametrize(
        "argv",
        [
            ["-f", "table", "detect", "Bonjour tout le monde"],
            ["detect", "-f", "table", "Bonjour tout le monde"],
            ["--format=table", "detect", "Bonjour tout le monde"],
            ["-f", "table", "Bonjour tout le monde"],
        ],
    )
    def test_table_format_in_any_position(self, capsys, argv):
        out = run(capsys, *argv)

        assert not out.lstrip().startswith("{")
        assert "Lang:" in out or "und" in out

    def test_subcommand_option_overrides_global(self, capsys):
        out = run(capsys, "-f", "table", "detect", "-f", "json", "Bonjour tout le monde")

        assert "lang" in json.loads(out)

    def test_default_output_is_compact_json(self, capsys):
        out = run(capsys, "Bonjour tout le monde")

        data = json.loads(out)
        assert data["text"] == "Bonjour tout le monde"
        assert "\n  " not in out

    def test_double_dash_detects_flag_like_text(self, capsys):
        out = run(capsys, "--", "--pretty")

        assert json.loads(out)["text"] == "--pretty"


class TestErrors:
    def test_batch_rejects_table_format(self, capsys, tmp_path):
        path = tmp_path / "in.txt"
        path.write_text("Hello there\n")

        with pytest.raises(SystemExit) as exc:
            main(["-f", "table", "batch", str(path)])

        assert exc.value.code == 2
        assert "batch supports --format json or jsonl" in capsys.readouterr().err

    def test_unknown_global_option_is_an_error(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--bogus", "backends"])

        assert exc.value.code == 2
        assert "--bogus" in capsys.readouterr().err
