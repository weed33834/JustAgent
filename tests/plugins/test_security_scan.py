"""Tests for the built-in security-scan plugin."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from myagent.core.context import CommandContext
from myagent.exceptions import SecurityScanError
from myagent.models.config import AppConfig, SecurityThreshold
from myagent.plugins import security_scan


@pytest.fixture
def security_context(app_config: AppConfig) -> CommandContext:
    """Return a CommandContext with security scanning enabled."""
    return CommandContext(
        command="commit",
        project_root=app_config.project_root,
        config=app_config,
    )


def _bandit_json(severity: str) -> str:
    return f'{{"results": [{{"issue_severity": "{severity}", "issue_confidence": "HIGH", "issue_text": "bad", "filename": "x.py", "line_number": 1}}]}}'


def test_security_scan_disabled(security_context: CommandContext) -> None:
    security_context.config.security.enabled = False
    with patch("subprocess.run") as mock_run:
        security_scan.plugin.pre_commit(security_context)
    mock_run.assert_not_called()


def test_security_scan_skips_missing_tool(security_context: CommandContext) -> None:
    with (
        patch("shutil.which", return_value=None),
        patch("subprocess.run") as mock_run,
    ):
        security_scan.plugin.pre_commit(security_context)
    mock_run.assert_not_called()


def test_bandit_high_triggers_error(security_context: CommandContext) -> None:
    with (
        patch("shutil.which", return_value="/usr/bin/bandit"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value.stdout = _bandit_json("HIGH")
        with pytest.raises(SecurityScanError, match="Security scan found"):
            security_scan.plugin.pre_commit(security_context)


def test_bandit_low_below_high_threshold(security_context: CommandContext) -> None:
    security_context.config.security.threshold = SecurityThreshold.HIGH
    with (
        patch("shutil.which", return_value="/usr/bin/bandit"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value.stdout = _bandit_json("LOW")
        security_scan.plugin.pre_commit(security_context)


def test_gitleaks_detects_secret(security_context: CommandContext) -> None:
    security_context.config.security.tools = ["gitleaks"]
    with (
        patch("shutil.which", return_value="/usr/bin/gitleaks"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = "leak"
        mock_run.return_value.stderr = ""
        with pytest.raises(SecurityScanError, match="Security scan found"):
            security_scan.plugin.pre_commit(security_context)


def _osv_payload(severity_value: object) -> str:
    severity_json = json.dumps(severity_value)
    return (
        '{"results": [{"packages": [{"vulnerabilities": ['
        f'{{"id": "GHSA-1", "summary": "bad lib", "severity": {severity_json}}}'
        "]}]}]}"
    )


def test_osv_scanner_detects_vulnerability(security_context: CommandContext) -> None:
    security_context.config.security.tools = ["osv-scanner"]
    # Real CVSS v3.1 vector that computes to a CRITICAL base score.
    payload = _osv_payload(
        [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"}]
    )
    with (
        patch("shutil.which", return_value="/usr/bin/osv-scanner"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value.stdout = payload
        with pytest.raises(SecurityScanError, match="Security scan found"):
            security_scan.plugin.pre_commit(security_context)


@pytest.mark.parametrize(
    ("vuln", "expected"),
    [
        (
            {
                "severity": [
                    {
                        "type": "CVSS_V3",
                        "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                    }
                ]
            },
            "CRITICAL",
        ),
        (
            {
                "severity": [
                    {
                        "type": "CVSS_V3",
                        "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:N/A:N",
                    }
                ]
            },
            "MEDIUM",
        ),
        ({"severity": "HIGH"}, "HIGH"),
        ({}, "MEDIUM"),
    ],
)
def test_osv_severity_parsing(vuln: dict[str, object], expected: str) -> None:
    assert security_scan._osv_severity(vuln) == expected


def test_unsupported_tool_is_skipped(security_context: CommandContext) -> None:
    security_context.config.security.tools = ["unknown-tool"]
    with patch("shutil.which", return_value="/usr/bin/unknown-tool"):
        security_scan.plugin.pre_commit(security_context)


def test_summarize_empty() -> None:
    assert security_scan._summarize([]) == {
        "total": 0,
        "low": 0,
        "medium": 0,
        "high": 0,
        "critical": 0,
        "max_severity": "LOW",
        "max_severity_value": 0,
    }


def test_summarize_multiple() -> None:
    findings = [
        {"severity": "LOW"},
        {"severity": "HIGH"},
        {"severity": "MEDIUM"},
    ]
    summary = security_scan._summarize(findings)
    assert summary["total"] == 3
    assert summary["max_severity"] == "HIGH"
    assert summary["max_severity_value"] == 3
    assert summary["critical"] == 0


def test_summarize_unknown_severity_falls_back_to_low() -> None:
    summary = security_scan._summarize([{"severity": "BANANA"}])
    # Unknown severity labels are counted as LOW.
    assert summary["total"] == 1
    assert summary["max_severity"] == "LOW"
    assert summary["max_severity_value"] == 1


def test_gitleaks_clean_returns_no_findings(security_context: CommandContext) -> None:
    """gitleaks exit code 0 means no secrets — scanner returns an empty list."""
    security_context.config.security.tools = ["gitleaks"]
    with (
        patch("shutil.which", return_value="/usr/bin/gitleaks"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""
        # HIGH threshold + no findings → must NOT raise.
        security_scan.plugin.pre_commit(security_context)


def test_osv_scanner_bad_json_returns_empty(security_context: CommandContext) -> None:
    """Malformed osv-scanner stdout is swallowed and yields no findings."""
    security_context.config.security.tools = ["osv-scanner"]
    with (
        patch("shutil.which", return_value="/usr/bin/osv-scanner"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value.stdout = "not-json"
        # No findings → no error raised even at LOW threshold.
        security_scan.plugin.pre_commit(security_context)


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (9.5, "CRITICAL"),
        (7.0, "HIGH"),
        (4.0, "MEDIUM"),
        (3.9, "LOW"),
        (0.0, "LOW"),
    ],
)
def test_score_to_severity_thresholds(score: float, expected: str) -> None:
    assert security_scan._score_to_severity(score) == expected


def test_cvss_v3_base_score_incomplete_vector_returns_none() -> None:
    # Missing several required metrics (AC, PR, UI, S, C, I, A).
    assert security_scan._cvss_v3_base_score("CVSS:3.1/AV:N") is None


def test_cvss_v3_base_score_invalid_metric_value_returns_none() -> None:
    # All required keys present but AV has an unsupported value.
    vector = "CVSS:3.1/AV:Q/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
    assert security_scan._cvss_v3_base_score(vector) is None


def test_cvss_v3_base_score_scope_changed_critical() -> None:
    """A scope-changed (S:C) vector exercises the alternate impact formula."""
    vector = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H"
    score = security_scan._cvss_v3_base_score(vector)
    assert score is not None
    assert score >= 9.0
    assert security_scan._score_to_severity(score) == "CRITICAL"


def test_cvss_v3_base_score_zero_impact_returns_zero() -> None:
    """All-zero confidentiality/integrity/availability yields a 0.0 score."""
    vector = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:N"
    assert security_scan._cvss_v3_base_score(vector) == 0.0


def test_osv_severity_string_is_cvss_vector() -> None:
    """A bare CVSS vector string is parsed and scored."""
    vuln = {"severity": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"}
    assert security_scan._osv_severity(vuln) == "CRITICAL"


def test_osv_severity_string_unknown_vector_returns_medium() -> None:
    vuln = {"severity": "CVSS:3.1/AV:Q"}
    assert security_scan._osv_severity(vuln) == "MEDIUM"


def test_osv_severity_list_of_string_labels() -> None:
    vuln = {"severity": ["HIGH", "LOW"]}
    assert security_scan._osv_severity(vuln) == "HIGH"


def test_osv_severity_list_with_non_dict_non_string_entry() -> None:
    """Non-string, non-dict list entries are skipped without crashing."""
    vuln = {"severity": [123, 4.5, None]}
    assert security_scan._osv_severity(vuln) == "MEDIUM"


def test_osv_severity_list_dict_with_string_label_score() -> None:
    """A dict whose ``score`` is a qualitative label is recognised directly."""
    vuln = {"severity": [{"type": "CVSS_V3", "score": "CRITICAL"}]}
    assert security_scan._osv_severity(vuln) == "CRITICAL"


def test_osv_severity_list_dict_with_type_label_fallback() -> None:
    """When ``score`` is unusable, ``type`` is consulted as a severity label."""
    vuln = {"severity": [{"type": "HIGH"}]}
    assert security_scan._osv_severity(vuln) == "HIGH"


def test_osv_severity_empty_list_returns_medium() -> None:
    assert security_scan._osv_severity({"severity": []}) == "MEDIUM"
