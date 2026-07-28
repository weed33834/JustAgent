"""Official security-scan plugin for JustAgent-CLI.

Runs lightweight security checks during the ``pre_commit`` hook. Tools are
invoked only when present on PATH, so the plugin degrades gracefully on
systems where ``semgrep``, ``gitleaks`` or ``osv-scanner`` are not installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from decimal import ROUND_UP, Decimal
from typing import Any, cast

import structlog

from justagent.core.context import CommandContext
from justagent.exceptions import SecurityScanError
from justagent.hookspec import hookimpl
from justagent.models.config import SecurityThreshold

logger = structlog.get_logger("justagent")

_SEVERITY_ORDER = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


class SecurityScanPlugin:
    """Built-in security scanner invoked before each commit."""

    @hookimpl
    def pre_commit(self, context: CommandContext) -> None:
        """Run configured security scanners and abort if issues exceed threshold."""
        config = context.config.security
        if not config.enabled:
            return

        findings: list[dict[str, Any]] = []
        for tool in config.tools:
            tool_findings = _run_scanner(tool, context.project_root, config.threshold)
            findings.extend(tool_findings)

        summary = _summarize(findings)
        logger.info(
            "Security scan completed: %s findings (%s critical, %s high, %s medium, %s low)",
            summary["total"],
            summary["critical"],
            summary["high"],
            summary["medium"],
            summary["low"],
        )

        if summary["max_severity_value"] >= _SEVERITY_ORDER[config.threshold.value.upper()]:
            message = (
                f"Security scan found {summary['total']} issue(s) "
                f"at or above '{config.threshold.value}' threshold."
            )
            raise SecurityScanError(
                message,
                details={"findings": findings, "summary": summary},
            )


def _run_scanner(
    tool: str, project_root: Any, threshold: SecurityThreshold
) -> list[dict[str, Any]]:
    """Dispatch to the requested scanner if it is available on PATH."""
    executable = shutil.which(tool)
    if executable is None:
        logger.warning("Security scanner %r not found on PATH; skipping.", tool)
        return []

    if tool == "semgrep":
        return _scan_semgrep(executable, project_root, threshold)
    if tool == "bandit":
        return _scan_bandit(executable, project_root)
    if tool == "gitleaks":
        return _scan_gitleaks(executable, project_root)
    if tool == "osv-scanner":
        return _scan_osv(executable, project_root)

    logger.warning("Unsupported security scanner %r; skipping.", tool)
    return []


def _scan_semgrep(
    executable: str, project_root: Any, threshold: SecurityThreshold
) -> list[dict[str, Any]]:
    """Run semgrep and return normalized findings."""
    severity_flag = {
        SecurityThreshold.LOW: "WARNING",
        SecurityThreshold.MEDIUM: "WARNING",
        SecurityThreshold.HIGH: "ERROR",
    }[threshold]

    result = subprocess.run(
        [
            executable,
            "scan",
            "--config",
            "auto",
            "--json",
            "--severity",
            severity_flag,
            "--quiet",
            ".",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    findings: list[dict[str, Any]] = []
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return findings

    for finding in data.get("results", []):
        extra = finding.get("extra", {})
        findings.append(
            {
                "tool": "semgrep",
                "severity": extra.get("severity", "WARNING").upper(),
                "confidence": extra.get("metadata", {}).get("confidence", "MEDIUM"),
                "summary": extra.get("message", ""),
                "filename": finding.get("path"),
                "line": finding.get("start", {}).get("line"),
            }
        )
    return findings


def _scan_bandit(executable: str, project_root: Any) -> list[dict[str, Any]]:
    """Run bandit and return normalized findings."""
    result = subprocess.run(
        [executable, "-r", ".", "-f", "json", "--quiet"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    findings: list[dict[str, Any]] = []
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return findings

    for issue in data.get("results", []):
        findings.append(
            {
                "tool": "bandit",
                "severity": issue.get("issue_severity", "LOW").upper(),
                "confidence": issue.get("issue_confidence", "MEDIUM").upper(),
                "summary": issue.get("issue_text", ""),
                "filename": issue.get("filename"),
                "line": issue.get("line_number"),
            }
        )
    return findings


def _scan_gitleaks(executable: str, project_root: Any) -> list[dict[str, Any]]:
    """Run gitleaks and return normalized findings."""
    result = subprocess.run(
        [executable, "detect", "--source", ".", "--no-banner", "--verbose"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        return []

    return [
        {
            "tool": "gitleaks",
            "severity": "HIGH",
            "summary": "Potential secret detected by gitleaks",
            "details": result.stdout or result.stderr,
        }
    ]


def _scan_osv(executable: str, project_root: Any) -> list[dict[str, Any]]:
    """Run osv-scanner and return normalized findings."""
    result = subprocess.run(
        [executable, "--format", "json", "-r", "."],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    findings: list[dict[str, Any]] = []
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return findings

    for entry in data.get("results", []):
        for pkg in entry.get("packages", []):
            for vuln in pkg.get("vulnerabilities", []):
                findings.append(
                    {
                        "tool": "osv-scanner",
                        "severity": _osv_severity(vuln),
                        "summary": vuln.get("summary", "OSV vulnerability"),
                        "id": vuln.get("id"),
                    }
                )
    return findings


def _score_to_severity(score: float) -> str:
    """Map a CVSS base score to a qualitative severity label."""
    if score >= 9.0:
        return "CRITICAL"
    if score >= 7.0:
        return "HIGH"
    if score >= 4.0:
        return "MEDIUM"
    return "LOW"


def _cvss_v3_base_score(vector: str) -> float | None:
    """Compute a CVSS v3 base score from a vector string.

    Supports both ``CVSS:3.0`` and ``CVSS:3.1`` vectors. Returns ``None`` when
    the vector is missing required metrics or uses unsupported values.
    """
    metrics: dict[str, str] = {}
    for part in vector.split("/"):
        if ":" in part:
            key, value = part.split(":", 1)
            metrics[key] = value

    required = {"AV", "AC", "PR", "UI", "S", "C", "I", "A"}
    if not required.issubset(metrics):
        return None

    av_map = {
        "N": Decimal("0.85"),
        "A": Decimal("0.62"),
        "L": Decimal("0.55"),
        "P": Decimal("0.2"),
    }
    ac_map = {"L": Decimal("0.77"), "H": Decimal("0.44")}
    pr_unchanged = {"N": Decimal("0.85"), "L": Decimal("0.62"), "H": Decimal("0.27")}
    pr_changed = {"N": Decimal("0.85"), "L": Decimal("0.68"), "H": Decimal("0.5")}
    ui_map = {"N": Decimal("0.85"), "R": Decimal("0.62")}
    cia_map = {"H": Decimal("0.56"), "L": Decimal("0.22"), "N": Decimal("0")}

    av = av_map.get(metrics["AV"])
    ac = ac_map.get(metrics["AC"])
    scope_changed = metrics["S"] == "C"
    pr = (pr_changed if scope_changed else pr_unchanged).get(metrics["PR"])
    ui = ui_map.get(metrics["UI"])
    c = cia_map.get(metrics["C"])
    i = cia_map.get(metrics["I"])
    a = cia_map.get(metrics["A"])
    if any(metric is None for metric in (av, ac, pr, ui, c, i, a)):
        return None
    av, ac, pr, ui, c, i, a = cast(
        tuple[Decimal, Decimal, Decimal, Decimal, Decimal, Decimal, Decimal],
        (av, ac, pr, ui, c, i, a),
    )

    iss = Decimal("1") - (Decimal("1") - c) * (Decimal("1") - i) * (Decimal("1") - a)
    if scope_changed:
        impact = (
            Decimal("7.52") * (iss - Decimal("0.029"))
            - Decimal("3.25") * (iss - Decimal("0.02")) ** 15
        )
    else:
        impact = Decimal("6.42") * iss

    exploitability = Decimal("8.22") * av * ac * pr * ui

    if impact <= 0:
        score = Decimal("0")
    elif scope_changed:
        score = min(Decimal("1.08") * (impact + exploitability), Decimal("10"))
    else:
        score = min(impact + exploitability, Decimal("10"))

    return float(score.quantize(Decimal("0.1"), rounding=ROUND_UP))


def _osv_severity(vuln: dict[str, Any]) -> str:
    """Extract the highest CVSS severity from an OSV vulnerability record."""
    severities = vuln.get("severity")

    if isinstance(severities, str):
        label = severities.strip().upper()
        if label in _SEVERITY_ORDER:
            return label
        base = _cvss_v3_base_score(severities)
        if base is not None:
            return _score_to_severity(base)
        return "MEDIUM"

    if not isinstance(severities, list) or not severities:
        return "MEDIUM"

    highest = 0
    severity_list: list[Any] = cast(list[Any], severities)
    entry: Any
    for entry in severity_list:
        if isinstance(entry, str):
            label = entry.strip().upper()
            if label in _SEVERITY_ORDER:
                highest = max(highest, _SEVERITY_ORDER[label])
            continue
        if not isinstance(entry, dict):
            continue

        entry_dict = cast(dict[str, Any], entry)
        score: Any = entry_dict.get("score")
        if isinstance(score, str):
            label = score.strip().upper()
            if label in _SEVERITY_ORDER:
                highest = max(highest, _SEVERITY_ORDER[label])
                continue
            base = _cvss_v3_base_score(score)
            if base is not None:
                highest = max(highest, _SEVERITY_ORDER[_score_to_severity(base)])
                continue

        type_label: Any = entry_dict.get("type")
        if isinstance(type_label, str):
            label = type_label.strip().upper()
            if label in _SEVERITY_ORDER:
                highest = max(highest, _SEVERITY_ORDER[label])

    if highest:
        for label, value in _SEVERITY_ORDER.items():
            if value == highest:
                return label
    return "MEDIUM"


def _summarize(findings: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate findings into a simple severity summary."""
    counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for finding in findings:
        severity = finding.get("severity", "LOW").upper()
        if severity in counts:
            counts[severity] += 1
        else:
            counts["LOW"] += 1

    max_value = max(
        (_SEVERITY_ORDER.get(severity, 0) for severity in counts if counts[severity] > 0),
        default=0,
    )
    max_severity = next(
        (label for label, value in _SEVERITY_ORDER.items() if value == max_value),
        "LOW",
    )

    return {
        "total": len(findings),
        "low": counts["LOW"],
        "medium": counts["MEDIUM"],
        "high": counts["HIGH"],
        "critical": counts["CRITICAL"],
        "max_severity": max_severity,
        "max_severity_value": max_value,
    }


plugin = SecurityScanPlugin()
