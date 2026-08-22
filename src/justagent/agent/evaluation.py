"""Evaluation layer — assess the quality, correctness and effectiveness of agent outputs.

This module implements the **Evaluation** layer of the JustAgent enterprise
AI agent architecture. It sits *after* generation and *before* delivery,
providing systematic assessment of agent-produced text, code, and legal
documents against configurable criteria.

The layer is split into four cooperating subsystems:

* **Evaluation criteria** — :class:`EvaluationCriterion`,
  :class:`CriterionWeight`, and :class:`CriterionSet` define *what* to
  evaluate and how much each dimension matters. Pre-built sets are
  provided for domain document generation, review workflows, case
  analysis, coding tasks, and general-purpose chat.

* **Evaluation results** — :class:`EvaluationScore` and
  :class:`EvaluationResult` capture the structured outcome of an
  evaluation run, including per-criterion scores, an overall weighted
  score, free-form feedback, and actionable recommendations.

* **Evaluators** — two complementary strategies:

  - :class:`LLMJudgeEvaluator` uses a :class:`ModelGateway` to perform
    LLM-as-a-judge evaluation. Supports single-output scoring,
    pairwise comparison, and rubric-based evaluation with structured
    JSON output parsing.

  - :class:`RuleBasedEvaluator` performs fast, deterministic checks:
    safety (harmful-content and jailbreak-attempt detection), legal
    compliance (citation-format verification and PII leakage detection),
    and code quality (syntax validation, style linting, test-coverage
    hints).

* **Pipeline & registry** — :class:`EvaluationPipeline` orchestrates
  multiple evaluators with weighted aggregation and configurable
  pass/fail thresholds; :class:`EvaluationRegistry` is a thread-safe
  store for criterion sets with built-in defaults for the domain-document
  domain.

All data structures are Pydantic models; all mutable registries are
guarded by :class:`threading.RLock`; LLM calls expose async variants.
The :class:`ModelGateway` is imported lazily so the module can be
imported without a configured backend.

Design::

    ┌─────────────────────────────────────────────────────┐
    │                   EvaluationPipeline                 │
    │  register_evaluator → evaluate → aggregate_scores    │
    ├──────────────────────┬──────────────────────────────┤
    │  LLMJudgeEvaluator   │     RuleBasedEvaluator        │
    │  (ModelGateway)      │  (regex / heuristic / ast)    │
    ├──────────────────────┴──────────────────────────────┤
    │               EvaluationRegistry                     │
    │  criterion sets: document / coding / general / ...   │
    └─────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import re
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from justagent.adapters.model_gateway import (
        ModelGateway,
    )

logger = logging.getLogger("justagent.agent.evaluation")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EvaluationError(Exception):
    """Raised for invalid evaluation configurations or evaluation failures."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EvaluationCriterion(str, Enum):  # noqa: UP042 - match existing codebase style
    """Dimensions along which agent outputs are assessed.

    Attributes:
        ACCURACY: Factual correctness — claims, citations, and code
            behaviour match ground truth.
        COMPLETENESS: All requested aspects of the task are addressed;
            no material omissions.
        RELEVANCE: The output stays on-topic and addresses the user's
            actual intent.
        COHERENCE: Internal logical consistency, clear structure, and
            readable flow.
        SAFETY: The output is free from harmful, dangerous, or
            policy-violating content.
        LEGAL_COMPLIANCE: Citations are valid, formatting conforms to
            legal-document standards, and no regulated data is leaked.
        EFFICIENCY: The output is produced and delivered with
            appropriate conciseness and resource use.
        HELPFULNESS: The output genuinely advances the user's goal and
            provides actionable value.
    """

    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    SAFETY = "safety"
    LEGAL_COMPLIANCE = "legal_compliance"
    EFFICIENCY = "efficiency"
    HELPFULNESS = "helpfulness"


class EvaluationStatus(str, Enum):  # noqa: UP042
    """Lifecycle status of an evaluation run.

    Attributes:
        PENDING: Evaluation queued but not yet started.
        IN_PROGRESS: Evaluation is currently running.
        COMPLETED: Evaluation finished successfully.
        FAILED: Evaluation could not be completed (error or timeout).
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluatorType(str, Enum):  # noqa: UP042
    """Who or what produced an evaluation result.

    Attributes:
        HUMAN: A human reviewer assessed the output.
        LLM: An LLM-as-a-judge evaluator assessed the output.
        AUTO: A rule-based / heuristic evaluator assessed the output.
    """

    HUMAN = "human"
    LLM = "llm"
    AUTO = "auto"


class EvaluationMode(str, Enum):  # noqa: UP042
    """Evaluation strategy for the :class:`LLMJudgeEvaluator`.

    Attributes:
        SINGLE: Score a single output against the criteria.
        PAIRWISE: Compare two outputs and pick the better one.
        RUBRIC: Score a single output against a detailed custom rubric.
    """

    SINGLE = "single"
    PAIRWISE = "pairwise"
    RUBRIC = "rubric"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> float:
    """Return the current Unix timestamp."""

    return time.time()


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp *value* into the ``[low, high]`` range."""

    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# Fallback dataclasses for ModelGateway compatibility
# ---------------------------------------------------------------------------


@dataclass
class _FallbackChatMessage:
    """Minimal stand-in for ``ChatMessage`` when the gateway module is unavailable.

    Mirrors the field layout of
    :class:`justagent.adapters.model_gateway.ChatMessage` so any gateway
    that accepts that type will also accept this one (duck typing).
    """

    role: str
    content: str


@dataclass
class _FallbackChatCompletionRequest:
    """Minimal stand-in for ``ChatCompletionRequest`` (duck-typed).

    Mirrors the field layout of
    :class:`justagent.adapters.model_gateway.ChatCompletionRequest`.
    """

    messages: list[_FallbackChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False


# ---------------------------------------------------------------------------
# Evaluation criteria models
# ---------------------------------------------------------------------------


class CriterionWeight(BaseModel):
    """A single criterion paired with its normalised weight.

    Attributes:
        criterion: The :class:`EvaluationCriterion` being weighted.
        weight: Importance in the range ``[0.0, 1.0]``. Weights within
            a :class:`CriterionSet` are normalised at aggregation time
            so they need not sum to 1.
    """

    criterion: EvaluationCriterion
    weight: float = Field(default=1.0)

    @field_validator("weight")
    @classmethod
    def _validate_weight(cls, v: float) -> float:
        """Clamp the weight into [0, 1]."""

        return _clamp(v)


class CriterionSet(BaseModel):
    """A named, reusable collection of weighted evaluation criteria.

    Attributes:
        id: Unique identifier (auto-generated UUID4 hex).
        name: Human-readable name (e.g. ``"domain_document"``).
        description: What the set is designed for.
        weights: List of :class:`CriterionWeight` entries. The set of
            criteria covered is derived from this list.
        created_at: Unix timestamp of creation.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    description: str = ""
    weights: list[CriterionWeight] = Field(default_factory=list)
    created_at: float = Field(default_factory=_now)
    _weight_cache: dict[EvaluationCriterion, float] | None = None

    @property
    def criteria(self) -> list[EvaluationCriterion]:
        """The criteria covered by this set (in declaration order)."""

        return [w.criterion for w in self.weights]

    def weight_for(self, criterion: EvaluationCriterion) -> float:
        """Return the weight for *criterion*, or ``0.0`` if absent."""

        if self._weight_cache is None:
            self._weight_cache = {w.criterion: w.weight for w in self.weights}
        return self._weight_cache.get(criterion, 0.0)

    def total_weight(self) -> float:
        """Return the sum of all criterion weights."""

        return sum(w.weight for w in self.weights)


# ---------------------------------------------------------------------------
# Evaluation result models
# ---------------------------------------------------------------------------


class EvaluationScore(BaseModel):
    """A score for a single criterion.

    Attributes:
        criterion: The :class:`EvaluationCriterion` scored.
        score: Numeric score in ``[0.0, 1.0]`` where 1.0 is perfect.
        reasoning: Free-form explanation of *why* this score was given.
        evidence: Concrete snippets or references from the output that
            justify the score (e.g. quoted text, line numbers).
    """

    criterion: EvaluationCriterion
    score: float = Field(default=0.0)
    reasoning: str = ""
    evidence: list[str] = Field(default_factory=list)

    @field_validator("score")
    @classmethod
    def _validate_score(cls, v: float) -> float:
        """Clamp the score into [0, 1]."""

        return _clamp(v)


class EvaluationResult(BaseModel):
    """The complete outcome of an evaluation run.

    Attributes:
        id: Unique result identifier (auto-generated UUID4 hex).
        task_id: Identifier of the task / agent run being evaluated.
        status: Current :class:`EvaluationStatus`.
        scores: Per-criterion :class:`EvaluationScore` entries.
        overall_score: Weighted aggregate score in ``[0.0, 1.0]``.
        passed: Whether ``overall_score`` meets the pass threshold.
        feedback: Summarised human- or LLM-readable feedback.
        recommendations: Actionable improvement suggestions.
        evaluator: The :class:`EvaluatorType` that produced this result.
        evaluator_name: Name/identifier of the specific evaluator
            instance (e.g. ``"rule_based"`` or a model name).
        timestamp: Unix timestamp of when the result was finalised.
        metadata: Arbitrary structured metadata (e.g. token usage).
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    task_id: str = ""
    status: EvaluationStatus = EvaluationStatus.PENDING
    scores: list[EvaluationScore] = Field(default_factory=list)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    passed: bool = False
    feedback: str = ""
    recommendations: list[str] = Field(default_factory=list)
    evaluator: EvaluatorType = EvaluatorType.AUTO
    evaluator_name: str = ""
    timestamp: float = Field(default_factory=_now)
    metadata: dict[str, Any] = Field(default_factory=dict)
    _score_cache: dict[EvaluationCriterion, EvaluationScore] | None = None

    def score_for(self, criterion: EvaluationCriterion) -> EvaluationScore | None:
        """Return the score entry for *criterion*, or ``None``."""

        if self._score_cache is None:
            self._score_cache = {s.criterion: s for s in self.scores}
        return self._score_cache.get(criterion)


# ---------------------------------------------------------------------------
# Pre-defined criterion sets
# ---------------------------------------------------------------------------


def _domain_document_criterion_set() -> CriterionSet:
    """Criterion set for domain document generation.

    Emphasises legal compliance (citation validity, format), accuracy,
    and completeness — the dimensions that matter most for formal legal
    documents such as indictments, judgments, and legal opinions.
    """

    return CriterionSet(
        name="domain_document",
        description="Criteria for evaluating generated documents.",
        weights=[
            CriterionWeight(criterion=EvaluationCriterion.LEGAL_COMPLIANCE, weight=1.0),
            CriterionWeight(criterion=EvaluationCriterion.ACCURACY, weight=0.9),
            CriterionWeight(criterion=EvaluationCriterion.COMPLETENESS, weight=0.8),
            CriterionWeight(criterion=EvaluationCriterion.COHERENCE, weight=0.6),
            CriterionWeight(criterion=EvaluationCriterion.SAFETY, weight=0.7),
            CriterionWeight(criterion=EvaluationCriterion.RELEVANCE, weight=0.6),
            CriterionWeight(criterion=EvaluationCriterion.HELPFULNESS, weight=0.4),
            CriterionWeight(criterion=EvaluationCriterion.EFFICIENCY, weight=0.2),
        ],
    )


def _coding_criterion_set() -> CriterionSet:
    """Criterion set for code-generation tasks.

    Prioritises accuracy (correct behaviour), safety (no insecure
    patterns), and efficiency (clean, performant code)."""

    return CriterionSet(
        name="coding",
        description="Criteria for evaluating generated code.",
        weights=[
            CriterionWeight(criterion=EvaluationCriterion.ACCURACY, weight=1.0),
            CriterionWeight(criterion=EvaluationCriterion.SAFETY, weight=0.8),
            CriterionWeight(criterion=EvaluationCriterion.COMPLETENESS, weight=0.7),
            CriterionWeight(criterion=EvaluationCriterion.EFFICIENCY, weight=0.6),
            CriterionWeight(criterion=EvaluationCriterion.COHERENCE, weight=0.5),
            CriterionWeight(criterion=EvaluationCriterion.HELPFULNESS, weight=0.4),
            CriterionWeight(criterion=EvaluationCriterion.RELEVANCE, weight=0.4),
        ],
    )


def _general_criterion_set() -> CriterionSet:
    """Criterion set for general-purpose chat / Q&A tasks."""

    return CriterionSet(
        name="general",
        description="General-purpose criteria for evaluating agent chat responses.",
        weights=[
            CriterionWeight(criterion=EvaluationCriterion.HELPFULNESS, weight=1.0),
            CriterionWeight(criterion=EvaluationCriterion.ACCURACY, weight=0.9),
            CriterionWeight(criterion=EvaluationCriterion.RELEVANCE, weight=0.8),
            CriterionWeight(criterion=EvaluationCriterion.COMPLETENESS, weight=0.7),
            CriterionWeight(criterion=EvaluationCriterion.COHERENCE, weight=0.6),
            CriterionWeight(criterion=EvaluationCriterion.SAFETY, weight=0.7),
            CriterionWeight(criterion=EvaluationCriterion.EFFICIENCY, weight=0.3),
        ],
    )


def _evidence_review_criterion_set() -> CriterionSet:
    """Criterion set for evidence-review outputs.

    Evidence review demands high accuracy (correct identification of
    evidence types and admissibility) and completeness (no evidence
    overlooked)."""

    return CriterionSet(
        name="evidence_review",
        description="Criteria for evaluating evidence-review analysis outputs.",
        weights=[
            CriterionWeight(criterion=EvaluationCriterion.ACCURACY, weight=1.0),
            CriterionWeight(criterion=EvaluationCriterion.COMPLETENESS, weight=0.9),
            CriterionWeight(criterion=EvaluationCriterion.LEGAL_COMPLIANCE, weight=0.8),
            CriterionWeight(criterion=EvaluationCriterion.COHERENCE, weight=0.5),
            CriterionWeight(criterion=EvaluationCriterion.SAFETY, weight=0.6),
            CriterionWeight(criterion=EvaluationCriterion.RELEVANCE, weight=0.5),
            CriterionWeight(criterion=EvaluationCriterion.HELPFULNESS, weight=0.4),
        ],
    )


def _case_analysis_criterion_set() -> CriterionSet:
    """Criterion set for case-analysis outputs.

        Case analysis must be accurate (correct application of law to
    facts), coherent (logical reasoning chain), and legally compliant."""

    return CriterionSet(
        name="case_analysis",
        description="Criteria for evaluating case-analysis and legal-reasoning outputs.",
        weights=[
            CriterionWeight(criterion=EvaluationCriterion.ACCURACY, weight=1.0),
            CriterionWeight(criterion=EvaluationCriterion.COHERENCE, weight=0.9),
            CriterionWeight(criterion=EvaluationCriterion.LEGAL_COMPLIANCE, weight=0.8),
            CriterionWeight(criterion=EvaluationCriterion.COMPLETENESS, weight=0.7),
            CriterionWeight(criterion=EvaluationCriterion.RELEVANCE, weight=0.6),
            CriterionWeight(criterion=EvaluationCriterion.HELPFULNESS, weight=0.5),
            CriterionWeight(criterion=EvaluationCriterion.SAFETY, weight=0.5),
        ],
    )


def _default_criterion_sets() -> dict[str, CriterionSet]:
    """Build all built-in default criterion sets keyed by name."""

    sets = [
        _domain_document_criterion_set(),
        _coding_criterion_set(),
        _general_criterion_set(),
        _evidence_review_criterion_set(),
        _case_analysis_criterion_set(),
    ]
    return {s.name: s for s in sets}


# ---------------------------------------------------------------------------
# Rule patterns for the RuleBasedEvaluator
# ---------------------------------------------------------------------------

#: Regex to extract statute citations like 《民法典》第143条 or 《刑法》第二百六十四条.
_CITATION_RE = re.compile(
    r"《([^》]+)》\s*(第[一二三四五六七八九十百千零\d]+条(?:之[一二三四五六七八九十\d]+)?)"
)

#: Email address pattern.
_PII_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

#: Chinese mobile phone number pattern (11 digits starting with 1).
_PII_PHONE_CN_RE = re.compile(r"\b1[3-9]\d{9}\b")

#: Chinese ID card number pattern (18 digits, last may be X).
_PII_ID_CARD_RE = re.compile(r"\b\d{17}[\dXx]\b")

#: Bank card number pattern (16-19 digits).
_PII_BANK_CARD_RE = re.compile(r"\b\d{16,19}\b")

#: International phone number pattern.
_PII_PHONE_INTL_RE = re.compile(r"\+\d{1,3}[\s-]?\d{4,14}")

#: Patterns indicating harmful / violent content.
_HARMFUL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("violence", re.compile(r"\b(kill|murder|assassinate|slaughter|bomb|massacre)\b", re.I)),
    ("self_harm", re.compile(r"\b(suicide|self[- ]?harm|kill myself|end my life)\b", re.I)),
    (
        "weapon_synthesis",
        re.compile(
            r"\b(how to make|synthesize|manufacture)\s+(bomb|weapon|explosive|poison)\b", re.I
        ),
    ),
    (
        "drug_synthesis",
        re.compile(r"\b(synthesize|manufacture|produce)\s+(meth|heroin|cocaine|fentanyl)\b", re.I),
    ),
    ("illegal_activity", re.compile(r"\b(how to (hack|steal|launder|smuggle|forge))\b", re.I)),
    ("hate_speech", re.compile(r"\b(subhuman|vermin|inferior race|ethnic cleansing)\b", re.I)),
]

#: Patterns indicating jailbreak / prompt-injection attempts.
_JAILBREAK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "ignore_instructions",
        re.compile(
            r"\b(ignore|disregard|forget)\b.{0,30}\b(previous|prior|above|all)\b.{0,30}\b(instruction|prompt|rule|direction)\b",
            re.I,
        ),
    ),
    (
        "role_override",
        re.compile(
            r"\byou are now\b.{0,40}\b(DAN|developer mode|jailbroken|unrestricted|no restrictions)\b",
            re.I,
        ),
    ),
    (
        "system_prompt_extraction",
        re.compile(
            r"\b(reveal|show|repeat|print|output)\b.{0,30}\b(system prompt|initial prompt|hidden instruction)\b",
            re.I,
        ),
    ),
    (
        "pretend_mode",
        re.compile(
            r"\b(pretend|act as|simulate)\b.{0,40}\b(no rules|no restrictions|unfiltered|unlimited)\b",
            re.I,
        ),
    ),
    (
        "encoding_bypass",
        re.compile(
            r"\b(base64|decode|rot13|hex decode)\b.{0,30}\b(prompt|instruction|command)\b", re.I
        ),
    ),
]


# ---------------------------------------------------------------------------
# Evaluator protocol (structural typing)
# ---------------------------------------------------------------------------


#: Structural protocol for any evaluator that can be registered with the
#: :class:`EvaluationPipeline`. An evaluator takes the agent output text,
#: optional task context, and a list of criteria, and returns an
#: :class:`EvaluationResult`.
Evaluator = Callable[..., EvaluationResult]

#: Async counterpart.
AsyncEvaluator = Callable[..., Awaitable[EvaluationResult]]


# ---------------------------------------------------------------------------
# Rule-based evaluator
# ---------------------------------------------------------------------------


class RuleBasedEvaluator:
    """Fast, deterministic evaluator using pattern-matching and heuristics.

    Performs three families of checks:

    * **Safety** — scans for harmful-content patterns (violence,
      self-harm, weapon/drug synthesis, illegal activity, hate speech)
      and jailbreak / prompt-injection attempts.
    * **Legal compliance** — validates statute-citation format (``《法
      律》第X条``), detects PII leakage (email, phone, ID card, bank
      card), and checks basic legal-document structure.
    * **Code quality** — validates Python syntax via :func:`ast.parse`,
      checks for common style issues (long lines, trailing whitespace,
      bare ``except``), and hints at missing test coverage.

    Each check produces :class:`EvaluationScore` objects in the
    ``[0, 1]`` range. A perfect output scores 1.0; each detected issue
    deducts from the score.

    Example::

        >>> evaluator = RuleBasedEvaluator()
        >>> result = evaluator.evaluate("Hello, world!")
        >>> result.evaluator is EvaluatorType.AUTO
        True
        >>> result.overall_score > 0
        True
    """

    def __init__(
        self,
        *,
        enable_safety: bool = True,
        enable_legal: bool = True,
        enable_code: bool = True,
        max_line_length: int = 100,
    ) -> None:
        """Initialise the rule-based evaluator.

        Args:
            enable_safety: Whether to run safety checks.
            enable_legal: Whether to run legal-compliance checks.
            enable_code: Whether to run code-quality checks.
            max_line_length: Threshold for the long-line style check.
        """

        self._enable_safety = enable_safety
        self._enable_legal = enable_legal
        self._enable_code = enable_code
        self._max_line_length = max_line_length
        self._name = "rule_based"

    @property
    def name(self) -> str:
        """The evaluator's identifier."""

        return self._name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        output: str,
        *,
        task_input: str | None = None,
        criteria: list[EvaluationCriterion] | None = None,
        task_id: str = "",
    ) -> EvaluationResult:
        """Run all enabled rule-based checks on *output*.

        Args:
            output: The agent output text to evaluate.
            task_input: The original user prompt (used for context).
            criteria: Optional subset of criteria to evaluate. If
                ``None``, all applicable criteria are checked.
            task_id: Optional task identifier for the result.

        Returns:
            An :class:`EvaluationResult` with per-criterion scores,
            aggregated overall score, feedback, and recommendations.
        """

        if not output:
            return self._empty_result(task_id)

        requested = criteria or list(EvaluationCriterion)
        scores: list[EvaluationScore] = []

        if self._enable_safety and (
            EvaluationCriterion.SAFETY in requested
            or EvaluationCriterion.LEGAL_COMPLIANCE in requested
        ):
            safety_score = self.check_safety(output, task_input)
            if EvaluationCriterion.SAFETY in requested:
                scores.append(safety_score)

        if self._enable_legal and EvaluationCriterion.LEGAL_COMPLIANCE in requested:
            legal_score = self.check_legal_compliance(output, task_input)
            scores.append(legal_score)

        if self._enable_code and EvaluationCriterion.ACCURACY in requested:
            code_score = self.check_code_quality(output)
            scores.append(code_score)

        # Ensure at least one score so the result is meaningful.
        if not scores:
            scores.append(
                EvaluationScore(
                    criterion=EvaluationCriterion.SAFETY,
                    score=1.0,
                    reasoning="No applicable rule-based checks were run.",
                )
            )

        overall = _clamp(sum(s.score for s in scores) / len(scores))
        recommendations = self._build_recommendations(scores)
        feedback = self._build_feedback(scores)

        return EvaluationResult(
            task_id=task_id,
            status=EvaluationStatus.COMPLETED,
            scores=scores,
            overall_score=overall,
            passed=overall >= 0.6,
            feedback=feedback,
            recommendations=recommendations,
            evaluator=EvaluatorType.AUTO,
            evaluator_name=self._name,
            timestamp=_now(),
            metadata={"checks_run": len(scores)},
        )

    async def evaluate_async(
        self,
        output: str,
        *,
        task_input: str | None = None,
        criteria: list[EvaluationCriterion] | None = None,
        task_id: str = "",
    ) -> EvaluationResult:
        """Async wrapper for :meth:`evaluate`."""

        return await asyncio.to_thread(
            self.evaluate,
            output,
            task_input=task_input,
            criteria=criteria,
            task_id=task_id,
        )

    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------

    def check_safety(
        self,
        output: str,
        task_input: str | None = None,
    ) -> EvaluationScore:
        """Scan for harmful content and jailbreak attempts.

        Deducts 0.2 per harmful-content match and 0.3 per jailbreak
        match, flooring at 0.0. Also inspects *task_input* for
        jailbreak attempts when provided.

        Args:
            output: The agent output to scan.
            task_input: The original user prompt (checked for
                injection attempts).

        Returns:
            An :class:`EvaluationScore` for
            :attr:`EvaluationCriterion.SAFETY`.
        """

        evidence: list[str] = []
        issues: list[str] = []
        score = 1.0

        # --- Harmful content in output ---
        for label, pattern in _HARMFUL_PATTERNS:
            matches = list(pattern.finditer(output))
            if matches:
                score -= 0.2 * len(matches)
                issues.append(f"harmful_content:{label} ({len(matches)} match(es))")
                for m in matches:
                    evidence.append(f"harmful [{label}]: '{m.group(0)}'")

        # --- Jailbreak attempts in output ---
        for label, pattern in _JAILBREAK_PATTERNS:
            matches = list(pattern.finditer(output))
            if matches:
                score -= 0.3 * len(matches)
                issues.append(f"jailbreak:{label} ({len(matches)} match(es))")
                for m in matches:
                    evidence.append(f"jailbreak [{label}]: '{m.group(0)}'")

        # --- Jailbreak attempts in user input ---
        if task_input:
            for label, pattern in _JAILBREAK_PATTERNS:
                matches = pattern.findall(task_input)
                if matches:
                    # The agent should *resist* these; if they appear
                    # in the output too, that's already caught above.
                    # Here we just note the attempt for context.
                    evidence.append(
                        f"jailbreak_attempt_in_input [{label}]: resisted"
                        if not pattern.search(output)
                        else f"jailbreak_attempt_in_input [{label}]: NOT resisted"
                    )

        score = _clamp(score)
        if issues:
            reasoning = f"Detected {len(issues)} safety issue(s): " + "; ".join(issues) + "."
        else:
            reasoning = "No harmful content or jailbreak patterns detected."

        return EvaluationScore(
            criterion=EvaluationCriterion.SAFETY,
            score=score,
            reasoning=reasoning,
            evidence=evidence,
        )

    # ------------------------------------------------------------------
    # Legal compliance checks
    # ------------------------------------------------------------------

    def check_legal_compliance(
        self,
        output: str,
        task_input: str | None = None,
    ) -> EvaluationScore:
        """Check citation format, PII leakage, and document structure.

        Validates that statute citations follow the ``《法律》第X条``
        format, detects PII (email, phone, ID card, bank card), and
        checks for a minimal document structure (presence of section
        headings).

        Args:
            output: The agent output to check.
            task_input: The original user prompt (unused, reserved for
                future context-aware checks).

        Returns:
            An :class:`EvaluationScore` for
            :attr:`EvaluationCriterion.LEGAL_COMPLIANCE`.
        """

        evidence: list[str] = []
        issues: list[str] = []
        score = 1.0

        # --- Citation format validation ---
        citations = _CITATION_RE.findall(output)
        if citations:
            # Citations exist and are well-formed — good.
            for law_name, article in citations:
                evidence.append(f"valid_citation: 《{law_name}》{article}")
        else:
            # Check if the text *mentions* law names without proper format.
            law_mentions = re.findall(r"《([^》]+)》", output)
            if law_mentions:
                score -= 0.2
                issues.append("malformed_citations: law names without article numbers")
                for name in law_mentions[:3]:
                    evidence.append(f"malformed_citation: 《{name}》 (missing article)")

        # --- PII detection ---
        pii_found: list[tuple[str, str]] = []
        for m in _PII_EMAIL_RE.finditer(output):
            pii_found.append(("email", m.group(0)))
        for m in _PII_PHONE_CN_RE.finditer(output):
            pii_found.append(("phone_cn", m.group(0)))
        for m in _PII_ID_CARD_RE.finditer(output):
            pii_found.append(("id_card", m.group(0)))
        for m in _PII_BANK_CARD_RE.finditer(output):
            # Avoid double-counting ID cards (18 digits).
            if not _PII_ID_CARD_RE.fullmatch(m.group(0)):
                pii_found.append(("bank_card", m.group(0)))
        for m in _PII_PHONE_INTL_RE.finditer(output):
            pii_found.append(("phone_intl", m.group(0)))

        if pii_found:
            score -= 0.3 * min(len(pii_found), 3)
            pii_types = sorted({t for t, _ in pii_found})
            issues.append(f"pii_detected: {', '.join(pii_types)} ({len(pii_found)} item(s))")
            for pii_type, value in pii_found[:5]:
                # Mask the PII in evidence.
                masked = (
                    value[:2] + "*" * (len(value) - 4) + value[-2:] if len(value) > 4 else "***"
                )
                evidence.append(f"pii [{pii_type}]: {masked}")

        # --- Document structure (heuristic) ---
        # Legal documents typically have section headings.
        if len(output) > 200:
            headings = re.findall(r"^[#一二三四五六七八九十、]+\s*\S+", output, re.MULTILINE)
            if not headings and "## " not in output:
                # No detectable structure — minor deduction.
                score -= 0.1
                issues.append("no_section_headings: long text without structure")

        score = _clamp(score)
        if issues:
            reasoning = (
                f"Legal compliance issues: {'; '.join(issues)}. "
                f"{len(citations)} valid citation(s) found."
            )
        else:
            reasoning = (
                f"No legal compliance issues. {len(citations)} valid citation(s), no PII detected."
            )

        return EvaluationScore(
            criterion=EvaluationCriterion.LEGAL_COMPLIANCE,
            score=score,
            reasoning=reasoning,
            evidence=evidence,
        )

    # ------------------------------------------------------------------
    # Code quality checks
    # ------------------------------------------------------------------

    def check_code_quality(self, output: str) -> EvaluationScore:
        """Validate Python code syntax, style, and test-coverage hints.

        Attempts to :func:`ast.parse` any Python code blocks found in
        *output* (delimited by `` ```python `` fences or inferred from
        content). Reports syntax errors, style issues (long lines,
        trailing whitespace, bare ``except``), and whether tests are
        present.

        Args:
            output: The agent output potentially containing code.

        Returns:
            An :class:`EvaluationScore` for
            :attr:`EvaluationCriterion.ACCURACY` (re-purposed for code
            correctness).
        """

        evidence: list[str] = []
        issues: list[str] = []
        score = 1.0
        code_blocks = self._extract_code_blocks(output)

        if not code_blocks:
            # No code detected — this check is not applicable.
            return EvaluationScore(
                criterion=EvaluationCriterion.ACCURACY,
                score=1.0,
                reasoning="No code blocks detected; code-quality check skipped.",
            )

        total_blocks = len(code_blocks)
        syntax_errors = 0

        for i, block in enumerate(code_blocks, start=1):
            # --- Syntax validation ---
            try:
                ast.parse(block)
                evidence.append(f"block_{i}: syntax OK ({len(block)} chars)")
            except SyntaxError as exc:
                syntax_errors += 1
                score -= 0.3
                issues.append(f"block_{i}: SyntaxError (line {exc.lineno}): {exc.msg}")
                evidence.append(f"block_{i}: SyntaxError at line {exc.lineno}: {exc.msg}")

            # --- Style checks ---
            lines = block.split("\n")
            long_lines = sum(1 for line in lines if len(line) > self._max_line_length)
            if long_lines:
                score -= 0.05 * min(long_lines, 5)
                issues.append(f"block_{i}: {long_lines} line(s) > {self._max_line_length} chars")

            trailing_ws = sum(1 for line in lines if line != line.rstrip() and line.strip())
            if trailing_ws:
                score -= 0.02 * min(trailing_ws, 5)
                issues.append(f"block_{i}: {trailing_ws} line(s) with trailing whitespace")

            # Bare except (anti-pattern).
            bare_excepts = len(re.findall(r"^\s*except\s*:", block, re.MULTILINE))
            if bare_excepts:
                score -= 0.1 * bare_excepts
                issues.append(f"block_{i}: {bare_excepts} bare 'except:' clause(s)")

        # --- Test coverage hints ---
        full_code = "\n".join(code_blocks)
        has_tests = bool(
            re.search(r"\b(def\s+test_|class\s+Test\w+|unittest|pytest|assert\s)", full_code)
        )
        if not has_tests and total_blocks > 0:
            # Only flag if the task seems to be about writing code.
            score -= 0.1
            issues.append("no_tests: code lacks test functions or assertions")
            evidence.append("hint: consider adding test cases")
        elif has_tests:
            evidence.append("tests_detected: code includes test patterns")

        score = _clamp(score)
        if syntax_errors:
            reasoning = (
                f"{syntax_errors}/{total_blocks} code block(s) have syntax errors. "
                + "; ".join(issues)
            )
        elif issues:
            reasoning = f"Code is syntactically valid but has style issues: {'; '.join(issues)}."
        else:
            reasoning = f"All {total_blocks} code block(s) passed syntax and style checks."

        return EvaluationScore(
            criterion=EvaluationCriterion.ACCURACY,
            score=score,
            reasoning=reasoning,
            evidence=evidence,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_code_blocks(output: str) -> list[str]:
        """Extract Python code from fenced blocks or infer from content.

        Recognises `` ```python `` and `` ```py `` fenced blocks. If no
        fenced blocks are found but the output looks like code (starts
        with ``def``, ``class``, ``import``, or ``from``), treats the
        whole output as a single block.
        """

        blocks: list[str] = []
        # Match ```python ... ``` or ```py ... ``` fences.
        fence_re = re.compile(
            r"```(?:python|py)\s*\n(.*?)```",
            re.DOTALL,
        )
        for m in fence_re.finditer(output):
            blocks.append(m.group(1))

        if blocks:
            return blocks

        # Heuristic: if the output looks like raw Python code.
        stripped = output.strip()
        if stripped and re.match(r"^(def |class |import |from |if __name__|#\s*!)", stripped):
            blocks.append(stripped)

        return blocks

    @staticmethod
    def _build_recommendations(scores: list[EvaluationScore]) -> list[str]:
        """Derive actionable recommendations from low-scoring criteria."""

        recs: list[str] = []
        for score in scores:
            if score.score < 0.5:
                if score.criterion is EvaluationCriterion.SAFETY:
                    recs.append(
                        "Review output for harmful content or jailbreak "
                        "vulnerabilities before delivery."
                    )
                elif score.criterion is EvaluationCriterion.LEGAL_COMPLIANCE:
                    recs.append(
                        "Fix citation formatting and remove any leaked PII from the output."
                    )
                elif score.criterion is EvaluationCriterion.ACCURACY:
                    recs.append("Address code syntax errors and style issues before delivery.")
            elif score.score < 0.8:
                if score.criterion is EvaluationCriterion.SAFETY:
                    recs.append("Minor safety concerns detected; review recommended.")
                elif score.criterion is EvaluationCriterion.LEGAL_COMPLIANCE:
                    recs.append("Minor legal-compliance issues; consider revising.")
        return recs

    @staticmethod
    def _build_feedback(scores: list[EvaluationScore]) -> str:
        """Build a concise feedback summary from the scores."""

        parts: list[str] = []
        for score in scores:
            label = score.criterion.value.replace("_", " ").title()
            parts.append(f"{label}: {score.score:.2f}")
        return "Rule-based checks — " + ", ".join(parts) + "."

    def _empty_result(self, task_id: str) -> EvaluationResult:
        """Return a result for an empty output."""

        return EvaluationResult(
            task_id=task_id,
            status=EvaluationStatus.COMPLETED,
            scores=[
                EvaluationScore(
                    criterion=EvaluationCriterion.COMPLETENESS,
                    score=0.0,
                    reasoning="Output is empty.",
                )
            ],
            overall_score=0.0,
            passed=False,
            feedback="Output is empty; cannot evaluate.",
            recommendations=["Provide a non-empty output for evaluation."],
            evaluator=EvaluatorType.AUTO,
            evaluator_name=self._name,
        )


# ---------------------------------------------------------------------------
# LLM-as-Judge evaluator
# ---------------------------------------------------------------------------


class LLMJudgeEvaluator:
    """Use an LLM (via :class:`ModelGateway`) to evaluate agent outputs.

    Supports three evaluation modes:

    * :attr:`EvaluationMode.SINGLE` — score a single output against a
      set of criteria. The LLM returns a JSON array of per-criterion
      scores with reasoning and evidence.
    * :attr:`EvaluationMode.PAIRWISE` — compare two outputs (A and B)
      for the same task and determine which is better, returning scores
      for the winner.
    * :attr:`EvaluationMode.RUBRIC` — score a single output against a
      detailed custom rubric (list of rubric items with descriptions
      and point values).

    The evaluator constructs a structured prompt, sends it to the
    gateway, and parses the JSON response into
    :class:`EvaluationScore` objects. If parsing fails, a fallback
    heuristic score is assigned.

    The :class:`ModelGateway` is imported lazily so this module can be
    imported without a configured backend.

    Example::

        >>> from justagent.adapters.model_gateway import ModelGateway
        >>> gateway = MyGateway(config)  # doctest: +SKIP
        >>> evaluator = LLMJudgeEvaluator(gateway)
        >>> result = evaluator.evaluate(
        ...     "The answer is 42.",
        ...     task_input="What is the meaning of life?",
        ...     criteria=[EvaluationCriterion.ACCURACY],
        ... )
    """

    #: System prompt instructing the LLM to act as a strict evaluator.
    _SYSTEM_PROMPT = (
        "You are an expert evaluation judge for an AI agent platform. "
        "Your task is to assess the quality of agent-generated outputs "
        "against specified criteria. You must be objective, rigorous, "
        "and provide specific evidence from the output to justify "
        "each score. Always respond with valid JSON only — no markdown, "
        "no explanations outside the JSON structure."
    )

    #: Instruction block appended to every single-output prompt.
    _SINGLE_INSTRUCTION = (
        "Evaluate the following agent output against each criterion. "
        "For each criterion, provide a score from 0.0 to 1.0 (where "
        "1.0 is perfect), a brief reasoning, and specific evidence "
        "(quoted text or references from the output).\n\n"
        "Respond with ONLY a JSON object in this exact format:\n"
        '{"scores": [{"criterion": "<criterion_name>", '
        '"score": <float>, "reasoning": "<text>", '
        '"evidence": ["<snippet1>", "<snippet2>"]}], '
        '"overall_feedback": "<summary>", '
        '"recommendations": ["<rec1>", "<rec2>"]}'
    )

    #: Instruction block for pairwise comparison.
    _PAIRWISE_INSTRUCTION = (
        "Compare Output A and Output B for the same task. Determine "
        "which output is better overall, then score the winning output "
        "against each criterion.\n\n"
        "Respond with ONLY a JSON object in this exact format:\n"
        '{"winner": "A" or "B", '
        '"scores": [{"criterion": "<criterion_name>", '
        '"score": <float>, "reasoning": "<text>", '
        '"evidence": ["<snippet>"]}], '
        '"overall_feedback": "<summary>", '
        '"recommendations": ["<rec>"]}'
    )

    #: Instruction block for rubric-based evaluation.
    _RUBRIC_INSTRUCTION = (
        "Evaluate the following agent output against the provided "
        "rubric. For each rubric item, determine whether the output "
        "meets the requirement and assign a score from 0.0 to 1.0.\n\n"
        "Respond with ONLY a JSON object in this exact format:\n"
        '{"scores": [{"criterion": "<rubric_item_name>", '
        '"score": <float>, "reasoning": "<text>", '
        '"evidence": ["<snippet>"]}], '
        '"overall_feedback": "<summary>", '
        '"recommendations": ["<rec>"]}'
    )

    def __init__(
        self,
        gateway: ModelGateway | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        evaluator_name: str = "llm_judge",
    ) -> None:
        """Initialise the LLM judge evaluator.

        Args:
            gateway: The :class:`ModelGateway` to use for LLM calls.
                May be ``None`` at construction time and set later
                via the :attr:`gateway` property.
            model: Override the model identifier (if the gateway
                supports multiple). Unused if ``None``.
            temperature: Sampling temperature for judge calls. Defaults
                to 0.0 for deterministic evaluation.
            max_tokens: Maximum tokens for the judge response.
            evaluator_name: Name for this evaluator instance.
        """

        self._gateway = gateway
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._name = evaluator_name

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def gateway(self) -> ModelGateway | None:
        """The LLM gateway, or ``None`` if not yet configured."""

        return self._gateway

    @gateway.setter
    def gateway(self, value: ModelGateway | None) -> None:
        self._gateway = value

    @property
    def name(self) -> str:
        """The evaluator's identifier."""

        return self._name

    # ------------------------------------------------------------------
    # Single-output evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        output: str,
        *,
        task_input: str | None = None,
        criteria: list[EvaluationCriterion] | None = None,
        criterion_set: CriterionSet | None = None,
        task_id: str = "",
        context: str | None = None,
    ) -> EvaluationResult:
        """Evaluate a single output via LLM-as-a-judge.

        Args:
            output: The agent output to evaluate.
            task_input: The original user prompt / task description.
            criteria: Criteria to evaluate against. If ``None``,
                derived from *criterion_set* or defaults to all
                criteria.
            criterion_set: Optional :class:`CriterionSet` for weighted
                scoring. When provided, its weights are used for
                aggregation.
            task_id: Optional task identifier.
            context: Additional context (e.g. reference answer, gold
                standard) to help the judge.

        Returns:
            An :class:`EvaluationResult` with LLM-generated scores.

        Raises:
            EvaluationError: If no gateway is configured or the LLM
                call fails irrecoverably.
        """

        if self._gateway is None:
            raise EvaluationError(
                "No ModelGateway configured; cannot perform LLM evaluation. "
                "Set the 'gateway' property before calling evaluate()."
            )

        resolved_criteria = self._resolve_criteria(criteria, criterion_set)
        prompt = self._build_single_prompt(output, task_input, resolved_criteria, context)
        raw_response = self._call_gateway(prompt)
        scores, feedback, recommendations = self._parse_judge_response(
            raw_response, resolved_criteria
        )
        overall = self._aggregate(scores, criterion_set)
        passed = overall >= 0.7

        return EvaluationResult(
            task_id=task_id,
            status=EvaluationStatus.COMPLETED,
            scores=scores,
            overall_score=overall,
            passed=passed,
            feedback=feedback,
            recommendations=recommendations,
            evaluator=EvaluatorType.LLM,
            evaluator_name=self._name,
            timestamp=_now(),
            metadata={
                "mode": EvaluationMode.SINGLE.value,
                "model": self._model or "default",
                "criteria_count": len(resolved_criteria),
                "raw_response_length": len(raw_response),
            },
        )

    async def evaluate_async(
        self,
        output: str,
        *,
        task_input: str | None = None,
        criteria: list[EvaluationCriterion] | None = None,
        criterion_set: CriterionSet | None = None,
        task_id: str = "",
        context: str | None = None,
    ) -> EvaluationResult:
        """Async wrapper for :meth:`evaluate`."""

        return await asyncio.to_thread(
            self.evaluate,
            output,
            task_input=task_input,
            criteria=criteria,
            criterion_set=criterion_set,
            task_id=task_id,
            context=context,
        )

    # ------------------------------------------------------------------
    # Pairwise comparison
    # ------------------------------------------------------------------

    def evaluate_pairwise(
        self,
        output_a: str,
        output_b: str,
        *,
        task_input: str,
        criteria: list[EvaluationCriterion] | None = None,
        criterion_set: CriterionSet | None = None,
        task_id: str = "",
    ) -> EvaluationResult:
        """Compare two outputs and score the winner.

        Args:
            output_a: The first candidate output.
            output_b: The second candidate output.
            task_input: The original task / prompt.
            criteria: Criteria for the comparison.
            criterion_set: Optional weighted criterion set.
            task_id: Optional task identifier.

        Returns:
            An :class:`EvaluationResult` scoring the winning output.
            The winner is recorded in ``metadata["winner"]``.
        """

        if self._gateway is None:
            raise EvaluationError("No ModelGateway configured; cannot perform pairwise evaluation.")

        resolved_criteria = self._resolve_criteria(criteria, criterion_set)
        prompt = self._build_pairwise_prompt(output_a, output_b, task_input, resolved_criteria)
        raw_response = self._call_gateway(prompt)
        scores, feedback, recommendations, winner = self._parse_pairwise_response(
            raw_response, resolved_criteria
        )
        overall = self._aggregate(scores, criterion_set)
        passed = overall >= 0.7

        return EvaluationResult(
            task_id=task_id,
            status=EvaluationStatus.COMPLETED,
            scores=scores,
            overall_score=overall,
            passed=passed,
            feedback=feedback,
            recommendations=recommendations,
            evaluator=EvaluatorType.LLM,
            evaluator_name=self._name,
            timestamp=_now(),
            metadata={
                "mode": EvaluationMode.PAIRWISE.value,
                "winner": winner,
                "model": self._model or "default",
            },
        )

    async def evaluate_pairwise_async(
        self,
        output_a: str,
        output_b: str,
        *,
        task_input: str,
        criteria: list[EvaluationCriterion] | None = None,
        criterion_set: CriterionSet | None = None,
        task_id: str = "",
    ) -> EvaluationResult:
        """Async wrapper for :meth:`evaluate_pairwise`."""

        return await asyncio.to_thread(
            self.evaluate_pairwise,
            output_a,
            output_b,
            task_input=task_input,
            criteria=criteria,
            criterion_set=criterion_set,
            task_id=task_id,
        )

    # ------------------------------------------------------------------
    # Rubric-based evaluation
    # ------------------------------------------------------------------

    def evaluate_rubric(
        self,
        output: str,
        *,
        rubric: list[dict[str, Any]],
        task_input: str | None = None,
        task_id: str = "",
    ) -> EvaluationResult:
        """Evaluate *output* against a custom rubric.

        Args:
            output: The agent output to evaluate.
            rubric: A list of rubric items, each a dict with keys:
                ``name`` (str), ``description`` (str), and optionally
                ``max_points`` (float, default 1.0).
            task_input: The original task / prompt.
            task_id: Optional task identifier.

        Returns:
            An :class:`EvaluationResult` with one score per rubric item.
            Criteria are mapped to the closest matching
            :class:`EvaluationCriterion` when possible, otherwise
            :attr:`EvaluationCriterion.HELPFULNESS`.
        """

        if self._gateway is None:
            raise EvaluationError("No ModelGateway configured; cannot perform rubric evaluation.")

        prompt = self._build_rubric_prompt(output, task_input, rubric)
        raw_response = self._call_gateway(prompt)
        scores, feedback, recommendations = self._parse_judge_response(
            raw_response,
            criteria=None,
            rubric_items=rubric,
        )
        overall = self._aggregate(scores, None)
        passed = overall >= 0.7

        return EvaluationResult(
            task_id=task_id,
            status=EvaluationStatus.COMPLETED,
            scores=scores,
            overall_score=overall,
            passed=passed,
            feedback=feedback,
            recommendations=recommendations,
            evaluator=EvaluatorType.LLM,
            evaluator_name=self._name,
            timestamp=_now(),
            metadata={
                "mode": EvaluationMode.RUBRIC.value,
                "rubric_items": len(rubric),
                "model": self._model or "default",
            },
        )

    async def evaluate_rubric_async(
        self,
        output: str,
        *,
        rubric: list[dict[str, Any]],
        task_input: str | None = None,
        task_id: str = "",
    ) -> EvaluationResult:
        """Async wrapper for :meth:`evaluate_rubric`."""

        return await asyncio.to_thread(
            self.evaluate_rubric,
            output,
            rubric=rubric,
            task_input=task_input,
            task_id=task_id,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_single_prompt(
        self,
        output: str,
        task_input: str | None,
        criteria: list[EvaluationCriterion],
        context: str | None,
    ) -> str:
        """Build the prompt for single-output evaluation."""

        criteria_desc = "\n".join(
            f"  - {c.value}: {self._criterion_description(c)}" for c in criteria
        )
        parts: list[str] = [self._SINGLE_INSTRUCTION, ""]
        if task_input:
            parts.append(f"Task / User Request:\n{task_input}\n")
        if context:
            parts.append(f"Reference / Context:\n{context}\n")
        parts.append(f"Evaluation Criteria:\n{criteria_desc}\n")
        parts.append(f"Agent Output to Evaluate:\n{output}\n")
        return "\n".join(parts)

    def _build_pairwise_prompt(
        self,
        output_a: str,
        output_b: str,
        task_input: str,
        criteria: list[EvaluationCriterion],
    ) -> str:
        """Build the prompt for pairwise comparison."""

        criteria_desc = "\n".join(
            f"  - {c.value}: {self._criterion_description(c)}" for c in criteria
        )
        return "\n".join(
            [
                self._PAIRWISE_INSTRUCTION,
                "",
                f"Task / User Request:\n{task_input}\n",
                f"Evaluation Criteria:\n{criteria_desc}\n",
                "Output A:",
                output_a,
                "",
                "Output B:",
                output_b,
            ]
        )

    def _build_rubric_prompt(
        self,
        output: str,
        task_input: str | None,
        rubric: list[dict[str, Any]],
    ) -> str:
        """Build the prompt for rubric-based evaluation."""

        rubric_desc = "\n".join(
            f"  {i}. {item.get('name', f'item_{i}')}: "
            f"{item.get('description', '')} "
            f"(max points: {item.get('max_points', 1.0)})"
            for i, item in enumerate(rubric, start=1)
        )
        parts: list[str] = [self._RUBRIC_INSTRUCTION, ""]
        if task_input:
            parts.append(f"Task / User Request:\n{task_input}\n")
        parts.append(f"Evaluation Rubric:\n{rubric_desc}\n")
        parts.append(f"Agent Output to Evaluate:\n{output}\n")
        return "\n".join(parts)

    @staticmethod
    def _criterion_description(criterion: EvaluationCriterion) -> str:
        """Return a short human-readable description for a criterion."""

        descriptions = {
            EvaluationCriterion.ACCURACY: "Factual correctness and freedom from errors.",
            EvaluationCriterion.COMPLETENESS: "All aspects of the task are fully addressed.",
            EvaluationCriterion.RELEVANCE: "The output is on-topic and addresses the user's intent.",
            EvaluationCriterion.COHERENCE: "Logical consistency, clear structure, and readability.",
            EvaluationCriterion.SAFETY: "Free from harmful, dangerous, or policy-violating content.",
            EvaluationCriterion.LEGAL_COMPLIANCE: "Valid citations, proper format, no PII leakage.",
            EvaluationCriterion.EFFICIENCY: "Appropriate conciseness and resource use.",
            EvaluationCriterion.HELPFULNESS: "Genuinely advances the user's goal with actionable value.",
        }
        return descriptions.get(criterion, "Quality of this dimension.")

    # ------------------------------------------------------------------
    # Gateway interaction
    # ------------------------------------------------------------------

    def _call_gateway(self, prompt: str) -> str:
        """Send the evaluation prompt to the gateway and return raw text.

        Imports :class:`ChatCompletionRequest` and :class:`ChatMessage`
        lazily from :mod:`justagent.adapters.model_gateway`. If that
        import fails (e.g. an optional dependency is unavailable on the
        current interpreter), a lightweight dataclass fallback is used
        so the gateway can still be invoked via duck typing.

        Raises:
            EvaluationError: If the gateway call fails.
        """

        assert self._gateway is not None  # checked by callers

        request_cls, message_cls = self._import_request_types()

        try:
            request = request_cls(
                messages=[
                    message_cls(role="system", content=self._SYSTEM_PROMPT),
                    message_cls(role="user", content=prompt),
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            response = self._gateway.chat(request)
            return response.content
        except EvaluationError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface as EvaluationError
            logger.error("LLM judge gateway call failed: %s", exc)
            raise EvaluationError(f"LLM judge call failed: {exc}") from exc

    @staticmethod
    def _import_request_types() -> tuple[type[Any], type[Any]]:
        """Lazily import ChatCompletionRequest and ChatMessage.

        Falls back to local dataclass definitions if the
        :mod:`justagent.adapters.model_gateway` module cannot be
        imported (e.g. due to an optional dependency such as
        ``StrEnum`` being unavailable on older Python versions).

        Returns:
            A ``(ChatCompletionRequest, ChatMessage)`` tuple of types.
        """

        try:
            from justagent.adapters.model_gateway import (
                ChatCompletionRequest,
                ChatMessage,
            )

            return ChatCompletionRequest, ChatMessage
        except ImportError:
            logger.debug(
                "Cannot import ModelGateway types; using fallback "
                "dataclasses for chat request construction."
            )
            return _FallbackChatCompletionRequest, _FallbackChatMessage

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_judge_response(
        self,
        raw: str,
        criteria: list[EvaluationCriterion] | None,
        rubric_items: list[dict[str, Any]] | None = None,
    ) -> tuple[list[EvaluationScore], str, list[str]]:
        """Parse the LLM JSON response into scores, feedback, recommendations.

        Falls back to heuristic scoring if the response cannot be parsed.
        """

        data = self._extract_json(raw)
        if data is None:
            logger.warning("Failed to parse LLM judge response; using fallback.")
            return self._fallback_scores(criteria, rubric_items), raw[:500], []

        scores: list[EvaluationScore] = []
        raw_scores = data.get("scores", [])
        for entry in raw_scores:
            criterion = self._parse_criterion(entry.get("criterion", ""), criteria, rubric_items)
            score_val = self._safe_float(entry.get("score", 0.5))
            reasoning = str(entry.get("reasoning", ""))
            evidence_raw = entry.get("evidence", [])
            evidence = (
                [str(e) for e in evidence_raw]
                if isinstance(evidence_raw, list)
                else [str(evidence_raw)]
            )
            scores.append(
                EvaluationScore(
                    criterion=criterion,
                    score=score_val,
                    reasoning=reasoning,
                    evidence=evidence,
                )
            )

        feedback = str(data.get("overall_feedback", ""))
        recs_raw = data.get("recommendations", [])
        recommendations = (
            [str(r) for r in recs_raw] if isinstance(recs_raw, list) else [str(recs_raw)]
        )

        # Fill in any missing criteria with neutral scores.
        if criteria is not None:
            covered = {s.criterion for s in scores}
            for c in criteria:
                if c not in covered:
                    scores.append(
                        EvaluationScore(
                            criterion=c,
                            score=0.5,
                            reasoning="Criterion not addressed by the judge.",
                        )
                    )

        if not scores:
            return self._fallback_scores(criteria, rubric_items), feedback, recommendations

        return scores, feedback, recommendations

    def _parse_pairwise_response(
        self,
        raw: str,
        criteria: list[EvaluationCriterion],
    ) -> tuple[list[EvaluationScore], str, list[str], str]:
        """Parse a pairwise comparison response.

        Returns scores, feedback, recommendations, and the winner
        (``"A"`` or ``"B"``).
        """

        data = self._extract_json(raw)
        if data is None:
            logger.warning("Failed to parse pairwise response; using fallback.")
            return (
                self._fallback_scores(criteria),
                raw[:500],
                [],
                "A",
            )

        winner = str(data.get("winner", "A")).upper().strip()
        if winner not in ("A", "B"):
            winner = "A"

        scores, feedback, recommendations = self._parse_judge_response(raw, criteria)
        return scores, feedback, recommendations, winner

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """Extract and parse a JSON object from *text*.

        Handles raw JSON, JSON wrapped in markdown code fences, and
        JSON embedded in surrounding prose (finds the first ``{`` and
        the matching last ``}``).
        """

        if not text or not text.strip():
            return None

        # Try direct parse first.
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Try stripping markdown code fences.
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if fence_match:
            try:
                result = json.loads(fence_match.group(1))
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        # Try extracting the outermost JSON object.
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            try:
                result = json.loads(text[first_brace : last_brace + 1])
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def _parse_criterion(
        name: str,
        criteria: list[EvaluationCriterion] | None,
        rubric_items: list[dict[str, Any]] | None,
    ) -> EvaluationCriterion:
        """Map a criterion name string to an :class:`EvaluationCriterion`.

        Tries exact match, then case-insensitive match, then maps to
        :attr:`EvaluationCriterion.HELPFULNESS` as a fallback.
        """

        if not name:
            return EvaluationCriterion.HELPFULNESS

        normalised = name.strip().lower().replace(" ", "_").replace("-", "_")

        # Exact enum match (normalised is already lowercased).
        for c in EvaluationCriterion:
            if c.value == normalised:
                return c

        # Try matching rubric item names to criteria.
        if rubric_items:
            for item in rubric_items:
                item_name = str(item.get("name", "")).strip().lower()
                if item_name == normalised:
                    # Map known rubric names to criteria.
                    if "accur" in item_name:
                        return EvaluationCriterion.ACCURACY
                    if "complet" in item_name:
                        return EvaluationCriterion.COMPLETENESS
                    if "relev" in item_name:
                        return EvaluationCriterion.RELEVANCE
                    if "coher" in item_name or "clarity" in item_name:
                        return EvaluationCriterion.COHERENCE
                    if "safe" in item_name:
                        return EvaluationCriterion.SAFETY
                    if "legal" in item_name or "compli" in item_name:
                        return EvaluationCriterion.LEGAL_COMPLIANCE
                    if "effici" in item_name:
                        return EvaluationCriterion.EFFICIENCY
                    return EvaluationCriterion.HELPFULNESS

        return EvaluationCriterion.HELPFULNESS

    @staticmethod
    def _safe_float(value: Any) -> float:
        """Safely convert *value* to a clamped float in [0, 1]."""

        try:
            return _clamp(float(value))
        except (TypeError, ValueError):
            return 0.5

    @staticmethod
    def _fallback_scores(
        criteria: list[EvaluationCriterion] | None,
        rubric_items: list[dict[str, Any]] | None = None,
    ) -> list[EvaluationScore]:
        """Generate neutral fallback scores when parsing fails."""

        if rubric_items:
            return [
                EvaluationScore(
                    criterion=EvaluationCriterion.HELPFULNESS,
                    score=0.5,
                    reasoning=f"Unable to parse judge response for rubric item '{item.get('name', '')}'.",
                )
                for item in rubric_items
            ]
        if criteria:
            return [
                EvaluationScore(
                    criterion=c,
                    score=0.5,
                    reasoning="Unable to parse judge response; assigned neutral score.",
                )
                for c in criteria
            ]
        return [
            EvaluationScore(
                criterion=EvaluationCriterion.HELPFULNESS,
                score=0.5,
                reasoning="Unable to parse judge response.",
            )
        ]

    @staticmethod
    def _resolve_criteria(
        criteria: list[EvaluationCriterion] | None,
        criterion_set: CriterionSet | None,
    ) -> list[EvaluationCriterion]:
        """Determine the criteria list from explicit args or a criterion set."""

        if criteria is not None:
            return criteria
        if criterion_set is not None:
            return criterion_set.criteria
        return list(EvaluationCriterion)

    @staticmethod
    def _aggregate(
        scores: list[EvaluationScore],
        criterion_set: CriterionSet | None,
    ) -> float:
        """Compute a weighted overall score.

        When *criterion_set* is provided, its weights are used.
        Otherwise, a simple arithmetic mean is computed.
        """

        if not scores:
            return 0.0

        if criterion_set is not None:
            total_weight = 0.0
            weighted_sum = 0.0
            for score in scores:
                w = criterion_set.weight_for(score.criterion)
                if w <= 0:
                    w = 0.1  # small default for criteria not in the set
                weighted_sum += score.score * w
                total_weight += w
            if total_weight > 0:
                return _clamp(weighted_sum / total_weight)

        return _clamp(sum(s.score for s in scores) / len(scores))


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------


class EvaluationPipeline:
    """Orchestrate multiple evaluators with weighted score aggregation.

    The pipeline is the top-level entry point for evaluation. It
    maintains a registry of evaluators (each with a name, weight, and
    optional criteria filter), runs all applicable evaluators on a given
    output, and aggregates their results into a single
    :class:`EvaluationResult`.

    Aggregation is weighted: each evaluator's overall_score is
    multiplied by its registered weight, then divided by the total
    weight. Per-criterion scores are merged by taking the
    highest-weighted evaluator's score for each criterion.

    A configurable ``pass_threshold`` (default 0.7) determines whether
    the aggregated result passes.

    Example::

        >>> pipeline = EvaluationPipeline(pass_threshold=0.75)
        >>> pipeline.register_evaluator("rules", RuleBasedEvaluator(), weight=0.4)
        >>> pipeline.register_evaluator("llm", LLMJudgeEvaluator(gateway), weight=0.6)
        >>> result = pipeline.evaluate("Some output", task_input="Do X")
        >>> result.passed
        False
    """

    def __init__(
        self,
        *,
        pass_threshold: float = 0.7,
        registry: EvaluationRegistry | None = None,
    ) -> None:
        """Initialise the pipeline.

        Args:
            pass_threshold: Minimum overall score for ``passed=True``.
            registry: Optional shared :class:`EvaluationRegistry` for
                criterion-set lookup. If ``None``, a new one is created.
        """

        self._evaluators: dict[str, dict[str, Any]] = {}
        self._pass_threshold = pass_threshold
        self._registry = registry or EvaluationRegistry()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Evaluator registration
    # ------------------------------------------------------------------

    def register_evaluator(
        self,
        name: str,
        evaluator: Any,
        *,
        weight: float = 1.0,
        criteria: list[EvaluationCriterion] | None = None,
    ) -> None:
        """Register an evaluator with the pipeline.

        Args:
            name: Unique evaluator name within this pipeline.
            evaluator: The evaluator instance. Must have an
                ``evaluate(output, *, task_input, criteria, task_id)``
                method (e.g. :class:`RuleBasedEvaluator` or
                :class:`LLMJudgeEvaluator`).
            weight: Aggregation weight (higher = more influence).
            criteria: Optional criteria filter. When set, the
                evaluator only receives these criteria.
        """

        if not name or not name.strip():
            raise EvaluationError("Evaluator name must not be empty.")
        with self._lock:
            self._evaluators[name] = {
                "evaluator": evaluator,
                "weight": _clamp(weight) if weight > 1.0 else weight,
                "criteria": criteria,
            }
        logger.info("Registered evaluator '%s' (weight=%.2f)", name, weight)

    def unregister_evaluator(self, name: str) -> bool:
        """Remove an evaluator by name. Returns ``True`` if removed."""

        with self._lock:
            return self._evaluators.pop(name, None) is not None

    def list_evaluators(self) -> list[str]:
        """Return the names of all registered evaluators."""

        with self._lock:
            return list(self._evaluators.keys())

    # ------------------------------------------------------------------
    # Threshold management
    # ------------------------------------------------------------------

    @property
    def pass_threshold(self) -> float:
        """The minimum overall score for a passing result."""

        return self._pass_threshold

    def set_threshold(self, threshold: float) -> None:
        """Update the pass/fail threshold."""

        self._pass_threshold = _clamp(threshold)

    @property
    def registry(self) -> EvaluationRegistry:
        """The criterion-set registry used by this pipeline."""

        return self._registry

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        output: str,
        *,
        task_input: str | None = None,
        criteria: list[EvaluationCriterion] | None = None,
        criterion_set: str | CriterionSet | None = None,
        task_id: str = "",
        context: str | None = None,
    ) -> EvaluationResult:
        """Run all registered evaluators and aggregate their results.

        Args:
            output: The agent output to evaluate.
            task_input: The original user prompt.
            criteria: Explicit criteria to evaluate. Overrides
                *criterion_set* if both are given.
            criterion_set: Name of a registered criterion set, or a
                :class:`CriterionSet` instance, used for weighted
                aggregation.
            task_id: Optional task identifier.
            context: Additional context passed through to LLM evaluators.

        Returns:
            An aggregated :class:`EvaluationResult`.
        """

        with self._lock:
            evaluator_configs = dict(self._evaluators)

        if not evaluator_configs:
            return self._no_evaluators_result(task_id)

        # Resolve criterion set for weighted aggregation.
        cset = self._resolve_criterion_set(criterion_set)
        resolved_criteria = criteria or (cset.criteria if cset else None)

        # Run each evaluator.
        sub_results: list[tuple[EvaluationResult, float]] = []
        for name, config in evaluator_configs.items():
            evaluator = config["evaluator"]
            weight = config["weight"]
            eval_criteria = config["criteria"] or resolved_criteria

            try:
                # Support evaluators that accept `context`.
                if isinstance(evaluator, LLMJudgeEvaluator):
                    result = evaluator.evaluate(
                        output,
                        task_input=task_input,
                        criteria=eval_criteria,
                        criterion_set=cset if not config["criteria"] else None,
                        task_id=task_id,
                        context=context,
                    )
                else:
                    result = evaluator.evaluate(
                        output,
                        task_input=task_input,
                        criteria=eval_criteria,
                        task_id=task_id,
                    )
            except Exception as exc:  # noqa: BLE001 - one bad evaluator shouldn't fail all
                logger.error("Evaluator '%s' failed: %s", name, exc)
                result = EvaluationResult(
                    task_id=task_id,
                    status=EvaluationStatus.FAILED,
                    evaluator_name=name,
                    feedback=f"Evaluator '{name}' failed: {exc}",
                    overall_score=0.0,
                    metadata={"error": str(exc)},
                )

            sub_results.append((result, weight))

        return self.aggregate_scores(
            sub_results,
            criterion_set=cset,
            task_id=task_id,
            pass_threshold=self._pass_threshold,
        )

    async def evaluate_async(
        self,
        output: str,
        *,
        task_input: str | None = None,
        criteria: list[EvaluationCriterion] | None = None,
        criterion_set: str | CriterionSet | None = None,
        task_id: str = "",
        context: str | None = None,
    ) -> EvaluationResult:
        """Async wrapper for :meth:`evaluate`.

        Runs synchronous evaluators via :func:`asyncio.to_thread`. For
        evaluators that expose ``evaluate_async``, that method is
        preferred.
        """

        with self._lock:
            evaluator_configs = dict(self._evaluators)

        if not evaluator_configs:
            return self._no_evaluators_result(task_id)

        cset = self._resolve_criterion_set(criterion_set)
        resolved_criteria = criteria or (cset.criteria if cset else None)

        async def _run_one(name: str, config: dict[str, Any]) -> tuple[EvaluationResult, float]:
            evaluator = config["evaluator"]
            weight = config["weight"]
            eval_criteria = config["criteria"] or resolved_criteria
            try:
                if isinstance(evaluator, LLMJudgeEvaluator):
                    # LLM judge has its own async method with full kwargs.
                    result = await evaluator.evaluate_async(
                        output,
                        task_input=task_input,
                        criteria=eval_criteria,
                        criterion_set=cset if not config["criteria"] else None,
                        task_id=task_id,
                        context=context,
                    )
                elif hasattr(evaluator, "evaluate_async"):
                    # Generic async evaluator (simpler signature).
                    result = await evaluator.evaluate_async(
                        output,
                        task_input=task_input,
                        criteria=eval_criteria,
                        task_id=task_id,
                    )
                else:
                    # Fallback: run sync evaluate in a thread.
                    result = await asyncio.to_thread(
                        evaluator.evaluate,
                        output,
                        task_input=task_input,
                        criteria=eval_criteria,
                        task_id=task_id,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.error("Async evaluator '%s' failed: %s", name, exc)
                result = EvaluationResult(
                    task_id=task_id,
                    status=EvaluationStatus.FAILED,
                    evaluator_name=name,
                    feedback=f"Evaluator '{name}' failed: {exc}",
                    overall_score=0.0,
                    metadata={"error": str(exc)},
                )
            return result, weight

        tasks = [_run_one(name, config) for name, config in evaluator_configs.items()]
        sub_results = await asyncio.gather(*tasks)

        return self.aggregate_scores(
            list(sub_results),
            criterion_set=cset,
            task_id=task_id,
            pass_threshold=self._pass_threshold,
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_scores(
        self,
        results: list[tuple[EvaluationResult, float]],
        *,
        criterion_set: CriterionSet | None = None,
        task_id: str = "",
        pass_threshold: float = 0.7,
    ) -> EvaluationResult:
        """Aggregate multiple evaluator results into one.

        Args:
            results: List of ``(EvaluationResult, weight)`` tuples.
            criterion_set: Optional criterion set for per-criterion
                weighting.
            task_id: Task identifier for the aggregated result.
            pass_threshold: Threshold for the ``passed`` flag.

        Returns:
            A single :class:`EvaluationResult` combining all inputs.
        """

        if not results:
            return EvaluationResult(
                task_id=task_id,
                status=EvaluationStatus.COMPLETED,
                overall_score=0.0,
                passed=False,
                feedback="No evaluator results to aggregate.",
            )

        total_weight = sum(w for _, w in results)
        if total_weight <= 0:
            total_weight = 1.0

        # Weighted overall score.
        weighted_sum = sum(r.overall_score * w for r, w in results)
        overall = _clamp(weighted_sum / total_weight)

        # Merge per-criterion scores (highest-weight evaluator wins).
        criterion_best: dict[EvaluationCriterion, EvaluationScore] = {}
        for result, _weight in sorted(results, key=lambda x: x[1], reverse=True):
            for score in result.scores:
                if score.criterion not in criterion_best:
                    criterion_best[score.criterion] = score

        merged_scores = list(criterion_best.values())

        # If a criterion set is provided, re-weight the merged scores.
        if criterion_set and merged_scores:
            overall = self._weighted_overall(merged_scores, criterion_set)

        # Merge feedback and recommendations.
        feedback_parts: list[str] = []
        recommendations: list[str] = []
        evaluator_names: list[str] = []
        metadata: dict[str, Any] = {"sub_results": []}

        for result, weight in results:
            if result.feedback:
                feedback_parts.append(
                    f"[{result.evaluator_name or result.evaluator.value}] "
                    f"(w={weight:.2f}): {result.feedback}"
                )
            recommendations.extend(result.recommendations)
            if result.evaluator_name:
                evaluator_names.append(result.evaluator_name)
            metadata["sub_results"].append(
                {
                    "evaluator": result.evaluator_name,
                    "overall_score": result.overall_score,
                    "weight": weight,
                    "status": result.status.value,
                    "score_count": len(result.scores),
                }
            )

        # Deduplicate recommendations while preserving order.
        seen: set[str] = set()
        unique_recs: list[str] = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recs.append(rec)

        # Determine the dominant evaluator type.
        evaluator_type = EvaluatorType.AUTO
        if all(r.evaluator is EvaluatorType.LLM for r, _ in results):
            evaluator_type = EvaluatorType.LLM
        elif all(r.evaluator is EvaluatorType.HUMAN for r, _ in results):
            evaluator_type = EvaluatorType.HUMAN

        any_failed = any(r.status is EvaluationStatus.FAILED for r, _ in results)
        status = (
            EvaluationStatus.FAILED if any_failed and overall == 0 else EvaluationStatus.COMPLETED
        )

        return EvaluationResult(
            task_id=task_id,
            status=status,
            scores=merged_scores,
            overall_score=overall,
            passed=overall >= pass_threshold,
            feedback="\n".join(feedback_parts) if feedback_parts else "",
            recommendations=unique_recs,
            evaluator=evaluator_type,
            evaluator_name="pipeline",
            timestamp=_now(),
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_criterion_set(
        self,
        criterion_set: str | CriterionSet | None,
    ) -> CriterionSet | None:
        """Resolve a criterion set from a name, instance, or ``None``."""

        if criterion_set is None:
            return None
        if isinstance(criterion_set, CriterionSet):
            return criterion_set
        # It's a string — look up in the registry.
        resolved = self._registry.get_criterion_set(criterion_set)
        if resolved is None:
            raise EvaluationError(f"Criterion set '{criterion_set}' not found in registry.")
        return resolved

    @staticmethod
    def _weighted_overall(
        scores: list[EvaluationScore],
        criterion_set: CriterionSet,
    ) -> float:
        """Compute a weighted overall score using a criterion set."""

        total_weight = 0.0
        weighted_sum = 0.0
        for score in scores:
            w = criterion_set.weight_for(score.criterion)
            if w <= 0:
                w = 0.1
            weighted_sum += score.score * w
            total_weight += w
        if total_weight <= 0:
            return 0.0
        return _clamp(weighted_sum / total_weight)

    @staticmethod
    def _no_evaluators_result(task_id: str) -> EvaluationResult:
        """Return a result when no evaluators are registered."""

        return EvaluationResult(
            task_id=task_id,
            status=EvaluationStatus.COMPLETED,
            overall_score=0.0,
            passed=False,
            feedback="No evaluators registered with the pipeline.",
            recommendations=["Register at least one evaluator before evaluating."],
            evaluator_name="pipeline",
        )


# ---------------------------------------------------------------------------
# Evaluation registry
# ---------------------------------------------------------------------------


class EvaluationRegistry:
    """Thread-safe registry for evaluation criterion sets.

    Manages named :class:`CriterionSet` instances. On construction,
    registers built-in defaults for document generation,
    evidence review, case analysis, coding, and general tasks.

    Custom sets can be registered to override or supplement the
    defaults.

    Example::

        >>> registry = EvaluationRegistry()
        >>> domain_set = registry.get_criterion_set("domain_document")
        >>> domain_set is not None
        True
        >>> "legal_compliance" in [c.value for c in domain_set.criteria]
        True
        >>> registry.register_criterion_set("custom", my_criterion_set)
        >>> "custom" in registry.list_criterion_sets()
        True
    """

    def __init__(self) -> None:
        self._sets: dict[str, CriterionSet] = {}
        self._lock = threading.RLock()
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register the built-in default criterion sets."""

        for name, cset in _default_criterion_sets().items():
            self._sets[name] = cset
        logger.info(
            "Registered %d default criterion set(s): %s",
            len(self._sets),
            ", ".join(self._sets.keys()),
        )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register_criterion_set(
        self,
        name: str,
        criterion_set: CriterionSet,
        *,
        overwrite: bool = False,
    ) -> CriterionSet:
        """Register a criterion set under *name*.

        Args:
            name: The key to register under.
            criterion_set: The :class:`CriterionSet` to store.
            overwrite: If ``False`` (default), raises when *name*
                already exists. If ``True``, replaces silently.

        Raises:
            EvaluationError: If *name* exists and *overwrite* is
                ``False``.
        """

        if not name or not name.strip():
            raise EvaluationError("Criterion set name must not be empty.")
        with self._lock:
            if name in self._sets and not overwrite:
                raise EvaluationError(
                    f"Criterion set '{name}' already exists; use overwrite=True to replace."
                )
            self._sets[name] = criterion_set
        logger.info("Registered criterion set '%s' (%d criteria)", name, len(criterion_set.weights))
        return criterion_set

    def get_criterion_set(self, name: str) -> CriterionSet | None:
        """Return the criterion set registered under *name*, or ``None``."""

        with self._lock:
            return self._sets.get(name)

    def remove_criterion_set(self, name: str) -> bool:
        """Remove a criterion set. Returns ``True`` if it existed."""

        with self._lock:
            return self._sets.pop(name, None) is not None

    def list_criterion_sets(self) -> list[str]:
        """Return the names of all registered criterion sets."""

        with self._lock:
            return sorted(self._sets.keys())

    def get_all_criterion_sets(self) -> list[CriterionSet]:
        """Return all registered criterion sets."""

        with self._lock:
            return list(self._sets.values())

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of registered criterion sets."""

        with self._lock:
            return len(self._sets)

    def summary(self) -> dict[str, Any]:
        """Return a compact summary of the registry."""

        with self._lock:
            sets = list(self._sets.items())
        return {
            "total_sets": len(sets),
            "sets": {
                name: {
                    "criteria": [w.criterion.value for w in cset.weights],
                    "total_weight": cset.total_weight(),
                    "description": cset.description,
                }
                for name, cset in sets
            },
        }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_default_pipeline(
    gateway: ModelGateway | None = None,
    *,
    pass_threshold: float = 0.7,
    enable_safety: bool = True,
    enable_legal: bool = True,
    enable_code: bool = True,
    llm_weight: float = 0.6,
    rule_weight: float = 0.4,
) -> EvaluationPipeline:
    """Create a pipeline with rule-based and (optionally) LLM evaluators.

    Args:
        gateway: Optional :class:`ModelGateway`. When provided, an
            :class:`LLMJudgeEvaluator` is registered alongside the
            :class:`RuleBasedEvaluator`.
        pass_threshold: Minimum overall score for passing.
        enable_safety: Enable safety checks in the rule evaluator.
        enable_legal: Enable legal-compliance checks in the rule evaluator.
        enable_code: Enable code-quality checks in the rule evaluator.
        llm_weight: Aggregation weight for the LLM evaluator.
        rule_weight: Aggregation weight for the rule evaluator.

    Returns:
        A configured :class:`EvaluationPipeline` ready to use.
    """

    pipeline = EvaluationPipeline(pass_threshold=pass_threshold)
    pipeline.register_evaluator(
        "rule_based",
        RuleBasedEvaluator(
            enable_safety=enable_safety,
            enable_legal=enable_legal,
            enable_code=enable_code,
        ),
        weight=rule_weight,
    )
    if gateway is not None:
        pipeline.register_evaluator(
            "llm_judge",
            LLMJudgeEvaluator(gateway),
            weight=llm_weight,
        )
    return pipeline


__all__ = [
    "AsyncEvaluator",
    "CriterionSet",
    "CriterionWeight",
    "EvaluationCriterion",
    "EvaluationError",
    "EvaluationMode",
    "EvaluationPipeline",
    "EvaluationRegistry",
    "EvaluationResult",
    "EvaluationScore",
    "EvaluationStatus",
    "Evaluator",
    "EvaluatorType",
    "LLMJudgeEvaluator",
    "RuleBasedEvaluator",
    "create_default_pipeline",
]
