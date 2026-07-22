"""Pydantic schemas for AutoShip configuration."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


class Provider(str, Enum):
    """Supported model backend providers."""

    LM_STUDIO = "lm_studio"
    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"
    VLLM = "vllm"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    OPENROUTER = "openrouter"


class ModelBackendConfig(BaseModel):
    """Configuration for a single model backend endpoint."""

    provider: Provider
    base_url: HttpUrl
    api_key: str | None = Field(default=None, repr=False)
    api_version: str | None = None
    model: str | None = None
    tier: Literal[1, 2, 3] = 2
    timeout: float = 30.0
    concurrency: int = 2
    priority: int = 0


SUPPORTED_CLEAN_TOOLS = frozenset({"autoflake", "black", "ruff", "isort"})


class CleanConfig(BaseModel):
    """Configuration for the `clean` command."""

    enabled: bool = True
    tools: list[str] = ["ruff"]
    dry_run: bool = False
    exclude: list[str] = Field(default_factory=list)

    @field_validator("tools", mode="after")
    @classmethod
    def _validate_tools(cls, tools: list[str]) -> list[str]:
        unsupported = [tool for tool in tools if tool not in SUPPORTED_CLEAN_TOOLS]
        if unsupported:
            raise ValueError(f"Unsupported clean tools: {', '.join(unsupported)}")
        return tools


class CommitConfig(BaseModel):
    """Configuration for the `commit` command."""

    enabled: bool = True
    max_tokens: int = 512
    conventional_commits: bool = True
    auto_push: bool = False
    allowed_editors: list[str] = Field(
        default_factory=lambda: [
            "vim",
            "nvim",
            "vi",
            "emacs",
            "nano",
            "code",
            "subl",
            "micro",
            "helix",
            "hx",
        ],
        description="Allowed editors for the commit command.",
    )


class SecurityThreshold(str, Enum):
    """Severity threshold for security scans."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SecurityConfig(BaseModel):
    """Configuration for the security-scan plugin."""

    enabled: bool = True
    tools: list[str] = ["semgrep"]
    threshold: SecurityThreshold = SecurityThreshold.MEDIUM
    fail_fast: bool = True


class AuditConfig(BaseModel):
    """Configuration for audit logging."""

    log_dir: Path | None = None
    retention_days: int = 30
    redact_unknown_fields: bool = False


class WebSearchProvider(str, Enum):
    """Supported web search backends."""

    DUCKDUCKGO = "duckduckgo"
    BRAVE = "brave"
    GOOGLE = "google"
    SEARXNG = "searxng"


class WebSearchConfig(BaseModel):
    """Configuration for the web-search plugin.

    Web search is disabled by default and must be explicitly enabled, because it
    sends error snippets to a public search service.
    """

    enabled: bool = False
    provider: WebSearchProvider = WebSearchProvider.DUCKDUCKGO
    api_key: str | None = Field(default=None, repr=False)
    cx: str | None = Field(default=None, repr=False)
    instance_url: str | None = None
    max_results: int = 3
    timeout: float = 10.0


class SandboxConfig(BaseModel):
    """Configuration for sandbox isolation requirements."""

    required: bool = True


class DockerShipConfig(BaseModel):
    """Configuration for the docker-ship plugin."""

    enabled: bool = True
    default_image: str | None = None
    default_tag: str = "latest"
    push: bool = False
    build_args: dict[str, str] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    """Configuration for model routing and fallback."""

    default_tier: Literal[1, 2, 3] = 2
    fallback: bool = True
    backends: list[ModelBackendConfig] = Field(default_factory=lambda: list[ModelBackendConfig]())


class RegistryConfig(BaseModel):
    """Configuration for the plugin registry client."""

    url: HttpUrl = Field(
        default="https://raw.githubusercontent.com/MS33834/autoship/main/registry/plugins.json",
        validate_default=True,
    )
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    public_key: str | None = Field(
        default=None,
        description="Base64-encoded Ed25519 public key used to verify the registry index.",
    )


class CacheConfig(BaseModel):
    """Configuration for the local disk cache."""

    enabled: bool = True
    ttl: int = 3600
    dir: Path | None = None


class LlmProvider(str, Enum):
    """Supported LLM providers for the fix command."""

    OPENAI = "openai"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"


class LlmConfig(BaseModel):
    """Configuration for the LLM-powered fix command."""

    provider: LlmProvider = LlmProvider.OPENAI
    model: str = "gpt-4o-mini"
    api_key: str | None = Field(default=None, repr=False)
    api_version: str | None = None
    base_url: HttpUrl | None = None
    timeout: float = 60.0
    max_tokens: int = 2048


class VerifyConfig(BaseModel):
    """Configuration for the ``verify`` command."""

    allowed_commands: list[str] = Field(
        default_factory=lambda: [
            "pytest",
            "python",
            "python3",
            "ruff",
            "mypy",
            "black",
            "isort",
            "tox",
            "nox",
            "npm",
            "yarn",
            "pnpm",
            "poetry",
            "make",
        ]
    )


class ToolConfig(BaseModel):
    """Configuration for a single external tool."""

    path: str | None = None
    sha256: str | None = None


class ToolsConfig(BaseModel):
    """Configuration for external tool paths and optional SHA-256 verification."""

    git: ToolConfig = Field(default_factory=ToolConfig)
    docker: ToolConfig = Field(default_factory=ToolConfig)
    twine: ToolConfig = Field(default_factory=ToolConfig)
    gh: ToolConfig = Field(default_factory=ToolConfig)
    patch: ToolConfig = Field(default_factory=ToolConfig)

    def get(self, name: str) -> ToolConfig:
        value = getattr(self, name, ToolConfig())
        if isinstance(value, ToolConfig):
            return value
        return ToolConfig()


class HookConfig(BaseModel):
    """A single run-on-save hook definition."""

    command: Literal["clean", "verify"]
    args: list[str] = Field(default_factory=list)
    include: list[str] = Field(default_factory=lambda: ["**/*"])
    exclude: list[str] = Field(default_factory=list)
    debounce_ms: int = Field(default=300, ge=0)
    verify_command: str | None = None

    @model_validator(mode="after")
    def _require_verify_command(self) -> HookConfig:
        if self.command == "verify" and not self.verify_command:
            raise ValueError("verify_command is required when command is 'verify'")
        return self


class HooksConfig(BaseModel):
    """Configuration for run-on-save hooks (``[hooks]`` section)."""

    enabled: bool = False
    on_save: list[HookConfig] = Field(default_factory=list[HookConfig])


class AppConfig(BaseModel):
    """Top-level application configuration."""

    schema_version: int = 1
    project_root: Path = Path(".")
    project_type: str = "unknown"
    log_level: str = "INFO"
    audit_log_dir: Path | None = None
    locale: str = "auto"
    clean: CleanConfig = Field(default_factory=CleanConfig)
    commit: CommitConfig = Field(default_factory=CommitConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    verify: VerifyConfig = Field(default_factory=VerifyConfig)
    docker_ship: DockerShipConfig = Field(default_factory=DockerShipConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    hooks: HooksConfig = Field(default_factory=HooksConfig)
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
