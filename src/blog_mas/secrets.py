"""Chapter 10 §1.1 — Centralised secrets management.

The chapter describes two production patterns:
  • Cloud provider solutions: AWS Secrets Manager, Azure Key Vault, GCP Secret Manager
  • HashiCorp Vault: platform-agnostic, highly secure

Both patterns share the same architecture: a sidecar/init-container fetches
secrets at pod startup and injects them into the application container's
environment.  The application code stays completely agnostic to which backend
supplied the values.

This module provides three concrete backends behind a common ``SecretsBackend``
protocol so the application code never branches on the secrets source:

  EnvBackend     — reads os.environ (default; what K8s injects after the
                   sidecar runs or what local .env provides)
  FileBackend    — reads files from a mounted directory (e.g. K8s Secret
                   volume at /var/run/secrets/blog-mas/)
  VaultBackend   — fetches from HashiCorp Vault's KV v2 API directly
                   (useful when the sidecar pattern is not available)

Usage at startup (called from api.py lifespan or celery_app.py):

  from blog_mas.secrets import load_secrets_into_env, VaultBackend

  # Production: init-container already wrote files; this picks them up.
  load_secrets_into_env(backend=FileBackend("/var/run/secrets/blog-mas"))

  # Or Vault directly:
  load_secrets_into_env(backend=VaultBackend(addr="https://vault.internal"))
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class SecretsBackend(Protocol):
    """Anything that can resolve a secret name to its string value."""

    def get(self, name: str) -> str | None:
        """Return the secret value, or None if not found."""
        ...

    def get_required(self, name: str) -> str:
        """Return the secret value, raise RuntimeError if not found."""
        ...


# ---------------------------------------------------------------------------
# Backend: environment variables (default)
# ---------------------------------------------------------------------------

class EnvBackend:
    """Reads secrets from os.environ.

    In Kubernetes this is the default because the sidecar/init-container has
    already populated the environment before the main container starts.
    """

    name = "env"

    def get(self, name: str) -> str | None:
        return os.environ.get(name)

    def get_required(self, name: str) -> str:
        val = os.environ.get(name)
        if not val:
            raise RuntimeError(
                f"[SecretsBackend/env] Required secret '{name}' not found in environment. "
                "Check your K8s Secret / ConfigMap or local .env file."
            )
        return val


# ---------------------------------------------------------------------------
# Backend: mounted file volume (K8s Secret volume / init-container output)
# ---------------------------------------------------------------------------

class FileBackend:
    """Reads secrets from individual files in a mounted directory.

    K8s can project a Secret as a volume where each key becomes a file.
    An init-container writing to /var/run/secrets/blog-mas/ and the app
    container reading from the same path decouples the secret storage
    backend from the application entirely — the app sees plain files.

    File naming: the secret name maps directly to the filename.
    E.g.  name="OPENAI_API_KEY" → reads /var/run/secrets/blog-mas/OPENAI_API_KEY
    """

    name = "file"

    def __init__(self, mount_path: str = "/var/run/secrets/blog-mas") -> None:
        self._base = Path(mount_path)

    def get(self, name: str) -> str | None:
        target = self._base / name
        if not target.exists():
            return None
        return target.read_text().strip() or None

    def get_required(self, name: str) -> str:
        val = self.get(name)
        if not val:
            raise RuntimeError(
                f"[SecretsBackend/file] Required secret file "
                f"'{self._base / name}' not found or empty. "
                "Check that the K8s Secret volume is mounted correctly."
            )
        return val


# ---------------------------------------------------------------------------
# Backend: HashiCorp Vault (KV v2 API)
# ---------------------------------------------------------------------------

class VaultBackend:
    """Fetches secrets from HashiCorp Vault KV v2.

    Ch10 §1.1: "HashiCorp Vault: a platform-agnostic, highly secure solution
    for managing secrets."

    This backend authenticates via a Vault token (VAULT_TOKEN env var) and
    reads from a configurable KV mount path.  In production, replace token
    auth with Kubernetes service-account auth (vault.auth.kubernetes).

    The httpx dependency is already in the project's requirements.
    """

    name = "vault"

    def __init__(
        self,
        addr: str | None = None,
        token: str | None = None,
        mount_path: str = "secret",
        secret_path: str = "blog-mas",
    ) -> None:
        self._addr = addr or os.environ.get("VAULT_ADDR", "http://127.0.0.1:8200")
        self._token = token or os.environ.get("VAULT_TOKEN", "")
        self._mount = mount_path
        self._secret_path = secret_path
        self._cache: dict[str, str] = {}

    def _fetch_all(self) -> dict[str, str]:
        """Fetch all KV pairs from the secret path in one call and cache them."""
        import httpx  # already a project dependency

        url = f"{self._addr}/v1/{self._mount}/data/{self._secret_path}"
        headers = {"X-Vault-Token": self._token}
        try:
            resp = httpx.get(url, headers=headers, timeout=5.0)
            resp.raise_for_status()
            data: dict = resp.json().get("data", {}).get("data", {})
            self._cache = {k: str(v) for k, v in data.items()}
            logger.info(
                "[SecretsBackend/vault] Fetched %d secrets from %s/%s",
                len(self._cache), self._mount, self._secret_path,
            )
        except Exception as exc:
            logger.error("[SecretsBackend/vault] Failed to fetch secrets: %s", exc)
        return self._cache

    def get(self, name: str) -> str | None:
        if not self._cache:
            self._fetch_all()
        return self._cache.get(name)

    def get_required(self, name: str) -> str:
        val = self.get(name)
        if not val:
            raise RuntimeError(
                f"[SecretsBackend/vault] Required secret '{name}' not found "
                f"at Vault path '{self._mount}/data/{self._secret_path}'. "
                "Verify the path and that the token has read access."
            )
        return val


# ---------------------------------------------------------------------------
# AWS Secrets Manager stub
# ---------------------------------------------------------------------------

class AWSSecretsManagerBackend:
    """AWS Secrets Manager adapter.

    Ch10 §1.1: "Cloud provider solutions: AWS Secrets Manager …"

    Requires boto3 installed (add to prod dependencies).  The secret is
    stored as a JSON object; each field becomes one secret name.

    Usage:
        load_secrets_into_env(backend=AWSSecretsManagerBackend(
            secret_name="blog-mas/prod",
            region="us-east-1",
        ))
    """

    name = "aws_secrets_manager"

    def __init__(
        self,
        secret_name: str = "blog-mas/prod",
        region: str | None = None,
    ) -> None:
        self._secret_name = secret_name
        self._region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self._cache: dict[str, str] = {}

    def _fetch_all(self) -> dict[str, str]:
        try:
            import boto3  # type: ignore[import]
            import json

            client = boto3.client("secretsmanager", region_name=self._region)
            response = client.get_secret_value(SecretId=self._secret_name)
            raw = response.get("SecretString", "{}")
            self._cache = {k: str(v) for k, v in json.loads(raw).items()}
            logger.info(
                "[SecretsBackend/aws] Fetched %d secrets from '%s'",
                len(self._cache), self._secret_name,
            )
        except Exception as exc:
            logger.error("[SecretsBackend/aws] Failed to fetch secrets: %s", exc)
        return self._cache

    def get(self, name: str) -> str | None:
        if not self._cache:
            self._fetch_all()
        return self._cache.get(name)

    def get_required(self, name: str) -> str:
        val = self.get(name)
        if not val:
            raise RuntimeError(
                f"[SecretsBackend/aws] Required secret '{name}' not found "
                f"in AWS Secrets Manager secret '{self._secret_name}'."
            )
        return val


# ---------------------------------------------------------------------------
# load_secrets_into_env — the startup injection function
# ---------------------------------------------------------------------------

# The canonical set of secret names the application needs.
# Any backend that provides all of these passes the completeness check.
_REQUIRED_SECRET_NAMES = [
    "OPENAI_API_KEY",
    "HF_TOKEN",
]

_OPTIONAL_SECRET_NAMES = [
    "API_KEY",
    "WEBHOOK_SECRET",
    "VAULT_TOKEN",
]


def load_secrets_into_env(
    backend: SecretsBackend | None = None,
    *,
    required: list[str] | None = None,
    optional: list[str] | None = None,
) -> None:
    """Fetch secrets from ``backend`` and inject them into os.environ.

    This is the function called once at pod startup (from api.py lifespan or
    celery_app.py) to bridge the secrets backend with the rest of the app.
    After this returns, every os.environ.get("OPENAI_API_KEY") call works
    as if the variable was always there — the app code stays agnostic to
    the secrets storage system.

    Ch10 §1.1: "The application should integrate with these systems at startup."
    """
    if backend is None:
        backend = EnvBackend()

    names_required = required if required is not None else _REQUIRED_SECRET_NAMES
    names_optional = optional if optional is not None else _OPTIONAL_SECRET_NAMES

    injected = 0
    for name in names_required:
        if os.environ.get(name):
            continue  # already set; env has priority over backend
        val = backend.get(name)
        if val:
            os.environ[name] = val
            injected += 1
            logger.debug("[secrets] Injected %s from %s", name, backend.name)
        else:
            logger.warning(
                "[secrets] Required secret '%s' not found in backend '%s'. "
                "The app may fail at startup validation.",
                name, backend.name,
            )

    for name in names_optional:
        if os.environ.get(name):
            continue
        val = backend.get(name)
        if val:
            os.environ[name] = val
            injected += 1

    logger.info(
        "[secrets] Startup injection complete: %d secrets loaded via %s backend.",
        injected, backend.name,
    )
