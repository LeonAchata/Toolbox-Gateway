from __future__ import annotations

import logging

from .config import Settings
from .errors import ProviderNotConfiguredError, UnknownModelError
from .providers import PROVIDER_CLASSES, Provider
from .schemas import ModelInfo

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Holds one long-lived instance per provider and resolves model aliases."""

    def __init__(self, settings: Settings, provider_classes=PROVIDER_CLASSES):
        self.settings = settings
        self._providers: dict[str, Provider] = {}
        self._aliases: dict[str, str] = {}
        for cls in provider_classes:
            provider = cls(settings)
            self._providers[provider.name] = provider
            for alias in (provider.name, *provider.aliases):
                self._aliases[alias.lower()] = provider.name

    def resolve(self, name: str | None) -> Provider:
        if not name:
            return self.default()
        canonical = self._aliases.get(name.strip().lower())
        if canonical is None:
            known = ", ".join(sorted(self._providers))
            raise UnknownModelError(f"Unknown model '{name}'. Known models: {known}")
        provider = self._providers[canonical]
        reason = provider.unavailable_reason()
        if reason:
            raise ProviderNotConfiguredError(f"Model '{canonical}' is not available: {reason}")
        return provider

    def default(self) -> Provider:
        if self.settings.default_model:
            return self.resolve(self.settings.default_model)
        for provider in self._providers.values():
            if provider.unavailable_reason() is None:
                return provider
        raise ProviderNotConfiguredError(
            "No provider is configured. Set OPENAI_API_KEY, GOOGLE_API_KEY or AWS credentials."
        )

    def default_name(self) -> str | None:
        try:
            return self.default().name
        except (ProviderNotConfiguredError, UnknownModelError):
            return None

    def describe(self) -> list[ModelInfo]:
        default = self.default_name()
        infos = []
        for provider in self._providers.values():
            reason = provider.unavailable_reason()
            infos.append(
                ModelInfo(
                    name=provider.name,
                    provider=provider.vendor,
                    provider_model=provider.model_id,
                    available=reason is None,
                    reason=reason,
                    default=provider.name == default,
                    aliases=list(provider.aliases),
                )
            )
        return infos

    def __iter__(self):
        return iter(self._providers.values())
