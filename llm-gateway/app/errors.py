class GatewayError(Exception):
    """Base error. ``status_code`` is what the HTTP layer returns."""

    status_code = 500

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code


class UnknownModelError(GatewayError):
    status_code = 400


class ProviderNotConfiguredError(GatewayError):
    status_code = 503


class ProviderError(GatewayError):
    """The upstream provider failed. Defaults to 502 Bad Gateway."""

    status_code = 502


def status_for_upstream(code: int | None) -> int:
    """Translate an upstream HTTP status into the one the gateway returns."""
    if code == 429:
        return 429
    if code in (408, 504):
        return 504
    if code is not None and 400 <= code < 500 and code not in (401, 403):
        # The request we built was rejected: surface it as a client error.
        return 400
    return 502
