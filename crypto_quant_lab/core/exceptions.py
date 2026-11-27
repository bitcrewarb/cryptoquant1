"""Domain-specific exception hierarchy for CRYPTO QUANT LAB."""


class QuantLabError(Exception):
    """Base exception for all platform errors."""


class DataValidationError(QuantLabError):
    """Raised when market data fails integrity or schema validation."""


class ExchangeAPIError(QuantLabError):
    """Raised on exchange API failure or rate limit exhaustion."""

    def __init__(self, message: str, status_code: int | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class StrategyGrammarError(QuantLabError):
    """Raised when a candidate strategy violates formal grammar or semantic rules."""


class LookAheadBiasError(QuantLabError):
    """Raised when a signal or split attempts to use future information."""


class RiskViolationError(QuantLabError):
    """Raised when an execution violates portfolio or position risk constraints."""


class AuthenticationError(QuantLabError):
    """Raised on authentication or authorization failure."""
