"""Typed SDK errors with stable machine-readable violation names."""


class AutoTreeError(Exception):
    """Base class for SDK errors."""


class TreeHTTPError(AutoTreeError):
    """An HTTP request failed; POST requests are never retried implicitly."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class TreeNotSupportedError(TreeHTTPError):
    """The server does not expose AutoTree's tree-completions endpoint."""

    def __init__(self) -> None:
        super().__init__(
            404,
            "/v1/tree/completions is unavailable; use the AutoTree fork "
            "instead of stock SGLang",
        )


class SSEParseError(AutoTreeError):
    """An SSE frame or typed event could not be parsed."""

    def __init__(self, violation: str, detail: str) -> None:
        self.violation = violation
        self.detail = detail
        super().__init__(f"{violation}: {detail}")


class TreeStreamError(AutoTreeError):
    """A typed terminal error event ended a tree stream before ``done``."""

    def __init__(
        self,
        *,
        code: str,
        detail: str,
        error_type: str,
        param: str | None,
        retry_after_seconds: int | None,
    ) -> None:
        self.code = code
        self.detail = detail
        self.error_type = error_type
        self.param = param
        self.retry_after_seconds = retry_after_seconds
        retry = (
            f"; retry after {retry_after_seconds}s"
            if retry_after_seconds is not None
            else ""
        )
        super().__init__(f"{code}: {detail}{retry}")


class TraceInvariantError(AutoTreeError):
    """A streamed trace violated the tree wire contract."""

    def __init__(
        self, violation: str, detail: str = "", *, branch_id: str | None = None
    ) -> None:
        self.violation = violation
        self.detail = detail
        self.branch_id = branch_id
        message = violation
        if branch_id is not None:
            message += f" [branch={branch_id}]"
        if detail:
            message += f": {detail}"
        super().__init__(message)


class ExportError(AutoTreeError):
    """A rollout cannot be represented honestly in the requested export."""

    def __init__(self, violation: str, detail: str) -> None:
        self.violation = violation
        self.detail = detail
        super().__init__(f"{violation}: {detail}")
