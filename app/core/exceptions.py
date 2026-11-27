"""Domain exceptions.

Services raise these; `app/main.py` converts them into safe HTTP responses.
Each carries a user-facing `message` that never contains internal details.
"""

from __future__ import annotations


class AppError(Exception):
    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class FileValidationError(AppError):
    status_code = 400
    error_code = "invalid_file"


class FileTooLargeError(FileValidationError):
    status_code = 413
    error_code = "file_too_large"


class UnsupportedFileTypeError(FileValidationError):
    status_code = 415
    error_code = "unsupported_file_type"


class ExtractionError(AppError):
    status_code = 422
    error_code = "extraction_failed"


class DocumentNotFoundError(AppError):
    status_code = 404
    error_code = "document_not_found"


class DocumentBusyError(AppError):
    """The document is still being indexed in the background."""

    status_code = 409
    error_code = "document_busy"


class InvalidModelError(AppError):
    """The request asked for a model that is not in the allowed list."""

    status_code = 400
    error_code = "invalid_model"


class LLMNotConfiguredError(AppError):
    status_code = 503
    error_code = "llm_not_configured"


class LLMTimeoutError(AppError):
    status_code = 504
    error_code = "llm_timeout"


class LLMRateLimitError(AppError):
    status_code = 429
    error_code = "llm_rate_limited"


class LLMProviderError(AppError):
    status_code = 502
    error_code = "llm_provider_error"


class EmbeddingModelMismatchError(AppError):
    """The vector store was built with a different embedding model."""

    error_code = "embedding_model_mismatch"
