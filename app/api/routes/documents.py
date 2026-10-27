"""Document endpoints. Handlers stay thin: HTTP in -> service call -> schema out."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Path, Response, UploadFile, status

from app.api.dependencies import get_document_service
from app.core.exceptions import FileTooLargeError
from app.schemas.common import ErrorResponse
from app.domain import DocumentStatus
from app.schemas.document import (
    DeleteResponse,
    DocumentListResponse,
    DocumentOut,
    OcrLanguagesResponse,
    UploadResponse,
)
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])

DocumentId = Annotated[str, Path(pattern=r"^[a-f0-9]{32}$", description="Document ID returned by upload")]
Service = Annotated[DocumentService, Depends(get_document_service)]

_ERRORS = {
    400: {"model": ErrorResponse, "description": "Invalid file"},
    413: {"model": ErrorResponse, "description": "File too large"},
    415: {"model": ErrorResponse, "description": "Unsupported file type"},
}


# `def` (not `async def`): reading and hashing the file is blocking work, so FastAPI
# runs this in its thread pool instead of blocking the event loop.
@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        200: {"model": UploadResponse, "description": "Identical file already indexed or being indexed"},
        201: {"model": UploadResponse, "description": "Indexed before the response (background indexing off)"},
        **_ERRORS,
    },
    summary="Upload a PDF, scanned PDF, PNG/JPG image, or DOCX; it is indexed in the background",
)
def upload_document(
    response: Response,
    service: Service,
    file: UploadFile = File(..., description="PDF, PNG, JPG/JPEG or DOCX"),
    ocr_language: str | None = Form(
        default=None,
        max_length=100,
        description='Tesseract language code(s) for scanned pages and images, e.g. "fra" or "hin+eng".',
    ),
) -> UploadResponse:
    # Read at most max+1 bytes so an oversized upload can't exhaust memory.
    content = file.file.read(service.max_upload_bytes + 1)
    if len(content) > service.max_upload_bytes:
        raise FileTooLargeError(
            f"File exceeds the maximum upload size of {service.max_upload_bytes / (1024 * 1024):.0f} MB."
        )
    record, created = service.upload(file.filename, file.content_type, content, ocr_language)
    if not created:
        response.status_code = status.HTTP_200_OK
    elif record.status is not DocumentStatus.PROCESSING:
        response.status_code = status.HTTP_201_CREATED  # already finished (background indexing off)
    return UploadResponse(document=DocumentOut.from_record(record), already_indexed=not created)


@router.get("", response_model=DocumentListResponse, summary="List uploaded documents")
def list_documents(service: Service) -> DocumentListResponse:
    docs = [DocumentOut.from_record(r) for r in service.list_documents()]
    return DocumentListResponse(documents=docs, total=len(docs))


@router.get(
    "/ocr-languages",
    response_model=OcrLanguagesResponse,
    summary="OCR languages installed on the server",
)
def ocr_languages(service: Service) -> OcrLanguagesResponse:
    return OcrLanguagesResponse(
        languages=service.extractor.available_languages(), default=service.extractor.ocr_language
    )


@router.get(
    "/{document_id}",
    response_model=DocumentOut,
    responses={404: {"model": ErrorResponse}},
    summary="Get one document (poll this to see when indexing has finished)",
)
def get_document(document_id: DocumentId, service: Service) -> DocumentOut:
    return DocumentOut.from_record(service.get_document(document_id))


@router.delete(
    "/{document_id}",
    response_model=DeleteResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse, "description": "Still indexing"}},
    summary="Delete a document, its vectors, and its stored file",
)
def delete_document(document_id: DocumentId, service: Service) -> DeleteResponse:
    service.delete(document_id)
    return DeleteResponse(document_id=document_id)
