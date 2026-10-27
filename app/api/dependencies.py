"""FastAPI dependencies: hand route handlers the services they need."""

from fastapi import Request

from app.core.container import Container
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_document_service(request: Request) -> DocumentService:
    return get_container(request).document_service


def get_chat_service(request: Request) -> ChatService:
    return get_container(request).chat_service
