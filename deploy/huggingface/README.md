---
title: RAG Document Chatbot
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Upload PDFs, scans, images or DOCX and ask questions
---

# RAG Document Chatbot

Upload PDFs, scanned PDFs, images or Word files, then ask questions about them.
Answers come only from your documents and show which pages they came from.

**Built with:** FastAPI, PyMuPDF, Tesseract OCR, python-docx, Sentence Transformers,
a cross-encoder re-ranker, ChromaDB, SQLite and Google Gemini.

Source code: https://github.com/vidhi-vats-11/rag-document-chatbot

**Demo notes**
- This is a shared public demo: documents you upload can be seen by other visitors. Don't upload private files.
- Files are cleared whenever the Space restarts.
- The AI model has a small free daily limit. If it's reached, try again tomorrow.
