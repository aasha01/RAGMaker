"""RAG Lab backend.

All pipeline logic (the swappable `stages`, and the `core` orchestration added
in later steps) plus the FastAPI service (`api`) that exposes it over HTTP. The
frontend never imports these modules directly — it talks to the API.
"""
