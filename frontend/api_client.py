"""Thin HTTP client the frontend uses to talk to the backend API.

Keeps all networking in one place so the Streamlit app stays about UI. Every
call raises `APIError` carrying the backend's own error message, so the UI can
show a learner *why* a stage failed (e.g. a bad overlap value) instead of a
generic stack trace — supporting the "fail loudly, clearly" Non-Negotiable.

The client accepts an injected httpx client, which lets tests drive it against
the FastAPI app in-process (ASGITransport) with no server running.
"""

from __future__ import annotations

import os

import httpx

DEFAULT_BASE_URL = os.environ.get("RAG_LAB_API_URL", "http://localhost:8000")


class APIError(Exception):
    """Raised when the backend returns an error; message is user-facing."""


class APIClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(base_url=self.base_url, timeout=timeout)

    # --- helpers ----------------------------------------------------------

    @staticmethod
    def _detail(response: httpx.Response) -> str:
        try:
            body = response.json()
            if isinstance(body, dict) and "detail" in body:
                return str(body["detail"])
        except Exception:
            pass
        return response.text or f"HTTP {response.status_code}"

    def _get(self, path: str) -> object:
        try:
            r = self._client.get(path)
        except httpx.HTTPError as e:
            raise APIError(f"Cannot reach backend at {self.base_url} ({e}).") from e
        if r.status_code >= 400:
            raise APIError(self._detail(r))
        return r.json()

    # --- endpoints --------------------------------------------------------

    def health(self) -> dict:
        return self._get("/health")  # type: ignore[return-value]

    def list_parsers(self) -> list[dict]:
        return self._get("/stages/parsers")  # type: ignore[return-value]

    def list_chunkers(self) -> list[dict]:
        return self._get("/stages/chunkers")  # type: ignore[return-value]

    def list_embedders(self) -> list[dict]:
        return self._get("/stages/embedders")  # type: ignore[return-value]

    def embedder_model_info(
        self,
        key: str,
        model_name: str = "all-MiniLM-L6-v2",
        normalize: bool = True,
        truncate_dim: int | None = None,
    ) -> dict:
        params: dict = {"model_name": model_name, "normalize": normalize}
        if truncate_dim is not None:
            params["truncate_dim"] = truncate_dim
        try:
            r = self._client.get(f"/embedders/{key}/model_info", params=params)
        except httpx.HTTPError as e:
            raise APIError(f"Cannot reach backend at {self.base_url} ({e}).") from e
        if r.status_code >= 400:
            raise APIError(self._detail(r))
        return r.json()

    def parse(self, filename: str, content: bytes, parser: str = "manual") -> dict:
        try:
            r = self._client.post(
                "/parse",
                params={"filename": filename, "parser": parser},
                content=content,
                headers={"content-type": "application/octet-stream"},
            )
        except httpx.HTTPError as e:
            raise APIError(f"Cannot reach backend at {self.base_url} ({e}).") from e
        if r.status_code >= 400:
            raise APIError(self._detail(r))
        return r.json()

    def chunk(
        self,
        document: dict,
        chunker: str = "recursive",
        params: dict | None = None,
        embedder: str | None = None,
        embedder_params: dict | None = None,
    ) -> dict:
        payload = {"document": document, "chunker": chunker, "params": params or {}}
        if embedder:
            payload["embedder"] = embedder
            payload["embedder_params"] = embedder_params or {}
        try:
            r = self._client.post("/chunk", json=payload)
        except httpx.HTTPError as e:
            raise APIError(f"Cannot reach backend at {self.base_url} ({e}).") from e
        if r.status_code >= 400:
            raise APIError(self._detail(r))
        return r.json()

    def embed(self, chunks: list[dict], embedder: str = "sentence_transformers", params: dict | None = None) -> dict:
        try:
            r = self._client.post(
                "/embed",
                json={"chunks": chunks, "embedder": embedder, "params": params or {}},
            )
        except httpx.HTTPError as e:
            raise APIError(f"Cannot reach backend at {self.base_url} ({e}).") from e
        if r.status_code >= 400:
            raise APIError(self._detail(r))
        return r.json()

    def list_vectorstores(self) -> list[dict]:
        return self._get("/stages/vectorstores")  # type: ignore[return-value]

    def list_retrievers(self) -> list[dict]:
        return self._get("/stages/retrievers")  # type: ignore[return-value]

    def list_llm_providers(self) -> list[dict]:
        return self._get("/stages/llm_providers")  # type: ignore[return-value]

    def build_vectorstore(
        self,
        vectors: list[list[float]],
        chunks: list[dict],
        vectorstore: str = "faiss",
        params: dict | None = None,
        embedder: str = "sentence_transformers",
        embed_params: dict | None = None,
        meta: dict | None = None,
    ) -> dict:
        payload = {
            "vectors": vectors,
            "chunks": chunks,
            "vectorstore": vectorstore,
            "params": params or {},
            "embedder": embedder,
            "embed_params": embed_params or {},
            "meta": meta or {},
        }
        try:
            r = self._client.post("/vectorstore/build", json=payload)
        except httpx.HTTPError as e:
            raise APIError(f"Cannot reach backend at {self.base_url} ({e}).") from e
        if r.status_code >= 400:
            raise APIError(self._detail(r))
        return r.json()

    def search(
        self,
        store_id: str,
        query: str,
        top_k: int = 4,
        retriever: str = "naive_topk",
        retriever_params: dict | None = None,
    ) -> dict:
        try:
            r = self._client.post(
                "/vectorstore/search",
                json={
                    "store_id": store_id,
                    "query": query,
                    "top_k": top_k,
                    "retriever": retriever,
                    "retriever_params": retriever_params or {},
                },
            )
        except httpx.HTTPError as e:
            raise APIError(f"Cannot reach backend at {self.base_url} ({e}).") from e
        if r.status_code >= 400:
            raise APIError(self._detail(r))
        return r.json()

    # --- recipes ----------------------------------------------------------

    def create_recipe(self, source_filename: str, source_b64: str, config: dict,
                      name: str | None = None, description: str | None = None) -> dict:
        payload = {
            "source_filename": source_filename,
            "source_b64": source_b64,
            "config": config,
            "name": name,
            "description": description,
        }
        try:
            r = self._client.post("/recipes", json=payload)
        except httpx.HTTPError as e:
            raise APIError(f"Cannot reach backend at {self.base_url} ({e}).") from e
        if r.status_code >= 400:
            raise APIError(self._detail(r))
        return r.json()

    def list_recipes(self) -> list[dict]:
        return self._get("/recipes")  # type: ignore[return-value]

    def get_recipe(self, recipe_id: str) -> dict:
        return self._get(f"/recipes/{recipe_id}")  # type: ignore[return-value]

    def search_recipe(
        self,
        recipe_id: str,
        query: str,
        top_k: int = 4,
        retriever: str = "naive_topk",
        retriever_params: dict | None = None,
    ) -> dict:
        try:
            r = self._client.post(
                f"/recipes/{recipe_id}/search",
                json={
                    "query": query,
                    "top_k": top_k,
                    "retriever": retriever,
                    "retriever_params": retriever_params or {},
                },
            )
        except httpx.HTTPError as e:
            raise APIError(f"Cannot reach backend at {self.base_url} ({e}).") from e
        if r.status_code >= 400:
            raise APIError(self._detail(r))
        return r.json()

    def generate_recipe(
        self,
        recipe_id: str,
        question: str,
        provider: str = "ollama",
        top_k: int = 4,
        retriever: str = "naive_topk",
        retriever_params: dict | None = None,
        provider_params: dict | None = None,
        gen_params: dict | None = None,
    ) -> dict:
        payload = {
            "question": question,
            "provider": provider,
            "top_k": top_k,
            "retriever": retriever,
            "retriever_params": retriever_params or {},
            "provider_params": provider_params or {},
            "gen_params": gen_params or {},
        }
        try:
            r = self._client.post(f"/recipes/{recipe_id}/generate", json=payload)
        except httpx.HTTPError as e:
            raise APIError(f"Cannot reach backend at {self.base_url} ({e}).") from e
        if r.status_code >= 400:
            raise APIError(self._detail(r))
        return r.json()

    # --- scoring (optional RAGAS-style quality metrics) -------------------

    def score_status(self) -> dict:
        """Whether optional scoring can run (i.e. ragas is installed)."""
        return self._get("/score/status")  # type: ignore[return-value]

    def score_cells(self, cells: list[dict], sort_by: str = "faithfulness") -> dict:
        """Score compare-grid cells. Returns per-cell scores + a ranking summary,
        or `{available: False, message: ...}` when ragas isn't installed."""
        try:
            r = self._client.post("/score", json={"cells": cells, "sort_by": sort_by})
        except httpx.HTTPError as e:
            raise APIError(f"Cannot reach backend at {self.base_url} ({e}).") from e
        if r.status_code >= 400:
            raise APIError(self._detail(r))
        return r.json()
