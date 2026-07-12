"""RAG Lab — Streamlit frontend (Build wizard + Recipes + Query & Compare).

Talks to the backend only over HTTP via APIClient. The Build tab walks the five
pipeline stages with live inspection and a per-stage "redo"; the Recipes tab
queries a saved recipe; the Compare tab runs one question across a grid of
recipes × LLM providers and can score every answer with optional RAGAS metrics.

Run (from the repo root, with the backend already running):
    uvicorn backend.api.main:app --reload         # terminal 1
    streamlit run frontend/app.py                 # terminal 2
"""

from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.api_client import APIClient, APIError, DEFAULT_BASE_URL

st.set_page_config(page_title="RAG Lab", page_icon="🧪", layout="wide")


def get_client(base_url: str) -> APIClient:
    return APIClient(base_url=base_url)


def option_picker(label: str, options: list[dict], state_key: str) -> str:
    """Radio of strategy options that shows each one's teaching description."""
    if not options:
        st.warning(f"No {label} strategies available from the backend.")
        return ""
    keys = [o["key"] for o in options]
    by_key = {o["key"]: o for o in options}
    chosen = st.radio(
        label,
        keys,
        format_func=lambda k: by_key[k]["name"],
        key=state_key,
    )
    st.caption(by_key[chosen]["description"])
    return chosen


def retriever_controls(client, key_prefix: str) -> tuple[str, dict]:
    """Render the query-time retriever picker + its strategy-specific params.

    Returns (retriever_key, retriever_params). Retrieval is chosen per query
    (SPEC.md §6), so this sits next to every query box, not in the build wizard —
    the same recipe can be probed with each strategy. `key_prefix` namespaces the
    Streamlit widget keys so the two query sections don't collide.
    """
    try:
        retrievers = client.list_retrievers()
    except APIError as e:
        st.error(f"Could not load retriever options: {e}")
        return "naive_topk", {}

    key = option_picker("Retriever (query-time)", retrievers, f"{key_prefix}_retriever")
    params: dict = {}

    if key == "mmr":
        c1, c2 = st.columns(2)
        params["fetch_k"] = int(c1.number_input(
            "fetch_k (candidate pool)", 1, 200, 20, key=f"{key_prefix}_mmr_fetchk"))
        params["lambda_mult"] = float(c2.slider(
            "lambda_mult (1=relevance, 0=diversity)", 0.0, 1.0, 0.5, 0.05,
            key=f"{key_prefix}_mmr_lambda"))
    elif key == "hybrid":
        c1, c2 = st.columns(2)
        params["fetch_k"] = int(c1.number_input(
            "fetch_k (per side)", 1, 200, 20, key=f"{key_prefix}_hy_fetchk"))
        fusion = c2.radio("fusion", ["rrf", "weighted"], horizontal=True,
                          key=f"{key_prefix}_hy_fusion")
        params["fusion"] = fusion
        if fusion == "rrf":
            params["rrf_k"] = int(st.number_input(
                "rrf_k (rank dampening)", 1, 1000, 60, key=f"{key_prefix}_hy_rrfk"))
        else:
            params["alpha"] = float(st.slider(
                "alpha (weight on vector vs BM25)", 0.0, 1.0, 0.5, 0.05,
                key=f"{key_prefix}_hy_alpha"))
    elif key == "rerank":
        c1, c2 = st.columns([1, 2])
        params["fetch_n"] = int(c1.number_input(
            "fetch_n (candidates to re-rank)", 1, 200, 20, key=f"{key_prefix}_rr_fetchn"))
        params["model_name"] = c2.text_input(
            "cross-encoder model", value="cross-encoder/ms-marco-MiniLM-L-6-v2",
            key=f"{key_prefix}_rr_model")
    elif key == "hyde":
        c1, c2 = st.columns(2)
        provider = c1.text_input("LLM provider", value="ollama", key=f"{key_prefix}_hyde_prov")
        model = c2.text_input("provider model (optional)", value="", key=f"{key_prefix}_hyde_model")
        params["provider"] = provider
        if model.strip():
            params["provider_params"] = {"model": model.strip()}
        st.caption("HyDE calls an LLM first, so it needs a running Ollama (default) or an API key.")

    return key, params


def highlight_overlap(text: str, overlap: int) -> str:
    """Return markdown with the leading `overlap` chars marked, for the UI."""
    if overlap <= 0:
        return text
    lead, rest = text[:overlap], text[overlap:]
    return f":orange[{lead}]{rest}"


def render_model_info(info: dict) -> None:
    """Show an embedding model's details (dimension, context window, size…)."""
    st.markdown("**Model details**")
    a, b, c, d = st.columns(4)
    a.metric("Default dim", info.get("default_dimension") or "—")
    b.metric("Output dim", info.get("output_dimension"))
    ctx = info.get("max_seq_length_tokens")
    c.metric("Context window (tok)", ctx if ctx is not None else "—")
    params = info.get("param_count")
    d.metric("Parameters", f"{params / 1e6:.1f}M" if params else "—")

    e, f, g, h = st.columns(4)
    e.metric("Approx size (MB)", info.get("approx_size_mb") or "—")
    f.metric("Backend", info.get("backend") or "—")
    g.metric("Normalize", str(info.get("normalize")))
    h.metric("Dim customizable", "yes" if info.get("dimension_customizable") else "no")

    st.caption(
        "Parameters sent to the model → "
        f"model_name=`{info.get('model_name')}`, "
        f"normalize=`{info.get('normalize')}`, "
        f"truncate_dim=`{info.get('truncate_dim')}`"
    )
    if info.get("notes"):
        st.warning(info["notes"])


def main() -> None:
    st.title("🧪 RAG Lab")
    st.caption(
        "Build RAG recipes stage by stage, inspect each step, then save & query "
        "them. Backend: FastAPI · Frontend: Streamlit."
    )

    # --- sidebar: backend connection ------------------------------------
    with st.sidebar:
        st.header("Backend")
        base_url = st.text_input("API URL", value=DEFAULT_BASE_URL)
        client = get_client(base_url)
        if st.button("Check connection"):
            try:
                st.success(f"Connected: {client.health()}")
            except APIError as e:
                st.error(str(e))

    tab_build, tab_recipes, tab_compare = st.tabs(["Build", "Recipes", "Compare"])
    with tab_build:
        render_build_tab(client)
    with tab_recipes:
        render_recipes_tab(client)
    with tab_compare:
        render_compare_tab(client)


def render_build_tab(client) -> None:
    # ------------------------------------------------------------------ #
    # Stage 1 — Parsing
    # ------------------------------------------------------------------ #
    st.header("Stage 1 · Parsing")
    uploaded = st.file_uploader(
        "Upload a document", type=["txt", "md", "pdf", "docx"]
    )

    try:
        parsers = client.list_parsers()
    except APIError as e:
        st.error(f"Could not load parser options: {e}")
        st.stop()

    parser_key = option_picker("Parser", parsers, "parser_choice")

    if st.button("Parse", type="primary", disabled=uploaded is None):
        try:
            result = client.parse(uploaded.name, uploaded.getvalue(), parser=parser_key)
            st.session_state["parsed"] = result
            st.session_state.pop("chunked", None)  # invalidate downstream stage
        except APIError as e:
            st.error(f"Parse failed: {e}")

    parsed = st.session_state.get("parsed")
    if parsed:
        s = parsed["summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Characters", f"{s['char_count']:,}")
        c2.metric("Words", f"{s['word_count']:,}")
        c3.metric("Pages", s.get("pages") or "—")
        c4.metric("Format", s.get("format") or "—")
        with st.expander("Extracted text (preview)", expanded=True):
            st.text(parsed["document"]["text"][:3000])
        with st.expander("Metadata"):
            st.json(parsed["document"]["metadata"])
        st.caption("↻ To redo: pick a different parser or file above and press Parse again.")

    # ------------------------------------------------------------------ #
    # Stage 2 — Chunking
    # ------------------------------------------------------------------ #
    st.header("Stage 2 · Chunking")
    if not parsed:
        st.info("Parse a document first to enable chunking.")
        return

    try:
        chunkers = client.list_chunkers()
    except APIError as e:
        st.error(f"Could not load chunker options: {e}")
        return

    chunker_key = option_picker("Chunker", chunkers, "chunker_choice")

    sem_embedder_key = None
    sem_embedder_params: dict = {}
    if chunker_key == "semantic":
        st.caption(
            "Semantic chunking needs an embedder *now* (to compare sentence "
            "similarity), before Stage 3's embedder choice. Pick one below — "
            "Stage 3 lets you pick a (possibly different) embedder for the "
            "final vectors."
        )
        try:
            embedders = client.list_embedders()
        except APIError as e:
            st.error(f"Could not load embedder options: {e}")
            return
        sem_embedder_key = option_picker(
            "Embedder (for semantic chunking)", embedders, "semantic_embedder_choice"
        )
        # Recommended default per embedder backend — applied when the backend
        # choice changes, never overwriting a value the user has since typed.
        default_model = (
            "nomic-embed-text:latest" if sem_embedder_key == "ollama" else "all-MiniLM-L6-v2"
        )
        if st.session_state.get("semantic_embedder_last") != sem_embedder_key:
            st.session_state["semantic_model_name"] = default_model
            st.session_state["semantic_embedder_last"] = sem_embedder_key
        sem_model_name = st.text_input("model_name (semantic)", key="semantic_model_name")
        similarity_threshold = st.slider(
            "similarity_threshold (lower = more, smaller chunks)",
            0.0, 1.0, 0.3, 0.05, key="semantic_sim_threshold",
        )
        sem_embedder_params = {"model_name": sem_model_name, "normalize": True}
        chunk_params = {"similarity_threshold": float(similarity_threshold)}
    else:
        col_a, col_b = st.columns(2)
        size = col_a.number_input("size (chars)", min_value=50, max_value=4000, value=512, step=50)
        overlap = col_b.number_input("overlap (chars)", min_value=0, max_value=1000, value=50, step=10)
        chunk_params = {"size": int(size), "overlap": int(overlap)}

    if st.button("Chunk", type="primary"):
        try:
            result = client.chunk(
                parsed["document"],
                chunker=chunker_key,
                params=chunk_params,
                embedder=sem_embedder_key,
                embedder_params=sem_embedder_params,
            )
            st.session_state["chunked"] = result
            st.session_state.pop("embedded", None)  # invalidate downstream stage
            if chunker_key == "semantic":
                # Prefill Stage 3's embedder choice to match what was just used
                # here; still fully overridable in Stage 3.
                st.session_state["embedder_choice"] = sem_embedder_key
                st.session_state["model_name_input"] = sem_embedder_params["model_name"]
                st.session_state["semantic_used_embedder"] = {
                    "key": sem_embedder_key,
                    "model_name": sem_embedder_params["model_name"],
                }
            else:
                st.session_state.pop("semantic_used_embedder", None)
        except APIError as e:
            st.error(f"Chunk failed: {e}")

    chunked = st.session_state.get("chunked")
    if chunked:
        s = chunked["summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Chunks", s["chunk_count"])
        c2.metric("Char len (min/mean/max)", f"{s['char_len_min']}/{s['char_len_mean']}/{s['char_len_max']}")
        c3.metric("Mean tokens", s["token_len_mean"])
        c4.metric("Total overlap (chars)", f"{s['total_overlap_chars']:,}")
        st.caption(
            "Orange text at the start of a chunk = characters shared with the "
            "previous chunk (the overlap)."
        )
        for ch in chunked["chunks"]:
            header = (
                f"#{ch['position']}  ·  {ch['char_len']} chars  ·  "
                f"~{ch['token_len']} tok  ·  overlap {ch['overlap_with_prev']}"
            )
            with st.expander(header):
                st.markdown(highlight_overlap(ch["text"], ch["overlap_with_prev"]))
        st.caption("↻ To redo: change chunker/size/overlap above and press Chunk again.")

    # ------------------------------------------------------------------ #
    # Stage 3 — Embedding
    # ------------------------------------------------------------------ #
    st.header("Stage 3 · Embedding")
    if not chunked:
        st.info("Chunk the document first to enable embedding.")
        return

    try:
        embedders = client.list_embedders()
    except APIError as e:
        st.error(f"Could not load embedder options: {e}")
        return

    sem_used = st.session_state.get("semantic_used_embedder")
    if sem_used:
        st.info(
            f"Semantic chunking (Stage 2) used **{sem_used['key']}** / "
            f"`{sem_used['model_name']}`. Pick the embedder/model for the final "
            "vectors below (same or different — your choice)."
        )

    embedder_key = option_picker("Embedder", embedders, "embedder_choice")
    col_m, col_n, col_d = st.columns([3, 1, 1.6])
    if "model_name_input" not in st.session_state:
        st.session_state["model_name_input"] = "all-MiniLM-L6-v2"
    model_name = col_m.text_input("model_name", key="model_name_input")
    normalize = col_n.checkbox("normalize", value=True)
    custom_dim = col_d.number_input(
        "output dim (0 = model default)",
        min_value=0,
        max_value=4096,
        value=0,
        step=1,
        help="Truncate each vector to this many dimensions (then re-normalize). "
        "0 = the model's native dimension. Lossy for non-Matryoshka models like MiniLM.",
    )
    truncate_dim = int(custom_dim) if custom_dim and custom_dim > 0 else None

    if st.button("Show model details"):
        with st.spinner("Loading model to read its details…"):
            try:
                st.session_state["model_info"] = client.embedder_model_info(
                    embedder_key,
                    model_name=model_name,
                    normalize=bool(normalize),
                    truncate_dim=truncate_dim,
                )
            except APIError as e:
                st.error(f"Could not load model details: {e}")
    if st.session_state.get("model_info"):
        render_model_info(st.session_state["model_info"])

    if st.button("Embed", type="primary"):
        with st.spinner("Loading model & embedding chunks… (first run may take a moment)"):
            try:
                params = {"model_name": model_name, "normalize": bool(normalize)}
                if truncate_dim:
                    params["truncate_dim"] = truncate_dim
                result = client.embed(chunked["chunks"], embedder=embedder_key, params=params)
                st.session_state["embedded"] = result
                st.session_state.pop("store", None)   # invalidate downstream stage
                st.session_state.pop("search", None)
            except APIError as e:
                st.error(f"Embed failed: {e}")

    embedded = st.session_state.get("embedded")
    if embedded:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Vectors", embedded["count"])
        c2.metric("Dimension", embedded["dimension"])
        c3.metric("Embed time (s)", embedded["embed_time_sec"])
        c4.metric("Cost (USD)", embedded.get("cost_usd") if embedded.get("cost_usd") is not None else "—")
        st.caption(f"Model: `{embedded['model_name']}`  ·  normalize = {embedded['normalize']}")
        render_model_info(embedded["model_info"])
        with st.expander("Raw vector preview (first vector, first 12 dims)", expanded=True):
            st.code(embedded["value_preview"])
            st.caption(
                f"L2 norms of first vectors: {embedded['norms_preview']}  "
                f"(≈1.0 confirms normalization)"
            )
        with st.expander("Embedding metadata (travels with the vectors, checked at query time)"):
            st.json(embedded["meta"])
        st.caption("↻ To redo: change embedder/model/normalize above and press Embed again.")

    # ------------------------------------------------------------------ #
    # Stage 4 — Vector store
    # ------------------------------------------------------------------ #
    st.header("Stage 4 · Vector store")
    if not embedded:
        st.info("Embed the chunks first to enable the vector store.")
        return

    try:
        vectorstores = client.list_vectorstores()
    except APIError as e:
        st.error(f"Could not load vector store options: {e}")
        return

    vs_key = option_picker("Vector store", vectorstores, "vs_choice")
    col_i, col_me = st.columns(2)
    index_type = col_i.radio("index_type", ["flat", "hnsw"], horizontal=True,
                             help="flat = exact search; hnsw = approximate but fast at scale.")
    metric = col_me.radio("metric", ["cosine", "l2", "dot"], horizontal=True)

    if st.button("Build vector store", type="primary"):
        with st.spinner("Building index…"):
            try:
                mi = embedded["model_info"]
                embed_params = {"model_name": mi["model_name"], "normalize": mi["normalize"]}
                if mi.get("truncate_dim"):
                    embed_params["truncate_dim"] = mi["truncate_dim"]
                result = client.build_vectorstore(
                    vectors=embedded["vectors"],
                    chunks=chunked["chunks"],
                    vectorstore=vs_key,
                    params={"index_type": index_type, "metric": metric},
                    embedder=embedded["embedder"],
                    embed_params=embed_params,
                    meta=embedded["meta"],
                )
                st.session_state["store"] = result
                st.session_state.pop("search", None)
            except APIError as e:
                st.error(f"Build failed: {e}")

    store = st.session_state.get("store")
    if store:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records", store["count"])
        c2.metric("Index", store["index_type"])
        c3.metric("Metric", store["metric"])
        c4.metric("Dimension", store["dimension"])
        st.caption(f"store_id `{store['store_id']}`  ·  model `{store['model_name']}`")
        with st.expander("Metadata table (first rows)"):
            st.dataframe(store["metadata_sample"], use_container_width=True)
        st.caption("↻ To redo: change index/metric above and press Build again.")

        # --- save as an immutable Recipe ---
        st.markdown("**Save this pipeline as a Recipe** (config + all artifacts, reproducible)")
        name = st.text_input("Name (optional)", key="recipe_name")
        desc = st.text_input("Description (optional)", key="recipe_desc")
        st.caption(
            "No name? It falls back to the description, then to the auto-generated "
            "recipe id."
        )
        save_disabled = uploaded is None
        if save_disabled:
            st.caption("Re-upload the document above to enable saving.")
        if st.button("💾 Build & Save Recipe", disabled=save_disabled):
            import base64
            config = {
                "parser": {"key": parser_key},
                # chunked["params"] echoes back whatever this chunker actually
                # ran with (size/overlap, similarity_threshold, ...) — generic
                # across chunkers rather than assuming size/overlap exist.
                "chunker": {"key": chunker_key, **chunked["params"]},
                "embedder": {
                    "key": embedder_key,
                    "model_name": model_name,
                    "normalize": bool(normalize),
                    **({"truncate_dim": truncate_dim} if truncate_dim else {}),
                },
                "vectorstore": {"key": vs_key, "index_type": index_type, "metric": metric},
            }
            with st.spinner("Building & saving recipe (re-runs the pipeline server-side)…"):
                try:
                    b64 = base64.b64encode(uploaded.getvalue()).decode()
                    rec = client.create_recipe(
                        uploaded.name, b64, config, name=name or None, description=desc or None
                    )
                    st.success(
                        f"Saved '{rec['name']}' ({rec['recipe_id']}) · {rec['chunk_count']} chunks · "
                        f"{rec['build_time_sec']}s. Open the **Recipes** tab to query it."
                    )
                except APIError as e:
                    st.error(f"Save failed: {e}")

        # --- query the store ---
        st.subheader("Query this store")
        query = st.text_input("Question", value="How was the pneumonia treated in hospital?")
        top_k = st.slider("top_k", min_value=1, max_value=10, value=4)
        retriever_key, retriever_params = retriever_controls(client, "store")
        if st.button("Search"):
            with st.spinner("Retrieving…"):
                try:
                    st.session_state["search"] = client.search(
                        store["store_id"], query, top_k=top_k,
                        retriever=retriever_key, retriever_params=retriever_params,
                    )
                except APIError as e:
                    st.error(f"Search failed: {e}")

        search = st.session_state.get("search")
        if search:
            better = "higher = more similar" if search["metric"] in ("cosine", "dot") else "lower = more similar"
            note = f" · scores are the {search.get('retriever')} strategy's own scale" if search.get("retriever") in ("rerank", "hybrid") else f" ({better})"
            st.caption(f"Retrieved {len(search['hits'])} chunks · retriever = {search.get('retriever')} · metric = {search['metric']}{note}")
            for rank, h in enumerate(search["hits"], 1):
                with st.expander(f"#{rank}  ·  score {h['score']:.4f}  ·  [{h['chunk_id']}]  pos {h['position']}"):
                    st.write(h["text"])


def render_recipes_tab(client) -> None:
    st.header("Saved recipes")
    st.caption("Recipes are immutable: config + all stage artifacts + the vector store, saved to disk.")

    try:
        recipes = client.list_recipes()
    except APIError as e:
        st.error(f"Could not load recipes: {e}")
        return

    if not recipes:
        st.info("No recipes yet. Build a pipeline in the **Build** tab and click "
                "'💾 Build & Save Recipe'.")
        return

    cols = ["recipe_id", "name", "source_filename", "chunker", "embedder", "model_name",
            "dimension", "vectorstore", "index_type", "metric", "chunk_count",
            "build_time_sec", "created_at", "description"]
    st.dataframe(
        [{c: r.get(c) for c in cols} for r in recipes],
        use_container_width=True,
    )

    st.subheader("Query a saved recipe")
    ids = [r["recipe_id"] for r in recipes]
    names_by_id = {r["recipe_id"]: r.get("name") or r["recipe_id"] for r in recipes}
    selected = st.selectbox(
        "Recipe", ids, key="recipe_select",
        format_func=lambda rid: f"{names_by_id[rid]} ({rid})",
    )
    query = st.text_input("Question", value="How was the pneumonia treated in hospital?",
                          key="recipe_query")
    top_k = st.slider("top_k", min_value=1, max_value=10, value=4, key="recipe_topk")
    retriever_key, retriever_params = retriever_controls(client, "recipe")
    st.caption(
        "Retrieval is chosen per query, not baked into the recipe — probe the "
        "same saved recipe with each strategy and compare."
    )
    if st.button("Search recipe"):
        with st.spinner("Retrieving from the saved store…"):
            try:
                st.session_state["recipe_search"] = client.search_recipe(
                    selected, query, top_k,
                    retriever=retriever_key, retriever_params=retriever_params,
                )
            except APIError as e:
                st.error(f"Search failed: {e}")

    res = st.session_state.get("recipe_search")
    if res:
        better = "higher = more similar" if res["metric"] in ("cosine", "dot") else "lower = more similar"
        st.caption(
            f"Recipe `{res['store_id']}` · model `{res['model_name']}` · "
            f"retriever {res.get('retriever')} · metric {res['metric']} ({better}) · "
            f"{len(res['hits'])} chunks"
        )
        for rank, h in enumerate(res["hits"], 1):
            with st.expander(f"#{rank}  ·  score {h['score']:.4f}  ·  [{h['chunk_id']}]  pos {h['position']}"):
                st.write(h["text"])


SCORE_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def _fmt_score(value) -> str:
    """A score cell for the grid/table: 3 decimals, or '—' when not computed."""
    return "—" if value is None else f"{value:.3f}"


def render_compare_tab(client) -> None:
    st.header("Query & Compare")
    st.caption(
        "Ask one question against a grid of saved recipes × LLM providers, then "
        "(optionally) score every answer with RAGAS to get a concrete ranking."
    )

    try:
        recipes = client.list_recipes()
        providers = client.list_llm_providers()
    except APIError as e:
        st.error(f"Could not load recipes/providers: {e}")
        return

    if not recipes:
        st.info("No recipes yet. Build and save at least one in the **Build** tab.")
        return

    ids = [r["recipe_id"] for r in recipes]
    names_by_id = {r["recipe_id"]: r.get("name") or r["recipe_id"] for r in recipes}
    chosen_recipes = st.multiselect(
        "Recipes (grid rows)", ids, default=ids[: min(2, len(ids))],
        format_func=lambda rid: f"{names_by_id[rid]} ({rid})",
    )
    prov_keys = [p["key"] for p in providers]
    prov_by_key = {p["key"]: p for p in providers}
    default_prov = ["ollama"] if "ollama" in prov_keys else prov_keys[:1]
    chosen_providers = st.multiselect(
        "LLM providers (grid columns)", prov_keys, default=default_prov,
        format_func=lambda k: prov_by_key[k]["name"],
    )

    question = st.text_input(
        "Question", value="How was the pneumonia treated in hospital?", key="cmp_q"
    )
    top_k = st.slider("top_k", min_value=1, max_value=10, value=4, key="cmp_topk")
    retriever_key, retriever_params = retriever_controls(client, "compare")
    ground_truth = st.text_area(
        "Reference answer (optional)",
        key="cmp_gt",
        help="A known-correct answer. Enables the context_recall metric; without "
        "it, recall is reported as '—' rather than guessed.",
    )

    if st.button("Run comparison grid", type="primary",
                 disabled=not (chosen_recipes and chosen_providers)):
        cells = []
        total = len(chosen_recipes) * len(chosen_providers)
        prog = st.progress(0.0, text="Generating…")
        done = 0
        for rid in chosen_recipes:
            for prov in chosen_providers:
                cell = {"recipe_id": rid, "provider": prov}
                try:
                    res = client.generate_recipe(
                        rid, question, provider=prov, top_k=top_k,
                        retriever=retriever_key, retriever_params=retriever_params,
                    )
                    cell.update(
                        answer=res["answer"],
                        latency_ms=res["latency_ms"],
                        cost_usd=res.get("cost_usd"),
                        model=res.get("model"),
                        contexts=[c["text"] for c in res["retrieved_chunks"]],
                        n_chunks=len(res["retrieved_chunks"]),
                    )
                except APIError as e:
                    # Fail loudly per cell (e.g. provider needs a key / server) but
                    # keep the rest of the grid running.
                    cell["error"] = str(e)
                cells.append(cell)
                done += 1
                prog.progress(done / total, text=f"Generating… ({done}/{total})")
        prog.empty()
        st.session_state["compare"] = {
            "question": question, "ground_truth": ground_truth.strip() or None,
            "cells": cells,
        }
        st.session_state.pop("compare_scores", None)  # invalidate old scores

    grid = st.session_state.get("compare")
    if not grid:
        return

    scores_by_key = st.session_state.get("compare_scores", {})

    # --- the grid: one row per recipe, one column per provider ---
    st.subheader("Grid")
    st.caption(f"Question: _{grid['question']}_")
    row_recipes = sorted({c["recipe_id"] for c in grid["cells"]})
    col_providers = sorted({c["provider"] for c in grid["cells"]})
    by_pos = {(c["recipe_id"], c["provider"]): c for c in grid["cells"]}

    for rid in row_recipes:
        st.markdown(f"**{rid}**")
        cols = st.columns(len(col_providers))
        for col, prov in zip(cols, col_providers):
            cell = by_pos.get((rid, prov))
            with col:
                st.markdown(f"`{prov}`")
                if cell is None:
                    st.caption("—")
                    continue
                if "error" in cell:
                    st.error(cell["error"])
                    continue
                st.write(cell["answer"])
                meta = f"⏱ {cell['latency_ms']:.0f} ms"
                if cell.get("cost_usd") is not None:
                    meta += f" · ${cell['cost_usd']:.4f}"
                meta += f" · {cell['n_chunks']} chunks"
                st.caption(meta)
                cell_scores = scores_by_key.get((rid, prov))
                if cell_scores:
                    st.caption(
                        " · ".join(
                            f"{m.split('_')[0]}={_fmt_score(cell_scores.get(m))}"
                            for m in SCORE_METRICS
                        )
                        + f" · mean={_fmt_score(cell_scores.get('mean'))}"
                    )
        st.divider()

    # --- optional scoring ---
    st.subheader("Automated quality scoring (RAGAS)")
    try:
        status = client.score_status()
    except APIError as e:
        st.error(f"Could not check scoring availability: {e}")
        return

    if not status.get("available"):
        st.info(status.get("message") or "Install ragas to enable scoring.")
        return

    st.caption(
        "Scores every answer for faithfulness, answer relevancy, and context "
        "precision/recall. Uses a judge LLM via ragas (needs OPENAI_API_KEY)."
    )
    if st.button("Score all answers"):
        cells_to_score = [
            {
                "recipe_id": c["recipe_id"],
                "provider": c["provider"],
                "question": grid["question"],
                "answer": c["answer"],
                "contexts": c.get("contexts", []),
                "ground_truth": grid.get("ground_truth"),
            }
            for c in grid["cells"]
            if "error" not in c
        ]
        if not cells_to_score:
            st.warning("No successful answers to score.")
        else:
            with st.spinner("Scoring answers with ragas (this calls a judge LLM)…"):
                try:
                    resp = client.score_cells(cells_to_score, sort_by="faithfulness")
                except APIError as e:
                    st.error(f"Scoring failed: {e}")
                    resp = None
            if resp is not None:
                if not resp.get("available"):
                    st.info(resp.get("message") or "Scoring unavailable.")
                else:
                    st.session_state["compare_scores"] = {
                        (c["recipe_id"], c["provider"]): c["scores"]
                        for c in resp["cells"]
                    }
                    st.rerun()

    # --- ranking summary (re-sortable by any metric, no backend round-trip) ---
    if scores_by_key:
        st.markdown("**Ranking summary**")
        sort_by = st.selectbox(
            "Sort by", list(SCORE_METRICS) + ["mean"], key="cmp_sortby",
            help="Higher is better for every RAGAS metric.",
        )
        rows = [
            {
                "recipe_id": rid, "provider": prov,
                **{m: sc.get(m) for m in SCORE_METRICS}, "mean": sc.get("mean"),
            }
            for (rid, prov), sc in scores_by_key.items()
        ]
        # Best first; cells missing the chosen metric sort last.
        rows.sort(key=lambda r: (r[sort_by] is None, -(r[sort_by] or 0.0)))
        table = [
            {
                "rank": i + 1, "recipe": r["recipe_id"], "provider": r["provider"],
                **{m: _fmt_score(r[m]) for m in list(SCORE_METRICS) + ["mean"]},
            }
            for i, r in enumerate(rows)
        ]
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption("'—' = metric not computed (e.g. context_recall with no reference answer).")


if __name__ == "__main__":
    main()
