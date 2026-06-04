"""
Kubernetes RAG Assistant — Streamlit UI (Phase 2)

Run with:
    streamlit run streamlit_app.py
"""
import asyncio
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="Kubernetes Documentation Assistant",
    layout="wide",
)

st.title("Kubernetes Documentation Assistant")
st.caption("Answers grounded in official Kubernetes documentation — no hallucinations.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=5)
    show_chunks = st.toggle("Show retrieved chunks", value=False)
    st.divider()
    st.markdown("**Model backend**")

    import app.config as cfg
    st.code(f"LLM:   {cfg.settings.llm_model}\nEmbed: {cfg.settings.embedding_model}", language="text")

# ── Example questions ─────────────────────────────────────────────────────────
EXAMPLES = [
    "What is a StatefulSet?",
    "Difference between Deployment and StatefulSet?",
    "How do I configure liveness probes?",
    "Explain taints and tolerations.",
    "How does kube-scheduler work?",
    "Show an example NetworkPolicy.",
]

with st.expander("Example questions"):
    cols = st.columns(3)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 3].button(example, key=f"ex{i}"):
            st.session_state["question"] = example

# ── Query input ───────────────────────────────────────────────────────────────
question = st.text_input(
    "Ask a question about Kubernetes:",
    value=st.session_state.get("question", ""),
    placeholder="What is a StatefulSet?",
)

if question:
    from app.retrieval.retriever import retrieve
    from app.generation.generator import generate_answer

    async def _run():
        chunks = await retrieve(question, top_k=top_k)
        if not chunks:
            return None, []
        result = await generate_answer(question, chunks)
        return result, chunks

    with st.spinner("Searching documentation and generating answer…"):
        result, chunks = asyncio.run(_run())

    if result is None:
        st.warning(
            "I could not find sufficient information in the indexed Kubernetes documentation "
            "to answer this question."
        )
    else:
        st.markdown("### Answer")
        st.markdown(result["answer"])

        if result["sources"]:
            st.markdown("### Sources")
            for source in result["sources"]:
                st.markdown(f"- [{source['title']}]({source['url']})")

        if show_chunks and chunks:
            st.markdown("### Retrieved Chunks")
            for i, chunk in enumerate(chunks, 1):
                score = chunk.get("rerank_score", 0.0)
                with st.expander(f"Chunk {i}: {chunk['title']}  (rerank score: {score:.3f})"):
                    st.markdown(f"**URL:** [{chunk['url']}]({chunk['url']})")
                    st.markdown(f"**Section:** {chunk.get('section', '')}")
                    st.text(chunk["content"][:1200])
