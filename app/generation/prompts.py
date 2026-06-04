SYSTEM_PROMPT = """\
You are a Kubernetes documentation assistant. Your sole purpose is to answer
questions about Kubernetes using the official documentation excerpts provided.

Rules:
- Answer ONLY using information present in the supplied documentation context.
- If the answer is not found in the context, respond with exactly:
  "I could not find sufficient information in the indexed Kubernetes documentation to answer this question."
- Always cite sources by referencing document titles and URLs from the context.
- Never invent kubectl commands, API fields, or YAML configurations.
- Format command-line examples and YAML manifests in fenced code blocks.
- Be precise and technical.\
"""


def build_context_block(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[DOCUMENT {i}]\n"
            f"Title: {chunk['title']}\n"
            f"URL: {chunk['url']}\n"
            f"Section: {chunk.get('section', '')}\n"
            f"---\n"
            f"{chunk['content']}"
        )
    return "\n\n".join(parts)


def build_user_message(query: str, chunks: list[dict]) -> str:
    context = build_context_block(chunks)
    return (
        f"Documentation context:\n\n{context}\n\n"
        f"{'─' * 60}\n\n"
        f"Question: {query}\n\n"
        f"Answer based only on the documentation above:"
    )
