from __future__ import annotations

LATE_INTERACTION_MODELS = {
    "colbert-ir/colbertv2.0",
    "lightonai/colbertv2.0",
    "lightonai/GTE-ModernColBERT-v1",
    "lightonai/Reason-ModernColBERT",
    "answerdotai/answerai-colbert-small-v1",
    "mixedbread-ai/mxbai-edge-colbert-v0-17m",
    "mixedbread-ai/mxbai-edge-colbert-v0-32m",
    "jinaai/jina-colbert-v2",
    "LiquidAI/LFM2-ColBERT-350M",
}

DENSE_MODELS = {
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "Alibaba-NLP/gte-large-en-v1.5",
    "intfloat/e5-large-v2",
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B",
    "nvidia/NV-Embed-v2",
}

PHASE1_MODELS = {
    "BAAI/bge-small-en-v1.5",
    "mixedbread-ai/mxbai-edge-colbert-v0-32m",
}


def infer_family(model_id: str, explicit_family: str | None = None) -> str:
    if explicit_family:
        if explicit_family not in {"dense", "late_interaction", "hybrid"}:
            raise ValueError("--family must be one of: dense, late_interaction, hybrid")
        return explicit_family
    if model_id in LATE_INTERACTION_MODELS or "colbert" in model_id.lower():
        return "late_interaction"
    return "dense"


def sanitize_model_id(model_id: str) -> str:
    return model_id.replace("/", "__").replace(" ", "_")
