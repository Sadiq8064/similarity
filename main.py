from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import openai
import os

# -----------------------------------------------------
# LOAD OPENAI KEY FROM ENV VARIABLES
# -----------------------------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise Exception("OPENAI_API_KEY not found in environment variables")

# -----------------------------------------------------
# FASTAPI APP
# -----------------------------------------------------
app = FastAPI(title="Embedding + Similarity API")

# -----------------------------------------------------
# MODELS
# -----------------------------------------------------

class ChunkItem(BaseModel):
    chunk_id: str
    text: str

class ChunkEmbedRequest(BaseModel):
    store_id: str
    document_id: str
    chunks: List[ChunkItem]

class ChunkEmbedResponseItem(BaseModel):
    chunk_id: str
    text: str
    embedding: List[float]

class ChunkEmbedResponse(BaseModel):
    store_id: str
    document_id: str
    embeddings: List[ChunkEmbedResponseItem]

# New model for /embed-query endpoint
class QueryEmbedRequest(BaseModel):
    query: str

class QueryEmbedResponse(BaseModel):
    embedding: List[float]

# -----------------------------------------------------
# FUNCTION: Compute Embedding
# -----------------------------------------------------
async def get_embedding(text: str):
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    except Exception as err:
        print("Embedding Error:", err)
        raise HTTPException(status_code=500, detail=str(err))


# -----------------------------------------------------
# FUNCTION: COSINE SIMILARITY
# -----------------------------------------------------
def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# -----------------------------------------------------
# ROUTE #1 — Similarity Check
# -----------------------------------------------------
@app.get("/similarity")
async def similarity(text1: str, text2: str):
    try:
        emb1 = await get_embedding(text1)
        emb2 = await get_embedding(text2)
        sim_value = cosine_similarity(emb1, emb2)
        return {"similarity": sim_value}

    except Exception as err:
        print("Similarity Error:", err)
        raise HTTPException(status_code=500, detail=str(err))


# -----------------------------------------------------
# ROUTE #2 — Chunk Embedding API
# -----------------------------------------------------
@app.post("/embed-chunks", response_model=ChunkEmbedResponse)
async def embed_chunks(req: ChunkEmbedRequest):

    try:
        output_items = []

        for chunk in req.chunks:
            emb = await get_embedding(chunk.text)
            output_items.append(
                ChunkEmbedResponseItem(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    embedding=emb
                )
            )

        return ChunkEmbedResponse(
            store_id=req.store_id,
            document_id=req.document_id,
            embeddings=output_items
        )

    except Exception as err:
        print("Chunk Embedding Error:", err)
        raise HTTPException(status_code=500, detail=str(err))


# -----------------------------------------------------
# 🚀 ROUTE #3 — Query Embedding API (NEW)
# -----------------------------------------------------
@app.post("/embed-query", response_model=QueryEmbedResponse)
async def embed_query(req: QueryEmbedRequest):
    """
    Accepts:
    {
        "query": "What is diabetes?"
    }

    Returns:
    {
        "embedding": [...]
    }
    """
    try:
        emb = await get_embedding(req.query)
        return QueryEmbedResponse(embedding=emb)

    except Exception as err:
        print("Query Embedding Error:", err)
        raise HTTPException(status_code=500, detail=str(err))
