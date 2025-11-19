from fastapi import FastAPI, HTTPException
import numpy as np
import openai
import os

# -----------------------------------------------------
# LOAD OPENAI KEY FROM ENV VARIABLES
# -----------------------------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise Exception("OPENAI_API_KEY not found in environment variables")


app = FastAPI()


# -----------------------------------------------------
# FUNCTION: Get Embedding Vector
# -----------------------------------------------------
async def get_embedding(text: str):
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",  # or gpt-4o-mini-embedding
            input=text
        )
        return response.data[0].embedding

    except Exception as e:
        print("Embedding error:", e)
        raise HTTPException(status_code=500, detail="Failed to generate embeddings")


# -----------------------------------------------------
# FUNCTION: Cosine Similarity
# -----------------------------------------------------
def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# -----------------------------------------------------
# GET Endpoint
# Example:
# /similarity?text1=hello&text2=hi
# -----------------------------------------------------
@app.get("/similarity")
async def similarity(text1: str, text2: str):
    emb1 = await get_embedding(text1)
    e
