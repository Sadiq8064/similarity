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
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    except Exception as err:
        print("Embedding Error:", err)
        raise HTTPException(status_code=500, detail=str(err))


# -----------------------------------------------------
# FUNCTION: Cosine Similarity
# -----------------------------------------------------
def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# -----------------------------------------------------
# GET Endpoint
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
