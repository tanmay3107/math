from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from vector_store import VectorStore

app = FastAPI(
    title="VectorStore REST API",
    description="REST API for VectorStore with HNSW ANN search, BM25 keyword search, and RRF hybrid search.",
    version="1.0.0"
)

# Global VectorStore instance with HNSW backend enabled
store = VectorStore(use_hnsw=True)

# --- Pydantic Request & Response Schemas ---

class AddRecordRequest(BaseModel):
    id: str
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None
    normalize: bool = False

class AddBatchRequest(BaseModel):
    records: List[Dict[str, Any]]
    normalize: bool = False

class UpdateRecordRequest(BaseModel):
    vector: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    normalize: bool = False

class DenseSearchRequest(BaseModel):
    query_vector: List[float]
    k: int = Field(default=5, ge=1)
    metric: str = "cosine"
    filter_metadata: Optional[Dict[str, Any]] = None
    backend: str = "exact"

class KeywordSearchRequest(BaseModel):
    query_text: str
    k: int = Field(default=5, ge=1)

class HybridSearchRequest(BaseModel):
    query_vector: List[float]
    query_text: str
    k: int = Field(default=5, ge=1)
    rrf_k: int = 60
    metric: str = "cosine"
    backend: str = "exact"

class FilePathRequest(BaseModel):
    filepath: str

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "VectorStore API",
        "total_documents": len(store.vectors),
        "hnsw_enabled": store.use_hnsw
    }

@app.post("/vectors", status_code=status.HTTP_201_CREATED)
def add_vector(req: AddRecordRequest):
    store.add(req.id, req.vector, req.metadata, normalize=req.normalize)
    return {"message": f"Vector '{req.id}' added successfully.", "id": req.id}

@app.post("/vectors/batch", status_code=status.HTTP_201_CREATED)
def add_vector_batch(req: AddBatchRequest):
    try:
        store.add_batch(req.records, normalize=req.normalize)
        return {"message": f"Batch of {len(req.records)} vectors added successfully."}
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/vectors/{doc_id}")
def get_vector(doc_id: str):
    record = store.get(doc_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Vector ID '{doc_id}' not found.")
    return record

@app.put("/vectors/{doc_id}")
def update_vector(doc_id: str, req: UpdateRecordRequest):
    success = store.update(doc_id, vector=req.vector, metadata=req.metadata, normalize=req.normalize)
    if not success:
        raise HTTPException(status_code=404, detail=f"Vector ID '{doc_id}' not found.")
    return {"message": f"Vector '{doc_id}' updated successfully.", "id": doc_id}

@app.delete("/vectors/{doc_id}")
def delete_vector(doc_id: str):
    success = store.delete(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Vector ID '{doc_id}' not found.")
    return {"message": f"Vector '{doc_id}' deleted successfully.", "id": doc_id}

@app.post("/search/dense")
def search_dense(req: DenseSearchRequest):
    try:
        results = store.search(
            query_vector=req.query_vector,
            k=req.k,
            metric=req.metric,
            filter_metadata=req.filter_metadata,
            backend=req.backend
        )
        return {"results": results, "count": len(results)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/search/keyword")
def search_keyword(req: KeywordSearchRequest):
    results = store.keyword_search(query_text=req.query_text, k=req.k)
    return {"results": results, "count": len(results)}

@app.post("/search/hybrid")
def search_hybrid(req: HybridSearchRequest):
    try:
        results = store.hybrid_search(
            query_vector=req.query_vector,
            query_text=req.query_text,
            k=req.k,
            rrf_k=req.rrf_k,
            metric=req.metric,
            backend=req.backend
        )
        return {"results": results, "count": len(results)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/store/save")
def save_store(req: FilePathRequest):
    try:
        store.save_to_json(req.filepath)
        return {"message": f"Store saved to '{req.filepath}' successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save store: {str(e)}")

@app.post("/store/load")
def load_store(req: FilePathRequest):
    try:
        store.load_from_json(req.filepath)
        return {"message": f"Store loaded from '{req.filepath}' successfully.", "total_documents": len(store.vectors)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load store: {str(e)}")