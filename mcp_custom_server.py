import nest_asyncio
import regex as re
import numpy as np
import faiss
from mcp.server.fastmcp import FastMCP
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from sentence_transformers import SentenceTransformer

nest_asyncio.apply()
mcp = FastMCP()

GITHUB_TOKEN = "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"
session_cache = {}  # Simple in-memory session cache

@mcp.tool()
def extract_documents(owner: str, repo: str, branch: str = "main") -> dict:
    github_client = GithubClient(github_token=GITHUB_TOKEN, verbose=True)

    documents = GithubRepositoryReader(
        github_client=github_client,
        owner=owner,
        repo=repo,
        use_parser=False,
        verbose=False,
        filter_file_extensions=([".py"], GithubRepositoryReader.FilterType.INCLUDE),
    ).load_data(branch=branch)

    if not documents:
        return {"error": "No Python files found in repo."}

    texts = [doc.text_resource.text for doc in documents]
    session_cache["documents"] = texts
    return {"message": f"{len(texts)} document(s) extracted.", "documents": texts}

@mcp.tool()
def preprocess_text() -> dict:
    documents = session_cache.get("documents")
    if not documents:
        return {"error": "No documents in session. Run extract_documents first."}
    
    full_text = "\n".join(documents)
    split_parts = [part.strip() for part in re.split(r'(?=\/\*\*)', full_text) if part.strip()]
    session_cache["split_parts"] = split_parts
    return {"parts_count": len(split_parts), "sample": split_parts[0][:200]}

@mcp.tool()
def generate_embeddings() -> dict:
    parts = session_cache.get("split_parts")
    if not parts:
        return {"error": "No split parts. Run preprocess_text first."}
    
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = model.encode(parts, show_progress_bar=True)
    embedding_array = np.array(embeddings).astype('float32')

    if embedding_array.ndim == 1:
        embedding_array = embedding_array.reshape(1, -1)

    session_cache["embedding_array"] = embedding_array
    return {"embedding_shape": embedding_array.shape}

@mcp.tool()
def create_faiss_index() -> dict:
    embedding_array = session_cache.get("embedding_array")
    if embedding_array is None:
        return {"error": "No embeddings found. Run generate_embeddings first."}

    dim = embedding_array.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embedding_array)

    session_cache["faiss_index"] = index
    return {"status": "Index created", "vectors_indexed": embedding_array.shape[0]}


@mcp.tool()
def multi_query_search(queries: list[str], top_k: int = 3) -> dict:
    index = session_cache.get("faiss_index")
    parts = session_cache.get("split_parts")
    
    if index is None or parts is None:
        return {"error": "Index or split parts not found. Run earlier steps first."}

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    query_embeddings = model.encode(queries, show_progress_bar=True).astype('float32')

    if query_embeddings.ndim == 1:
        query_embeddings = query_embeddings.reshape(1, -1)

    results = []
    for i, query_embedding in enumerate(query_embeddings):
        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = index.search(query_embedding, top_k)

        matched = [
            {
                "text": parts[idx][:200],  # Preview first 200 chars
                "distance": float(dist)
            }
            for idx, dist in zip(indices[0], distances[0])
        ]

        results.append({
            "query": queries[i],
            "matches": matched
        })

    return {"results": results}

if __name__ == "__main__":
    mcp.run(transport="sse")
