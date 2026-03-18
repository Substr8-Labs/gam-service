#!/usr/bin/env python3
"""
GAM pgvector Service
- Indexes git commits into pgvector for fast semantic search
- Provides retrieval API
"""

import os
import hashlib
import json
from datetime import datetime
from typing import Optional, List
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
import git
from openai import OpenAI
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Config
DB_HOST = os.getenv("GAM_DB_HOST", "localhost")
DB_PORT = int(os.getenv("GAM_DB_PORT", "5433"))
DB_USER = os.getenv("GAM_DB_USER", "gam")
DB_PASSWORD = os.getenv("GAM_DB_PASSWORD", "gam_dev_password")
DB_NAME = os.getenv("GAM_DB_NAME", "gam")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"

# FastAPI app
app = FastAPI(title="GAM pgvector Service", version="0.1.0")


def get_db_connection():
    """Get database connection with pgvector support."""
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME
    )
    register_vector(conn)
    return conn


def get_openai_client():
    """Get OpenAI client for embeddings."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")
    return OpenAI(api_key=OPENAI_API_KEY)


def compute_content_hash(content: str) -> str:
    """SHA256 hash of content for deduplication."""
    return hashlib.sha256(content.encode()).hexdigest()


def generate_embedding(client: OpenAI, text: str) -> List[float]:
    """Generate embedding using OpenAI."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def chunk_content(content: str, max_chars: int = 2000) -> List[str]:
    """Split content into chunks for embedding."""
    if len(content) <= max_chars:
        return [content]
    
    chunks = []
    lines = content.split('\n')
    current_chunk = []
    current_length = 0
    
    for line in lines:
        if current_length + len(line) + 1 > max_chars and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [line]
            current_length = len(line)
        else:
            current_chunk.append(line)
            current_length += len(line) + 1
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks


class IndexRequest(BaseModel):
    repo_path: str
    agent_id: str
    file_patterns: List[str] = ["*.md", "*.txt", "*.json"]


class SearchRequest(BaseModel):
    agent_id: str
    query: str
    limit: int = 10
    customer_id: Optional[str] = None
    min_similarity: float = 0.1


class SearchResult(BaseModel):
    content: str
    file_path: str
    commit_hash: str
    committed_at: datetime
    similarity: float


@app.get("/health")
def health():
    """Health check."""
    try:
        conn = get_db_connection()
        conn.close()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.post("/index")
def index_repo(request: IndexRequest):
    """Index a git repository into pgvector."""
    repo_path = Path(request.repo_path)
    if not repo_path.exists():
        raise HTTPException(status_code=404, detail=f"Repo not found: {repo_path}")
    
    try:
        repo = git.Repo(repo_path)
    except git.InvalidGitRepositoryError:
        raise HTTPException(status_code=400, detail=f"Not a git repo: {repo_path}")
    
    conn = get_db_connection()
    cur = conn.cursor()
    openai_client = get_openai_client()
    
    indexed_count = 0
    skipped_count = 0
    
    try:
        # Get last indexed commit
        cur.execute(
            "SELECT last_commit_hash FROM indexing_state WHERE repo_path = %s",
            (str(repo_path),)
        )
        row = cur.fetchone()
        last_commit = row[0] if row else None
        
        # Get commits to process
        if last_commit:
            try:
                commits = list(repo.iter_commits(f"{last_commit}..HEAD"))
            except git.GitCommandError:
                commits = list(repo.iter_commits())
        else:
            commits = list(repo.iter_commits())
        
        # Process each commit
        for commit in reversed(commits):  # Oldest first
            commit_hash = commit.hexsha
            committed_at = datetime.fromtimestamp(commit.committed_date)
            
            # Get files changed in this commit
            for item in commit.tree.traverse():
                if item.type != 'blob':
                    continue
                
                file_path = item.path
                
                # Check file pattern
                if not any(Path(file_path).match(p) for p in request.file_patterns):
                    continue
                
                try:
                    content = item.data_stream.read().decode('utf-8')
                except:
                    continue
                
                # Chunk and embed
                chunks = chunk_content(content)
                for chunk in chunks:
                    content_hash = compute_content_hash(chunk)
                    
                    # Check if already indexed
                    cur.execute(
                        "SELECT 1 FROM memory_entries WHERE commit_hash = %s AND file_path = %s AND content_hash = %s",
                        (commit_hash, file_path, content_hash)
                    )
                    if cur.fetchone():
                        skipped_count += 1
                        continue
                    
                    # Generate embedding
                    embedding = generate_embedding(openai_client, chunk)
                    
                    # Insert
                    cur.execute("""
                        INSERT INTO memory_entries 
                        (commit_hash, file_path, content, agent_id, committed_at, embedding, content_hash)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (commit_hash, file_path, chunk, request.agent_id, committed_at, embedding, content_hash))
                    
                    indexed_count += 1
                    
                    # Commit every 10 entries to avoid losing progress
                    if indexed_count % 10 == 0:
                        conn.commit()
            
            # Update indexing state after each commit
            cur.execute("""
                INSERT INTO indexing_state (repo_path, last_commit_hash, last_indexed_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (repo_path) DO UPDATE SET 
                    last_commit_hash = EXCLUDED.last_commit_hash,
                    last_indexed_at = NOW()
            """, (str(repo_path), commit_hash))
            conn.commit()  # Commit after each git commit processed
        
        conn.commit()  # Final commit
        
        return {
            "status": "success",
            "indexed": indexed_count,
            "skipped": skipped_count,
            "commits_processed": len(commits)
        }
    
    finally:
        cur.close()
        conn.close()


@app.post("/search", response_model=List[SearchResult])
def search_memory(request: SearchRequest):
    """Semantic search over indexed memory."""
    conn = get_db_connection()
    cur = conn.cursor()
    openai_client = get_openai_client()
    
    try:
        # Generate query embedding
        query_embedding = generate_embedding(openai_client, request.query)
        
        # Search using pgvector
        cur.execute("""
            SELECT content, file_path, commit_hash, committed_at,
                   1 - (embedding <=> %s::vector) as similarity
            FROM memory_entries
            WHERE agent_id = %s
              AND (%s IS NULL OR customer_id = %s)
              AND embedding IS NOT NULL
              AND 1 - (embedding <=> %s::vector) >= %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (
            query_embedding, 
            request.agent_id, 
            request.customer_id,
            request.customer_id,
            query_embedding,
            request.min_similarity,
            query_embedding,
            request.limit
        ))
        
        results = []
        for row in cur.fetchall():
            results.append(SearchResult(
                content=row[0],
                file_path=row[1],
                commit_hash=row[2],
                committed_at=row[3],
                similarity=row[4]
            ))
        
        # Audit log
        cur.execute("""
            INSERT INTO memory_audit (operation, agent_id, customer_id, details)
            VALUES ('search', %s, %s, %s)
        """, (request.agent_id, request.customer_id, json.dumps({
            "query": request.query,
            "results_count": len(results)
        })))
        conn.commit()
        
        return results
    
    finally:
        cur.close()
        conn.close()


@app.get("/stats")
def get_stats(agent_id: str = Query(...)):
    """Get memory stats for an agent."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            SELECT 
                COUNT(*) as total_entries,
                COUNT(DISTINCT commit_hash) as total_commits,
                COUNT(DISTINCT file_path) as total_files,
                MIN(committed_at) as earliest,
                MAX(committed_at) as latest
            FROM memory_entries
            WHERE agent_id = %s
        """, (agent_id,))
        
        row = cur.fetchone()
        return {
            "agent_id": agent_id,
            "total_entries": row[0],
            "total_commits": row[1],
            "total_files": row[2],
            "earliest_commit": row[3],
            "latest_commit": row[4]
        }
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8091"))
    uvicorn.run(app, host="0.0.0.0", port=port)

@app.get("/debug/embedding-status")
def embedding_status(agent_id: str = "ada"):
    """Debug endpoint to check embedding status."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(embedding) as with_embedding,
                COUNT(*) - COUNT(embedding) as without_embedding
            FROM memory_entries 
            WHERE agent_id = %s
        """, (agent_id,))
        row = cur.fetchone()
        return {
            "agent_id": agent_id,
            "total": row[0],
            "with_embedding": row[1],
            "without_embedding": row[2]
        }
    finally:
        cur.close()
        conn.close()
