#!/usr/bin/env python3
"""
GAM Service v2 - Multi-tenant Memory Infrastructure PoC

Changes from v1:
- Connection pooling (fixes 500 errors)
- Multi-tenant isolation
- Canonical memory envelope
- Attention/salience scoring
- Idempotent writes
- Better error handling
"""

import os
import hashlib
import json
import secrets
import logging
from datetime import datetime, timezone
from typing import Optional, List
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_values, RealDictCursor
from pgvector.psycopg2 import register_vector
from openai import OpenAI
from fastapi import FastAPI, HTTPException, Query, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gam-service")

# Config
DB_HOST = os.getenv("GAM_DB_HOST", os.getenv("PGHOST", "localhost"))
DB_PORT = int(os.getenv("GAM_DB_PORT", os.getenv("PGPORT", "5432")))
DB_USER = os.getenv("GAM_DB_USER", os.getenv("PGUSER", "gam"))
DB_PASSWORD = os.getenv("GAM_DB_PASSWORD", os.getenv("PGPASSWORD", ""))
DB_NAME = os.getenv("GAM_DB_NAME", os.getenv("PGDATABASE", "gam"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536

# Attention scoring config
DEFAULT_SALIENCE = 0.5
ATTENTION_WEIGHT = float(os.getenv("GAM_ATTENTION_WEIGHT", "0.3"))

# Connection pool
db_pool: pool.ThreadedConnectionPool = None

# FastAPI app
app = FastAPI(
    title="GAM Service v2",
    version="2.0.0",
    description="Multi-tenant Memory Infrastructure for Agents"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Database Connection Pooling
# ============================================================

def init_db_pool():
    """Initialize connection pool on startup."""
    global db_pool
    try:
        db_pool = pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            dbname=DB_NAME,
            connect_timeout=10
        )
        logger.info(f"Database pool initialized: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    except Exception as e:
        logger.error(f"Failed to initialize DB pool: {e}")
        raise


@contextmanager
def get_db():
    """Get connection from pool with automatic cleanup."""
    conn = None
    try:
        conn = db_pool.getconn()
        register_vector(conn)
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            db_pool.putconn(conn)


@app.on_event("startup")
async def startup():
    init_db_pool()
    # Run migrations
    with get_db() as conn:
        run_migrations(conn)


@app.on_event("shutdown")
async def shutdown():
    if db_pool:
        db_pool.closeall()


# ============================================================
# Migrations
# ============================================================

def run_migrations(conn):
    """Run schema migrations."""
    cur = conn.cursor()
    try:
        # Check if tenants table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'tenants'
            )
        """)
        if not cur.fetchone()[0]:
            logger.info("Running v2 migrations...")
            
            # Create tenants table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tenants (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(255) NOT NULL,
                    api_key VARCHAR(64) UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    config JSONB DEFAULT '{}'
                )
            """)
            
            # Add columns to memory_entries if they don't exist
            migrations = [
                "ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS tenant_id UUID",
                "ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS source_channel VARCHAR(50) DEFAULT 'api'",
                "ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS memory_kind VARCHAR(50) DEFAULT 'note'",
                "ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS salience_score FLOAT DEFAULT 0.5",
                "ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS session_id VARCHAR(100)",
                "ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'",
                "CREATE INDEX IF NOT EXISTS idx_memory_tenant ON memory_entries(tenant_id)",
                "CREATE INDEX IF NOT EXISTS idx_memory_salience ON memory_entries(salience_score DESC)",
            ]
            
            for sql in migrations:
                try:
                    cur.execute(sql)
                except Exception as e:
                    logger.warning(f"Migration warning: {e}")
            
            # Create default tenant for Ada (backwards compatibility)
            cur.execute("""
                INSERT INTO tenants (id, name, api_key)
                VALUES ('00000000-0000-0000-0000-000000000001', 'ada', 'ada-legacy-key')
                ON CONFLICT (id) DO NOTHING
            """)
            
            # Update existing entries to default tenant
            cur.execute("""
                UPDATE memory_entries 
                SET tenant_id = '00000000-0000-0000-0000-000000000001'
                WHERE tenant_id IS NULL
            """)
            
            conn.commit()
            logger.info("Migrations complete")
        
    finally:
        cur.close()


# ============================================================
# OpenAI Client
# ============================================================

_openai_client = None

def get_openai_client():
    """Singleton OpenAI client."""
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def generate_embedding(text: str) -> List[float]:
    """Generate embedding using OpenAI."""
    client = get_openai_client()
    # Truncate to avoid token limits
    text = text[:8000]
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


# ============================================================
# Authentication
# ============================================================

async def get_tenant(x_api_key: str = Header(None, alias="X-API-Key")):
    """Validate API key and return tenant."""
    if not x_api_key:
        # Allow legacy agent_id parameter for backwards compatibility
        return None
    
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM tenants WHERE api_key = %s", (x_api_key,))
        tenant = cur.fetchone()
        cur.close()
        
        if not tenant:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return tenant


# ============================================================
# Attention Scoring
# ============================================================

def compute_salience(content: str, metadata: dict = None) -> float:
    """
    Compute salience score for a memory entry.
    
    Based on whitepaper heuristics:
    - 0.0-0.3: Low importance (routine queries, transient details)
    - 0.4-0.6: Medium importance (standard task completion)
    - 0.7-1.0: High importance (strategic decisions, emotional moments)
    """
    score = DEFAULT_SALIENCE
    content_lower = content.lower()
    
    # High importance signals
    high_signals = [
        'decided', 'decision', 'strategy', 'strategic',
        'important', 'critical', 'milestone', 'breakthrough',
        'agreed', 'confirmed', 'committed', 'promise',
        'learned', 'realized', 'insight', 'discovered',
        'remember this', 'never forget', 'key point'
    ]
    
    # Medium importance signals
    medium_signals = [
        'completed', 'finished', 'shipped', 'deployed',
        'meeting', 'discussed', 'reviewed', 'feedback',
        'todo', 'task', 'action item', 'follow up'
    ]
    
    # Low importance signals (demote)
    low_signals = [
        'heartbeat', 'health check', 'status ok',
        'routine', 'daily', 'weekly'
    ]
    
    # Check signals
    for signal in high_signals:
        if signal in content_lower:
            score = min(1.0, score + 0.15)
    
    for signal in medium_signals:
        if signal in content_lower:
            score = min(1.0, score + 0.05)
    
    for signal in low_signals:
        if signal in content_lower:
            score = max(0.1, score - 0.2)
    
    # Metadata boosts
    if metadata:
        if metadata.get('pinned'):
            score = 1.0
        if metadata.get('source_channel') == 'decision':
            score = min(1.0, score + 0.2)
    
    return round(score, 2)


# ============================================================
# Models
# ============================================================

class TenantCreate(BaseModel):
    name: str


class TenantResponse(BaseModel):
    id: str
    name: str
    api_key: str
    created_at: datetime


class MemoryEntry(BaseModel):
    content: str
    file_path: str = "api"
    source_channel: str = "api"
    memory_kind: str = "note"
    session_id: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    salience_score: Optional[float] = None  # Auto-computed if not provided


class BatchEntry(BaseModel):
    file_path: str = "api"
    content: str
    commit_hash: Optional[str] = None
    committed_at: Optional[str] = None
    source_channel: str = "api"
    memory_kind: str = "note"
    metadata: dict = Field(default_factory=dict)


class BatchIngestRequest(BaseModel):
    agent_id: str  # Kept for backwards compatibility
    entries: List[BatchEntry]


class SearchRequest(BaseModel):
    agent_id: Optional[str] = None  # Legacy, use X-API-Key instead
    query: str
    limit: int = 10
    min_similarity: float = 0.1
    include_scores: bool = False  # Return both similarity and salience
    attention_weight: Optional[float] = None  # Override default


class SearchResult(BaseModel):
    id: int
    content: str
    file_path: str
    committed_at: datetime
    similarity: float
    salience_score: float = 0.5
    final_score: float = 0.0
    source_channel: str = "api"
    memory_kind: str = "note"


# ============================================================
# Endpoints
# ============================================================

@app.get("/health")
def health():
    """Health check with pool status."""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
        return {
            "status": "healthy",
            "database": "connected",
            "version": "2.0.0",
            "pool_size": db_pool.maxconn if db_pool else 0
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


@app.post("/tenants", response_model=TenantResponse)
def create_tenant(request: TenantCreate):
    """Register a new tenant."""
    api_key = secrets.token_urlsafe(32)
    
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        try:
            cur.execute("""
                INSERT INTO tenants (name, api_key)
                VALUES (%s, %s)
                RETURNING id, name, api_key, created_at
            """, (request.name, api_key))
            tenant = cur.fetchone()
            conn.commit()
            return TenantResponse(**tenant)
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            cur.close()


@app.get("/tenants/{tenant_id}")
def get_tenant_info(tenant_id: str, tenant: dict = Depends(get_tenant)):
    """Get tenant info (requires auth)."""
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT id, name, created_at, config FROM tenants WHERE id = %s", (tenant_id,))
        result = cur.fetchone()
        cur.close()
        if not result:
            raise HTTPException(status_code=404, detail="Tenant not found")
        return result


@app.post("/memory")
def store_memory(entry: MemoryEntry, tenant: dict = Depends(get_tenant)):
    """Store a single memory entry."""
    if not tenant:
        raise HTTPException(status_code=401, detail="API key required")
    
    content_hash = hashlib.sha256(entry.content.encode()).hexdigest()
    
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        try:
            # Check for duplicate (idempotent)
            cur.execute(
                "SELECT id FROM memory_entries WHERE content_hash = %s AND tenant_id = %s",
                (content_hash, tenant['id'])
            )
            existing = cur.fetchone()
            if existing:
                return {"id": existing['id'], "status": "duplicate"}
            
            # Compute salience if not provided
            salience = entry.salience_score or compute_salience(entry.content, entry.metadata)
            
            # Generate embedding
            embedding = generate_embedding(entry.content)
            
            # Generate commit hash for tracking
            commit_hash = f"mem-{content_hash[:16]}"
            
            # Insert
            cur.execute("""
                INSERT INTO memory_entries 
                (tenant_id, agent_id, commit_hash, file_path, content, content_hash, 
                 embedding, source_channel, memory_kind, session_id, 
                 salience_score, metadata, committed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                RETURNING id
            """, (
                tenant['id'], tenant['name'], commit_hash, entry.file_path, entry.content,
                content_hash, embedding, entry.source_channel, entry.memory_kind,
                entry.session_id, salience, json.dumps(entry.metadata)
            ))
            
            result = cur.fetchone()
            conn.commit()
            
            return {
                "id": result['id'],
                "status": "created",
                "salience_score": salience
            }
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Store memory failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            cur.close()


@app.post("/batch")
def batch_ingest(request: BatchIngestRequest):
    """
    Batch ingest entries (backwards compatible).
    Uses agent_id to find tenant, or creates entries under legacy tenant.
    """
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Find tenant by agent_id (backwards compat)
            cur.execute("SELECT id FROM tenants WHERE name = %s", (request.agent_id,))
            tenant = cur.fetchone()
            
            if not tenant:
                # Use legacy Ada tenant
                tenant_id = '00000000-0000-0000-0000-000000000001'
            else:
                tenant_id = tenant['id']
            
            inserted = 0
            duplicates = 0
            
            for entry in request.entries:
                content_hash = hashlib.sha256(entry.content.encode()).hexdigest()
                
                # Check for duplicate (idempotent)
                cur.execute(
                    "SELECT 1 FROM memory_entries WHERE content_hash = %s AND tenant_id = %s",
                    (content_hash, tenant_id)
                )
                if cur.fetchone():
                    duplicates += 1
                    continue
                
                # Compute salience
                salience = compute_salience(entry.content, entry.metadata)
                
                # Generate embedding
                try:
                    embedding = generate_embedding(entry.content)
                except Exception as e:
                    logger.error(f"Embedding failed: {e}")
                    continue
                
                commit_hash = entry.commit_hash or f"batch-{hash(entry.content) & 0xffffffff:08x}"
                committed_at = entry.committed_at or datetime.now(timezone.utc).isoformat()
                
                cur.execute("""
                    INSERT INTO memory_entries 
                    (tenant_id, agent_id, commit_hash, file_path, content, content_hash,
                     embedding, source_channel, memory_kind, salience_score, 
                     metadata, committed_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    tenant_id, request.agent_id, commit_hash, entry.file_path,
                    entry.content, content_hash, embedding, entry.source_channel,
                    entry.memory_kind, salience, json.dumps(entry.metadata), committed_at
                ))
                inserted += 1
                
                # Commit every 10 to avoid large transactions
                if inserted % 10 == 0:
                    conn.commit()
            
            conn.commit()
            logger.info(f"Batch ingest: {inserted} inserted, {duplicates} duplicates")
            return {"inserted": inserted, "duplicates": duplicates, "total": len(request.entries)}
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Batch ingest failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            cur.close()


@app.post("/search", response_model=List[SearchResult])
def search_memory(request: SearchRequest, tenant: dict = Depends(get_tenant)):
    """
    Hybrid search with attention ranking.
    
    final_score = similarity * (1 - α) + salience * α
    """
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Determine tenant
            if tenant:
                tenant_id = tenant['id']
            elif request.agent_id:
                # Legacy: find by agent_id
                cur.execute("SELECT id FROM tenants WHERE name = %s", (request.agent_id,))
                t = cur.fetchone()
                tenant_id = t['id'] if t else '00000000-0000-0000-0000-000000000001'
            else:
                raise HTTPException(status_code=400, detail="agent_id or API key required")
            
            # Generate query embedding
            query_embedding = generate_embedding(request.query)
            
            # Search with hybrid scoring
            attention_weight = request.attention_weight or ATTENTION_WEIGHT
            
            cur.execute("""
                SELECT 
                    id,
                    content,
                    file_path,
                    committed_at,
                    source_channel,
                    memory_kind,
                    COALESCE(salience_score, 0.5) as salience_score,
                    1 - (embedding <=> %s::vector) as similarity
                FROM memory_entries
                WHERE tenant_id = %s
                  AND embedding IS NOT NULL
                  AND 1 - (embedding <=> %s::vector) >= %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (
                query_embedding,
                tenant_id,
                query_embedding,
                request.min_similarity,
                query_embedding,
                request.limit * 2  # Fetch more for reranking
            ))
            
            rows = cur.fetchall()
            
            # Compute final scores and rerank
            results = []
            for row in rows:
                similarity = row['similarity']
                salience = row['salience_score']
                final_score = similarity * (1 - attention_weight) + salience * attention_weight
                
                results.append(SearchResult(
                    id=row['id'],
                    content=row['content'],
                    file_path=row['file_path'],
                    committed_at=row['committed_at'],
                    source_channel=row['source_channel'] or 'api',
                    memory_kind=row['memory_kind'] or 'note',
                    similarity=round(similarity, 4),
                    salience_score=salience,
                    final_score=round(final_score, 4)
                ))
            
            # Sort by final score and limit
            results.sort(key=lambda x: x.final_score, reverse=True)
            results = results[:request.limit]
            
            return results
            
        finally:
            cur.close()


@app.get("/stats")
def get_stats(
    agent_id: str = Query(None),
    tenant: dict = Depends(get_tenant)
):
    """Get memory stats for a tenant/agent."""
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            if tenant:
                tenant_id = tenant['id']
            elif agent_id:
                cur.execute("SELECT id FROM tenants WHERE name = %s", (agent_id,))
                t = cur.fetchone()
                tenant_id = t['id'] if t else '00000000-0000-0000-0000-000000000001'
            else:
                raise HTTPException(status_code=400, detail="agent_id or API key required")
            
            cur.execute("""
                SELECT 
                    COUNT(*) as total_entries,
                    COUNT(DISTINCT commit_hash) as total_commits,
                    COUNT(DISTINCT file_path) as total_files,
                    AVG(salience_score) as avg_salience,
                    MIN(committed_at) as earliest,
                    MAX(committed_at) as latest
                FROM memory_entries
                WHERE tenant_id = %s
            """, (tenant_id,))
            
            row = cur.fetchone()
            return {
                "tenant_id": str(tenant_id),
                "total_entries": row['total_entries'],
                "total_commits": row['total_commits'],
                "total_files": row['total_files'],
                "avg_salience": round(row['avg_salience'] or 0.5, 2),
                "earliest_commit": row['earliest'],
                "latest_commit": row['latest']
            }
        finally:
            cur.close()


@app.get("/debug/embedding-status")
def embedding_status(agent_id: str = "ada"):
    """Debug endpoint to check embedding status."""
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        try:
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(embedding) as with_embedding,
                    COUNT(*) - COUNT(embedding) as without_embedding,
                    AVG(salience_score) as avg_salience
                FROM memory_entries 
                WHERE agent_id = %s
            """, (agent_id,))
            row = cur.fetchone()
            return {
                "agent_id": agent_id,
                "total": row['total'],
                "with_embedding": row['with_embedding'],
                "without_embedding": row['without_embedding'],
                "avg_salience": round(row['avg_salience'] or 0.5, 2)
            }
        finally:
            cur.close()


# Legacy endpoint compatibility
@app.post("/index")
def index_repo():
    """Placeholder for git repo indexing (not in PoC)."""
    return {"status": "not_implemented", "message": "Use /batch or /memory for direct ingestion"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8091"))
    uvicorn.run(app, host="0.0.0.0", port=port)
