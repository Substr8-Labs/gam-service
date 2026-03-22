#!/usr/bin/env python3
"""
GAM Service v2.1 - Multi-tenant Memory Infrastructure with Telemetry

v2.1 adds:
- Structured JSON logging
- Correlation IDs for request tracing
- Failure taxonomy
- Ingestion counters (Prometheus-compatible)
- Health dashboard endpoint

Previous (v2.0):
- Connection pooling
- Multi-tenant isolation
- Attention/salience scoring
- Idempotent writes
"""

import os
import hashlib
import json
import secrets
import uuid
import time
import logging
import sys
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from collections import defaultdict
from threading import Lock

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from openai import OpenAI
from fastapi import FastAPI, HTTPException, Query, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Configuration
# ============================================================

SERVICE_NAME = "gam-service"
SERVICE_VERSION = "2.7.0"

DB_HOST = os.getenv("GAM_DB_HOST", os.getenv("PGHOST", "localhost"))
DB_PORT = int(os.getenv("GAM_DB_PORT", os.getenv("PGPORT", "5432")))
DB_USER = os.getenv("GAM_DB_USER", os.getenv("PGUSER", "gam"))
DB_PASSWORD = os.getenv("GAM_DB_PASSWORD", os.getenv("PGPASSWORD", ""))
DB_NAME = os.getenv("GAM_DB_NAME", os.getenv("PGDATABASE", "gam"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536

DEFAULT_SALIENCE = 0.5
ATTENTION_WEIGHT = float(os.getenv("GAM_ATTENTION_WEIGHT", "0.3"))

# ============================================================
# Failure Taxonomy
# ============================================================

class ErrorCode:
    """Canonical error codes for failure taxonomy."""
    VALIDATION_ERROR = "validation_error"
    AUTH_ERROR = "auth_error"
    RATE_LIMIT = "rate_limit_exceeded"
    DB_ERROR = "db_error"
    DB_TIMEOUT = "db_timeout"
    EMBEDDING_ERROR = "embedding_error"
    SEARCH_ERROR = "search_error"
    EXTERNAL_ERROR = "external_dependency_error"
    UNKNOWN_ERROR = "unknown_error"

# ============================================================
# Structured Logging
# ============================================================

class StructuredLogger:
    """JSON structured logger for telemetry."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
    
    def _emit(self, level: str, event_type: str, **kwargs):
        """Emit structured log event."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "event_type": event_type,
            **kwargs
        }
        self.logger.log(getattr(logging, level), json.dumps(event))
    
    def info(self, event_type: str, **kwargs):
        self._emit("INFO", event_type, **kwargs)
    
    def error(self, event_type: str, **kwargs):
        self._emit("ERROR", event_type, **kwargs)
    
    def warning(self, event_type: str, **kwargs):
        self._emit("WARNING", event_type, **kwargs)

log = StructuredLogger(SERVICE_NAME)

# ============================================================
# Metrics Counters (Thread-safe)
# ============================================================

class MetricsCollector:
    """Simple in-memory metrics collector for Prometheus-style export."""
    
    def __init__(self):
        self._lock = Lock()
        self._counters: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._start_time = time.time()
    
    def inc(self, name: str, labels: Dict[str, str] = None, value: int = 1):
        """Increment a counter."""
        labels = labels or {}
        label_key = json.dumps(labels, sort_keys=True)
        with self._lock:
            self._counters[name][label_key] += value
    
    def observe(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram observation."""
        labels = labels or {}
        label_key = json.dumps(labels, sort_keys=True)
        with self._lock:
            self._histograms[f"{name}:{label_key}"].append(value)
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = [
            f"# GAM Service Metrics",
            f"# Generated at {datetime.now(timezone.utc).isoformat()}",
            f"gam_service_uptime_seconds {time.time() - self._start_time:.0f}",
            ""
        ]
        
        with self._lock:
            # Export counters
            for name, label_values in self._counters.items():
                lines.append(f"# TYPE {name} counter")
                for label_key, value in label_values.items():
                    labels = json.loads(label_key) if label_key != "{}" else {}
                    label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
                    if label_str:
                        lines.append(f"{name}{{{label_str}}} {value}")
                    else:
                        lines.append(f"{name} {value}")
            
            # Export histogram summaries
            for name_key, values in self._histograms.items():
                name, label_key = name_key.rsplit(":", 1)
                if values:
                    labels = json.loads(label_key) if label_key != "{}" else {}
                    label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
                    prefix = f"{name}{{{label_str}}}" if label_str else name
                    lines.append(f"# TYPE {name} summary")
                    lines.append(f"{prefix}_count {len(values)}")
                    lines.append(f"{prefix}_sum {sum(values):.2f}")
                    if values:
                        sorted_values = sorted(values)
                        lines.append(f"{prefix}_p50 {sorted_values[len(values)//2]:.2f}")
                        lines.append(f"{prefix}_p95 {sorted_values[int(len(values)*0.95)]:.2f}")
        
        return "\n".join(lines)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary as JSON."""
        with self._lock:
            summary = {
                "uptime_seconds": int(time.time() - self._start_time),
                "counters": {},
                "histograms": {}
            }
            
            for name, label_values in self._counters.items():
                summary["counters"][name] = {}
                for label_key, value in label_values.items():
                    labels = json.loads(label_key) if label_key != "{}" else {}
                    key = str(labels) if labels else "total"
                    summary["counters"][name][key] = value
            
            for name_key, values in self._histograms.items():
                name, _ = name_key.rsplit(":", 1)
                if name not in summary["histograms"]:
                    summary["histograms"][name] = {"count": 0, "sum": 0}
                summary["histograms"][name]["count"] += len(values)
                summary["histograms"][name]["sum"] += sum(values)
            
            return summary

metrics = MetricsCollector()

# ============================================================
# Correlation ID
# ============================================================

def generate_correlation_id() -> str:
    """Generate unique correlation ID for request tracing."""
    return f"gam-{uuid.uuid4().hex[:12]}"

# ============================================================
# Database Connection Pooling
# ============================================================

db_pool: pool.ThreadedConnectionPool = None

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
        log.info("db.pool.initialized", host=DB_HOST, port=DB_PORT, database=DB_NAME)
    except Exception as e:
        log.error("db.pool.failed", error=str(e), error_code=ErrorCode.DB_ERROR)
        raise

@contextmanager
def get_db(correlation_id: str = None):
    """Get connection from pool with automatic cleanup."""
    conn = None
    try:
        conn = db_pool.getconn()
        register_vector(conn)
        yield conn
    except psycopg2.OperationalError as e:
        if conn:
            conn.rollback()
        log.error("db.connection.error", 
                  correlation_id=correlation_id,
                  error=str(e), 
                  error_code=ErrorCode.DB_ERROR)
        metrics.inc("gam_db_error_total", {"error_code": ErrorCode.DB_ERROR})
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        log.error("db.error", 
                  correlation_id=correlation_id,
                  error=str(e), 
                  error_code=ErrorCode.UNKNOWN_ERROR)
        raise
    finally:
        if conn:
            db_pool.putconn(conn)

# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="GAM Service",
    version=SERVICE_VERSION,
    description="Multi-tenant Memory Infrastructure for Agents"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    init_db_pool()
    with get_db() as conn:
        run_migrations(conn)
    log.info("service.started", version=SERVICE_VERSION)

@app.on_event("shutdown")
async def shutdown():
    if db_pool:
        db_pool.closeall()
    log.info("service.stopped")

# ============================================================
# Migrations
# ============================================================

def run_migrations(conn):
    """Run schema migrations."""
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'tenants'
            )
        """)
        if not cur.fetchone()[0]:
            log.info("db.migrations.running")
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tenants (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(255) NOT NULL,
                    api_key VARCHAR(64) UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    config JSONB DEFAULT '{}'
                )
            """)
            
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
                    log.warning("db.migration.warning", sql=sql[:50], error=str(e))
            
            cur.execute("""
                INSERT INTO tenants (id, name, api_key)
                VALUES ('00000000-0000-0000-0000-000000000001', 'ada', 'ada-legacy-key')
                ON CONFLICT (id) DO NOTHING
            """)
            
            cur.execute("""
                UPDATE memory_entries 
                SET tenant_id = '00000000-0000-0000-0000-000000000001'
                WHERE tenant_id IS NULL
            """)
            
            conn.commit()
            log.info("db.migrations.complete")
        
        # V2.3 Enrichment migrations
        run_enrichment_migrations(conn)
    finally:
        cur.close()


def run_enrichment_migrations(conn):
    """Run v2.3 enrichment schema migrations."""
    cur = conn.cursor()
    try:
        # Check if enrichment_jobs exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'enrichment_jobs'
            )
        """)
        if not cur.fetchone()[0]:
            log.info("db.migrations.enrichment.running")
            
            # Enrichment Jobs table
            cur.execute("""
                CREATE TABLE enrichment_jobs (
                    id SERIAL PRIMARY KEY,
                    job_id VARCHAR(50) UNIQUE NOT NULL,
                    job_type VARCHAR(50) NOT NULL,
                    status VARCHAR(20) DEFAULT 'pending',
                    memory_id INTEGER REFERENCES memory_entries(id),
                    tenant_id UUID NOT NULL,
                    priority INTEGER DEFAULT 1,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    result JSONB,
                    error TEXT,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3
                )
            """)
            cur.execute("CREATE INDEX idx_jobs_status ON enrichment_jobs(status)")
            cur.execute("CREATE INDEX idx_jobs_tenant ON enrichment_jobs(tenant_id)")
            cur.execute("CREATE INDEX idx_jobs_memory ON enrichment_jobs(memory_id)")
            cur.execute("CREATE INDEX idx_jobs_type_status ON enrichment_jobs(job_type, status)")
            
            # Entities table
            cur.execute("""
                CREATE TABLE entities (
                    id SERIAL PRIMARY KEY,
                    tenant_id UUID NOT NULL,
                    memory_id INTEGER NOT NULL REFERENCES memory_entries(id),
                    entity_type VARCHAR(50) NOT NULL,
                    entity_value TEXT NOT NULL,
                    confidence FLOAT DEFAULT 1.0,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'
                )
            """)
            cur.execute("CREATE INDEX idx_entities_tenant ON entities(tenant_id)")
            cur.execute("CREATE INDEX idx_entities_memory ON entities(memory_id)")
            cur.execute("CREATE INDEX idx_entities_type_value ON entities(entity_type, entity_value)")
            
            # Relations table
            cur.execute("""
                CREATE TABLE relations (
                    id SERIAL PRIMARY KEY,
                    tenant_id UUID NOT NULL,
                    source_memory_id INTEGER NOT NULL REFERENCES memory_entries(id),
                    subject_entity_id INTEGER REFERENCES entities(id),
                    predicate VARCHAR(50) NOT NULL,
                    object_entity_id INTEGER REFERENCES entities(id),
                    object_memory_id INTEGER REFERENCES memory_entries(id),
                    confidence FLOAT DEFAULT 1.0,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'
                )
            """)
            cur.execute("CREATE INDEX idx_relations_tenant ON relations(tenant_id)")
            cur.execute("CREATE INDEX idx_relations_subject ON relations(subject_entity_id)")
            cur.execute("CREATE INDEX idx_relations_object ON relations(object_entity_id)")
            cur.execute("CREATE INDEX idx_relations_predicate ON relations(predicate)")
            
            conn.commit()
            log.info("db.migrations.enrichment.complete")
        
        # V2.7 Retrieval Events table (Holy Smokes demo)
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'retrieval_events'
            )
        """)
        if not cur.fetchone()[0]:
            log.info("db.migrations.retrieval.running")
            cur.execute("""
                CREATE TABLE retrieval_events (
                    id SERIAL PRIMARY KEY,
                    memory_id INTEGER REFERENCES memory_entries(id),
                    tenant_id UUID NOT NULL,
                    run_id VARCHAR(100),
                    session_id VARCHAR(100),
                    retrieved_at TIMESTAMPTZ DEFAULT NOW(),
                    influenced_response BOOLEAN DEFAULT true,
                    context JSONB DEFAULT '{}'
                )
            """)
            cur.execute("CREATE INDEX idx_retrieval_memory ON retrieval_events(memory_id)")
            cur.execute("CREATE INDEX idx_retrieval_tenant ON retrieval_events(tenant_id)")
            cur.execute("CREATE INDEX idx_retrieval_run ON retrieval_events(run_id)")
            conn.commit()
            log.info("db.migrations.retrieval.complete")
    except Exception as e:
        log.error("db.migrations.enrichment.error", error=str(e))
        conn.rollback()
    finally:
        cur.close()

# ============================================================
# OpenAI Client
# ============================================================

_openai_client = None

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client

def generate_embedding(text: str, correlation_id: str = None) -> List[float]:
    """Generate embedding using OpenAI."""
    start_time = time.time()
    client = get_openai_client()
    text = text[:8000]
    
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        duration_ms = int((time.time() - start_time) * 1000)
        metrics.observe("gam_embedding_duration_ms", duration_ms)
        metrics.inc("gam_embedding_generated_total")
        return response.data[0].embedding
    except Exception as e:
        log.error("embedding.failed", 
                  correlation_id=correlation_id,
                  error=str(e),
                  error_code=ErrorCode.EMBEDDING_ERROR)
        metrics.inc("gam_embedding_error_total", {"error_code": ErrorCode.EMBEDDING_ERROR})
        raise

# ============================================================
# Attention Scoring
# ============================================================

def compute_salience(content: str, metadata: dict = None) -> float:
    score = DEFAULT_SALIENCE
    content_lower = content.lower()
    
    high_signals = [
        'decided', 'decision', 'strategy', 'strategic',
        'important', 'critical', 'milestone', 'breakthrough',
        'agreed', 'confirmed', 'committed', 'promise',
        'learned', 'realized', 'insight', 'discovered'
    ]
    
    medium_signals = [
        'completed', 'finished', 'shipped', 'deployed',
        'meeting', 'discussed', 'reviewed', 'feedback',
        'todo', 'task', 'action item', 'follow up'
    ]
    
    low_signals = ['heartbeat', 'health check', 'status ok', 'routine', 'daily']
    
    for signal in high_signals:
        if signal in content_lower:
            score = min(1.0, score + 0.15)
    
    for signal in medium_signals:
        if signal in content_lower:
            score = min(1.0, score + 0.05)
    
    for signal in low_signals:
        if signal in content_lower:
            score = max(0.1, score - 0.2)
    
    if metadata:
        if metadata.get('pinned'):
            score = 1.0
        if metadata.get('source_channel') == 'decision':
            score = min(1.0, score + 0.2)
    
    return round(score, 2)

# ============================================================
# Authentication
# ============================================================

async def get_tenant(x_api_key: str = Header(None, alias="X-API-Key")):
    if not x_api_key:
        return None
    
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM tenants WHERE api_key = %s", (x_api_key,))
        tenant = cur.fetchone()
        cur.close()
        
        if not tenant:
            metrics.inc("gam_auth_error_total")
            raise HTTPException(status_code=401, detail="Invalid API key")
        return tenant

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
    salience_score: Optional[float] = None

class BatchEntry(BaseModel):
    file_path: str = "api"
    content: str
    commit_hash: Optional[str] = None
    committed_at: Optional[str] = None
    source_channel: str = "api"
    memory_kind: str = "note"
    metadata: dict = Field(default_factory=dict)

class BatchIngestRequest(BaseModel):
    agent_id: str
    entries: List[BatchEntry]

class SearchRequest(BaseModel):
    agent_id: Optional[str] = None
    query: str
    limit: int = 10
    min_similarity: float = 0.1
    attention_weight: Optional[float] = None

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
    """Health check with telemetry status."""
    correlation_id = generate_correlation_id()
    try:
        with get_db(correlation_id) as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
        return {
            "status": "healthy",
            "database": "connected",
            "version": SERVICE_VERSION,
            "pool_size": db_pool.maxconn if db_pool else 0
        }
    except Exception as e:
        log.error("health.check.failed", correlation_id=correlation_id, error=str(e))
        return {"status": "unhealthy", "error": str(e)}

@app.get("/metrics", response_class=PlainTextResponse)
def get_metrics():
    """Prometheus-compatible metrics endpoint."""
    return metrics.export_prometheus()

@app.get("/dashboard")
def get_dashboard():
    """Ingestion health dashboard data."""
    summary = metrics.get_summary()
    
    counters = summary.get("counters", {})
    ingest = counters.get("gam_ingest_attempt_total", {})
    success = counters.get("gam_ingest_success_total", {})
    failures = counters.get("gam_ingest_failure_total", {})
    
    total_attempted = sum(ingest.values()) if ingest else 0
    total_succeeded = sum(success.values()) if success else 0
    total_failed = sum(failures.values()) if failures else 0
    
    success_rate = (total_succeeded / total_attempted * 100) if total_attempted > 0 else 100
    
    return {
        "version": SERVICE_VERSION,
        "uptime_seconds": summary.get("uptime_seconds", 0),
        "ingestion": {
            "attempted": total_attempted,
            "succeeded": total_succeeded,
            "failed": total_failed,
            "success_rate_percent": round(success_rate, 1),
            "failures_by_reason": failures
        },
        "search": {
            "total": counters.get("gam_search_total", {}).get("total", 0),
            "empty_results": counters.get("gam_search_empty_total", {}).get("total", 0)
        },
        "embeddings": {
            "generated": counters.get("gam_embedding_generated_total", {}).get("total", 0),
            "errors": counters.get("gam_embedding_error_total", {}).get("total", 0)
        }
    }

@app.post("/tenants", response_model=TenantResponse)
def create_tenant(request: TenantCreate):
    """Register a new tenant."""
    correlation_id = generate_correlation_id()
    api_key = secrets.token_urlsafe(32)
    
    log.info("tenant.create.started", correlation_id=correlation_id, name=request.name)
    
    with get_db(correlation_id) as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        try:
            cur.execute("""
                INSERT INTO tenants (name, api_key)
                VALUES (%s, %s)
                RETURNING id, name, api_key, created_at
            """, (request.name, api_key))
            tenant = cur.fetchone()
            conn.commit()
            
            log.info("tenant.create.succeeded", 
                     correlation_id=correlation_id,
                     tenant_id=str(tenant['id']),
                     name=tenant['name'])
            metrics.inc("gam_tenant_created_total")
            
            return TenantResponse(**tenant)
        except Exception as e:
            conn.rollback()
            log.error("tenant.create.failed",
                      correlation_id=correlation_id,
                      error=str(e),
                      error_code=ErrorCode.DB_ERROR)
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            cur.close()

@app.get("/tenants/{tenant_id}")
def get_tenant_info(tenant_id: str, tenant: dict = Depends(get_tenant)):
    """Get tenant info."""
    correlation_id = generate_correlation_id()
    with get_db(correlation_id) as conn:
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
    correlation_id = generate_correlation_id()
    start_time = time.time()
    
    if not tenant:
        metrics.inc("gam_auth_error_total")
        raise HTTPException(status_code=401, detail="API key required")
    
    content_hash = hashlib.sha256(entry.content.encode()).hexdigest()
    
    log.info("memory.store.started",
             correlation_id=correlation_id,
             tenant_id=str(tenant['id']),
             content_length=len(entry.content),
             memory_kind=entry.memory_kind)
    
    metrics.inc("gam_ingest_attempt_total", {"tenant": tenant['name'], "kind": entry.memory_kind})
    
    with get_db(correlation_id) as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        try:
            # Check for duplicate
            cur.execute(
                "SELECT id FROM memory_entries WHERE content_hash = %s AND tenant_id = %s",
                (content_hash, tenant['id'])
            )
            existing = cur.fetchone()
            if existing:
                duration_ms = int((time.time() - start_time) * 1000)
                log.info("memory.store.duplicate",
                         correlation_id=correlation_id,
                         memory_id=existing['id'],
                         duration_ms=duration_ms)
                return {"id": existing['id'], "status": "duplicate"}
            
            salience = entry.salience_score or compute_salience(entry.content, entry.metadata)
            embedding = generate_embedding(entry.content, correlation_id)
            commit_hash = f"mem-{content_hash[:16]}"
            
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
            
            duration_ms = int((time.time() - start_time) * 1000)
            log.info("memory.store.succeeded",
                     correlation_id=correlation_id,
                     tenant_id=str(tenant['id']),
                     memory_id=result['id'],
                     salience_score=salience,
                     duration_ms=duration_ms)
            
            metrics.inc("gam_ingest_success_total", {"tenant": tenant['name'], "kind": entry.memory_kind})
            metrics.observe("gam_ingest_duration_ms", duration_ms, {"tenant": tenant['name']})
            
            return {"id": result['id'], "status": "created", "salience_score": salience}
            
        except Exception as e:
            conn.rollback()
            duration_ms = int((time.time() - start_time) * 1000)
            error_code = ErrorCode.DB_ERROR if "psycopg" in str(type(e)) else ErrorCode.UNKNOWN_ERROR
            
            log.error("memory.store.failed",
                      correlation_id=correlation_id,
                      tenant_id=str(tenant['id']),
                      error=str(e),
                      error_code=error_code,
                      duration_ms=duration_ms)
            
            metrics.inc("gam_ingest_failure_total", {"tenant": tenant['name'], "error_code": error_code})
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            cur.close()

@app.post("/batch")
def batch_ingest(request: BatchIngestRequest):
    """Batch ingest entries."""
    correlation_id = generate_correlation_id()
    start_time = time.time()
    
    log.info("batch.ingest.started",
             correlation_id=correlation_id,
             agent_id=request.agent_id,
             entry_count=len(request.entries))
    
    with get_db(correlation_id) as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cur.execute("SELECT id FROM tenants WHERE name = %s", (request.agent_id,))
            tenant = cur.fetchone()
            tenant_id = tenant['id'] if tenant else '00000000-0000-0000-0000-000000000001'
            
            inserted = 0
            duplicates = 0
            failed = 0
            failure_reasons = defaultdict(int)
            
            for entry in request.entries:
                metrics.inc("gam_ingest_attempt_total", {"tenant": request.agent_id, "kind": entry.memory_kind})
                
                content_hash = hashlib.sha256(entry.content.encode()).hexdigest()
                
                cur.execute(
                    "SELECT 1 FROM memory_entries WHERE content_hash = %s AND tenant_id = %s",
                    (content_hash, tenant_id)
                )
                if cur.fetchone():
                    duplicates += 1
                    continue
                
                salience = compute_salience(entry.content, entry.metadata)
                
                try:
                    embedding = generate_embedding(entry.content, correlation_id)
                except Exception as e:
                    failed += 1
                    failure_reasons[ErrorCode.EMBEDDING_ERROR] += 1
                    metrics.inc("gam_ingest_failure_total", 
                               {"tenant": request.agent_id, "error_code": ErrorCode.EMBEDDING_ERROR})
                    continue
                
                commit_hash = entry.commit_hash or f"batch-{hash(entry.content) & 0xffffffff:08x}"
                committed_at = entry.committed_at or datetime.now(timezone.utc).isoformat()
                
                try:
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
                    metrics.inc("gam_ingest_success_total", {"tenant": request.agent_id, "kind": entry.memory_kind})
                    
                    if inserted % 10 == 0:
                        conn.commit()
                        
                except Exception as e:
                    failed += 1
                    failure_reasons[ErrorCode.DB_ERROR] += 1
                    metrics.inc("gam_ingest_failure_total",
                               {"tenant": request.agent_id, "error_code": ErrorCode.DB_ERROR})
            
            conn.commit()
            duration_ms = int((time.time() - start_time) * 1000)
            
            status = "success" if failed == 0 else ("partial_success" if inserted > 0 else "failed")
            
            log.info("batch.ingest.completed",
                     correlation_id=correlation_id,
                     agent_id=request.agent_id,
                     status=status,
                     attempted=len(request.entries),
                     inserted=inserted,
                     duplicates=duplicates,
                     failed=failed,
                     failure_reasons=dict(failure_reasons),
                     duration_ms=duration_ms)
            
            metrics.observe("gam_batch_duration_ms", duration_ms, {"tenant": request.agent_id})
            
            return {
                "status": status,
                "inserted": inserted, 
                "duplicates": duplicates, 
                "failed": failed,
                "total": len(request.entries),
                "failure_reasons": dict(failure_reasons) if failure_reasons else None
            }
            
        except Exception as e:
            conn.rollback()
            duration_ms = int((time.time() - start_time) * 1000)
            
            log.error("batch.ingest.failed",
                      correlation_id=correlation_id,
                      agent_id=request.agent_id,
                      error=str(e),
                      error_code=ErrorCode.DB_ERROR,
                      duration_ms=duration_ms)
            
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            cur.close()

# ============================================================
# Typed Retrieval (Milestone D.3)
# ============================================================

# Query type detection patterns
QUERY_TYPE_PATTERNS = {
    "decision": ["decision", "decided", "chose", "choice", "concluded", "conclusion", "agreed"],
    "milestone": ["milestone", "achieved", "completed", "shipped", "launched", "released", "finished"],
    "task": ["task", "todo", "do next", "need to", "action item", "pending", "assigned"],
    "insight": ["insight", "learned", "realized", "understood", "discovery", "finding", "lesson"],
    "fact": ["fact", "data", "information", "reference", "definition", "what is", "how does"],
    "issue": ["issue", "problem", "bug", "error", "concern", "broken", "failed", "failing"]
}

# Typed hint boost factor (how much to boost matching hints)
TYPED_HINT_BOOST = 0.15


def detect_query_type(query: str) -> Optional[str]:
    """Detect what type of memory the query is looking for."""
    query_lower = query.lower()
    
    scores = {}
    for hint_type, patterns in QUERY_TYPE_PATTERNS.items():
        score = sum(1 for p in patterns if p in query_lower)
        if score > 0:
            scores[hint_type] = score
    
    if scores:
        return max(scores, key=scores.get)
    return None


@app.post("/search", response_model=List[SearchResult])
def search_memory(request: SearchRequest, tenant: dict = Depends(get_tenant)):
    """Hybrid search with attention ranking and typed retrieval."""
    correlation_id = generate_correlation_id()
    start_time = time.time()
    
    with get_db(correlation_id) as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            if tenant:
                tenant_id = tenant['id']
            elif request.agent_id:
                cur.execute("SELECT id FROM tenants WHERE name = %s", (request.agent_id,))
                t = cur.fetchone()
                tenant_id = t['id'] if t else '00000000-0000-0000-0000-000000000001'
            else:
                raise HTTPException(status_code=400, detail="agent_id or API key required")
            
            # Detect query type for typed retrieval boost
            detected_type = detect_query_type(request.query)
            
            log.info("search.started",
                     correlation_id=correlation_id,
                     tenant_id=str(tenant_id),
                     query_length=len(request.query),
                     detected_type=detected_type,
                     limit=request.limit)
            
            metrics.inc("gam_search_total", {"tenant": request.agent_id or "api"})
            if detected_type:
                metrics.inc("gam_typed_search_total", {"type": detected_type})
            
            query_embedding = generate_embedding(request.query, correlation_id)
            attention_weight = request.attention_weight or ATTENTION_WEIGHT
            
            cur.execute("""
                SELECT 
                    id, content, file_path, committed_at,
                    source_channel, memory_kind, metadata,
                    COALESCE(salience_score, 0.5) as salience_score,
                    1 - (embedding <=> %s::vector) as similarity
                FROM memory_entries
                WHERE tenant_id = %s
                  AND embedding IS NOT NULL
                  AND 1 - (embedding <=> %s::vector) >= %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (
                query_embedding, tenant_id, query_embedding,
                request.min_similarity, query_embedding, request.limit * 2
            ))
            
            rows = cur.fetchall()
            
            results = []
            for row in rows:
                similarity = row['similarity']
                salience = row['salience_score']
                metadata = row['metadata'] or {}
                typed_hint = metadata.get('typed_hint')
                
                # Base score: similarity + salience
                base_score = similarity * (1 - attention_weight) + salience * attention_weight
                
                # Typed hint boost: if query type matches memory's typed hint
                typed_boost = 0.0
                if detected_type and typed_hint == detected_type:
                    typed_boost = TYPED_HINT_BOOST
                
                final_score = min(1.0, base_score + typed_boost)
                
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
            
            results.sort(key=lambda x: x.final_score, reverse=True)
            results = results[:request.limit]
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            if not results:
                metrics.inc("gam_search_empty_total", {"tenant": request.agent_id or "api"})
            
            log.info("search.completed",
                     correlation_id=correlation_id,
                     tenant_id=str(tenant_id),
                     result_count=len(results),
                     detected_type=detected_type,
                     duration_ms=duration_ms)
            
            metrics.observe("gam_search_duration_ms", duration_ms, {"tenant": request.agent_id or "api"})
            
            return results
            
        finally:
            cur.close()


class TypedSearchRequest(BaseModel):
    """Typed search request with explicit type filter."""
    query: str
    typed_hint: Optional[str] = None  # Explicit type filter
    agent_id: Optional[str] = None
    limit: int = Field(default=10, le=50)
    min_similarity: float = Field(default=0.3, ge=0.0, le=1.0)
    show_scoring: bool = False  # Show scoring breakdown


@app.post("/v3/typed-search")
def typed_search(request: TypedSearchRequest, tenant: dict = Depends(get_tenant)):
    """Search with explicit typed hint filter and scoring breakdown."""
    correlation_id = generate_correlation_id()
    start_time = time.time()
    
    with get_db(correlation_id) as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            if tenant:
                tenant_id = tenant['id']
            elif request.agent_id:
                cur.execute("SELECT id FROM tenants WHERE name = %s", (request.agent_id,))
                t = cur.fetchone()
                tenant_id = t['id'] if t else '00000000-0000-0000-0000-000000000001'
            else:
                raise HTTPException(status_code=400, detail="agent_id or API key required")
            
            # Use explicit type or detect from query
            target_type = request.typed_hint or detect_query_type(request.query)
            
            log.info("typed_search.started",
                     correlation_id=correlation_id,
                     tenant_id=str(tenant_id),
                     query_length=len(request.query),
                     target_type=target_type,
                     limit=request.limit)
            
            metrics.inc("gam_typed_search_total", {"type": target_type or "none"})
            
            query_embedding = generate_embedding(request.query, correlation_id)
            
            # Build query with optional typed hint filter
            if target_type:
                # Typed search: get type-matching first, then others
                cur.execute("""
                    SELECT 
                        id, content, file_path, committed_at,
                        source_channel, memory_kind, metadata,
                        COALESCE(salience_score, 0.5) as salience_score,
                        1 - (embedding <=> %s::vector) as similarity
                    FROM memory_entries
                    WHERE tenant_id = %s
                      AND embedding IS NOT NULL
                      AND 1 - (embedding <=> %s::vector) >= %s
                    ORDER BY 
                        CASE WHEN metadata->>'typed_hint' = %s THEN 0 ELSE 1 END,
                        embedding <=> %s::vector
                    LIMIT %s
                """, (
                    query_embedding, tenant_id, query_embedding,
                    request.min_similarity, target_type, query_embedding, request.limit * 2
                ))
            else:
                cur.execute("""
                    SELECT 
                        id, content, file_path, committed_at,
                        source_channel, memory_kind, metadata,
                        COALESCE(salience_score, 0.5) as salience_score,
                        1 - (embedding <=> %s::vector) as similarity
                    FROM memory_entries
                    WHERE tenant_id = %s
                      AND embedding IS NOT NULL
                      AND 1 - (embedding <=> %s::vector) >= %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (
                    query_embedding, tenant_id, query_embedding,
                    request.min_similarity, query_embedding, request.limit * 2
                ))
            
            rows = cur.fetchall()
            
            results = []
            for row in rows:
                similarity = row['similarity']
                salience = row['salience_score']
                metadata = row['metadata'] or {}
                typed_hint = metadata.get('typed_hint')
                
                # Calculate scores
                semantic_score = similarity
                salience_weight = 0.3
                base_score = semantic_score * 0.7 + salience * salience_weight
                
                # Typed boost
                typed_boost = TYPED_HINT_BOOST if (target_type and typed_hint == target_type) else 0.0
                final_score = min(1.0, base_score + typed_boost)
                
                result = {
                    "id": row['id'],
                    "content": row['content'][:500] + "..." if len(row['content']) > 500 else row['content'],
                    "memory_kind": row['memory_kind'] or 'note',
                    "typed_hint": typed_hint,
                    "final_score": round(final_score, 4)
                }
                
                if request.show_scoring:
                    result["scoring"] = {
                        "semantic_similarity": round(semantic_score, 4),
                        "salience": round(salience, 4),
                        "base_score": round(base_score, 4),
                        "typed_boost": round(typed_boost, 4),
                        "target_type": target_type,
                        "memory_hint": typed_hint,
                        "boost_applied": typed_boost > 0,
                        "formula": "final = (similarity * 0.7 + salience * 0.3) + typed_boost"
                    }
                
                results.append(result)
            
            results.sort(key=lambda x: x['final_score'], reverse=True)
            results = results[:request.limit]
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            log.info("typed_search.completed",
                     correlation_id=correlation_id,
                     result_count=len(results),
                     target_type=target_type,
                     duration_ms=duration_ms)
            
            metrics.observe("gam_typed_search_duration_ms", duration_ms)
            
            return {
                "query": request.query,
                "target_type": target_type,
                "results": results,
                "total": len(results),
                "duration_ms": duration_ms
            }
            
        finally:
            cur.close()

@app.get("/stats")
def get_stats(agent_id: str = Query(None), tenant: dict = Depends(get_tenant)):
    """Get memory stats for a tenant/agent."""
    correlation_id = generate_correlation_id()
    
    with get_db(correlation_id) as conn:
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

@app.post("/index")
def index_repo():
    """Placeholder for git repo indexing."""
    return {"status": "not_implemented", "message": "Use /batch or /memory for direct ingestion"}

# ============================================================
# New MCP v1 Endpoints
# ============================================================

class ContextRequest(BaseModel):
    task: str
    max_tokens: int = 2000
    include_decisions: bool = True
    include_milestones: bool = True

@app.post("/context")
def build_context(request: ContextRequest, tenant: dict = Depends(get_tenant)):
    """Build a context pack for a task."""
    correlation_id = generate_correlation_id()
    start_time = time.time()
    
    if not tenant:
        metrics.inc("gam_auth_error_total")
        raise HTTPException(status_code=401, detail="API key required")
    
    log.info("context.build.started",
             correlation_id=correlation_id,
             tenant_id=str(tenant['id']),
             task_length=len(request.task),
             max_tokens=request.max_tokens)
    
    metrics.inc("gam_context_build_total", {"tenant": tenant['name']})
    
    with get_db(correlation_id) as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Generate embedding for task
            task_embedding = generate_embedding(request.task, correlation_id)
            
            # Build filter for memory kinds
            kinds_filter = []
            if request.include_decisions:
                kinds_filter.append("'decision'")
            if request.include_milestones:
                kinds_filter.append("'milestone'")
            kinds_filter.extend(["'note'", "'task'"])
            kinds_sql = f"memory_kind IN ({','.join(kinds_filter)})"
            
            # Search for relevant memories with preference for high salience
            cur.execute(f"""
                SELECT 
                    id, content, memory_kind, salience_score,
                    1 - (embedding <=> %s::vector) as similarity
                FROM memory_entries
                WHERE tenant_id = %s
                  AND embedding IS NOT NULL
                  AND {kinds_sql}
                  AND 1 - (embedding <=> %s::vector) >= 0.2
                ORDER BY 
                    (1 - (embedding <=> %s::vector)) * 0.7 + salience_score * 0.3 DESC
                LIMIT 20
            """, (task_embedding, tenant['id'], task_embedding, task_embedding))
            
            rows = cur.fetchall()
            
            # Build context pack with token limit
            context_parts = []
            memory_ids = []
            estimated_tokens = 0
            
            # Group by kind
            decisions = [r for r in rows if r['memory_kind'] == 'decision']
            milestones = [r for r in rows if r['memory_kind'] == 'milestone']
            other = [r for r in rows if r['memory_kind'] not in ('decision', 'milestone')]
            
            # Add decisions first (most important)
            if decisions and request.include_decisions:
                context_parts.append("## Relevant Decisions\n")
                for r in decisions[:5]:
                    content = r['content'][:500]
                    tokens = len(content) // 4
                    if estimated_tokens + tokens > request.max_tokens:
                        break
                    context_parts.append(f"- {content}\n")
                    memory_ids.append(r['id'])
                    estimated_tokens += tokens
                context_parts.append("\n")
            
            # Add milestones
            if milestones and request.include_milestones:
                context_parts.append("## Recent Milestones\n")
                for r in milestones[:3]:
                    content = r['content'][:300]
                    tokens = len(content) // 4
                    if estimated_tokens + tokens > request.max_tokens:
                        break
                    context_parts.append(f"- {content}\n")
                    memory_ids.append(r['id'])
                    estimated_tokens += tokens
                context_parts.append("\n")
            
            # Add other context
            if other:
                context_parts.append("## Related Context\n")
                for r in other[:5]:
                    content = r['content'][:300]
                    tokens = len(content) // 4
                    if estimated_tokens + tokens > request.max_tokens:
                        break
                    context_parts.append(f"- {content}\n")
                    memory_ids.append(r['id'])
                    estimated_tokens += tokens
            
            context = "".join(context_parts)
            duration_ms = int((time.time() - start_time) * 1000)
            
            log.info("context.build.completed",
                     correlation_id=correlation_id,
                     tenant_id=str(tenant['id']),
                     memory_count=len(memory_ids),
                     token_count=estimated_tokens,
                     duration_ms=duration_ms)
            
            metrics.observe("gam_context_build_duration_ms", duration_ms, {"tenant": tenant['name']})
            
            return {
                "context": context,
                "memory_ids": memory_ids,
                "token_count": estimated_tokens
            }
            
        finally:
            cur.close()

@app.post("/memory/{memory_id}/pin")
def pin_memory(memory_id: int, tenant: dict = Depends(get_tenant)):
    """Pin a memory (set salience to 1.0)."""
    correlation_id = generate_correlation_id()
    start_time = time.time()
    
    if not tenant:
        metrics.inc("gam_auth_error_total")
        raise HTTPException(status_code=401, detail="API key required")
    
    log.info("memory.pin.started",
             correlation_id=correlation_id,
             tenant_id=str(tenant['id']),
             memory_id=memory_id)
    
    with get_db(correlation_id) as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Get current state
            cur.execute("""
                SELECT id, salience_score, metadata 
                FROM memory_entries 
                WHERE id = %s AND tenant_id = %s
            """, (memory_id, tenant['id']))
            
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Memory not found")
            
            previous_salience = row['salience_score']
            metadata = row['metadata'] or {}
            metadata['pinned'] = True
            metadata['pinned_at'] = datetime.now(timezone.utc).isoformat()
            
            # Update salience to 1.0 and mark as pinned
            cur.execute("""
                UPDATE memory_entries 
                SET salience_score = 1.0, metadata = %s
                WHERE id = %s AND tenant_id = %s
            """, (json.dumps(metadata), memory_id, tenant['id']))
            
            conn.commit()
            duration_ms = int((time.time() - start_time) * 1000)
            
            log.info("memory.pin.completed",
                     correlation_id=correlation_id,
                     tenant_id=str(tenant['id']),
                     memory_id=memory_id,
                     previous_salience=previous_salience,
                     duration_ms=duration_ms)
            
            metrics.inc("gam_governance_pin_total", {"tenant": tenant['name']})
            
            return {
                "memory_id": memory_id,
                "status": "pinned",
                "previous_salience": previous_salience,
                "new_salience": 1.0
            }
            
        finally:
            cur.close()

@app.get("/memory/{memory_id}/related")
def get_related(memory_id: int, limit: int = 5, tenant: dict = Depends(get_tenant)):
    """Get memories related to a given memory."""
    correlation_id = generate_correlation_id()
    start_time = time.time()
    
    if not tenant:
        metrics.inc("gam_auth_error_total")
        raise HTTPException(status_code=401, detail="API key required")
    
    log.info("memory.related.started",
             correlation_id=correlation_id,
             tenant_id=str(tenant['id']),
             memory_id=memory_id,
             limit=limit)
    
    with get_db(correlation_id) as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Get source memory (convert embedding to list for reuse)
            cur.execute("""
                SELECT id, content, memory_kind, embedding::text, salience_score
                FROM memory_entries 
                WHERE id = %s AND tenant_id = %s
            """, (memory_id, tenant['id']))
            
            source = cur.fetchone()
            if not source:
                raise HTTPException(status_code=404, detail="Memory not found")
            
            if not source['embedding']:
                raise HTTPException(status_code=400, detail="Memory has no embedding")
            
            # Find related by semantic similarity using subquery
            cur.execute("""
                WITH source AS (
                    SELECT embedding FROM memory_entries WHERE id = %s
                )
                SELECT 
                    m.id, m.content, m.memory_kind, m.salience_score,
                    1 - (m.embedding <=> source.embedding) as similarity
                FROM memory_entries m, source
                WHERE m.tenant_id = %s
                  AND m.id != %s
                  AND m.embedding IS NOT NULL
                  AND 1 - (m.embedding <=> source.embedding) >= 0.3
                ORDER BY m.embedding <=> source.embedding
                LIMIT %s
            """, (memory_id, tenant['id'], memory_id, limit))
            
            related = []
            for row in cur.fetchall():
                related.append({
                    "id": row['id'],
                    "content": row['content'][:500],
                    "kind": row['memory_kind'],
                    "relation": "semantic_similarity",
                    "score": round(row['similarity'], 4)
                })
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            log.info("memory.related.completed",
                     correlation_id=correlation_id,
                     tenant_id=str(tenant['id']),
                     memory_id=memory_id,
                     related_count=len(related),
                     duration_ms=duration_ms)
            
            metrics.inc("gam_related_query_total", {"tenant": tenant['name']})
            
            return {
                "source_memory": {
                    "id": source['id'],
                    "content": source['content'][:500],
                    "kind": source['memory_kind']
                },
                "related": related
            }
            
        finally:
            cur.close()

# ============================================================
# Admin Endpoints (Milestone C)
# ============================================================

@app.get("/admin/tenants")
def admin_list_tenants(tenant: dict = Depends(get_tenant)):
    """List all tenants with stats."""
    correlation_id = generate_correlation_id()
    
    if not tenant:
        raise HTTPException(status_code=401, detail="API key required")
    
    log.info("admin.tenants.list", correlation_id=correlation_id)
    
    with get_db(correlation_id) as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cur.execute("""
                SELECT 
                    t.id, t.name, t.created_at,
                    COUNT(m.id) as memory_count,
                    COUNT(CASE WHEN m.metadata->>'pinned' = 'true' THEN 1 END) as pinned_count,
                    COUNT(CASE WHEN m.metadata->>'suppressed' = 'true' THEN 1 END) as suppressed_count,
                    COUNT(CASE WHEN m.memory_kind = 'decision' THEN 1 END) as decision_count,
                    COUNT(CASE WHEN m.memory_kind = 'milestone' THEN 1 END) as milestone_count
                FROM tenants t
                LEFT JOIN memory_entries m ON t.id = m.tenant_id
                GROUP BY t.id, t.name, t.created_at
                ORDER BY t.name
            """)
            
            tenants = []
            for row in cur.fetchall():
                tenants.append({
                    "id": str(row['id']),
                    "name": row['name'],
                    "created_at": row['created_at'].isoformat() if row['created_at'] else None,
                    "stats": {
                        "total": row['memory_count'],
                        "pinned": row['pinned_count'],
                        "suppressed": row['suppressed_count'],
                        "decisions": row['decision_count'],
                        "milestones": row['milestone_count']
                    }
                })
            
            return {"tenants": tenants}
        finally:
            cur.close()


@app.get("/admin/memories")
def admin_list_memories(
    tenant_id: Optional[str] = None,
    kind: Optional[str] = None,
    min_salience: Optional[float] = None,
    pinned: Optional[bool] = None,
    suppressed: bool = False,
    q: Optional[str] = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    sort: str = "created_at",
    order: str = "desc",
    tenant: dict = Depends(get_tenant)
):
    """Paginated memory list with filters."""
    correlation_id = generate_correlation_id()
    
    if not tenant:
        raise HTTPException(status_code=401, detail="API key required")
    
    log.info("admin.memories.list", 
             correlation_id=correlation_id,
             tenant_id=tenant_id,
             kind=kind,
             limit=limit,
             offset=offset)
    
    with get_db(correlation_id) as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Build query
            where_clauses = []
            params = []
            
            if tenant_id:
                where_clauses.append("tenant_id = %s")
                params.append(tenant_id)
            
            if kind and kind != "all":
                where_clauses.append("memory_kind = %s")
                params.append(kind)
            
            if min_salience is not None:
                where_clauses.append("salience_score >= %s")
                params.append(min_salience)
            
            if pinned is True:
                where_clauses.append("metadata->>'pinned' = 'true'")
            
            if not suppressed:
                where_clauses.append("(metadata->>'suppressed' IS NULL OR metadata->>'suppressed' != 'true')")
            
            if q:
                where_clauses.append("content ILIKE %s")
                params.append(f"%{q}%")
            
            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            # Validate sort column
            valid_sorts = {"created_at": "committed_at", "salience": "salience_score", "id": "id"}
            sort_col = valid_sorts.get(sort, "committed_at")
            order_dir = "DESC" if order.lower() == "desc" else "ASC"
            
            # Count total
            cur.execute(f"SELECT COUNT(*) FROM memory_entries WHERE {where_sql}", params)
            total = cur.fetchone()['count']
            
            # Fetch page
            cur.execute(f"""
                SELECT 
                    m.id, m.content, m.memory_kind, m.source_channel,
                    m.committed_at, m.salience_score, m.metadata,
                    t.name as tenant_name, t.id as tenant_id
                FROM memory_entries m
                JOIN tenants t ON m.tenant_id = t.id
                WHERE {where_sql}
                ORDER BY {sort_col} {order_dir}
                LIMIT %s OFFSET %s
            """, params + [limit, offset])
            
            memories = []
            for row in cur.fetchall():
                metadata = row['metadata'] or {}
                memories.append({
                    "id": row['id'],
                    "content": row['content'][:300] + "..." if len(row['content']) > 300 else row['content'],
                    "kind": row['memory_kind'],
                    "source": row['source_channel'],
                    "created_at": row['committed_at'].isoformat() if row['committed_at'] else None,
                    "salience": row['salience_score'],
                    "pinned": metadata.get('pinned', False),
                    "suppressed": metadata.get('suppressed', False),
                    "tenant": {
                        "id": str(row['tenant_id']),
                        "name": row['tenant_name']
                    }
                })
            
            return {
                "data": memories,
                "meta": {
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "tenant_id": tenant_id
                }
            }
        finally:
            cur.close()


@app.get("/admin/memories/{memory_id}")
def admin_get_memory(memory_id: int, tenant: dict = Depends(get_tenant)):
    """Full memory detail with provenance."""
    correlation_id = generate_correlation_id()
    
    if not tenant:
        raise HTTPException(status_code=401, detail="API key required")
    
    log.info("admin.memory.detail", correlation_id=correlation_id, memory_id=memory_id)
    
    with get_db(correlation_id) as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cur.execute("""
                SELECT 
                    m.id, m.content, m.memory_kind, m.source_channel,
                    m.committed_at, m.salience_score, m.metadata,
                    m.embedding IS NOT NULL as has_embedding,
                    m.file_path,
                    t.name as tenant_name, t.id as tenant_id
                FROM memory_entries m
                JOIN tenants t ON m.tenant_id = t.id
                WHERE m.id = %s
            """, (memory_id,))
            
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Memory not found")
            
            metadata = row['metadata'] or {}
            content_hash = hashlib.sha256(row['content'].encode()).hexdigest()[:16]
            
            # Get related memories (explicit relations + semantic fallback)
            related = []
            seen_ids = set()
            
            # First: explicit relations (highest priority)
            cur.execute("""
                SELECT r.object_memory_id as id, r.predicate, r.confidence,
                       m.content, m.memory_kind
                FROM relations r
                JOIN memory_entries m ON r.object_memory_id = m.id
                WHERE r.source_memory_id = %s AND r.tenant_id = %s
                ORDER BY r.confidence DESC
                LIMIT 3
            """, (memory_id, row['tenant_id']))
            
            for rel in cur.fetchall():
                related.append({
                    "id": rel['id'],
                    "score": round(rel['confidence'], 4),
                    "preview": rel['content'][:100] + "..." if len(rel['content']) > 100 else rel['content'],
                    "kind": rel['memory_kind'],
                    "source": "explicit_relation",
                    "predicate": rel['predicate']
                })
                seen_ids.add(rel['id'])
            
            # Then: semantic similarity (fill remaining slots)
            if row['has_embedding'] and len(related) < 5:
                cur.execute("""
                    WITH source AS (
                        SELECT embedding FROM memory_entries WHERE id = %s
                    )
                    SELECT 
                        m.id, m.content, m.memory_kind,
                        1 - (m.embedding <=> source.embedding) as similarity
                    FROM memory_entries m, source
                    WHERE m.tenant_id = %s
                      AND m.id != %s
                      AND m.embedding IS NOT NULL
                      AND 1 - (m.embedding <=> source.embedding) >= 0.3
                    ORDER BY m.embedding <=> source.embedding
                    LIMIT %s
                """, (memory_id, row['tenant_id'], memory_id, 5 - len(related)))
                
                for rel in cur.fetchall():
                    if rel['id'] not in seen_ids:
                        related.append({
                            "id": rel['id'],
                            "score": round(rel['similarity'], 4),
                            "preview": rel['content'][:100] + "..." if len(rel['content']) > 100 else rel['content'],
                            "kind": rel['memory_kind'],
                            "source": "semantic"
                        })
            
            return {
                "id": row['id'],
                "content": row['content'],
                "memory_kind": row['memory_kind'],
                "source_channel": row['source_channel'],
                "created_at": row['committed_at'].isoformat() if row['committed_at'] else None,
                "tenant": {
                    "id": str(row['tenant_id']),
                    "name": row['tenant_name']
                },
                "salience": {
                    "score": row['salience_score'],
                    "pinned": metadata.get('pinned', False),
                    "pinned_at": metadata.get('pinned_at'),
                    "previous_score": metadata.get('previous_salience')
                },
                "provenance": {
                    "memory_id": row['id'],
                    "tenant_id": str(row['tenant_id']),
                    "source_channel": row['source_channel'],
                    "created_at": row['committed_at'].isoformat() if row['committed_at'] else None,
                    "updated_at": metadata.get('updated_at'),
                    "has_embedding": row['has_embedding'],
                    "embedding_dims": EMBEDDING_DIMS if row['has_embedding'] else None,
                    "content_hash": content_hash,
                    "pinned": metadata.get('pinned', False),
                    "suppressed": metadata.get('suppressed', False)
                },
                "related": related,
                "governance": {
                    "suppressed": metadata.get('suppressed', False),
                    "suppressed_at": metadata.get('suppressed_at'),
                    "suppressed_reason": metadata.get('suppressed_reason')
                },
                "enrichment": {
                    "typed_hint": metadata.get('typed_hint'),
                    "typed_hint_at": metadata.get('typed_hint_at'),
                    "entities": get_memory_entities_internal(cur, memory_id, row['tenant_id']),
                    "relations": get_memory_relations_internal(cur, memory_id, row['tenant_id'])
                },
                # Holy Smokes: Source provenance for memory inspection
                "source": {
                    "message_excerpt": metadata.get('source_message') or metadata.get('source_message_excerpt'),
                    "run_id": metadata.get('run_id') or metadata.get('source_run_id'),
                    "session_id": metadata.get('session_id') or row.get('session_id') or metadata.get('source_session_id'),
                    "actor": metadata.get('created_by') or metadata.get('source_actor'),
                    "policy_decision": metadata.get('policy_decision') or metadata.get('store_policy_decision'),
                    "policy_reason": metadata.get('policy_reason') or metadata.get('store_policy_reason')
                },
                # Lineage for supersession
                "lineage": {
                    "supersedes": metadata.get('supersedes') or metadata.get('supersedes_memory_id'),
                    "superseded_by": metadata.get('superseded_by') or metadata.get('superseded_by_memory_id')
                },
                # Raw metadata for debugging
                "raw_metadata": metadata
            }
        finally:
            cur.close()


def get_memory_entities_internal(cur, memory_id: int, tenant_id) -> List[dict]:
    """Get entities for a memory (internal helper)."""
    cur.execute("""
        SELECT id, entity_type, entity_value, confidence, metadata
        FROM entities
        WHERE memory_id = %s AND tenant_id = %s
        ORDER BY confidence DESC
    """, (memory_id, tenant_id))
    
    entities = []
    for row in cur.fetchall():
        entities.append({
            "id": row['id'],
            "type": row['entity_type'],
            "value": row['entity_value'],
            "confidence": row['confidence'],
            "excerpt": row['metadata'].get('excerpt') if row['metadata'] else None
        })
    return entities


def get_memory_relations_internal(cur, memory_id: int, tenant_id) -> dict:
    """Get relations for a memory (internal helper)."""
    # Outgoing relations
    cur.execute("""
        SELECT r.object_memory_id as target_id, r.predicate, r.confidence,
               m.content as target_preview, m.memory_kind as target_kind
        FROM relations r
        JOIN memory_entries m ON r.object_memory_id = m.id
        WHERE r.source_memory_id = %s AND r.tenant_id = %s
        ORDER BY r.confidence DESC
        LIMIT 5
    """, (memory_id, tenant_id))
    
    outgoing = []
    for row in cur.fetchall():
        outgoing.append({
            "target_id": row['target_id'],
            "predicate": row['predicate'],
            "confidence": row['confidence'],
            "preview": row['target_preview'][:80] + "..." if len(row['target_preview']) > 80 else row['target_preview']
        })
    
    # Incoming relations
    cur.execute("""
        SELECT r.source_memory_id as source_id, r.predicate, r.confidence,
               m.content as source_preview
        FROM relations r
        JOIN memory_entries m ON r.source_memory_id = m.id
        WHERE r.object_memory_id = %s AND r.tenant_id = %s
        ORDER BY r.confidence DESC
        LIMIT 5
    """, (memory_id, tenant_id))
    
    incoming = []
    for row in cur.fetchall():
        incoming.append({
            "source_id": row['source_id'],
            "predicate": row['predicate'],
            "confidence": row['confidence'],
            "preview": row['source_preview'][:80] + "..." if len(row['source_preview']) > 80 else row['source_preview']
        })
    
    return {
        "outgoing": outgoing,
        "incoming": incoming,
        "total": len(outgoing) + len(incoming)
    }


@app.post("/admin/memories/{memory_id}/suppress")
def admin_suppress_memory(memory_id: int, reason: Optional[str] = None, tenant: dict = Depends(get_tenant)):
    """Suppress a memory (soft delete, reversible)."""
    correlation_id = generate_correlation_id()
    
    if not tenant:
        raise HTTPException(status_code=401, detail="API key required")
    
    log.info("admin.memory.suppress", correlation_id=correlation_id, memory_id=memory_id)
    
    with get_db(correlation_id) as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cur.execute("SELECT id, metadata FROM memory_entries WHERE id = %s", (memory_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Memory not found")
            
            metadata = row['metadata'] or {}
            metadata['suppressed'] = True
            metadata['suppressed_at'] = datetime.now(timezone.utc).isoformat()
            if reason:
                metadata['suppressed_reason'] = reason
            
            cur.execute("""
                UPDATE memory_entries 
                SET metadata = %s
                WHERE id = %s
            """, (json.dumps(metadata), memory_id))
            
            conn.commit()
            
            log.info("admin.memory.suppressed", 
                     correlation_id=correlation_id, 
                     memory_id=memory_id)
            
            metrics.inc("gam_governance_suppress_total")
            
            return {
                "memory_id": memory_id,
                "status": "suppressed",
                "suppressed_at": metadata['suppressed_at']
            }
        finally:
            cur.close()


@app.post("/admin/memories/{memory_id}/unsuppress")
def admin_unsuppress_memory(memory_id: int, tenant: dict = Depends(get_tenant)):
    """Unsuppress a memory (restore from soft delete)."""
    correlation_id = generate_correlation_id()
    
    if not tenant:
        raise HTTPException(status_code=401, detail="API key required")
    
    log.info("admin.memory.unsuppress", correlation_id=correlation_id, memory_id=memory_id)
    
    with get_db(correlation_id) as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cur.execute("SELECT id, metadata FROM memory_entries WHERE id = %s", (memory_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Memory not found")
            
            metadata = row['metadata'] or {}
            metadata['suppressed'] = False
            metadata['unsuppressed_at'] = datetime.now(timezone.utc).isoformat()
            
            cur.execute("""
                UPDATE memory_entries 
                SET metadata = %s
                WHERE id = %s
            """, (json.dumps(metadata), memory_id))
            
            conn.commit()
            
            log.info("admin.memory.unsuppressed", 
                     correlation_id=correlation_id, 
                     memory_id=memory_id)
            
            return {
                "memory_id": memory_id,
                "status": "restored"
            }
        finally:
            cur.close()


@app.post("/admin/memories/{memory_id}/reindex")
def admin_reindex_memory(memory_id: int, tenant: dict = Depends(get_tenant)):
    """Regenerate embedding for a memory (async-safe)."""
    correlation_id = generate_correlation_id()
    
    if not tenant:
        raise HTTPException(status_code=401, detail="API key required")
    
    log.info("admin.memory.reindex.started", correlation_id=correlation_id, memory_id=memory_id)
    
    with get_db(correlation_id) as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cur.execute("SELECT id, content, metadata FROM memory_entries WHERE id = %s", (memory_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Memory not found")
            
            # Generate new embedding
            try:
                new_embedding = generate_embedding(row['content'], correlation_id)
            except Exception as e:
                log.error("admin.memory.reindex.embedding_failed",
                          correlation_id=correlation_id,
                          memory_id=memory_id,
                          error=str(e))
                return {
                    "memory_id": memory_id,
                    "status": "failed",
                    "error": "Embedding generation failed"
                }
            
            # Update embedding
            metadata = row['metadata'] or {}
            metadata['reindexed_at'] = datetime.now(timezone.utc).isoformat()
            metadata['updated_at'] = datetime.now(timezone.utc).isoformat()
            
            cur.execute("""
                UPDATE memory_entries 
                SET embedding = %s::vector, metadata = %s
                WHERE id = %s
            """, (new_embedding, json.dumps(metadata), memory_id))
            
            conn.commit()
            
            log.info("admin.memory.reindex.completed", 
                     correlation_id=correlation_id, 
                     memory_id=memory_id)
            
            metrics.inc("gam_governance_reindex_total")
            
            return {
                "memory_id": memory_id,
                "status": "reindexed",
                "reindexed_at": metadata['reindexed_at']
            }
        finally:
            cur.close()


# ============================================================
# Holy Smokes: Retrieval Event Tracking
# ============================================================

class RetrievalEvent(BaseModel):
    """Record when a memory was retrieved in a run."""
    run_id: str
    session_id: Optional[str] = None
    influenced_response: bool = True
    context: dict = Field(default_factory=dict)


@app.post("/admin/memories/{memory_id}/retrievals")
def admin_record_retrieval(
    memory_id: int, 
    event: RetrievalEvent,
    tenant: dict = Depends(get_tenant)
):
    """Record a retrieval event for a memory (Holy Smokes tracking)."""
    correlation_id = generate_correlation_id()
    
    if not tenant:
        raise HTTPException(status_code=401, detail="API key required")
    
    log.info("admin.memory.retrieval.record", 
             correlation_id=correlation_id, 
             memory_id=memory_id,
             run_id=event.run_id)
    
    with get_db(correlation_id) as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Verify memory exists
            cur.execute("SELECT id, tenant_id FROM memory_entries WHERE id = %s", (memory_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Memory not found")
            
            # Record retrieval event
            cur.execute("""
                INSERT INTO retrieval_events 
                (memory_id, tenant_id, run_id, session_id, influenced_response, context)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id, retrieved_at
            """, (memory_id, row['tenant_id'], event.run_id, event.session_id, 
                  event.influenced_response, json.dumps(event.context)))
            
            result = cur.fetchone()
            conn.commit()
            
            return {
                "event_id": result['id'],
                "memory_id": memory_id,
                "run_id": event.run_id,
                "retrieved_at": result['retrieved_at'].isoformat()
            }
        finally:
            cur.close()


@app.get("/admin/memories/{memory_id}/retrievals")
def admin_get_retrievals(
    memory_id: int,
    limit: int = 10,
    tenant: dict = Depends(get_tenant)
):
    """Get retrieval events for a memory (Holy Smokes demo)."""
    correlation_id = generate_correlation_id()
    
    if not tenant:
        raise HTTPException(status_code=401, detail="API key required")
    
    log.info("admin.memory.retrievals.list", 
             correlation_id=correlation_id, 
             memory_id=memory_id)
    
    with get_db(correlation_id) as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cur.execute("""
                SELECT id, run_id, session_id, retrieved_at, influenced_response, context
                FROM retrieval_events
                WHERE memory_id = %s
                ORDER BY retrieved_at DESC
                LIMIT %s
            """, (memory_id, limit))
            
            events = []
            for row in cur.fetchall():
                events.append({
                    "event_id": row['id'],
                    "run_id": row['run_id'],
                    "session_id": row['session_id'],
                    "retrieved_at": row['retrieved_at'].isoformat(),
                    "influenced_response": row['influenced_response'],
                    "context": row['context'] or {}
                })
            
            return {
                "memory_id": memory_id,
                "events": events,
                "total": len(events)
            }
        finally:
            cur.close()


@app.get("/admin/stats")
def admin_stats(tenant: dict = Depends(get_tenant)):
    """Global stats dashboard."""
    correlation_id = generate_correlation_id()
    
    if not tenant:
        raise HTTPException(status_code=401, detail="API key required")
    
    log.info("admin.stats", correlation_id=correlation_id)
    
    with get_db(correlation_id) as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Total counts
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN metadata->>'pinned' = 'true' THEN 1 END) as pinned,
                    COUNT(CASE WHEN metadata->>'suppressed' = 'true' THEN 1 END) as suppressed,
                    COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as embedded
                FROM memory_entries
            """)
            counts = cur.fetchone()
            
            # By kind
            cur.execute("""
                SELECT memory_kind, COUNT(*) as count
                FROM memory_entries
                WHERE metadata->>'suppressed' IS NULL OR metadata->>'suppressed' != 'true'
                GROUP BY memory_kind
                ORDER BY count DESC
            """)
            by_kind = {row['memory_kind']: row['count'] for row in cur.fetchall()}
            
            # Tenant count
            cur.execute("SELECT COUNT(*) as count FROM tenants")
            tenant_count = cur.fetchone()['count']
            
            # Recent activity (last 24h)
            cur.execute("""
                SELECT COUNT(*) as count
                FROM memory_entries
                WHERE committed_at > NOW() - INTERVAL '24 hours'
            """)
            recent = cur.fetchone()['count']
            
            return {
                "totals": {
                    "memories": counts['total'],
                    "pinned": counts['pinned'],
                    "suppressed": counts['suppressed'],
                    "embedded": counts['embedded'],
                    "tenants": tenant_count
                },
                "by_kind": by_kind,
                "activity": {
                    "last_24h": recent
                }
            }
        finally:
            cur.close()


class AdminSearchRequest(BaseModel):
    query: str
    tenant_id: Optional[str] = None
    limit: int = 10
    show_scoring: bool = True

@app.post("/admin/search")
def admin_search(request: AdminSearchRequest, tenant: dict = Depends(get_tenant)):
    """Debug search with scoring breakdown."""
    correlation_id = generate_correlation_id()
    
    if not tenant:
        raise HTTPException(status_code=401, detail="API key required")
    
    log.info("admin.search", 
             correlation_id=correlation_id, 
             query=request.query[:50],
             tenant_id=request.tenant_id)
    
    with get_db(correlation_id) as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Generate query embedding
            query_embedding = generate_embedding(request.query, correlation_id)
            
            # Build tenant filter
            tenant_filter = ""
            params = [query_embedding, query_embedding, query_embedding]
            if request.tenant_id:
                tenant_filter = "AND tenant_id = %s"
                params.append(request.tenant_id)
            
            params.append(request.limit)
            
            cur.execute(f"""
                SELECT 
                    m.id, m.content, m.memory_kind, m.salience_score,
                    1 - (m.embedding <=> %s::vector) as similarity,
                    (1 - (m.embedding <=> %s::vector)) * 0.7 + m.salience_score * 0.3 as final_score,
                    t.name as tenant_name
                FROM memory_entries m
                JOIN tenants t ON m.tenant_id = t.id
                WHERE m.embedding IS NOT NULL
                  AND (m.metadata->>'suppressed' IS NULL OR m.metadata->>'suppressed' != 'true')
                  AND 1 - (m.embedding <=> %s::vector) >= 0.2
                  {tenant_filter}
                ORDER BY final_score DESC
                LIMIT %s
            """, params)
            
            results = []
            for row in cur.fetchall():
                result = {
                    "id": row['id'],
                    "content": row['content'][:200] + "..." if len(row['content']) > 200 else row['content'],
                    "kind": row['memory_kind'],
                    "tenant": row['tenant_name']
                }
                
                if request.show_scoring:
                    result["scoring"] = {
                        "similarity": round(row['similarity'], 4),
                        "salience": round(row['salience_score'], 4),
                        "final_score": round(row['final_score'], 4),
                        "formula": "final = similarity * 0.7 + salience * 0.3"
                    }
                
                results.append(result)
            
            return {
                "query": request.query,
                "total": len(results),
                "attention_weight": ATTENTION_WEIGHT,
                "results": results
            }
        finally:
            cur.close()


# ============================================================
# Entity Endpoints (Milestone D.2)
# ============================================================

@app.get("/memory/{memory_id}/entities")
def get_memory_entities(memory_id: int, tenant: dict = Depends(get_tenant)):
    """Get extracted entities for a memory."""
    if not tenant:
        raise HTTPException(status_code=401, detail="API key required")
    
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cur.execute("""
                SELECT e.*, m.content as memory_preview
                FROM entities e
                JOIN memory_entries m ON e.memory_id = m.id
                WHERE e.memory_id = %s AND e.tenant_id = %s
                ORDER BY e.confidence DESC
            """, (memory_id, tenant['id']))
            
            entities = []
            memory_preview = None
            for row in cur.fetchall():
                memory_preview = row['memory_preview'][:200] if row['memory_preview'] else None
                entities.append({
                    "id": row['id'],
                    "type": row['entity_type'],
                    "value": row['entity_value'],
                    "confidence": row['confidence'],
                    "excerpt": row['metadata'].get('excerpt') if row['metadata'] else None,
                    "created_at": row['created_at'].isoformat() if row['created_at'] else None
                })
            
            return {
                "memory_id": memory_id,
                "memory_preview": memory_preview,
                "entities": entities,
                "count": len(entities)
            }
        finally:
            cur.close()


@app.get("/admin/entities")
def admin_list_entities(
    entity_type: Optional[str] = None,
    value: Optional[str] = None,
    memory_id: Optional[int] = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    tenant: dict = Depends(get_tenant)
):
    """List entities with filters (admin)."""
    if not tenant:
        raise HTTPException(status_code=401, detail="API key required")
    
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            where_clauses = []
            params = []
            
            if entity_type:
                where_clauses.append("e.entity_type = %s")
                params.append(entity_type)
            
            if value:
                where_clauses.append("e.entity_value ILIKE %s")
                params.append(f"%{value}%")
            
            if memory_id:
                where_clauses.append("e.memory_id = %s")
                params.append(memory_id)
            
            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            cur.execute(f"SELECT COUNT(*) FROM entities e WHERE {where_sql}", params)
            total = cur.fetchone()['count']
            
            cur.execute(f"""
                SELECT e.*, m.content as memory_preview, t.name as tenant_name
                FROM entities e
                JOIN memory_entries m ON e.memory_id = m.id
                JOIN tenants t ON e.tenant_id = t.id
                WHERE {where_sql}
                ORDER BY e.created_at DESC
                LIMIT %s OFFSET %s
            """, params + [limit, offset])
            
            entities = []
            for row in cur.fetchall():
                entities.append({
                    "id": row['id'],
                    "memory_id": row['memory_id'],
                    "type": row['entity_type'],
                    "value": row['entity_value'],
                    "confidence": row['confidence'],
                    "memory_preview": row['memory_preview'][:100] + "..." if row['memory_preview'] and len(row['memory_preview']) > 100 else row['memory_preview'],
                    "tenant": row['tenant_name'],
                    "created_at": row['created_at'].isoformat() if row['created_at'] else None
                })
            
            return {
                "data": entities,
                "meta": {
                    "total": total,
                    "limit": limit,
                    "offset": offset
                }
            }
        finally:
            cur.close()


@app.get("/admin/typed-hints/stats")
def admin_typed_hint_stats(tenant: dict = Depends(get_tenant)):
    """Get typed hint statistics."""
    if not tenant:
        raise HTTPException(status_code=401, detail="API key required")
    
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Total with hints
            cur.execute("""
                SELECT COUNT(*) as count
                FROM memory_entries
                WHERE metadata->>'typed_hint' IS NOT NULL
            """)
            total_hinted = cur.fetchone()['count']
            
            # By type
            cur.execute("""
                SELECT metadata->>'typed_hint' as hint_type, COUNT(*) as count
                FROM memory_entries
                WHERE metadata->>'typed_hint' IS NOT NULL
                GROUP BY metadata->>'typed_hint'
                ORDER BY count DESC
            """)
            by_type = {row['hint_type']: row['count'] for row in cur.fetchall()}
            
            # Total memories
            cur.execute("SELECT COUNT(*) as count FROM memory_entries")
            total_memories = cur.fetchone()['count']
            
            # Coverage
            coverage = round(total_hinted / total_memories * 100, 2) if total_memories > 0 else 0
            
            return {
                "total_hinted": total_hinted,
                "total_memories": total_memories,
                "coverage_percent": coverage,
                "by_type": by_type,
                "available_types": TYPED_HINT_CATEGORIES
            }
        finally:
            cur.close()


@app.post("/jobs/typed-hint/{memory_id}")
def enqueue_typed_hint(memory_id: int, tenant: dict = Depends(get_tenant)):
    """Enqueue a typed hint job (convenience endpoint)."""
    if not tenant:
        raise HTTPException(status_code=401, detail="API key required")
    
    request = EnqueueJobRequest(job_type="typed_hint", memory_id=memory_id, priority=1)
    return enqueue_job(request, tenant)


@app.post("/jobs/relation-link/{memory_id}")
def enqueue_relation_link(memory_id: int, tenant: dict = Depends(get_tenant)):
    """Enqueue a relation linking job (convenience endpoint)."""
    if not tenant:
        raise HTTPException(status_code=401, detail="API key required")
    
    request = EnqueueJobRequest(job_type="relation_link", memory_id=memory_id, priority=1)
    return enqueue_job(request, tenant)


@app.get("/memory/{memory_id}/relations")
def get_memory_relations(memory_id: int, tenant: dict = Depends(get_tenant)):
    """Get explicit relations for a memory."""
    if not tenant:
        raise HTTPException(status_code=401, detail="API key required")
    
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Get outgoing relations
            cur.execute("""
                SELECT r.*, m.content as target_content, m.memory_kind as target_kind
                FROM relations r
                JOIN memory_entries m ON r.object_memory_id = m.id
                WHERE r.source_memory_id = %s AND r.tenant_id = %s
                ORDER BY r.confidence DESC
            """, (memory_id, tenant['id']))
            
            outgoing = []
            for row in cur.fetchall():
                outgoing.append({
                    "id": row['id'],
                    "target_id": row['object_memory_id'],
                    "predicate": row['predicate'],
                    "confidence": row['confidence'],
                    "target_preview": row['target_content'][:100] + "..." if len(row['target_content']) > 100 else row['target_content'],
                    "target_kind": row['target_kind'],
                    "metadata": row['metadata']
                })
            
            # Get incoming relations
            cur.execute("""
                SELECT r.*, m.content as source_content, m.memory_kind as source_kind
                FROM relations r
                JOIN memory_entries m ON r.source_memory_id = m.id
                WHERE r.object_memory_id = %s AND r.tenant_id = %s
                ORDER BY r.confidence DESC
            """, (memory_id, tenant['id']))
            
            incoming = []
            for row in cur.fetchall():
                incoming.append({
                    "id": row['id'],
                    "source_id": row['source_memory_id'],
                    "predicate": row['predicate'],
                    "confidence": row['confidence'],
                    "source_preview": row['source_content'][:100] + "..." if len(row['source_content']) > 100 else row['source_content'],
                    "source_kind": row['source_kind']
                })
            
            return {
                "memory_id": memory_id,
                "outgoing": outgoing,
                "incoming": incoming,
                "total": len(outgoing) + len(incoming)
            }
        finally:
            cur.close()


@app.get("/admin/relations")
def admin_list_relations(
    predicate: Optional[str] = None,
    memory_id: Optional[int] = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    tenant: dict = Depends(get_tenant)
):
    """List relations with filters (admin)."""
    if not tenant:
        raise HTTPException(status_code=401, detail="API key required")
    
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            where_clauses = []
            params = []
            
            if predicate:
                where_clauses.append("r.predicate = %s")
                params.append(predicate)
            
            if memory_id:
                where_clauses.append("(r.source_memory_id = %s OR r.object_memory_id = %s)")
                params.extend([memory_id, memory_id])
            
            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            cur.execute(f"SELECT COUNT(*) FROM relations r WHERE {where_sql}", params)
            total = cur.fetchone()['count']
            
            cur.execute(f"""
                SELECT r.*, 
                       s.content as source_preview, s.memory_kind as source_kind,
                       o.content as object_preview, o.memory_kind as object_kind
                FROM relations r
                JOIN memory_entries s ON r.source_memory_id = s.id
                JOIN memory_entries o ON r.object_memory_id = o.id
                WHERE {where_sql}
                ORDER BY r.created_at DESC
                LIMIT %s OFFSET %s
            """, params + [limit, offset])
            
            relations = []
            for row in cur.fetchall():
                relations.append({
                    "id": row['id'],
                    "source_id": row['source_memory_id'],
                    "object_id": row['object_memory_id'],
                    "predicate": row['predicate'],
                    "confidence": row['confidence'],
                    "source_preview": row['source_preview'][:80] + "..." if len(row['source_preview']) > 80 else row['source_preview'],
                    "object_preview": row['object_preview'][:80] + "..." if len(row['object_preview']) > 80 else row['object_preview'],
                    "source_kind": row['source_kind'],
                    "object_kind": row['object_kind'],
                    "metadata": row['metadata'],
                    "created_at": row['created_at'].isoformat() if row['created_at'] else None
                })
            
            return {
                "data": relations,
                "meta": {
                    "total": total,
                    "limit": limit,
                    "offset": offset
                }
            }
        finally:
            cur.close()


@app.get("/admin/relations/stats")
def admin_relation_stats(tenant: dict = Depends(get_tenant)):
    """Get relation statistics."""
    if not tenant:
        raise HTTPException(status_code=401, detail="API key required")
    
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Total
            cur.execute("SELECT COUNT(*) as count FROM relations")
            total = cur.fetchone()['count']
            
            # By predicate
            cur.execute("""
                SELECT predicate, COUNT(*) as count
                FROM relations
                GROUP BY predicate
                ORDER BY count DESC
            """)
            by_predicate = {row['predicate']: row['count'] for row in cur.fetchall()}
            
            # Memories with relations
            cur.execute("""
                SELECT COUNT(DISTINCT source_memory_id) as count
                FROM relations
            """)
            memories_linked = cur.fetchone()['count']
            
            # Average confidence
            cur.execute("SELECT AVG(confidence) as avg FROM relations")
            avg_confidence = round(cur.fetchone()['avg'] or 0, 4)
            
            return {
                "total": total,
                "by_predicate": by_predicate,
                "memories_linked": memories_linked,
                "avg_confidence": avg_confidence,
                "available_predicates": RELATION_TYPES_V1
            }
        finally:
            cur.close()


@app.get("/admin/entities/stats")
def admin_entity_stats(tenant: dict = Depends(get_tenant)):
    """Get entity statistics."""
    if not tenant:
        raise HTTPException(status_code=401, detail="API key required")
    
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Total
            cur.execute("SELECT COUNT(*) as count FROM entities")
            total = cur.fetchone()['count']
            
            # By type
            cur.execute("""
                SELECT entity_type, COUNT(*) as count
                FROM entities
                GROUP BY entity_type
                ORDER BY count DESC
            """)
            by_type = {row['entity_type']: row['count'] for row in cur.fetchall()}
            
            # Memories with entities
            cur.execute("""
                SELECT COUNT(DISTINCT memory_id) as count
                FROM entities
            """)
            memories_enriched = cur.fetchone()['count']
            
            # Top entities
            cur.execute("""
                SELECT entity_value, entity_type, COUNT(*) as mentions
                FROM entities
                GROUP BY entity_value, entity_type
                ORDER BY mentions DESC
                LIMIT 10
            """)
            top_entities = [
                {"value": row['entity_value'], "type": row['entity_type'], "mentions": row['mentions']}
                for row in cur.fetchall()
            ]
            
            return {
                "total": total,
                "by_type": by_type,
                "memories_enriched": memories_enriched,
                "top_entities": top_entities
            }
        finally:
            cur.close()


# ============================================================
# Enrichment Job Infrastructure (Milestone D)
# ============================================================

# Job types
JOB_TYPES = ["entity_extract", "relation_link", "typed_hint", "summary_generate", "reindex"]
JOB_STATUSES = ["pending", "running", "completed", "failed"]

# Typed hint categories (locked)
TYPED_HINT_CATEGORIES = ["decision", "milestone", "task", "insight", "fact", "issue"]

# Relation types (v1)
RELATION_TYPES = ["related_to", "supports", "follows", "references"]


def generate_job_id() -> str:
    """Generate unique job ID."""
    return f"job-{uuid.uuid4().hex[:12]}"


class EnqueueJobRequest(BaseModel):
    job_type: str
    memory_id: int
    priority: int = 1


@app.post("/jobs/enqueue")
def enqueue_job(request: EnqueueJobRequest, tenant: dict = Depends(get_tenant)):
    """Enqueue an enrichment job."""
    correlation_id = generate_correlation_id()
    
    if not tenant:
        raise HTTPException(status_code=401, detail="API key required")
    
    if request.job_type not in JOB_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid job_type. Must be one of: {JOB_TYPES}")
    
    log.info("job.enqueue.started",
             correlation_id=correlation_id,
             job_type=request.job_type,
             memory_id=request.memory_id)
    
    job_id = generate_job_id()
    
    with get_db(correlation_id) as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Verify memory exists and belongs to tenant
            cur.execute("""
                SELECT id FROM memory_entries 
                WHERE id = %s AND tenant_id = %s
            """, (request.memory_id, tenant['id']))
            
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Memory not found")
            
            cur.execute("""
                INSERT INTO enrichment_jobs 
                (job_id, job_type, memory_id, tenant_id, priority, status)
                VALUES (%s, %s, %s, %s, %s, 'pending')
                RETURNING id, job_id, created_at
            """, (job_id, request.job_type, request.memory_id, tenant['id'], request.priority))
            
            row = cur.fetchone()
            conn.commit()
            
            log.info("job.enqueue.completed",
                     correlation_id=correlation_id,
                     job_id=job_id,
                     job_type=request.job_type)
            
            metrics.inc("gam_jobs_enqueued_total", {"job_type": request.job_type})
            
            return {
                "job_id": row['job_id'],
                "status": "pending",
                "created_at": row['created_at'].isoformat()
            }
        finally:
            cur.close()


@app.get("/jobs/{job_id}")
def get_job(job_id: str, tenant: dict = Depends(get_tenant)):
    """Get job status."""
    if not tenant:
        raise HTTPException(status_code=401, detail="API key required")
    
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cur.execute("""
                SELECT * FROM enrichment_jobs 
                WHERE job_id = %s AND tenant_id = %s
            """, (job_id, tenant['id']))
            
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Job not found")
            
            return {
                "job_id": row['job_id'],
                "job_type": row['job_type'],
                "status": row['status'],
                "memory_id": row['memory_id'],
                "priority": row['priority'],
                "created_at": row['created_at'].isoformat() if row['created_at'] else None,
                "started_at": row['started_at'].isoformat() if row['started_at'] else None,
                "completed_at": row['completed_at'].isoformat() if row['completed_at'] else None,
                "result": row['result'],
                "error": row['error'],
                "retry_count": row['retry_count']
            }
        finally:
            cur.close()


@app.get("/admin/jobs")
def admin_list_jobs(
    status: Optional[str] = None,
    job_type: Optional[str] = None,
    memory_id: Optional[int] = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    tenant: dict = Depends(get_tenant)
):
    """List enrichment jobs (admin)."""
    if not tenant:
        raise HTTPException(status_code=401, detail="API key required")
    
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            where_clauses = []
            params = []
            
            if status:
                where_clauses.append("status = %s")
                params.append(status)
            
            if job_type:
                where_clauses.append("job_type = %s")
                params.append(job_type)
            
            if memory_id:
                where_clauses.append("memory_id = %s")
                params.append(memory_id)
            
            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            cur.execute(f"SELECT COUNT(*) FROM enrichment_jobs WHERE {where_sql}", params)
            total = cur.fetchone()['count']
            
            cur.execute(f"""
                SELECT * FROM enrichment_jobs
                WHERE {where_sql}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """, params + [limit, offset])
            
            jobs = []
            for row in cur.fetchall():
                jobs.append({
                    "job_id": row['job_id'],
                    "job_type": row['job_type'],
                    "status": row['status'],
                    "memory_id": row['memory_id'],
                    "priority": row['priority'],
                    "created_at": row['created_at'].isoformat() if row['created_at'] else None,
                    "started_at": row['started_at'].isoformat() if row['started_at'] else None,
                    "completed_at": row['completed_at'].isoformat() if row['completed_at'] else None,
                    "error": row['error'],
                    "retry_count": row['retry_count']
                })
            
            return {
                "data": jobs,
                "meta": {
                    "total": total,
                    "limit": limit,
                    "offset": offset
                }
            }
        finally:
            cur.close()


@app.get("/admin/jobs/stats")
def admin_job_stats(tenant: dict = Depends(get_tenant)):
    """Get job queue statistics."""
    if not tenant:
        raise HTTPException(status_code=401, detail="API key required")
    
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # By status
            cur.execute("""
                SELECT status, COUNT(*) as count
                FROM enrichment_jobs
                GROUP BY status
            """)
            by_status = {row['status']: row['count'] for row in cur.fetchall()}
            
            # By type
            cur.execute("""
                SELECT job_type, COUNT(*) as count
                FROM enrichment_jobs
                GROUP BY job_type
            """)
            by_type = {row['job_type']: row['count'] for row in cur.fetchall()}
            
            # Recent activity
            cur.execute("""
                SELECT COUNT(*) as count
                FROM enrichment_jobs
                WHERE created_at > NOW() - INTERVAL '1 hour'
            """)
            last_hour = cur.fetchone()['count']
            
            # Failed jobs
            cur.execute("""
                SELECT COUNT(*) as count
                FROM enrichment_jobs
                WHERE status = 'failed' AND retry_count >= max_retries
            """)
            exhausted = cur.fetchone()['count']
            
            return {
                "by_status": by_status,
                "by_type": by_type,
                "last_hour": last_hour,
                "exhausted_retries": exhausted
            }
        finally:
            cur.close()


@app.post("/jobs/{job_id}/retry")
def retry_job(job_id: str, tenant: dict = Depends(get_tenant)):
    """Retry a failed job."""
    if not tenant:
        raise HTTPException(status_code=401, detail="API key required")
    
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cur.execute("""
                UPDATE enrichment_jobs
                SET status = 'pending', 
                    error = NULL,
                    started_at = NULL,
                    completed_at = NULL
                WHERE job_id = %s AND tenant_id = %s AND status = 'failed'
                RETURNING job_id
            """, (job_id, tenant['id']))
            
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Job not found or not in failed state")
            
            conn.commit()
            
            log.info("job.retry.queued", job_id=job_id)
            metrics.inc("gam_jobs_retried_total")
            
            return {"job_id": job_id, "status": "pending"}
        finally:
            cur.close()


# Wire reindex action to job queue
@app.post("/jobs/reindex/{memory_id}")
def enqueue_reindex(memory_id: int, tenant: dict = Depends(get_tenant)):
    """Enqueue a reindex job (convenience endpoint)."""
    if not tenant:
        raise HTTPException(status_code=401, detail="API key required")
    
    # Use the standard enqueue path
    request = EnqueueJobRequest(job_type="reindex", memory_id=memory_id, priority=2)
    return enqueue_job(request, tenant)


# ============================================================
# Worker Loop (Background Processing)
# ============================================================

import threading
import traceback

_worker_running = False
_worker_thread = None

def process_reindex_job(job: dict, conn) -> dict:
    """Process a reindex job."""
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute("SELECT content FROM memory_entries WHERE id = %s", (job['memory_id'],))
        row = cur.fetchone()
        if not row:
            return {"error": "Memory not found"}
        
        # Generate new embedding
        correlation_id = generate_correlation_id()
        embedding = generate_embedding(row['content'], correlation_id)
        
        # Update embedding
        cur.execute("""
            UPDATE memory_entries 
            SET embedding = %s::vector
            WHERE id = %s
        """, (embedding, job['memory_id']))
        
        conn.commit()
        return {"status": "reindexed", "memory_id": job['memory_id']}
    finally:
        cur.close()


# Entity types (v1)
ENTITY_TYPES = ["project", "person", "service", "component", "tool", "concept"]

ENTITY_EXTRACTION_PROMPT = """Extract entities from the following memory content. Return a JSON array of entities.

Each entity should have:
- type: one of [project, person, service, component, tool, concept]
- value: the entity name/value
- confidence: 0.0-1.0 confidence score
- excerpt: the relevant text span where this entity appears

Only extract entities that are clearly present. Be conservative.

Content:
{content}

Return ONLY valid JSON array, no other text:
[{{"type": "project", "value": "GAM", "confidence": 0.9, "excerpt": "GAM Platform"}}]
"""


def extract_entities_llm(content: str, correlation_id: str) -> List[dict]:
    """Extract entities using LLM."""
    client = get_openai_client()
    
    # Truncate content for extraction
    content = content[:4000]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an entity extraction system. Extract named entities from text and return JSON."},
                {"role": "user", "content": ENTITY_EXTRACTION_PROMPT.format(content=content)}
            ],
            temperature=0,
            max_tokens=1000
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse JSON (handle potential markdown wrapping)
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        
        entities = json.loads(result_text)
        
        # Validate and normalize
        valid_entities = []
        for e in entities:
            if isinstance(e, dict) and 'type' in e and 'value' in e:
                entity_type = e.get('type', 'concept').lower()
                if entity_type not in ENTITY_TYPES:
                    entity_type = 'concept'
                
                valid_entities.append({
                    'type': entity_type,
                    'value': str(e.get('value', ''))[:500],
                    'confidence': min(1.0, max(0.0, float(e.get('confidence', 0.8)))),
                    'excerpt': str(e.get('excerpt', ''))[:200]
                })
        
        return valid_entities
        
    except Exception as e:
        log.error("entity.extraction.llm_error",
                  correlation_id=correlation_id,
                  error=str(e))
        raise


def process_entity_extract_job(job: dict, conn) -> dict:
    """Process an entity extraction job."""
    correlation_id = generate_correlation_id()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Get memory content
        cur.execute("""
            SELECT id, content, tenant_id 
            FROM memory_entries 
            WHERE id = %s
        """, (job['memory_id'],))
        
        row = cur.fetchone()
        if not row:
            return {"error": "Memory not found"}
        
        memory_id = row['id']
        tenant_id = row['tenant_id']
        content = row['content']
        
        # Skip if too short
        if len(content) < 20:
            return {"status": "skipped", "reason": "content_too_short", "entities_count": 0}
        
        # Extract entities via LLM
        start_time = time.time()
        entities = extract_entities_llm(content, correlation_id)
        extraction_ms = int((time.time() - start_time) * 1000)
        
        log.info("entity.extraction.completed",
                 correlation_id=correlation_id,
                 memory_id=memory_id,
                 entities_count=len(entities),
                 extraction_ms=extraction_ms)
        
        # Idempotent upsert: delete existing entities for this memory first
        cur.execute("DELETE FROM entities WHERE memory_id = %s", (memory_id,))
        
        # Insert new entities
        inserted = 0
        for entity in entities:
            cur.execute("""
                INSERT INTO entities (tenant_id, memory_id, entity_type, entity_value, confidence, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                tenant_id,
                memory_id,
                entity['type'],
                entity['value'],
                entity['confidence'],
                json.dumps({'excerpt': entity.get('excerpt', '')})
            ))
            inserted += 1
        
        conn.commit()
        
        metrics.inc("gam_entities_extracted_total", {"count": str(inserted)})
        metrics.observe("gam_entity_extraction_ms", extraction_ms)
        
        return {
            "status": "extracted",
            "memory_id": memory_id,
            "entities_count": inserted,
            "extraction_ms": extraction_ms,
            "entities": entities
        }
        
    except Exception as e:
        conn.rollback()
        log.error("entity.extraction.failed",
                  correlation_id=correlation_id,
                  memory_id=job['memory_id'],
                  error=str(e))
        raise
    finally:
        cur.close()


TYPED_HINT_PROMPT = """Classify this memory content into exactly ONE category. Return ONLY the category name, nothing else.

Categories:
- decision: A choice or conclusion that was made
- milestone: An achievement or significant event completed
- task: Something that needs to be done or was done
- insight: A realization, learning, or understanding
- fact: A piece of information or reference data
- issue: A problem, bug, or concern

Content:
{content}

Category:"""


def classify_typed_hint(content: str, correlation_id: str) -> Optional[str]:
    """Classify memory content into a typed hint category."""
    client = get_openai_client()
    
    # Truncate content
    content = content[:2000]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a classification system. Return only the category name."},
                {"role": "user", "content": TYPED_HINT_PROMPT.format(content=content)}
            ],
            temperature=0,
            max_tokens=20
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        # Normalize and validate
        if result in TYPED_HINT_CATEGORIES:
            return result
        
        # Try to extract category from response
        for cat in TYPED_HINT_CATEGORIES:
            if cat in result:
                return cat
        
        log.warning("typed_hint.classification.unknown",
                    correlation_id=correlation_id,
                    result=result)
        return None
        
    except Exception as e:
        log.error("typed_hint.classification.error",
                  correlation_id=correlation_id,
                  error=str(e))
        raise


def process_typed_hint_job(job: dict, conn) -> dict:
    """Process a typed hint classification job."""
    correlation_id = generate_correlation_id()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Get memory content
        cur.execute("""
            SELECT id, content, tenant_id, metadata
            FROM memory_entries 
            WHERE id = %s
        """, (job['memory_id'],))
        
        row = cur.fetchone()
        if not row:
            return {"error": "Memory not found"}
        
        memory_id = row['id']
        content = row['content']
        metadata = row['metadata'] or {}
        
        # Skip if too short
        if len(content) < 10:
            return {"status": "skipped", "reason": "content_too_short"}
        
        # Classify via LLM
        start_time = time.time()
        hint_type = classify_typed_hint(content, correlation_id)
        classification_ms = int((time.time() - start_time) * 1000)
        
        if hint_type:
            # Store hint in metadata
            metadata['typed_hint'] = hint_type
            metadata['typed_hint_at'] = datetime.now(timezone.utc).isoformat()
            
            cur.execute("""
                UPDATE memory_entries
                SET metadata = %s
                WHERE id = %s
            """, (json.dumps(metadata), memory_id))
            
            conn.commit()
            
            log.info("typed_hint.completed",
                     correlation_id=correlation_id,
                     memory_id=memory_id,
                     hint_type=hint_type,
                     classification_ms=classification_ms)
            
            metrics.inc("gam_typed_hints_total", {"type": hint_type})
            metrics.observe("gam_typed_hint_ms", classification_ms)
            
            return {
                "status": "classified",
                "memory_id": memory_id,
                "typed_hint": hint_type,
                "classification_ms": classification_ms
            }
        else:
            return {
                "status": "unclassified",
                "memory_id": memory_id,
                "reason": "no_matching_category"
            }
        
    except Exception as e:
        conn.rollback()
        log.error("typed_hint.failed",
                  correlation_id=correlation_id,
                  memory_id=job['memory_id'],
                  error=str(e))
        raise
    finally:
        cur.close()


# ============================================================
# Relation Linking (Milestone D.4)
# ============================================================

# Relation types (v1 - locked)
RELATION_TYPES_V1 = ["related_to", "supports", "follows", "references"]

# Minimum similarity for relation creation
RELATION_MIN_SIMILARITY = 0.5
RELATION_MIN_ENTITY_OVERLAP = 1


def find_relation_candidates(memory_id: int, tenant_id, cur) -> List[dict]:
    """Find candidate memories to link based on entities and semantic similarity."""
    candidates = []
    
    # Get source memory's entities
    cur.execute("""
        SELECT entity_type, entity_value 
        FROM entities 
        WHERE memory_id = %s AND tenant_id = %s
    """, (memory_id, tenant_id))
    source_entities = {(r['entity_type'], r['entity_value']) for r in cur.fetchall()}
    
    # Get source memory's typed hint
    cur.execute("""
        SELECT metadata->>'typed_hint' as typed_hint
        FROM memory_entries
        WHERE id = %s
    """, (memory_id,))
    source_row = cur.fetchone()
    source_hint = source_row['typed_hint'] if source_row else None
    
    if not source_entities and not source_hint:
        return []
    
    # Find memories with overlapping entities
    if source_entities:
        entity_values = [v for _, v in source_entities]
        cur.execute("""
            SELECT DISTINCT e.memory_id, COUNT(*) as overlap_count
            FROM entities e
            WHERE e.tenant_id = %s
              AND e.memory_id != %s
              AND e.entity_value = ANY(%s)
            GROUP BY e.memory_id
            HAVING COUNT(*) >= %s
            ORDER BY overlap_count DESC
            LIMIT 10
        """, (tenant_id, memory_id, entity_values, RELATION_MIN_ENTITY_OVERLAP))
        
        for row in cur.fetchall():
            candidates.append({
                'memory_id': row['memory_id'],
                'entity_overlap': row['overlap_count'],
                'relation_type': 'related_to',
                'confidence': min(1.0, 0.5 + row['overlap_count'] * 0.1)
            })
    
    # Find semantically similar memories with same typed hint
    if source_hint:
        cur.execute("""
            WITH source AS (
                SELECT embedding FROM memory_entries WHERE id = %s
            )
            SELECT 
                m.id as memory_id,
                1 - (m.embedding <=> source.embedding) as similarity,
                m.metadata->>'typed_hint' as hint
            FROM memory_entries m, source
            WHERE m.tenant_id = %s
              AND m.id != %s
              AND m.embedding IS NOT NULL
              AND m.metadata->>'typed_hint' = %s
              AND 1 - (m.embedding <=> source.embedding) >= %s
            ORDER BY m.embedding <=> source.embedding
            LIMIT 5
        """, (memory_id, tenant_id, memory_id, source_hint, RELATION_MIN_SIMILARITY))
        
        for row in cur.fetchall():
            # Check if already in candidates
            existing = next((c for c in candidates if c['memory_id'] == row['memory_id']), None)
            if existing:
                existing['confidence'] = min(1.0, existing['confidence'] + 0.2)
                existing['has_hint_match'] = True
            else:
                candidates.append({
                    'memory_id': row['memory_id'],
                    'similarity': row['similarity'],
                    'relation_type': 'supports' if source_hint in ['decision', 'milestone'] else 'related_to',
                    'confidence': round(row['similarity'] * 0.8, 4),
                    'has_hint_match': True
                })
    
    return candidates


def process_relation_link_job(job: dict, conn) -> dict:
    """Process a relation linking job."""
    correlation_id = generate_correlation_id()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Get memory
        cur.execute("""
            SELECT id, tenant_id, content
            FROM memory_entries 
            WHERE id = %s
        """, (job['memory_id'],))
        
        row = cur.fetchone()
        if not row:
            return {"error": "Memory not found"}
        
        memory_id = row['id']
        tenant_id = row['tenant_id']
        
        # Find relation candidates
        start_time = time.time()
        candidates = find_relation_candidates(memory_id, tenant_id, cur)
        
        if not candidates:
            return {
                "status": "no_candidates",
                "memory_id": memory_id,
                "relations_created": 0
            }
        
        # Idempotent: delete existing relations from this source
        cur.execute("""
            DELETE FROM relations 
            WHERE source_memory_id = %s AND tenant_id = %s
        """, (memory_id, tenant_id))
        
        # Insert new relations
        created = 0
        for candidate in candidates:
            cur.execute("""
                INSERT INTO relations 
                (tenant_id, source_memory_id, object_memory_id, predicate, confidence, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                tenant_id,
                memory_id,
                candidate['memory_id'],
                candidate['relation_type'],
                candidate['confidence'],
                json.dumps({
                    'entity_overlap': candidate.get('entity_overlap'),
                    'similarity': candidate.get('similarity'),
                    'has_hint_match': candidate.get('has_hint_match', False)
                })
            ))
            created += 1
        
        conn.commit()
        
        linking_ms = int((time.time() - start_time) * 1000)
        
        log.info("relation_link.completed",
                 correlation_id=correlation_id,
                 memory_id=memory_id,
                 relations_created=created,
                 linking_ms=linking_ms)
        
        metrics.inc("gam_relations_created_total", {"count": str(created)})
        metrics.observe("gam_relation_link_ms", linking_ms)
        
        return {
            "status": "linked",
            "memory_id": memory_id,
            "relations_created": created,
            "relations": [
                {
                    "target_id": c['memory_id'],
                    "type": c['relation_type'],
                    "confidence": c['confidence']
                }
                for c in candidates
            ],
            "linking_ms": linking_ms
        }
        
    except Exception as e:
        conn.rollback()
        log.error("relation_link.failed",
                  correlation_id=correlation_id,
                  memory_id=job['memory_id'],
                  error=str(e))
        raise
    finally:
        cur.close()


def process_job(job: dict) -> dict:
    """Process a single enrichment job."""
    with get_db() as conn:
        if job['job_type'] == 'reindex':
            return process_reindex_job(job, conn)
        elif job['job_type'] == 'entity_extract':
            return process_entity_extract_job(job, conn)
        elif job['job_type'] == 'typed_hint':
            return process_typed_hint_job(job, conn)
        elif job['job_type'] == 'relation_link':
            return process_relation_link_job(job, conn)
        else:
            return {"status": "not_implemented", "job_type": job['job_type']}


def worker_loop():
    """Background worker loop for processing jobs."""
    global _worker_running
    
    log.info("worker.started")
    
    while _worker_running:
        try:
            with get_db() as conn:
                cur = conn.cursor(cursor_factory=RealDictCursor)
                
                # Claim a pending job
                cur.execute("""
                    UPDATE enrichment_jobs
                    SET status = 'running', started_at = NOW()
                    WHERE id = (
                        SELECT id FROM enrichment_jobs
                        WHERE status = 'pending'
                        ORDER BY priority DESC, created_at ASC
                        LIMIT 1
                        FOR UPDATE SKIP LOCKED
                    )
                    RETURNING *
                """)
                
                job = cur.fetchone()
                conn.commit()
                cur.close()
                
                if job:
                    log.info("worker.job.started", 
                             job_id=job['job_id'], 
                             job_type=job['job_type'])
                    
                    start_time = time.time()
                    
                    try:
                        result = process_job(dict(job))
                        duration_ms = int((time.time() - start_time) * 1000)
                        
                        # Mark completed
                        cur2 = conn.cursor()
                        cur2.execute("""
                            UPDATE enrichment_jobs
                            SET status = 'completed', 
                                completed_at = NOW(),
                                result = %s
                            WHERE job_id = %s
                        """, (json.dumps(result), job['job_id']))
                        conn.commit()
                        cur2.close()
                        
                        log.info("worker.job.completed",
                                 job_id=job['job_id'],
                                 job_type=job['job_type'],
                                 duration_ms=duration_ms)
                        
                        metrics.inc("gam_jobs_completed_total", {"job_type": job['job_type']})
                        metrics.observe("gam_job_duration_ms", duration_ms, {"job_type": job['job_type']})
                        
                    except Exception as e:
                        duration_ms = int((time.time() - start_time) * 1000)
                        error_msg = str(e)
                        
                        # Check retries
                        retry_count = job['retry_count'] + 1
                        new_status = 'pending' if retry_count < job['max_retries'] else 'failed'
                        
                        cur2 = conn.cursor()
                        cur2.execute("""
                            UPDATE enrichment_jobs
                            SET status = %s, 
                                error = %s,
                                retry_count = %s,
                                completed_at = CASE WHEN %s = 'failed' THEN NOW() ELSE NULL END
                            WHERE job_id = %s
                        """, (new_status, error_msg, retry_count, new_status, job['job_id']))
                        conn.commit()
                        cur2.close()
                        
                        log.error("worker.job.failed",
                                  job_id=job['job_id'],
                                  job_type=job['job_type'],
                                  error=error_msg,
                                  retry_count=retry_count,
                                  duration_ms=duration_ms)
                        
                        metrics.inc("gam_jobs_failed_total", {"job_type": job['job_type']})
                else:
                    # No jobs, sleep
                    time.sleep(1)
                    
        except Exception as e:
            log.error("worker.error", error=str(e), traceback=traceback.format_exc())
            time.sleep(5)
    
    log.info("worker.stopped")


def start_worker():
    """Start background worker thread."""
    global _worker_running, _worker_thread
    
    if _worker_running:
        return
    
    _worker_running = True
    _worker_thread = threading.Thread(target=worker_loop, daemon=True)
    _worker_thread.start()
    log.info("worker.thread.started")


def stop_worker():
    """Stop background worker thread."""
    global _worker_running
    _worker_running = False
    log.info("worker.thread.stopping")


# Start worker on app startup
@app.on_event("startup")
async def start_enrichment_worker():
    start_worker()


@app.on_event("shutdown")
async def stop_enrichment_worker():
    stop_worker()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8091"))
    uvicorn.run(app, host="0.0.0.0", port=port)
