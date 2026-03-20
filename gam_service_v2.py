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
SERVICE_VERSION = "2.1.0"

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

@app.post("/search", response_model=List[SearchResult])
def search_memory(request: SearchRequest, tenant: dict = Depends(get_tenant)):
    """Hybrid search with attention ranking."""
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
            
            log.info("search.started",
                     correlation_id=correlation_id,
                     tenant_id=str(tenant_id),
                     query_length=len(request.query),
                     limit=request.limit)
            
            metrics.inc("gam_search_total", {"tenant": request.agent_id or "api"})
            
            query_embedding = generate_embedding(request.query, correlation_id)
            attention_weight = request.attention_weight or ATTENTION_WEIGHT
            
            cur.execute("""
                SELECT 
                    id, content, file_path, committed_at,
                    source_channel, memory_kind,
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
            
            results.sort(key=lambda x: x.final_score, reverse=True)
            results = results[:request.limit]
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            if not results:
                metrics.inc("gam_search_empty_total", {"tenant": request.agent_id or "api"})
            
            log.info("search.completed",
                     correlation_id=correlation_id,
                     tenant_id=str(tenant_id),
                     result_count=len(results),
                     duration_ms=duration_ms)
            
            metrics.observe("gam_search_duration_ms", duration_ms, {"tenant": request.agent_id or "api"})
            
            return results
            
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
            # Get source memory
            cur.execute("""
                SELECT id, content, memory_kind, embedding, salience_score
                FROM memory_entries 
                WHERE id = %s AND tenant_id = %s
            """, (memory_id, tenant['id']))
            
            source = cur.fetchone()
            if not source:
                raise HTTPException(status_code=404, detail="Memory not found")
            
            if not source['embedding']:
                raise HTTPException(status_code=400, detail="Memory has no embedding")
            
            # Find related by semantic similarity
            cur.execute("""
                SELECT 
                    id, content, memory_kind, salience_score,
                    1 - (embedding <=> %s::vector) as similarity
                FROM memory_entries
                WHERE tenant_id = %s
                  AND id != %s
                  AND embedding IS NOT NULL
                  AND 1 - (embedding <=> %s::vector) >= 0.3
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (source['embedding'], tenant['id'], memory_id, 
                  source['embedding'], source['embedding'], limit))
            
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8091"))
    uvicorn.run(app, host="0.0.0.0", port=port)
