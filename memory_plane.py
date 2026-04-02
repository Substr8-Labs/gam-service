"""
Ada Memory Plane v1.2 — Canonical Memory Operations

This module implements:
- MemoryEnvelope schema
- Canonical write path
- Scoped retrieval
- Recall envelope generation
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, Any
import asyncpg


class MemoryClass(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryScope(str, Enum):
    RUN = "run"
    AGENT = "agent"
    PROJECT = "project"
    ORG = "org"


class MemoryStatus(str, Enum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    PROMOTED = "promoted"
    SUPPRESSED = "suppressed"
    ARCHIVED = "archived"


class SourceType(str, Enum):
    CONVERSATION = "conversation"
    DOCUMENT = "document"
    RUN = "run"
    MANUAL = "manual"
    PROMOTION = "promotion"
    SYSTEM = "system"


@dataclass
class ScopeContext:
    """Current scope context for memory operations."""
    org_id: str
    project_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class MemoryEnvelope:
    """
    Canonical memory envelope per Ada Memory Plane v1.2 spec.
    """
    # Required fields
    content: str
    memory_class: MemoryClass
    scope: MemoryScope
    source_type: SourceType
    source_ref: str
    actor_id: str
    org_id: str
    
    # Identity (auto-generated)
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_version: int = 1
    
    # Content hash (auto-generated)
    content_hash: str = ""
    
    # Optional fields
    summary: Optional[str] = None
    source_channel: Optional[str] = None
    project_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    
    # Temporal
    created_at: Optional[datetime] = None
    observed_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Governance
    status: MemoryStatus = MemoryStatus.ACTIVE
    confidence: float = 0.5
    canonical_status_reason: Optional[str] = None
    access_policy_ref: Optional[str] = None
    visibility: str = "default"
    authority_scope: Optional[MemoryScope] = None
    
    # Promotion lineage
    promotion_path: List[str] = field(default_factory=list)
    promoted_from: Optional[str] = None
    supersedes: Optional[str] = None
    
    # Relationships
    related_concepts: List[str] = field(default_factory=list)
    related_memories: List[str] = field(default_factory=list)
    
    # Usage
    used_in_runs: List[str] = field(default_factory=list)
    last_recalled_at: Optional[datetime] = None
    recall_count: int = 0
    
    # Class-specific metadata
    class_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Auto-generate computed fields."""
        now = datetime.now(timezone.utc)
        
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()
        
        if not self.created_at:
            self.created_at = now
        if not self.observed_at:
            self.observed_at = now
        if not self.updated_at:
            self.updated_at = now
        
        # Apply default status rules from spec
        if self.memory_class in (MemoryClass.SEMANTIC, MemoryClass.PROCEDURAL):
            # Semantic and procedural default to candidate unless explicitly set
            if self.status == MemoryStatus.ACTIVE and not self._is_auto_approved():
                self.status = MemoryStatus.CANDIDATE
    
    def _is_auto_approved(self) -> bool:
        """Check if this memory should be auto-approved."""
        # Agent-scope procedural with high confidence can auto-approve
        if self.memory_class == MemoryClass.PROCEDURAL:
            if self.scope == MemoryScope.AGENT and self.confidence >= 0.8:
                return True
        # High-confidence semantic from trusted sources
        if self.memory_class == MemoryClass.SEMANTIC:
            if self.source_type == SourceType.MANUAL and self.confidence >= 0.9:
                return True
        return False
    
    def to_db_row(self) -> Dict[str, Any]:
        """Convert to database row format."""
        return {
            'memory_id': self.memory_id,
            'memory_version': self.memory_version,
            'memory_class': self.memory_class.value,
            'scope': self.scope.value,
            'content': self.content,
            'content_hash': self.content_hash,
            'summary': self.summary,
            'source_type': self.source_type.value,
            'source_ref': self.source_ref,
            'source_channel': self.source_channel,
            'actor_id': self.actor_id,
            'created_at': self.created_at,
            'observed_at': self.observed_at,
            'updated_at': self.updated_at,
            'expires_at': self.expires_at,
            'org_id': self.org_id,
            'project_id': self.project_id,
            'agent_id': self.agent_id,
            'run_id': self.run_id,
            'status': self.status.value,
            'confidence': self.confidence,
            'canonical_status_reason': self.canonical_status_reason,
            'access_policy_ref': self.access_policy_ref,
            'visibility': self.visibility,
            'authority_scope': self.authority_scope.value if self.authority_scope else None,
            'promotion_path': self.promotion_path,
            'promoted_from': self.promoted_from,
            'supersedes': self.supersedes,
            'related_concepts': self.related_concepts,
            'related_memories': self.related_memories,
            'used_in_runs': self.used_in_runs,
            'last_recalled_at': self.last_recalled_at,
            'recall_count': self.recall_count,
            'class_metadata': json.dumps(self.class_metadata),
        }


@dataclass
class RecallEnvelope:
    """Envelope for tracking memory recall (for RunProof binding)."""
    run_id: str
    query: str
    scope_context: ScopeContext
    retrieval_method: str = "gam"
    memories_retrieved: List[str] = field(default_factory=list)
    memories_injected: List[str] = field(default_factory=list)
    recalled_at: Optional[datetime] = None
    proof_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.recalled_at:
            self.recalled_at = datetime.now(timezone.utc)
    
    def to_db_row(self) -> Dict[str, Any]:
        return {
            'run_id': self.run_id,
            'recalled_at': self.recalled_at,
            'query': self.query,
            'scope_context': json.dumps(self.scope_context.to_dict()),
            'retrieval_method': self.retrieval_method,
            'memories_retrieved': self.memories_retrieved,
            'memories_injected': self.memories_injected,
            'proof_id': self.proof_id,
        }


class MemoryPlane:
    """
    Ada Memory Plane v1.2 — Core operations.
    
    Handles:
    - Canonical write path
    - Scoped retrieval
    - Recall envelope management
    """
    
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
    
    async def write_canonical(self, envelope: MemoryEnvelope) -> str:
        """
        Write memory to canonical store.
        Returns memory_id.
        """
        row = envelope.to_db_row()
        
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO canonical_memories (
                    memory_id, memory_version, memory_class, scope,
                    content, content_hash, summary,
                    source_type, source_ref, source_channel, actor_id,
                    created_at, observed_at, updated_at, expires_at,
                    org_id, project_id, agent_id, run_id,
                    status, confidence, canonical_status_reason,
                    access_policy_ref, visibility, authority_scope,
                    promotion_path, promoted_from, supersedes,
                    related_concepts, related_memories,
                    used_in_runs, last_recalled_at, recall_count,
                    class_metadata
                ) VALUES (
                    $1, $2, $3, $4,
                    $5, $6, $7,
                    $8, $9, $10, $11,
                    $12, $13, $14, $15,
                    $16, $17, $18, $19,
                    $20, $21, $22,
                    $23, $24, $25,
                    $26, $27, $28,
                    $29, $30,
                    $31, $32, $33,
                    $34
                )
            ''',
                row['memory_id'], row['memory_version'], row['memory_class'], row['scope'],
                row['content'], row['content_hash'], row['summary'],
                row['source_type'], row['source_ref'], row['source_channel'], row['actor_id'],
                row['created_at'], row['observed_at'], row['updated_at'], row['expires_at'],
                row['org_id'], row['project_id'], row['agent_id'], row['run_id'],
                row['status'], row['confidence'], row['canonical_status_reason'],
                row['access_policy_ref'], row['visibility'], row['authority_scope'],
                row['promotion_path'], row['promoted_from'], row['supersedes'],
                row['related_concepts'], row['related_memories'],
                row['used_in_runs'], row['last_recalled_at'], row['recall_count'],
                row['class_metadata']
            )
        
        return envelope.memory_id
    
    async def recall_scoped(
        self,
        context: ScopeContext,
        memory_classes: Optional[List[MemoryClass]] = None,
        min_confidence: float = 0.5,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories using scoped inheritance.
        
        Order: agent → project → org (procedural before semantic)
        """
        classes_param = [c.value for c in memory_classes] if memory_classes else None
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT * FROM recall_scoped($1, $2, $3, $4, $5, $6)
            ''',
                context.org_id,
                context.project_id,
                context.agent_id,
                classes_param,
                min_confidence,
                limit
            )
        
        return [dict(row) for row in rows]
    
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a single memory by ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('''
                SELECT * FROM canonical_memories WHERE memory_id = $1
            ''', memory_id)
        
        return dict(row) if row else None
    
    async def update_recall_usage(self, memory_id: str, run_id: str) -> None:
        """Update usage tracking when memory is recalled."""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                UPDATE canonical_memories 
                SET 
                    used_in_runs = array_append(used_in_runs, $2),
                    last_recalled_at = NOW(),
                    recall_count = recall_count + 1
                WHERE memory_id = $1
            ''', memory_id, run_id)
    
    async def save_recall_envelope(self, envelope: RecallEnvelope) -> int:
        """Save recall envelope for proof binding."""
        row = envelope.to_db_row()
        
        async with self.pool.acquire() as conn:
            result = await conn.fetchrow('''
                INSERT INTO recall_envelopes (
                    run_id, recalled_at, query, scope_context,
                    retrieval_method, memories_retrieved, memories_injected,
                    proof_id
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
            ''',
                row['run_id'], row['recalled_at'], row['query'], row['scope_context'],
                row['retrieval_method'], row['memories_retrieved'], row['memories_injected'],
                row['proof_id']
            )
        
        return result['id']
    
    async def bind_to_proof(self, envelope_id: int, proof_id: str) -> None:
        """Bind recall envelope to RunProof."""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                UPDATE recall_envelopes 
                SET proof_id = $2, bound_at = NOW()
                WHERE id = $1
            ''', envelope_id, proof_id)


# Factory functions for common memory types

def create_episodic_memory(
    content: str,
    session_id: str,
    actor_id: str,
    org_id: str,
    agent_id: str,
    project_id: Optional[str] = None,
    source_channel: str = "conversation",
    confidence: float = 0.6,
    event_type: str = "conversation",
    participants: Optional[List[str]] = None
) -> MemoryEnvelope:
    """Create an episodic memory from a session."""
    return MemoryEnvelope(
        content=content,
        memory_class=MemoryClass.EPISODIC,
        scope=MemoryScope.AGENT if not project_id else MemoryScope.PROJECT,
        source_type=SourceType.CONVERSATION,
        source_ref=session_id,
        source_channel=source_channel,
        actor_id=actor_id,
        org_id=org_id,
        project_id=project_id,
        agent_id=agent_id,
        confidence=confidence,
        status=MemoryStatus.ACTIVE,  # Episodic goes active immediately
        class_metadata={
            "event_type": event_type,
            "session_id": session_id,
            "participants": participants or []
        }
    )


def create_semantic_memory(
    content: str,
    source_ref: str,
    actor_id: str,
    org_id: str,
    scope: MemoryScope = MemoryScope.AGENT,
    project_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    confidence: float = 0.7,
    concepts: Optional[List[str]] = None,
    fact_type: str = "definition"
) -> MemoryEnvelope:
    """Create a semantic (durable fact) memory."""
    return MemoryEnvelope(
        content=content,
        memory_class=MemoryClass.SEMANTIC,
        scope=scope,
        source_type=SourceType.MANUAL,
        source_ref=source_ref,
        actor_id=actor_id,
        org_id=org_id,
        project_id=project_id,
        agent_id=agent_id,
        confidence=confidence,
        related_concepts=concepts or [],
        class_metadata={
            "fact_type": fact_type
        }
    )


def create_procedural_memory(
    content: str,
    source_ref: str,
    actor_id: str,
    org_id: str,
    agent_id: str,
    applies_to: str = "behavior",
    trigger_conditions: Optional[List[str]] = None,
    exceptions: Optional[List[str]] = None,
    confidence: float = 0.7
) -> MemoryEnvelope:
    """Create a procedural (behavior-shaping) memory."""
    return MemoryEnvelope(
        content=content,
        memory_class=MemoryClass.PROCEDURAL,
        scope=MemoryScope.AGENT,
        source_type=SourceType.MANUAL,
        source_ref=source_ref,
        actor_id=actor_id,
        org_id=org_id,
        agent_id=agent_id,
        confidence=confidence,
        class_metadata={
            "applies_to": applies_to,
            "trigger_conditions": trigger_conditions or [],
            "exceptions": exceptions or []
        }
    )
