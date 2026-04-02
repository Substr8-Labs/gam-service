-- Ada Memory Plane v1.2 — Canonical Memories Table
-- This is the source of truth for all agent memory
-- GAM indexes from this; Neo4j projects from this

-- Memory class enum
CREATE TYPE memory_class AS ENUM ('working', 'episodic', 'semantic', 'procedural');

-- Memory scope enum
CREATE TYPE memory_scope AS ENUM ('run', 'agent', 'project', 'org');

-- Memory status enum
CREATE TYPE memory_status AS ENUM ('candidate', 'active', 'promoted', 'suppressed', 'archived');

-- Source type enum
CREATE TYPE source_type AS ENUM ('conversation', 'document', 'run', 'manual', 'promotion', 'system');

-- Canonical memories table (Delta Lake equivalent for Phase 1)
CREATE TABLE canonical_memories (
    -- Identity
    memory_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_version INTEGER NOT NULL DEFAULT 1,
    
    -- Classification
    memory_class memory_class NOT NULL,
    scope memory_scope NOT NULL,
    
    -- Content
    content TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,  -- SHA256
    summary TEXT,
    
    -- Provenance
    source_type source_type NOT NULL,
    source_ref TEXT NOT NULL,           -- session_id, doc_path, etc.
    source_channel VARCHAR(50),         -- telegram, discord, api, system
    actor_id VARCHAR(100) NOT NULL,     -- who/what created this
    
    -- Temporal
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    observed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,  -- for working memory TTL
    
    -- Hierarchy (scope tags)
    org_id VARCHAR(100) NOT NULL,
    project_id VARCHAR(100),
    agent_id VARCHAR(100),
    run_id UUID,
    
    -- Governance
    status memory_status NOT NULL DEFAULT 'active',
    confidence FLOAT NOT NULL DEFAULT 0.5 CHECK (confidence >= 0 AND confidence <= 1),
    canonical_status_reason TEXT,
    access_policy_ref VARCHAR(100),
    visibility VARCHAR(20) DEFAULT 'default' CHECK (visibility IN ('default', 'restricted', 'private')),
    authority_scope memory_scope,
    
    -- Promotion lineage
    promotion_path TEXT[],              -- history of scope promotions
    promoted_from UUID REFERENCES canonical_memories(memory_id),
    supersedes UUID REFERENCES canonical_memories(memory_id),
    
    -- Relationships (denormalized for query)
    related_concepts TEXT[],
    related_memories UUID[],
    
    -- Proof / Usage tracking
    used_in_runs UUID[],
    last_recalled_at TIMESTAMP WITH TIME ZONE,
    recall_count INTEGER DEFAULT 0,
    
    -- Class-specific fields (JSONB for flexibility)
    class_metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Uniqueness
    UNIQUE(content_hash, org_id, scope, memory_class)
);

-- Indexes for common queries
CREATE INDEX idx_canonical_org ON canonical_memories(org_id);
CREATE INDEX idx_canonical_project ON canonical_memories(org_id, project_id) WHERE project_id IS NOT NULL;
CREATE INDEX idx_canonical_agent ON canonical_memories(org_id, agent_id) WHERE agent_id IS NOT NULL;
CREATE INDEX idx_canonical_class ON canonical_memories(memory_class);
CREATE INDEX idx_canonical_scope ON canonical_memories(scope);
CREATE INDEX idx_canonical_status ON canonical_memories(status);
CREATE INDEX idx_canonical_created ON canonical_memories(created_at DESC);
CREATE INDEX idx_canonical_confidence ON canonical_memories(confidence DESC) WHERE status = 'active';

-- GIN index for concept search
CREATE INDEX idx_canonical_concepts ON canonical_memories USING GIN (related_concepts);

-- Partial index for active memories only (most common query)
CREATE INDEX idx_canonical_active ON canonical_memories(org_id, scope, memory_class) 
    WHERE status = 'active';

-- Memory version history (for audit trail)
CREATE TABLE memory_versions (
    id SERIAL PRIMARY KEY,
    memory_id UUID NOT NULL REFERENCES canonical_memories(memory_id),
    version INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    changed_by VARCHAR(100) NOT NULL,
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    change_reason TEXT,
    
    UNIQUE(memory_id, version)
);

CREATE INDEX idx_versions_memory ON memory_versions(memory_id, version DESC);

-- Recall envelopes (for RunProof binding)
CREATE TABLE recall_envelopes (
    id SERIAL PRIMARY KEY,
    run_id UUID NOT NULL,
    recalled_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    query TEXT NOT NULL,
    scope_context JSONB NOT NULL,       -- {org_id, project_id, agent_id, run_id}
    retrieval_method VARCHAR(20) NOT NULL CHECK (retrieval_method IN ('gam', 'graph', 'both')),
    
    -- Memory refs
    memories_retrieved UUID[] NOT NULL,
    memories_injected UUID[] NOT NULL,
    
    -- Proof binding
    proof_id UUID,
    bound_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_recall_run ON recall_envelopes(run_id);
CREATE INDEX idx_recall_proof ON recall_envelopes(proof_id) WHERE proof_id IS NOT NULL;

-- Scoped recall function
CREATE OR REPLACE FUNCTION recall_scoped(
    p_org_id VARCHAR(100),
    p_project_id VARCHAR(100) DEFAULT NULL,
    p_agent_id VARCHAR(100) DEFAULT NULL,
    p_memory_classes memory_class[] DEFAULT NULL,
    p_min_confidence FLOAT DEFAULT 0.5,
    p_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
    memory_id UUID,
    content TEXT,
    summary TEXT,
    memory_class memory_class,
    scope memory_scope,
    confidence FLOAT,
    source_scope memory_scope,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    WITH scope_matches AS (
        -- Agent scope
        SELECT cm.*, 'agent'::memory_scope as source_scope, 1 as scope_priority
        FROM canonical_memories cm
        WHERE cm.org_id = p_org_id
          AND cm.agent_id = p_agent_id
          AND cm.status = 'active'
          AND cm.confidence >= p_min_confidence
          AND (p_memory_classes IS NULL OR cm.memory_class = ANY(p_memory_classes))
          AND p_agent_id IS NOT NULL
        
        UNION ALL
        
        -- Project scope
        SELECT cm.*, 'project'::memory_scope as source_scope, 2 as scope_priority
        FROM canonical_memories cm
        WHERE cm.org_id = p_org_id
          AND cm.project_id = p_project_id
          AND cm.scope = 'project'
          AND cm.status = 'active'
          AND cm.confidence >= p_min_confidence
          AND (p_memory_classes IS NULL OR cm.memory_class = ANY(p_memory_classes))
          AND p_project_id IS NOT NULL
        
        UNION ALL
        
        -- Org scope
        SELECT cm.*, 'org'::memory_scope as source_scope, 3 as scope_priority
        FROM canonical_memories cm
        WHERE cm.org_id = p_org_id
          AND cm.scope = 'org'
          AND cm.status = 'active'
          AND cm.confidence >= p_min_confidence
          AND (p_memory_classes IS NULL OR cm.memory_class = ANY(p_memory_classes))
    )
    SELECT 
        sm.memory_id,
        sm.content,
        sm.summary,
        sm.memory_class,
        sm.scope,
        sm.confidence,
        sm.source_scope,
        sm.created_at
    FROM scope_matches sm
    ORDER BY sm.scope_priority, sm.confidence DESC, sm.created_at DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Comment for documentation
COMMENT ON TABLE canonical_memories IS 'Ada Memory Plane v1.2 - Canonical memory store. Source of truth for all agent memory.';
