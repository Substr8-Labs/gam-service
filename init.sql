-- GAM (Git-native Agent Memory) pgvector schema
-- Provides fast semantic search over git-committed memory

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Memory entries table
-- Each row = one piece of memory (from a git commit)
CREATE TABLE memory_entries (
    id SERIAL PRIMARY KEY,
    
    -- Git provenance
    commit_hash VARCHAR(40) NOT NULL,
    file_path TEXT NOT NULL,
    
    -- Content
    content TEXT NOT NULL,
    content_type VARCHAR(50) DEFAULT 'text',  -- text, conversation, decision, etc.
    
    -- Metadata
    agent_id VARCHAR(100) NOT NULL,           -- Which agent's memory
    customer_id VARCHAR(100),                  -- Optional: for customer-scoped memory
    session_id VARCHAR(100),                   -- Optional: session context
    
    -- Timestamps
    committed_at TIMESTAMP WITH TIME ZONE NOT NULL,
    indexed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Embedding (OpenAI text-embedding-3-small = 1536 dimensions)
    embedding vector(1536),
    
    -- Deduplication
    content_hash VARCHAR(64) NOT NULL,
    
    UNIQUE(commit_hash, file_path, content_hash)
);

-- Indexes for common queries
CREATE INDEX idx_memory_agent ON memory_entries(agent_id);
CREATE INDEX idx_memory_customer ON memory_entries(customer_id);
CREATE INDEX idx_memory_committed ON memory_entries(committed_at DESC);

-- Vector similarity index (IVFFlat for good perf up to ~1M rows)
CREATE INDEX idx_memory_embedding ON memory_entries 
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Semantic search function
CREATE OR REPLACE FUNCTION search_memory(
    p_agent_id VARCHAR(100),
    p_query_embedding vector(1536),
    p_limit INTEGER DEFAULT 10,
    p_customer_id VARCHAR(100) DEFAULT NULL,
    p_min_similarity FLOAT DEFAULT 0.7
)
RETURNS TABLE (
    id INTEGER,
    content TEXT,
    file_path TEXT,
    commit_hash VARCHAR(40),
    committed_at TIMESTAMP WITH TIME ZONE,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        me.id,
        me.content,
        me.file_path,
        me.commit_hash,
        me.committed_at,
        1 - (me.embedding <=> p_query_embedding) AS similarity
    FROM memory_entries me
    WHERE me.agent_id = p_agent_id
      AND (p_customer_id IS NULL OR me.customer_id = p_customer_id)
      AND me.embedding IS NOT NULL
      AND 1 - (me.embedding <=> p_query_embedding) >= p_min_similarity
    ORDER BY me.embedding <=> p_query_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Track indexing state (which commits have been processed)
CREATE TABLE indexing_state (
    repo_path TEXT PRIMARY KEY,
    last_commit_hash VARCHAR(40),
    last_indexed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Audit log for memory operations
CREATE TABLE memory_audit (
    id SERIAL PRIMARY KEY,
    operation VARCHAR(20) NOT NULL,  -- insert, search, delete
    agent_id VARCHAR(100) NOT NULL,
    customer_id VARCHAR(100),
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_audit_agent ON memory_audit(agent_id, created_at DESC);
