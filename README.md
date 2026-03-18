# GAM pgvector Service

Fast semantic search over Git-native Agent Memory using PostgreSQL + pgvector.

## Architecture

```
Git Repo (source of truth)
         ↓
    GAM Indexer
         ↓
PostgreSQL + pgvector (search index)
         ↓
   Retrieval API
```

## Quick Start

```bash
# 1. Set your OpenAI API key
export OPENAI_API_KEY=sk-...

# 2. Start services
docker-compose up -d

# 3. Wait for postgres to be ready (~10s)
docker-compose logs -f postgres

# 4. Index a repo
curl -X POST http://localhost:8091/index \
  -H "Content-Type: application/json" \
  -d '{
    "repo_path": "/repos/ada-workspace",
    "agent_id": "ada",
    "file_patterns": ["*.md", "*.txt"]
  }'

# 5. Search memory
curl -X POST http://localhost:8091/search \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "ada",
    "query": "What was the decision about sidecar architecture?",
    "limit": 5
  }'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/index` | POST | Index a git repo |
| `/search` | POST | Semantic search |
| `/stats` | GET | Memory stats for an agent |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GAM_DB_HOST` | localhost | PostgreSQL host |
| `GAM_DB_PORT` | 5433 | PostgreSQL port |
| `GAM_DB_USER` | gam | Database user |
| `GAM_DB_PASSWORD` | gam_dev_password | Database password |
| `GAM_DB_NAME` | gam | Database name |
| `OPENAI_API_KEY` | (required) | For embeddings |

## Indexing Strategy

The indexer:
1. Walks through git commits chronologically
2. Extracts content from files matching patterns (default: `*.md`, `*.txt`, `*.json`)
3. Chunks large files (~2000 chars per chunk)
4. Generates embeddings via OpenAI `text-embedding-3-small`
5. Stores in pgvector with commit provenance
6. Tracks indexing state to avoid re-processing

## Search

Uses cosine similarity with configurable threshold (default: 0.7).

Returns:
- Content chunk
- File path
- Git commit hash (provenance!)
- Committed timestamp
- Similarity score

## For Ada's Memory

To use this as Ada's memory system:

1. Deploy on VPS alongside ThreadHQ
2. Mount Ada's workspace: `/home/claw/.openclaw/workspace`
3. Run initial index
4. Set up cron/hook to re-index on new commits
5. Create OpenClaw skill that calls `/search` instead of `memory_search`

## Production Notes

- Set a real password in production
- Consider adding auth to the API
- For >1M entries, tune pgvector index (adjust `lists` parameter)
- Embeddings cost ~$0.02 per 1M tokens
# trigger rebuild 1773872077
