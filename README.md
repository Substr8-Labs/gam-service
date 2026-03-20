# GAM Service v2

Multi-tenant Memory Infrastructure for Agents.

## What's New in v2

- **Connection Pooling** - Fixes intermittent 500 errors
- **Multi-tenancy** - Isolated memory per tenant
- **Attention Scoring** - Salience-based ranking
- **Idempotent Writes** - Content hash deduplication
- **Better Error Handling** - Graceful failures

## API

### Tenant Management
```
POST /tenants          - Register new tenant, get API key
GET  /tenants/:id      - Get tenant info
```

### Memory Operations
```
POST /memory           - Store single memory (requires X-API-Key header)
POST /batch            - Batch ingest (backwards compatible with agent_id)
POST /search           - Hybrid search with attention ranking
GET  /stats            - Memory statistics
```

### Health
```
GET  /health           - Pool status + DB connection
```

## Hybrid Search

Search uses attention-weighted ranking:

```
final_score = similarity * (1 - α) + salience * α
```

Default α = 0.3 (configurable via `GAM_ATTENTION_WEIGHT`).

## Migration

Existing data is automatically migrated to the default "ada" tenant on first startup.

## Environment Variables

- `OPENAI_API_KEY` - Required for embeddings
- `GAM_ATTENTION_WEIGHT` - Attention weight (default: 0.3)
- `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE` - Database connection
