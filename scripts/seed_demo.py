#!/usr/bin/env python3
"""
GAM Demo Seed Script

Creates a demo tenant with curated memories showing:
- Different memory kinds
- Different salience levels
- Hybrid retrieval in action
"""

import requests
import json
import sys

GAM_URL = "https://gam-service-production.up.railway.app"

# Demo memories with expected salience behavior
DEMO_MEMORIES = [
    # High salience - decisions
    {
        "content": "We decided to use PostgreSQL with pgvector for the memory layer. This gives us hybrid search capabilities and proven reliability.",
        "memory_kind": "decision",
        "source_channel": "meeting"
    },
    {
        "content": "Strategic decision: Focus on MCP integration first, admin console second. Agent access is the critical path.",
        "memory_kind": "decision",
        "source_channel": "planning"
    },
    {
        "content": "Architecture decision: Attention scoring will use α=0.3 weight by default. This balances semantic relevance with importance signals.",
        "memory_kind": "decision",
        "source_channel": "technical"
    },
    
    # High salience - milestones
    {
        "content": "Critical milestone achieved: GAM v2 deployed to production with multi-tenant support and connection pooling. This is a breakthrough for platform stability.",
        "memory_kind": "milestone",
        "source_channel": "deploy"
    },
    {
        "content": "Milestone: First external agent successfully stored and retrieved memories. The platform is no longer just Ada's personal store.",
        "memory_kind": "milestone",
        "source_channel": "integration"
    },
    
    # Medium salience - tasks
    {
        "content": "Task completed: Implemented idempotent writes via content hash deduplication. No more duplicate entries on retry.",
        "memory_kind": "task",
        "source_channel": "development"
    },
    {
        "content": "Task: Set up metrics dashboard for memory operations. Need to track write success rate and search latency.",
        "memory_kind": "task",
        "source_channel": "ops"
    },
    {
        "content": "Follow up needed: Review the typed hints implementation from the whitepaper benchmark and port to production.",
        "memory_kind": "task",
        "source_channel": "backlog"
    },
    
    # Low salience - routine
    {
        "content": "Daily standup: No blockers. Continuing work on MCP wrapper.",
        "memory_kind": "note",
        "source_channel": "standup"
    },
    {
        "content": "Routine health check passed. All services responding normally.",
        "memory_kind": "note",
        "source_channel": "monitoring"
    },
    
    # Mixed content - for retrieval testing
    {
        "content": "The hybrid search combines semantic similarity with salience scoring. Final score = similarity * 0.7 + salience * 0.3. This surfaces important memories even when semantic match is lower.",
        "memory_kind": "note",
        "source_channel": "documentation"
    },
    {
        "content": "Memory kinds help categorize entries: notes for general, decisions for choices, milestones for achievements, tasks for action items.",
        "memory_kind": "note",
        "source_channel": "documentation"
    },
    {
        "content": "Important insight: Agents need context packs, not just search results. The memory_context tool assembles relevant memories into prompt-ready format.",
        "memory_kind": "note",
        "source_channel": "design"
    },
    {
        "content": "Learned: Connection pooling is essential for production. Without it, each request opens a new connection and eventually exhausts the pool.",
        "memory_kind": "note",
        "source_channel": "incident"
    },
    {
        "content": "Customer signal: Operators want to see WHY a memory ranked highly. Need to expose salience breakdown in admin UI.",
        "memory_kind": "note",
        "source_channel": "feedback"
    },
]

def create_demo_tenant():
    """Create the demo tenant."""
    response = requests.post(
        f"{GAM_URL}/tenants",
        json={"name": "demo-showcase"}
    )
    if response.status_code != 200:
        print(f"Failed to create tenant: {response.text}")
        sys.exit(1)
    
    tenant = response.json()
    print(f"Created tenant: {tenant['name']}")
    print(f"API Key: {tenant['api_key']}")
    print(f"Tenant ID: {tenant['id']}")
    return tenant

def seed_memories(api_key):
    """Seed demo memories."""
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key
    }
    
    results = []
    for i, memory in enumerate(DEMO_MEMORIES):
        response = requests.post(
            f"{GAM_URL}/memory",
            headers=headers,
            json=memory
        )
        if response.status_code == 200:
            result = response.json()
            results.append(result)
            print(f"  [{i+1}/{len(DEMO_MEMORIES)}] {memory['memory_kind']}: salience={result['salience_score']}")
        else:
            print(f"  [{i+1}] FAILED: {response.text}")
    
    return results

def test_search(api_key):
    """Test search with demo queries."""
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key
    }
    
    queries = [
        "what decisions have we made",
        "what milestones achieved",
        "database architecture",
        "how does ranking work",
    ]
    
    print("\n=== Search Tests ===")
    for query in queries:
        response = requests.post(
            f"{GAM_URL}/search",
            headers=headers,
            json={"query": query, "limit": 3}
        )
        results = response.json()
        print(f"\nQuery: '{query}'")
        for r in results:
            print(f"  [{r['memory_kind']}] score={r['final_score']:.2f} (sim={r['similarity']:.2f}, sal={r['salience_score']:.2f})")
            print(f"       {r['content'][:60]}...")

def main():
    print("=== GAM Demo Seed Script ===\n")
    
    # Create tenant
    tenant = create_demo_tenant()
    api_key = tenant['api_key']
    
    print("\n=== Seeding Memories ===")
    results = seed_memories(api_key)
    
    print(f"\nSeeded {len(results)} memories")
    
    # Show salience distribution
    saliences = [r['salience_score'] for r in results]
    high = len([s for s in saliences if s >= 0.7])
    medium = len([s for s in saliences if 0.4 <= s < 0.7])
    low = len([s for s in saliences if s < 0.4])
    
    print(f"\nSalience distribution:")
    print(f"  High (≥0.7): {high}")
    print(f"  Medium (0.4-0.7): {medium}")
    print(f"  Low (<0.4): {low}")
    
    # Test search
    test_search(api_key)
    
    print("\n=== Demo Setup Complete ===")
    print(f"Tenant: demo-showcase")
    print(f"API Key: {api_key}")
    print(f"Use this key for demo queries")

if __name__ == "__main__":
    main()
