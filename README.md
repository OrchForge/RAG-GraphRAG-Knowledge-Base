# RAG + GraphRAG Knowledge Base

A self-hosted, privacy-first knowledge base that combines **vector search (RAG)** with a **knowledge graph (GraphRAG)**. All processing runs locally via [Ollama](https://ollama.com) — your data never leaves your machine unless you explicitly choose a cloud embedding provider.

## Architecture

```
Your documents (Markdown, PDF, TXT)
        │
        ▼
   migrate.js          ← chunk + embed → Supabase (pgvector)
        │
        ▼
 graph_builder.js      ← LLM extracts entities & relations → Supabase (graph tables)
        │
   ┌────┴────┐
   ▼         ▼
chat.js   agents.js    ← RAG query interfaces (CLI)
              │
         server.js     ← MCP server (Claude Desktop integration)
              │
      graph_server.js  ← Visual graph editor (web UI)
```

**Pipeline for each query (agents.js):**

```
Question
   → Router         (classify question type)
   → Researcher     (vector similarity search)     ┐ parallel
   → GraphExplorer  (BFS over knowledge graph)     ┘
   → Enrich         (merge graph-linked documents)
   → Reranker       (per-chunk LLM scoring)
   → Synthesizer    (streaming answer generation)
```

## Requirements

- **Node.js** 20+
- **Ollama** running locally
- **Supabase** project with the `pgvector` extension enabled
- `pdftotext` for PDF ingestion: `sudo apt-get install -y poppler-utils`

### Recommended Ollama models

| Role | Model |
|------|-------|
| Embeddings | `bge-m3` or `nomic-embed-text` |
| Chat / extraction | `qwen3:1.7b` (recommended) or `qwen2.5:1.5b` |

## Setup

### 1. Install dependencies

```bash
npm install
```

### 2. Configure environment

Create a `.env` file:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=bge-m3
OLLAMA_CHAT_MODEL=qwen3:1.7b

# Optional: ollama (default) | openai | huggingface
EMBEDDING_PROVIDER=ollama

# Optional: local | internal (default) | public
DEFAULT_PRIVACY=internal

# Optional: port for the graph editor UI (default: 3003)
GRAPH_PORT=3003
```

### 3. Create Supabase tables

Run `supabase_setup.sql` in the **Supabase SQL Editor**. This creates:

- `documents` — chunked text with pgvector embeddings
- `entities` — knowledge graph nodes with embeddings
- `relationships` — knowledge graph edges
- `document_entities` — links between chunks and entities
- SQL functions: `match_documents`, `match_entities`, `graph_neighbors`

### 4. Pull Ollama models

```bash
ollama pull bge-m3
ollama pull qwen3:1.7b
```

## Usage

### Step 1 — Ingest documents

Place your `.md`, `.txt`, or `.pdf` files in a `docs/` folder, then run:

```bash
node migrate.js 'docs/**/*.{md,txt,pdf}'
```

### Step 2 — Build the knowledge graph

```bash
node graph_builder.js

# Enable verbose model output for debugging:
DEBUG_EXTRACT=1 node graph_builder.js
```

### Step 3 — Query your knowledge base

**Simple RAG chat** (query expansion + reranking):

```bash
node chat.js
```

**Multi-agent RAG + GraphRAG** (Router → Researcher ‖ GraphExplorer → Reranker → Synthesizer):

```bash
node agents.js
```

Special commands inside `agents.js`:
- `/graph` — show graph statistics
- `exit` — quit

### Visual graph editor

```bash
node graph_server.js
# Open: http://localhost:3003
```

Features: interactive graph visualization, entity detail panel, type filtering, add/delete relations, rebuild graph from UI, export PNG.

### MCP server (Claude Desktop)

```bash
node server.js
```

Add to your Claude Desktop `config.json`:

```json
{
  "mcpServers": {
    "rag-assistant": {
      "command": "node",
      "args": ["/absolute/path/to/server.js"]
    }
  }
}
```

Available MCP tools: `query_knowledge_base`, `store_memory`, `delete_memory`, `knowledge_base_stats`.

### Database management

```bash
node manage.js list                        # list all ingested files
node manage.js show docs/file.md           # preview chunks of a file
node manage.js delete docs/file.md         # delete a file's chunks
node manage.js delete-tag nginx            # delete chunks by tag
```

## Privacy model

Documents are classified into three privacy levels:

| Level | Description |
|-------|-------------|
| `local` | Never leaves the machine. Detected automatically (card numbers, passwords, etc.) |
| `internal` | Local LLM only (Ollama). Default for most documents. |
| `public` | Can be sent to cloud APIs (OpenAI, HuggingFace). |

Set per-document via frontmatter:

```markdown
---
privacy: internal
tags: [networking, tcp]
project: my-project
---

Your document content here.
```

## Project structure

```
├── server.js          MCP server (Claude Desktop integration)
├── agents.js          Multi-agent RAG + GraphRAG CLI
├── chat.js            Simple RAG chat CLI
├── graph_builder.js   Extracts entities & relations from documents
├── graph_server.js    Web-based graph editor and viewer
├── migrate.js         Ingests documents into Supabase
├── manage.js          CLI database management tool
├── package.json
└── .env               (not committed)
```

## License

MIT
