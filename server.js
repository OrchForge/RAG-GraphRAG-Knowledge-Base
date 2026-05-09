import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { createClient } from "@supabase/supabase-js";
import { z } from "zod";
import "dotenv/config";

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY,
  { auth: { persistSession: false } }
);

// Privacy levels (ascending): local → internal → public
const PRIVACY_LEVELS = {
  local:    0,
  internal: 1,
  public:   2,
};

function classifyContent(content, metadata = {}) {
  if (metadata.privacy && PRIVACY_LEVELS[metadata.privacy] !== undefined) {
    return metadata.privacy;
  }
  const sensitivePatterns = [
    /\b\d{16}\b/,
    /\b[A-Z]{2}\d{6,9}\b/,
    /password|secret|api[\s_-]?key/i,
    /\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b/,
    /internal use only|not for distribution/i,
  ];
  if (sensitivePatterns.some((p) => p.test(content))) {
    return "local";
  }
  return process.env.DEFAULT_PRIVACY || "internal";
}

function canUseExternalAPI(privacyLevel) {
  return PRIVACY_LEVELS[privacyLevel] >= PRIVACY_LEVELS["public"];
}

async function generateEmbedding(text, forceLocal = false) {
  const provider = forceLocal ? "ollama" : (process.env.EMBEDDING_PROVIDER || "ollama");

  if (provider === "ollama") {
    const res = await fetch(
      `${process.env.OLLAMA_BASE_URL || "http://localhost:11434"}/api/embeddings`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: process.env.OLLAMA_EMBED_MODEL || "nomic-embed-text",
          prompt: text,
        }),
      }
    );
    if (!res.ok) {
      throw new Error(`Ollama embedding HTTP ${res.status}: ${await res.text()}`);
    }
    const data = await res.json();
    if (!Array.isArray(data.embedding)) {
      throw new Error(`Unexpected Ollama response: ${JSON.stringify(data)}`);
    }
    return data.embedding;
  }

  if (provider === "openai") {
    if (!canUseExternalAPI("public")) {
      throw new Error("Privacy policy: cannot send data to OpenAI");
    }
    const res = await fetch("https://api.openai.com/v1/embeddings", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      },
      body: JSON.stringify({
        model: "text-embedding-3-small",
        input: text,
        dimensions: 768,
      }),
    });
    if (!res.ok) throw new Error(`OpenAI HTTP ${res.status}`);
    const data = await res.json();
    return data.data[0].embedding;
  }

  if (provider === "huggingface") {
    const res = await fetch(
      "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${process.env.HF_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ inputs: text, options: { wait_for_model: true } }),
      }
    );
    if (!res.ok) throw new Error(`HuggingFace HTTP ${res.status}`);
    const data = await res.json();
    return Array.isArray(data[0]) ? data[0] : data;
  }

  throw new Error(`Unknown embedding provider: ${provider}`);
}

function chunkText(text, chunkSize = 800, overlap = 80) {
  if (text.length <= chunkSize) {
    const piece = text.trim();
    return piece.length > 20 ? [piece] : [];
  }

  const chunks = [];
  let start = 0;

  while (start < text.length) {
    const end = Math.min(start + chunkSize, text.length);
    let actualEnd = end;

    if (end < text.length) {
      const boundary = Math.max(
        text.lastIndexOf("\n", end),
        text.lastIndexOf(". ", end)
      );
      if (boundary > start + chunkSize * 0.5) actualEnd = boundary + 1;
    }

    const chunk = text.slice(start, actualEnd).trim();
    if (chunk.length > 20) chunks.push(chunk);

    const nextStart = actualEnd - overlap;
    if (nextStart <= start) break;
    start = nextStart;
  }

  return chunks;
}

const server = new McpServer({
  name: "rag-assistant",
  version: "1.0.0",
});

server.tool(
  "query_knowledge_base",
  "Semantic search over the knowledge base. Returns relevant text fragments.",
  {
    query: z.string().min(1).max(2000).describe("Natural language search query"),
    top_k: z.number().int().min(1).max(10).default(5).describe("Number of results to return"),
    threshold: z.number().min(0).max(1).default(0.3).describe("Minimum cosine similarity (0–1)"),
    privacy_filter: z
      .array(z.enum(["local", "internal", "public"]))
      .default(["internal", "public"])
      .describe("Privacy levels to include in search"),
  },
  async ({ query, top_k, threshold, privacy_filter }) => {
    const forceLocal = privacy_filter.includes("local");

    let embedding;
    try {
      embedding = await generateEmbedding(query, forceLocal);
    } catch (err) {
      return {
        content: [{ type: "text", text: `Embedding error: ${err.message}` }],
        isError: true,
      };
    }

    const { data, error } = await supabase.rpc("match_documents", {
      query_embedding: embedding,
      match_count:     top_k,
      match_threshold: threshold,
      filter_privacy:  privacy_filter,
    });

    if (error) {
      return {
        content: [{ type: "text", text: `Supabase RPC error: ${error.message}` }],
        isError: true,
      };
    }

    if (!data || data.length === 0) {
      return {
        content: [{ type: "text", text: "No results found in the knowledge base." }],
      };
    }

    const results = data
      .map((doc, i) => {
        const meta = doc.metadata ? JSON.stringify(doc.metadata) : "{}";
        return (
          `[${i + 1}] Similarity: ${(doc.similarity * 100).toFixed(1)}% | ` +
          `Privacy: ${doc.privacy} | Meta: ${meta}\n${doc.content}`
        );
      })
      .join("\n\n---\n\n");

    return {
      content: [
        {
          type: "text",
          text: `Found ${data.length} fragments:\n\n${results}`,
        },
      ],
    };
  }
);

server.tool(
  "store_memory",
  "Saves a knowledge fragment to the database. Automatically splits text into chunks and generates embeddings.",
  {
    content: z.string().min(1).max(50000).describe("Text to store"),
    metadata: z
      .object({
        title:   z.string().optional(),
        tags:    z.array(z.string()).optional(),
        source:  z.string().optional(),
        privacy: z.enum(["local", "internal", "public"]).optional(),
        project: z.string().optional(),
      })
      .default({})
      .describe("Metadata: title, tags, source, privacy, project"),
    chunk_size: z
      .number().int().min(200).max(4000).default(800)
      .describe("Chunk size in characters"),
  },
  async ({ content, metadata, chunk_size }) => {
    const privacy    = classifyContent(content, metadata);
    const forceLocal = !canUseExternalAPI(privacy);

    if (forceLocal && process.env.EMBEDDING_PROVIDER !== "ollama") {
      console.error(
        `[PRIVACY FILTER] Document classified as '${privacy}' — forcing local Ollama`
      );
    }

    const chunks   = chunkText(content, chunk_size, Math.floor(chunk_size * 0.1));
    const inserted = [];
    const errors   = [];

    for (let i = 0; i < chunks.length; i++) {
      try {
        const embedding = await generateEmbedding(chunks[i], forceLocal);
        const chunkMeta = {
          ...metadata,
          privacy,
          chunk_index:  i,
          total_chunks: chunks.length,
        };
        const { data, error } = await supabase
          .from("documents")
          .insert({ content: chunks[i], metadata: chunkMeta, embedding, privacy })
          .select("id")
          .single();

        if (error) throw error;
        inserted.push(data.id);
      } catch (err) {
        errors.push(`Chunk ${i}: ${err.message}`);
      }
    }

    const lines = [
      `Saved: ${inserted.length}/${chunks.length} chunks`,
      `Privacy level: ${privacy}`,
      `Embedder: ${forceLocal ? "Ollama (local)" : process.env.EMBEDDING_PROVIDER || "ollama"}`,
    ];
    if (errors.length > 0) lines.push(`Errors: ${errors.join("; ")}`);

    return { content: [{ type: "text", text: lines.join("\n") }] };
  }
);

server.tool(
  "delete_memory",
  "Deletes records from the database by UUID or metadata filter.",
  {
    ids: z.array(z.string().uuid()).optional().describe("List of UUIDs to delete"),
    filter_tag: z.string().optional().describe("Delete all records with this tag"),
    filter_project: z.string().optional().describe("Delete all records for this project"),
  },
  async ({ ids, filter_tag, filter_project }) => {
    if (!ids?.length && !filter_tag && !filter_project) {
      return {
        content: [
          { type: "text", text: "Provide at least one parameter: ids, filter_tag, or filter_project" },
        ],
        isError: true,
      };
    }

    let query = supabase.from("documents").delete().neq("id", "00000000-0000-0000-0000-000000000000");

    if (ids?.length)    query = query.in("id", ids);
    if (filter_tag)     query = query.contains("metadata", { tags: [filter_tag] });
    if (filter_project) query = query.eq("metadata->>project", filter_project);

    const { error, count } = await query.select("id", { count: "exact" });
    if (error) {
      return { content: [{ type: "text", text: `Error: ${error.message}` }], isError: true };
    }

    return { content: [{ type: "text", text: `Deleted records: ${count ?? "unknown"}` }] };
  }
);

server.tool(
  "knowledge_base_stats",
  "Knowledge base statistics: record count, distribution by privacy and project.",
  {},
  async () => {
    const { data, error } = await supabase
      .from("documents")
      .select("privacy, metadata->>project");

    if (error) {
      return { content: [{ type: "text", text: `Error: ${error.message}` }], isError: true };
    }

    const byPrivacy = {};
    const byProject = {};

    for (const row of data || []) {
      byPrivacy[row.privacy] = (byPrivacy[row.privacy] || 0) + 1;
      const proj = row.project || "no project";
      byProject[proj] = (byProject[proj] || 0) + 1;
    }

    const lines = [
      `Total documents: ${data?.length ?? 0}`,
      "",
      "By privacy level:",
      ...Object.entries(byPrivacy).map(([k, v]) => `  ${k}: ${v}`),
      "",
      "By project:",
      ...Object.entries(byProject).map(([k, v]) => `  ${k}: ${v}`),
    ];

    return { content: [{ type: "text", text: lines.join("\n") }] };
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
console.error("[RAG MCP] Server started on stdio");
