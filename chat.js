import * as readline from "readline/promises";
import os from "os";
import "dotenv/config";

const OLLAMA      = process.env.OLLAMA_BASE_URL    || "http://localhost:11434";
const EMBED_MODEL = process.env.OLLAMA_EMBED_MODEL || "bge-m3";
const CHAT_MODEL  = process.env.OLLAMA_CHAT_MODEL  || "qwen2.5:1.5b";
const SB_URL      = process.env.SUPABASE_URL;
const SB_KEY      = process.env.SUPABASE_SERVICE_ROLE_KEY;

const PHYSICAL_CORES = Math.max(1, Math.floor(os.cpus().length / 2));

const LLM_OPTIONS = {
  num_ctx:        4096,
  num_thread:     PHYSICAL_CORES,
  temperature:    0.1,
  top_p:          0.5,
  repeat_penalty: 1.15,
  top_k:          20,
};

const SEARCH_CANDIDATES = 15;
const SEARCH_THRESHOLD  = 0.25;
const FINAL_TOP_K       = 5;

async function ollamaFetch(path, body, timeoutMs = 60000) {
  const ctrl  = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${OLLAMA}${path}`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(body),
      signal:  ctrl.signal,
    });
    clearTimeout(timer);
    if (!res.ok) throw new Error(`Ollama HTTP ${res.status}: ${await res.text()}`);
    return res;
  } catch (err) {
    clearTimeout(timer);
    if (err.name === "AbortError")
      throw new Error(`Ollama timeout (${timeoutMs / 1000}s). Check: systemctl status ollama`);
    throw err;
  }
}

async function embed(text) {
  const res  = await ollamaFetch("/api/embeddings", { model: EMBED_MODEL, prompt: text });
  const data = await res.json();
  if (!Array.isArray(data.embedding))
    throw new Error(`Unexpected embedding response: ${JSON.stringify(data).slice(0, 120)}`);
  return data.embedding;
}

// Generates 2-3 rephrasings of the question to improve recall
async function expandQuery(question) {
  const prompt =
    `You are a technical search assistant. Rephrase the question in two different ways.\n` +
    `Preserve all technical terms as-is (TCP, DNS, RJ45, Slow Start, etc.).\n` +
    `Return ONLY a JSON array of 2 strings.\n` +
    `Example: ["rephrasing 1", "rephrasing 2"]\n\n` +
    `QUESTION: ${question}`;

  try {
    const res = await ollamaFetch(
      "/api/generate",
      { model: CHAT_MODEL, prompt, stream: false, options: { temperature: 0.3, num_ctx: 512 } },
      15000
    );
    const data  = await res.json();
    const match = data.response?.match(/\[.*?\]/s);
    if (!match) return [question];
    const variants = JSON.parse(match[0]);
    if (!Array.isArray(variants)) return [question];
    return [question, ...variants.filter(v => typeof v === "string").slice(0, 2)];
  } catch {
    return [question];
  }
}

async function searchByEmbedding(embedding, topK = SEARCH_CANDIDATES) {
  const res = await fetch(`${SB_URL}/rest/v1/rpc/match_documents`, {
    method:  "POST",
    headers: {
      "Content-Type": "application/json",
      "apikey":        SB_KEY,
      "Authorization": `Bearer ${SB_KEY}`,
    },
    body: JSON.stringify({
      query_embedding: embedding,
      match_count:     topK,
      match_threshold: SEARCH_THRESHOLD,
      filter_privacy:  ["internal", "public"],
    }),
  });
  if (!res.ok) throw new Error(`Supabase HTTP ${res.status}: ${await res.text()}`);
  return res.json();
}

// Searches with query expansion: merges and deduplicates results from multiple queries
async function search(question) {
  process.stdout.write("🔍 Query expansion...");
  const queries = await expandQuery(question);
  process.stdout.write(` ${queries.length} queries\n`);

  const seen    = new Set();
  const results = [];

  for (const q of queries) {
    process.stdout.write(`   🗄  Searching: "${q.slice(0, 60)}"...`);
    const embedding = await embed(q);
    const hits      = await searchByEmbedding(embedding, SEARCH_CANDIDATES);
    process.stdout.write(` ${hits.length} found\n`);

    for (const hit of hits) {
      if (!seen.has(hit.id)) {
        seen.add(hit.id);
        results.push(hit);
      }
    }
  }

  results.sort((a, b) => b.similarity - a.similarity);
  return results.slice(0, SEARCH_CANDIDATES);
}

// Scores each chunk individually — more reliable than batch scoring for small models
async function scoreChunk(question, chunkText) {
  const prompt =
    `Rate the relevance of this fragment for answering the question.\n` +
    `Return ONLY a single integer from 0 to 10.\n` +
    `10 = the fragment directly answers the question.\n` +
    `0  = the fragment is completely unrelated.\n\n` +
    `QUESTION: ${question}\n\n` +
    `FRAGMENT: ${chunkText.slice(0, 350)}\n\n` +
    `SCORE (digits only):`;

  try {
    const res = await ollamaFetch(
      "/api/generate",
      {
        model:   CHAT_MODEL,
        prompt,
        stream:  false,
        options: { temperature: 0, num_ctx: 512, num_predict: 3 },
      },
      10000
    );
    const data  = await res.json();
    const match = data.response?.match(/\b([0-9]|10)\b/);
    return match ? parseInt(match[1], 10) / 10 : 0.5;
  } catch {
    return 0.5;
  }
}

async function rerank(question, candidates, topN = FINAL_TOP_K) {
  if (candidates.length <= topN) return candidates;

  process.stdout.write(`🎯 Reranking ${candidates.length} chunks`);

  const scored = [];
  for (let i = 0; i < candidates.length; i++) {
    process.stdout.write(".");
    const score = await scoreChunk(question, candidates[i].content);
    scored.push({ ...candidates[i], rerank_score: score });
  }

  process.stdout.write(" ✓\n");

  return scored
    .sort((a, b) => b.rerank_score - a.rerank_score)
    .slice(0, topN);
}

function buildSystemPrompt(question, chunks) {
  if (chunks.length === 0) {
    return (
      `You are a technical assistant. You have no knowledge base data for this question.\n` +
      `Tell the user: "The loaded documentation contains no information on this topic."\n` +
      `Do not answer from general knowledge. Do not invent.`
    );
  }

  const contextBlock = chunks
    .map((c, i) => {
      const score = c.rerank_score ?? c.similarity;
      const src   = c.metadata?.source ? ` | source: ${c.metadata.source}` : "";
      return (
        `--- FRAGMENT ${i + 1} (relevance ${(score * 100).toFixed(0)}%${src}) ---\n` +
        c.content
      );
    })
    .join("\n\n");

  return (
    `You are a strict technical assistant. Answer EXCLUSIVELY based on the fragments below.\n\n` +
    `RULES:\n` +
    `1. Use ONLY information from the FRAGMENTS. No external knowledge.\n` +
    `2. If the fragments contain no answer — reply ONLY: "The loaded documentation contains no information on this topic."\n` +
    `3. Do not invent, supplement, or generalize beyond the context.\n` +
    `4. NEVER translate established IT terms. Keep as-is: TCP, UDP, DNS, RJ45, Slow Start,\n` +
    `   Handshake, AIMD, Three-Way Handshake, Round-Trip Time, Congestion Window, TLS, ACK, etc.\n` +
    `5. Be precise and concise. No filler phrases like "Sure!" or "Great question!".\n` +
    `6. If different fragments contradict each other — state that explicitly.\n\n` +
    `DOCUMENTATION FRAGMENTS:\n${contextBlock}\n\n` +
    `QUESTION: ${question}`
  );
}

async function ask(question, chunks) {
  const prompt = buildSystemPrompt(question, chunks);

  const res = await ollamaFetch(
    "/api/chat",
    {
      model:    CHAT_MODEL,
      messages: [{ role: "user", content: prompt }],
      stream:   true,
      options:  LLM_OPTIONS,
    },
    180000
  );

  process.stdout.write("\n🤖 ");

  const reader  = res.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    for (const line of decoder.decode(value, { stream: true }).split("\n").filter(Boolean)) {
      try {
        const json = JSON.parse(line);
        if (json.message?.content) process.stdout.write(json.message.content);
      } catch { /* incomplete JSON chunk */ }
    }
  }

  console.log("\n");
}

async function main() {
  if (!SB_URL || !SB_KEY) {
    console.error("✗ SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set in .env");
    process.exit(1);
  }

  const modelPad = CHAT_MODEL.padEnd(18).slice(0, 18);
  console.log(`╔══════════════════════════════════════════╗`);
  console.log(`║  RAG Chat  |  ${modelPad}  ║`);
  console.log(`║  Embed: ${EMBED_MODEL.padEnd(16).slice(0,16)}  |  CPU: ${PHYSICAL_CORES} cores  ║`);
  console.log(`║  KV cache: ${LLM_OPTIONS.num_ctx} tokens | temp: ${LLM_OPTIONS.temperature}    ║`);
  console.log(`║  Reranker: per-chunk individual scoring  ║`);
  console.log(`║  Type /help for available commands       ║`);
  console.log(`╚══════════════════════════════════════════╝\n`);

  const rl = readline.createInterface({
    input:  process.stdin,
    output: process.stdout,
  });

  process.on("SIGINT", () => { console.log("\nExiting."); rl.close(); process.exit(0); });

  while (true) {
    let question;
    try {
      question = await rl.question("❓ Question: ");
    } catch {
      console.log("\nExiting."); break;
    }

    if (!question.trim()) continue;

    if (question === "/help") {
      console.log(`
Commands:
  /help    — this help
  /list    — all files in the knowledge base
  /stats   — database statistics
  /model   — current model and parameters
  exit     — quit
      `);
      continue;
    }

    if (question === "/model") {
      console.log(`\nModel:       ${CHAT_MODEL}`);
      console.log(`Embed:       ${EMBED_MODEL}`);
      console.log(`num_ctx:     ${LLM_OPTIONS.num_ctx}`);
      console.log(`temperature: ${LLM_OPTIONS.temperature}`);
      console.log(`Candidates:  ${SEARCH_CANDIDATES} → top ${FINAL_TOP_K}\n`);
      continue;
    }

    if (question === "/list") {
      const res  = await fetch(`${SB_URL}/rest/v1/documents?select=metadata,privacy`, {
        headers: { "apikey": SB_KEY, "Authorization": `Bearer ${SB_KEY}` },
      });
      const rows = await res.json();
      const sources = {};
      for (const row of rows) {
        const src = row.metadata?.source || "unknown";
        sources[src] = (sources[src] || 0) + 1;
      }
      console.log("\n📚 Files in the knowledge base:");
      for (const [src, count] of Object.entries(sources)) {
        console.log(`   ${String(count).padStart(4)} chunks  ${src}`);
      }
      console.log(`   Total: ${rows.length} chunks\n`);
      continue;
    }

    if (question === "/stats") {
      const res  = await fetch(`${SB_URL}/rest/v1/documents?select=privacy,metadata`, {
        headers: { "apikey": SB_KEY, "Authorization": `Bearer ${SB_KEY}` },
      });
      const rows      = await res.json();
      const byPrivacy = {};
      for (const row of rows) {
        byPrivacy[row.privacy] = (byPrivacy[row.privacy] || 0) + 1;
      }
      console.log(`\n📊 Stats: total ${rows.length} chunks`);
      for (const [k, v] of Object.entries(byPrivacy)) {
        console.log(`   ${k}: ${v}`);
      }
      console.log();
      continue;
    }

    if (question.toLowerCase() === "exit") {
      console.log("Exiting."); rl.close(); break;
    }

    try {
      const candidates = await search(question);

      if (candidates.length === 0) {
        console.log("⚠️  No relevant fragments found in the knowledge base\n");
        await ask(question, []);
        continue;
      }

      const topChunks = await rerank(question, candidates, FINAL_TOP_K);

      console.log("📚 Sources (after rerank):");
      topChunks.forEach((r, i) => {
        const score = r.rerank_score ?? r.similarity;
        const src   = r.metadata?.source || "unknown";
        console.log(`   [${i + 1}] ${(score * 100).toFixed(0)}% — ${src}`);
      });

      await ask(question, topChunks);

    } catch (err) {
      console.error(`\n✗ Error: ${err.message}\n`);
    }
  }
}

main().catch(console.error);
