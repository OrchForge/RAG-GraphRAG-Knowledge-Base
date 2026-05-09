import * as readline from "readline/promises";
import os            from "os";
import "dotenv/config";

const OLLAMA      = process.env.OLLAMA_BASE_URL    || "http://localhost:11434";
const EMBED_MODEL = process.env.OLLAMA_EMBED_MODEL || "bge-m3";
const CHAT_MODEL  = process.env.OLLAMA_CHAT_MODEL  || "qwen3:1.7b";
const SB_URL      = process.env.SUPABASE_URL;
const SB_KEY      = process.env.SUPABASE_SERVICE_ROLE_KEY;
const CORES       = Math.max(1, Math.floor(os.cpus().length / 2));
const IS_QWEN3    = CHAT_MODEL.toLowerCase().includes("qwen3");

const LLM_OPTS = {
  num_ctx:        4096,
  num_thread:     CORES,
  temperature:    0.1,
  top_p:          0.5,
  repeat_penalty: 1.15,
  top_k:          20,
};

async function ollamaGen(prompt, opts = {}, timeoutMs = 60000) {
  const ctrl       = new AbortController();
  const timer      = setTimeout(() => ctrl.abort(), timeoutMs);
  const fullPrompt = IS_QWEN3 ? `/no_think\n${prompt}` : prompt;
  try {
    const res = await fetch(`${OLLAMA}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model:   CHAT_MODEL,
        prompt:  fullPrompt,
        stream:  false,
        options: { ...LLM_OPTS, temperature: 0, num_predict: 200, ...opts },
      }),
      signal: ctrl.signal,
    });
    clearTimeout(timer);
    const data = await res.json();
    let text = data.response?.trim() || "";
    text = text.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
    return text;
  } catch (e) {
    clearTimeout(timer);
    if (e.name === "AbortError")
      throw new Error(`Ollama timeout (${timeoutMs / 1000}s)`);
    throw e;
  }
}

async function embed(text) {
  const res = await fetch(`${OLLAMA}/api/embeddings`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ model: EMBED_MODEL, prompt: text }),
  });
  if (!res.ok) throw new Error(`Embed HTTP ${res.status}`);
  const data = await res.json();
  if (!Array.isArray(data.embedding))
    throw new Error(`No embedding in response: ${JSON.stringify(data).slice(0, 80)}`);
  return data.embedding;
}

async function sbRpc(fn, params) {
  const res = await fetch(`${SB_URL}/rest/v1/rpc/${fn}`, {
    method:  "POST",
    headers: {
      "Content-Type":  "application/json",
      "apikey":        SB_KEY,
      "Authorization": `Bearer ${SB_KEY}`,
    },
    body: JSON.stringify(params),
  });

  const body = await res.json();

  if (!res.ok) {
    throw new Error(
      `Supabase RPC '${fn}' → ${res.status}: ${body?.message || JSON.stringify(body)}`
    );
  }

  return body;
}

async function sbGet(path) {
  const res = await fetch(`${SB_URL}/rest/v1/${path}`, {
    headers: {
      "apikey":        SB_KEY,
      "Authorization": `Bearer ${SB_KEY}`,
    },
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`Supabase GET '${path}' → ${res.status}: ${txt}`);
  }
  return res.json();
}

// Classifies the question type to guide the synthesizer's response style
async function agentRouter(question) {
  const prompt =
    `Classify the type of technical question. Reply ONLY with JSON, no explanation.\n\n` +
    `Types:\n` +
    `- "factual"     — a specific fact ("What is RJ45?")\n` +
    `- "relational"  — connections between concepts ("How is TCP related to IP?")\n` +
    `- "procedural"  — processes ("How does DNS work?")\n` +
    `- "comparative" — comparison ("How does TCP differ from UDP?")\n` +
    `- "global"      — broad overview ("What topics are covered?")\n\n` +
    `QUESTION: ${question}\n` +
    `JSON (single object):`;

  const raw = await ollamaGen(prompt, { num_predict: 80 }).catch(() => "");
  try {
    const m = raw.match(/\{[\s\S]*?\}/);
    if (m) {
      const parsed = JSON.parse(m[0]);
      if (parsed.type) return parsed;
    }
  } catch { /* fallback below */ }

  const q = question.toLowerCase();
  if (q.includes("related") || q.includes("depends") || q.includes("consists"))
    return { type: "relational" };
  if (q.includes("how does") || q.includes("describe") || q.includes("process"))
    return { type: "procedural" };
  if (q.includes("differ") || q.includes("difference") || q.includes("compare"))
    return { type: "comparative" };
  if (q.includes("what topics") || q.includes("overview") || q.includes("all"))
    return { type: "global" };
  return { type: "factual" };
}

async function agentResearcher(question, topK = 15) {
  const emb  = await embed(question);
  const docs = await sbRpc("match_documents", {
    query_embedding: emb,
    match_count:     topK,
    match_threshold: 0.25,
    filter_privacy:  ["internal", "public"],
  });
  return Array.isArray(docs) ? docs : [];
}

async function agentGraphExplorer(question) {
  const emb = await embed(question);

  let matchedEntities;
  try {
    matchedEntities = await sbRpc("match_entities", {
      query_embedding: emb,
      match_count:     6,
      match_threshold: 0.30,
    });
  } catch (err) {
    if (err.message.includes("404") || err.message.includes("does not exist") ||
        err.message.includes("function") || err.message.includes("PGRST")) {
      console.error(`\n  ╔════════════════════════════════════════════════════╗`);
      console.error(`  ║ ERROR: SQL function match_entities not found!       ║`);
      console.error(`  ║ → Run supabase_setup.sql in the Supabase SQL Editor ║`);
      console.error(`  ╚════════════════════════════════════════════════════╝`);
    } else {
      console.error(`  [Graph] match_entities: ${err.message}`);
    }
    return { entities: [], graphLines: [], relatedDocIds: [] };
  }

  if (!Array.isArray(matchedEntities) || matchedEntities.length === 0) {
    return { entities: [], graphLines: [], relatedDocIds: [] };
  }

  // BFS traversal from top-3 matched entities
  const graphLines = [];
  const seen       = new Set();

  for (const ent of matchedEntities.slice(0, 3)) {
    let neighbors;
    try {
      neighbors = await sbRpc("graph_neighbors", {
        start_entity_id: ent.id,
        max_depth:       2,
        max_results:     20,
      });
    } catch (err) {
      if (err.message.includes("404") || err.message.includes("does not exist") ||
          err.message.includes("PGRST")) {
        console.error(`  [Graph] graph_neighbors not found → run supabase_setup.sql`);
      } else {
        console.error(`  [Graph] graph_neighbors(${ent.name}): ${err.message}`);
      }
      continue;
    }

    for (const n of neighbors || []) {
      if (!n.relation || n.depth === 0) continue;
      const key = `${ent.id}→${n.entity_id}`;
      if (!seen.has(key)) {
        seen.add(key);
        graphLines.push(
          `${ent.name} --[${n.relation}]--> ${n.entity_name} (${n.entity_type})`
        );
      }
    }
  }

  // IDs of documents linked to the matched entities (used to enrich the candidate pool)
  let relatedDocIds = [];
  try {
    const entityIds = matchedEntities.map(e => e.id).join(",");
    const rows = await sbGet(
      `document_entities?entity_id=in.(${entityIds})&select=document_id`
    );
    relatedDocIds = [...new Set(rows.map(r => r.document_id))];
  } catch (err) {
    console.error(`  [Graph] document_entities: ${err.message}`);
  }

  return { entities: matchedEntities, graphLines, relatedDocIds };
}

async function scoreChunk(question, content) {
  const prompt =
    `Rate how useful this fragment is for answering the question.\n` +
    `Return ONLY a single digit from 0 to 10.\n\n` +
    `QUESTION: ${question}\n\n` +
    `FRAGMENT: ${content.slice(0, 400)}\n\n` +
    `SCORE:`;

  try {
    const raw   = await ollamaGen(prompt, { num_predict: 4, num_ctx: 768 }, 12000);
    const match = raw.match(/\b(10|[0-9])\b/);
    return match ? parseInt(match[1], 10) / 10 : 0.5;
  } catch {
    return 0.5;
  }
}

async function agentReranker(question, candidates, topN = 5) {
  if (candidates.length === 0) return [];
  if (candidates.length <= topN) return candidates;

  process.stdout.write(`🎯 [Reranker] Scoring ${candidates.length} fragments`);

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

async function agentSynthesizer(question, chunks, graphResult, queryType) {
  const contextBlock = chunks.length > 0
    ? chunks.map((c, i) => {
        const score = c.rerank_score ?? c.similarity;
        const src   = c.metadata?.source ? ` | ${c.metadata.source}` : "";
        return `[Document ${i + 1} | ${(score * 100).toFixed(0)}%${src}]\n${c.content}`;
      }).join("\n\n---\n\n")
    : "No documents found.";

  const graphBlock = graphResult?.graphLines?.length > 0
    ? `\n\nKNOWLEDGE GRAPH RELATIONS:\n${graphResult.graphLines.join("\n")}`
    : "";

  const styleMap = {
    factual:     "Give a concise, precise answer. Facts from the context only.",
    relational:  "Describe the connections between concepts using the graph and text.",
    procedural:  "Describe the process step-by-step with a numbered list.",
    comparative: "Compare the concepts by key parameters. A table is acceptable.",
    global:      "Give a broad overview covering all aspects of the topic.",
  };
  const style = styleMap[queryType] || styleMap.factual;

  const prompt =
    `You are a strict technical assistant. Answer ONLY based on the CONTEXT and GRAPH below.\n\n` +
    `RULES:\n` +
    `1. Use ONLY information from the CONTEXT and GRAPH. No external knowledge.\n` +
    `2. If the answer is not found — say: "No information on this topic in the documentation."\n` +
    `3. Do not invent. Do not translate technical terms.\n` +
    `4. ${style}\n\n` +
    `CONTEXT:\n${contextBlock}` +
    `${graphBlock}\n\n` +
    `QUESTION: ${question}`;

  const res = await fetch(`${OLLAMA}/api/chat`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({
      model:    CHAT_MODEL,
      messages: [{ role: "user", content: IS_QWEN3 ? `/no_think\n${prompt}` : prompt }],
      stream:   true,
      options:  LLM_OPTS,
    }),
  });

  if (!res.ok) throw new Error(`Ollama chat HTTP ${res.status}`);

  const reader  = res.body.getReader();
  const decoder = new TextDecoder();

  process.stdout.write("\n🤖 ");

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    for (const line of decoder.decode(value, { stream: true }).split("\n").filter(Boolean)) {
      try {
        const j = JSON.parse(line);
        if (j.message?.content) {
          const text = j.message.content.replace(/<think>[\s\S]*?<\/think>/g, "");
          if (text) process.stdout.write(text);
        }
      } catch { /* incomplete JSON frame */ }
    }
  }

  console.log("\n");
}

async function orchestrate(question) {
  console.log("\n" + "─".repeat(60));

  process.stdout.write("🔀 [Router] Question type... ");
  const route = await agentRouter(question);
  console.log(`${route.type}`);

  process.stdout.write("🔍 [Researcher] Vector search...\n");
  process.stdout.write("🕸  [GraphExplorer] Graph search...\n");

  const [rawDocs, graphResult] = await Promise.all([
    agentResearcher(question, 15).catch(err => {
      console.error(`  [Researcher] error: ${err.message}`);
      return [];
    }),
    agentGraphExplorer(question).catch(err => {
      console.error(`  [GraphExplorer] error: ${err.message}`);
      return { entities: [], graphLines: [], relatedDocIds: [] };
    }),
  ]);

  console.log(
    `   Documents from vector search: ${rawDocs.length}\n` +
    `   Entities in graph:            ${graphResult.entities?.length ?? 0}\n` +
    `   Graph relations:              ${graphResult.graphLines?.length ?? 0}\n` +
    `   Bonus docs from graph:        ${graphResult.relatedDocIds?.length ?? 0}`
  );

  // Enrich the candidate pool with documents linked via the knowledge graph
  let enrichedDocs = rawDocs;

  if (graphResult.relatedDocIds?.length > 0) {
    const existingIds = new Set(rawDocs.map(d => d.id));
    const missingIds  = graphResult.relatedDocIds.filter(id => !existingIds.has(id));

    if (missingIds.length > 0) {
      try {
        const extra = await sbGet(
          `documents?id=in.(${missingIds.slice(0, 5).join(",")})&select=id,content,metadata,privacy`
        );
        const extraWithScore = extra.map(d => ({ ...d, similarity: 0.1 }));
        enrichedDocs = [...rawDocs, ...extraWithScore];
        console.log(`   + ${extraWithScore.length} documents added from graph`);
      } catch (err) {
        console.error(`  [Enrich] ${err.message}`);
      }
    }
  }

  const topChunks = await agentReranker(question, enrichedDocs, 5);

  console.log("\n📚 Sources:");
  topChunks.forEach((c, i) => {
    const score = c.rerank_score ?? c.similarity;
    const src   = c.metadata?.source || "unknown";
    console.log(`   [${i + 1}] ${(score * 100).toFixed(0)}% — ${src}`);
  });

  if (graphResult.graphLines?.length > 0) {
    console.log("\n🕸  Knowledge Graph:");
    graphResult.graphLines.slice(0, 6).forEach(l => console.log(`   ${l}`));
    if (graphResult.graphLines.length > 6)
      console.log(`   ... ${graphResult.graphLines.length - 6} more relations`);
  } else {
    console.log("\n🕸  Graph: no matches (graph is empty or threshold not reached)");
  }

  process.stdout.write("\n✍️  [Synthesizer] Generating answer...");
  await agentSynthesizer(question, topChunks, graphResult, route.type);
}

if (!SB_URL || !SB_KEY) {
  console.error("✗ Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env");
  process.exit(1);
}

const modelPad = CHAT_MODEL.padEnd(20).slice(0, 20);
console.log(`╔═══════════════════════════════════════════════════╗`);
console.log(`║  Multi-Agent RAG + GraphRAG                       ║`);
console.log(`║  Model: ${modelPad}${IS_QWEN3 ? " [/no_think]" : "            "}  ║`);
console.log(`║  Pipeline: Router → [Researcher ‖ Graph]          ║`);
console.log(`║            → Enrich → Rerank → Synthesize         ║`);
console.log(`║  Type 'exit' to quit, '/graph' for graph status   ║`);
console.log(`╚═══════════════════════════════════════════════════╝`);

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
process.on("SIGINT", () => { console.log("\nExiting."); rl.close(); process.exit(0); });

while (true) {
  let q;
  try { q = await rl.question("\n❓ Question: "); }
  catch { break; }

  if (!q?.trim()) continue;

  if (q.toLowerCase() === "exit") {
    rl.close();
    break;
  }

  if (q.toLowerCase() === "/graph") {
    try {
      const [
        { count: entCount },
        { count: relCount },
        { count: deCount },
      ] = await Promise.all([
        fetch(`${SB_URL}/rest/v1/entities?select=id`, {
          headers: { "apikey": SB_KEY, "Authorization": `Bearer ${SB_KEY}`,
                     "Prefer": "count=exact", "Range": "0-0" },
        }).then(r => ({ count: r.headers.get("content-range")?.split("/")[1] ?? "?" })),
        fetch(`${SB_URL}/rest/v1/relationships?select=id`, {
          headers: { "apikey": SB_KEY, "Authorization": `Bearer ${SB_KEY}`,
                     "Prefer": "count=exact", "Range": "0-0" },
        }).then(r => ({ count: r.headers.get("content-range")?.split("/")[1] ?? "?" })),
        fetch(`${SB_URL}/rest/v1/document_entities?select=document_id`, {
          headers: { "apikey": SB_KEY, "Authorization": `Bearer ${SB_KEY}`,
                     "Prefer": "count=exact", "Range": "0-0" },
        }).then(r => ({ count: r.headers.get("content-range")?.split("/")[1] ?? "?" })),
      ]);
      console.log(`\n📊 Graph status:`);
      console.log(`   Entities:           ${entCount}`);
      console.log(`   Relations:          ${relCount}`);
      console.log(`   Doc↔entity links:   ${deCount}`);
      if (entCount === "0" || entCount === "?")
        console.log(`\n   ⚠  Graph is empty → run: node graph_builder.js`);
    } catch (err) {
      console.error(`  Error: ${err.message}`);
    }
    continue;
  }

  try {
    await orchestrate(q);
  } catch (err) {
    console.error(`\n✗ Error: ${err.message}\n`);
  }
}
