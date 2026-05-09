import { createClient } from "@supabase/supabase-js";
import "dotenv/config";

const sb = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY,
  { auth: { persistSession: false } }
);

const OLLAMA      = process.env.OLLAMA_BASE_URL    || "http://localhost:11434";
const EMBED_MODEL = process.env.OLLAMA_EMBED_MODEL || "bge-m3";
const CHAT_MODEL  = process.env.OLLAMA_CHAT_MODEL  || "qwen3:1.7b";
const DEBUG       = process.env.DEBUG_EXTRACT === "1";
const IS_QWEN3    = CHAT_MODEL.toLowerCase().includes("qwen3");

// Supported entity types for the access-control system domain:
//   device, card, software, procedure, role, concept, network, log_system, setting
//
// Supported relation types:
//   opens, encodes, reads, configures, manages, requires, uses,
//   assigns_to, part_of, connects_to, logs, runs_on, triggers, stores

function normalizeName(name) {
  if (typeof name !== "string") return "";
  const trimmed = name.trim().replace(/\s+/g, " ");
  if (trimmed.length === 0) return "";
  return trimmed.charAt(0).toUpperCase() + trimmed.slice(1);
}

async function generate(prompt, maxTokens = 700, timeoutMs = 90000) {
  const ctrl  = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  const fullPrompt = IS_QWEN3 ? `/no_think\n${prompt}` : prompt;

  try {
    const res = await fetch(`${OLLAMA}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: CHAT_MODEL,
        prompt: fullPrompt,
        stream: false,
        options: { temperature: 0, num_predict: maxTokens, num_ctx: 2048, top_k: 10, top_p: 0.5 },
      }),
      signal: ctrl.signal,
    });
    clearTimeout(timer);
    const data = await res.json();
    let text = data.response?.trim() || "";
    text = text.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
    return text;
  } catch (err) {
    clearTimeout(timer);
    if (err.name === "AbortError")
      throw new Error(`Ollama timeout ${timeoutMs / 1000}s — model not responding`);
    throw err;
  }
}

async function embed(text, attempt = 1) {
  const ctrl  = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 20000);
  try {
    const res = await fetch(`${OLLAMA}/api/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: EMBED_MODEL, prompt: text }),
      signal: ctrl.signal,
    });
    clearTimeout(timer);
    const data = await res.json();
    if (!Array.isArray(data.embedding))
      throw new Error(`No embedding in response: ${JSON.stringify(data).slice(0, 80)}`);
    return data.embedding;
  } catch (err) {
    clearTimeout(timer);
    if (attempt < 3 && err.name !== "AbortError") {
      await new Promise(r => setTimeout(r, 1000 * attempt));
      return embed(text, attempt + 1);
    }
    throw err;
  }
}

// Extracts the first valid JSON object from raw model output.
// Handles markdown code fences and attempts to repair truncated JSON.
function parseModelJSON(raw) {
  let cleaned = raw.replace(/```json[\s\S]*?```/g, match =>
    match.replace(/```json\s*/g, "").replace(/\s*```/g, "")
  ).replace(/```[\s\S]*?```/g, match =>
    match.replace(/```\s*/g, "")
  );

  const match = cleaned.match(/\{[\s\S]*\}/);
  if (!match) return null;

  try {
    return JSON.parse(match[0]);
  } catch {
    const partial = match[0];
    for (const suffix of ["]}}", "]}", "}"]) {
      try {
        return JSON.parse(partial + suffix);
      } catch { /* continue */ }
    }
    return null;
  }
}

// Few-shot example uses a real domain (lock management + access cards)
// to guide small models (1.5B–1.7B) toward the correct output structure.
function buildMainPrompt(excerpt) {
  return `Extract entities and relations from the text about an access control system. Return ONLY JSON without explanation.

Example format:
{"entities":[{"name":"Wristband","type":"card","description":"carrier for encoding lock access"},{"name":"Reader","type":"device","description":"device for reading and writing cards"},{"name":"Lock","type":"device","description":"electronic locker lock"},{"name":"Card encoding","type":"procedure","description":"process of writing access data to a card"}],"relationships":[{"source":"Reader","target":"Wristband","type":"encodes"},{"source":"Wristband","target":"Lock","type":"opens"},{"source":"Card encoding","target":"Reader","type":"requires"}]}

Allowed entity types: device, card, software, procedure, role, concept, network, log_system, setting
Allowed relation types: opens, encodes, reads, configures, manages, requires, uses, assigns_to, part_of, connects_to, logs, runs_on, triggers, stores

TEXT:
${excerpt}

JSON:`;
}

function buildFallbackPrompt(excerpt) {
  return `List all important objects from the text. Return ONLY JSON.

{"entities":[{"name":"Name","type":"type","description":"description"}],"relationships":[{"source":"A","target":"B","type":"uses"}]}

Types: device, card, software, procedure, role, concept, network, log_system, setting
Relations: opens, encodes, configures, manages, requires, uses, connects_to, logs

TEXT:
${excerpt}

JSON:`;
}

async function extractEntities(text) {
  const excerpt = text.slice(0, 1500);

  let raw = await generate(buildMainPrompt(excerpt), 800);

  if (DEBUG) {
    console.log("\n┌─── RAW MODEL OUTPUT (attempt 1) ───────────────────────");
    console.log(raw.slice(0, 600));
    console.log("└────────────────────────────────────────────────────────\n");
  }

  let parsed = parseModelJSON(raw);

  if (!parsed || !Array.isArray(parsed.entities) || parsed.entities.length === 0) {
    if (DEBUG) console.log("  [retry] First attempt empty, running fallback prompt...");

    await new Promise(r => setTimeout(r, 300));
    raw = await generate(buildFallbackPrompt(excerpt), 600);

    if (DEBUG) {
      console.log("\n┌─── RAW MODEL OUTPUT (attempt 2 / fallback) ────────────");
      console.log(raw.slice(0, 600));
      console.log("└────────────────────────────────────────────────────────\n");
    }

    parsed = parseModelJSON(raw);
  }

  if (!parsed) {
    if (DEBUG) console.log("  [warn] Both attempts produced no valid JSON.");
    return { entities: [], relationships: [] };
  }

  const VALID_ENTITY_TYPES = new Set([
    "device", "card", "software", "procedure",
    "role", "concept", "network", "log_system", "setting",
    "protocol", "algorithm", "standard", "term",
  ]);

  const entities = (parsed.entities || [])
    .filter(e =>
      e &&
      typeof e.name === "string" && e.name.trim().length > 0 &&
      typeof e.type === "string" && e.type.trim().length > 0
    )
    .map(e => ({
      name:        normalizeName(e.name),
      type:        e.type.trim().toLowerCase(),
      description: typeof e.description === "string" ? e.description.trim() : "",
    }))
    .filter((e, idx, arr) =>
      arr.findIndex(x => x.name.toLowerCase() === e.name.toLowerCase()) === idx
    );

  const entityNames = new Set(entities.map(e => e.name.toLowerCase()));

  const relationships = (parsed.relationships || [])
    .filter(r =>
      r &&
      typeof r.source === "string" && r.source.trim().length > 0 &&
      typeof r.target === "string" && r.target.trim().length > 0 &&
      typeof r.type   === "string" && r.type.trim().length > 0
    )
    .map(r => ({
      source: normalizeName(r.source),
      target: normalizeName(r.target),
      type:   r.type.trim().toLowerCase(),
    }))
    .filter(r =>
      entityNames.has(r.source.toLowerCase()) &&
      entityNames.has(r.target.toLowerCase()) &&
      r.source.toLowerCase() !== r.target.toLowerCase()
    );

  return { entities, relationships };
}

async function upsertEntity(entity) {
  const { data: existing } = await sb
    .from("entities")
    .select("id")
    .eq("name", entity.name)
    .eq("type", entity.type)
    .maybeSingle();

  if (existing) return existing.id;

  const embText   = `${entity.name}: ${entity.description || entity.type}`;
  const embedding = await embed(embText);

  const { data, error } = await sb
    .from("entities")
    .insert({
      name:        entity.name,
      type:        entity.type,
      description: entity.description,
      embedding,
    })
    .select("id")
    .single();

  if (error) {
    if (error.code === "23505") {
      const { data: retry } = await sb
        .from("entities")
        .select("id")
        .eq("name", entity.name)
        .eq("type", entity.type)
        .single();
      return retry?.id ?? null;
    }
    throw new Error(`upsertEntity(${entity.name}): ${error.message}`);
  }

  return data.id;
}

async function upsertRelationship(sourceId, targetId, relType) {
  if (!sourceId || !targetId || sourceId === targetId) return;

  const { error } = await sb
    .from("relationships")
    .upsert(
      { source_id: sourceId, target_id: targetId, relation_type: relType },
      { onConflict: "source_id,target_id,relation_type", ignoreDuplicates: true }
    );

  if (error && error.code !== "23505") {
    console.error(`  [!] relationship(${relType}): ${error.message}`);
  }
}

async function main() {
  console.log(`\n${"═".repeat(60)}`);
  console.log(`  Graph Builder`);
  console.log(`  Model:  ${CHAT_MODEL}${IS_QWEN3 ? " (Qwen3 /no_think)" : ""}`);
  console.log(`  Embed:  ${EMBED_MODEL}`);
  console.log(`  Debug:  ${DEBUG ? "ON (DEBUG_EXTRACT=1)" : "off"}`);
  console.log(`${"═".repeat(60)}`);

  const { error: tableCheck } = await sb.from("entities").select("id").limit(1);
  if (tableCheck) {
    console.error(`\n✗ Table 'entities' not found: ${tableCheck.message}`);
    console.error(`  → Run supabase_setup.sql in the Supabase SQL Editor!\n`);
    process.exit(1);
  }

  const { data: docs, error } = await sb
    .from("documents")
    .select("id, content, metadata")
    .order("created_at", { ascending: true });

  if (error) {
    console.error(`\n✗ Failed to load documents: ${error.message}`);
    process.exit(1);
  }

  if (!docs || docs.length === 0) {
    console.log("\n⚠  No data in the documents table. Run migrate.js first.\n");
    process.exit(0);
  }

  console.log(`\nDocuments to process: ${docs.length}`);
  console.log("─".repeat(60));

  let totalEntities = 0;
  let totalRel      = 0;
  let totalErrors   = 0;
  let totalSkipped  = 0;

  for (let i = 0; i < docs.length; i++) {
    const doc    = docs[i];
    const label  = doc.metadata?.source || doc.metadata?.filename || doc.id.slice(0, 8);
    const padLen = docs.length.toString().length;
    const prefix = `[${String(i + 1).padStart(padLen)}/${docs.length}]`;

    process.stdout.write(`${prefix} ${label} ... `);

    try {
      const { entities, relationships } = await extractEntities(doc.content);

      if (entities.length === 0) {
        console.log("⚠  no entities extracted");
        totalSkipped++;
        continue;
      }

      const entityMap = new Map();

      for (const ent of entities) {
        const id = await upsertEntity(ent);
        if (id) {
          entityMap.set(ent.name.toLowerCase(), id);
          totalEntities++;

          await sb
            .from("document_entities")
            .upsert(
              { document_id: doc.id, entity_id: id },
              { onConflict: "document_id,entity_id", ignoreDuplicates: true }
            );
        }
      }

      let savedRel = 0;
      for (const rel of relationships) {
        const srcId = entityMap.get(rel.source.toLowerCase());
        const tgtId = entityMap.get(rel.target.toLowerCase());
        if (srcId && tgtId) {
          await upsertRelationship(srcId, tgtId, rel.type);
          totalRel++;
          savedRel++;
        }
      }

      console.log(`✓  ${entities.length} entities, ${savedRel} relations`);

    } catch (err) {
      console.log(`✗  ${err.message}`);
      totalErrors++;
    }

    if (i < docs.length - 1) {
      await new Promise(r => setTimeout(r, 200));
    }
  }

  console.log("─".repeat(60));
  console.log(`Session:`);
  console.log(`  Processed:  ${docs.length - totalSkipped - totalErrors} of ${docs.length}`);
  console.log(`  Skipped:    ${totalSkipped} (no entities extracted)`);
  console.log(`  Errors:     ${totalErrors}`);
  console.log(`  Entities:   ${totalEntities} (new in this run)`);
  console.log(`  Relations:  ${totalRel} (new in this run)`);
  console.log();

  const [{ count: entCount }, { count: relCount }] = await Promise.all([
    sb.from("entities").select("*", { count: "exact", head: true }),
    sb.from("relationships").select("*", { count: "exact", head: true }),
  ]);

  console.log(`Database totals:`);
  console.log(`  Entities:  ${entCount}`);
  console.log(`  Relations: ${relCount}`);

  if (totalSkipped > 0) {
    console.log(`\n💡 Tip: ${totalSkipped} documents produced no entities.`);
    console.log(`   Run with DEBUG_EXTRACT=1 to inspect raw model output.`);
  }
  console.log();
}

main().catch(err => {
  console.error("\nFatal error:", err.message);
  process.exit(1);
});
