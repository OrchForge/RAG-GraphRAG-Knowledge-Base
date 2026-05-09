import { createRequire }  from "module";
import { glob }           from "glob";
import fs                 from "fs/promises";
import path               from "path";
import { execFile }       from "child_process";
import { promisify }      from "util";
import "dotenv/config";

const require       = createRequire(import.meta.url);
const matter        = require("gray-matter");
const execFileAsync = promisify(execFile);

const SB_URL     = process.env.SUPABASE_URL;
const SB_KEY     = process.env.SUPABASE_SERVICE_ROLE_KEY;
const SB_HEADERS = {
  "Content-Type":  "application/json",
  "apikey":        SB_KEY,
  "Authorization": `Bearer ${SB_KEY}`,
  "Prefer":        "return=minimal",
};

async function sbInsert(row) {
  const res = await fetch(`${SB_URL}/rest/v1/documents`, {
    method:  "POST",
    headers: SB_HEADERS,
    body:    JSON.stringify(row),
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`Supabase ${res.status}: ${txt}`);
  }
}

async function sbCheck() {
  const res = await fetch(`${SB_URL}/rest/v1/documents?select=id&limit=1`, {
    headers: { ...SB_HEADERS, "Prefer": "return=representation" },
  });
  return res.ok;
}

async function embed(text, attempt = 1) {
  const ctrl  = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 20_000);
  try {
    const res = await fetch(
      `${process.env.OLLAMA_BASE_URL || "http://localhost:11434"}/api/embeddings`,
      {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          model:  process.env.OLLAMA_EMBED_MODEL || "nomic-embed-text",
          prompt: text,
        }),
        signal: ctrl.signal,
      }
    );
    clearTimeout(timer);
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    const data = await res.json();
    if (!Array.isArray(data.embedding))
      throw new Error(`Unexpected response: ${JSON.stringify(data).slice(0, 100)}`);
    return data.embedding;
  } catch (err) {
    clearTimeout(timer);
    if (err.name === "AbortError" && attempt < 3) {
      console.log(`   ⟳ Ollama timeout, attempt ${attempt + 1}/3…`);
      await new Promise(r => setTimeout(r, 2000));
      return embed(text, attempt + 1);
    }
    throw new Error(
      err.name === "AbortError"
        ? "Ollama not responding after 3 attempts. Check: systemctl status ollama"
        : err.message
    );
  }
}

function chunk(text, size = 800, overlap = 80) {
  if (text.length <= size) {
    const piece = text.trim();
    return piece.length > 20 ? [piece] : [];
  }

  const chunks = [];
  let start    = 0;

  while (start < text.length) {
    const end = Math.min(start + size, text.length);
    let actualEnd = end;

    if (end < text.length) {
      const boundary = Math.max(
        text.lastIndexOf("\n", end),
        text.lastIndexOf(". ", end)
      );
      if (boundary > start + size * 0.5) actualEnd = boundary + 1;
    }

    const piece = text.slice(start, actualEnd).trim();
    if (piece.length > 20) chunks.push(piece);

    const nextStart = actualEnd - overlap;
    if (nextStart <= start) break;
    start = nextStart;
  }

  return chunks;
}

async function parsePdf(filePath) {
  try {
    const { stdout } = await execFileAsync(
      "pdftotext",
      ["-enc", "UTF-8", filePath, "-"],
      { maxBuffer: 500 * 1024 * 1024 }
    );
    return stdout;
  } catch (err) {
    throw new Error(
      `pdftotext error: ${err.message}\n` +
      `Install with: sudo apt-get install -y poppler-utils`
    );
  }
}

async function parseFile(filePath) {
  const ext = path.extname(filePath).toLowerCase();

  if (ext === ".pdf") {
    const content = await parsePdf(filePath);
    return {
      content,
      metadata: { source: filePath, type: "pdf" },
    };
  }

  const raw    = await fs.readFile(filePath, "utf8");
  const parsed = matter(raw);
  return {
    content:  parsed.content,
    metadata: {
      source: filePath,
      type:   ext.slice(1) || "text",
      ...parsed.data,
    },
  };
}

const mem = () => {
  const m = process.memoryUsage();
  return `RSS:${(m.rss / 1024 / 1024).toFixed(0)}MB Heap:${(m.heapUsed / 1024 / 1024).toFixed(0)}MB`;
};

async function migrate() {
  console.log(`[MEM start] ${mem()}`);

  if (!SB_URL || !SB_KEY) {
    console.error("✗ SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set in .env");
    process.exit(1);
  }

  const ok = await sbCheck();
  console.log(`[Supabase]  ${ok ? "✓ OK" : "✗ Connection error"} | ${mem()}`);
  if (!ok) process.exit(1);

  const pattern = process.argv[2] || "docs/**/*.{md,txt,pdf}";
  const files   = await glob(pattern, { nodir: true });

  if (files.length === 0) {
    console.log(`\nNo files found for pattern: ${pattern}`);
    console.log("Usage: node migrate.js 'docs/**/*.{md,txt,pdf}'");
    return;
  }

  console.log(`\nFiles found: ${files.length}\n`);

  let totalChunks   = 0;
  let totalInserted = 0;
  let totalErrors   = 0;

  for (const filePath of files) {
    console.log(`📄 ${filePath}`);

    let content, metadata;
    try {
      ({ content, metadata } = await parseFile(filePath));
    } catch (err) {
      console.error(`   ✗ Read error: ${err.message}`);
      totalErrors++;
      continue;
    }

    if (!content?.trim()) {
      console.log("   ⚠ Empty file, skipping");
      continue;
    }

    const privacy = metadata.privacy ?? process.env.DEFAULT_PRIVACY ?? "internal";
    const chunks  = chunk(content);

    console.log(`   Chunks: ${chunks.length} | Privacy: ${privacy} | ${mem()}`);
    totalChunks += chunks.length;

    for (let i = 0; i < chunks.length; i++) {
      try {
        const embedding = await embed(chunks[i]);
        await sbInsert({
          content:  chunks[i],
          metadata: {
            ...metadata,
            privacy,
            chunk_index:  i,
            total_chunks: chunks.length,
          },
          embedding,
          privacy,
        });
        totalInserted++;
        process.stdout.write(`\r   Inserted: ${i + 1}/${chunks.length}  `);
      } catch (err) {
        console.error(`\n   ✗ Chunk ${i}: ${err.message}`);
        totalErrors++;
      }
    }

    console.log(`\n   [MEM after file] ${mem()}`);
    if (global.gc) global.gc();
  }

  console.log("\n─────────────────────────────────────");
  console.log(`Files processed: ${files.length}`);
  console.log(`Total chunks:    ${totalChunks}`);
  console.log(`Inserted:        ${totalInserted}`);
  console.log(`Errors:          ${totalErrors}`);
}

migrate().catch(err => {
  console.error("\nFatal error:", err.message);
  process.exit(1);
});
