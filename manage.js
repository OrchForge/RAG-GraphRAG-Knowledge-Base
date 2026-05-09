import "dotenv/config";

const SB_URL = process.env.SUPABASE_URL;
const SB_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const HEADERS = {
  "apikey":        SB_KEY,
  "Authorization": `Bearer ${SB_KEY}`,
  "Content-Type":  "application/json",
};

const cmd = process.argv[2];
const arg = process.argv[3];

async function list() {
  const res  = await fetch(`${SB_URL}/rest/v1/documents?select=metadata,privacy`, { headers: HEADERS });
  const rows = await res.json();
  const sources = {};
  for (const row of rows) {
    const src = row.metadata?.source || "unknown";
    sources[src] = (sources[src] || 0) + 1;
  }
  console.log(`\nFiles in database: ${Object.keys(sources).length} | Chunks: ${rows.length}\n`);
  for (const [src, count] of Object.entries(sources)) {
    console.log(`  ${String(count).padStart(4)} chunks  ${src}`);
  }
  console.log();
}

async function deleteFile(source) {
  const res = await fetch(
    `${SB_URL}/rest/v1/documents?metadata->>source=eq.${encodeURIComponent(source)}`,
    { method: "DELETE", headers: { ...HEADERS, "Prefer": "return=representation" } }
  );
  const deleted = await res.json();
  console.log(`✓ Deleted ${deleted.length} chunks from: ${source}`);
}

async function deleteTag(tag) {
  const res = await fetch(
    `${SB_URL}/rest/v1/documents?metadata->tags=cs.["${tag}"]`,
    { method: "DELETE", headers: { ...HEADERS, "Prefer": "return=representation" } }
  );
  const deleted = await res.json();
  console.log(`✓ Deleted ${deleted.length} chunks with tag: ${tag}`);
}

async function show(source) {
  const res  = await fetch(
    `${SB_URL}/rest/v1/documents?metadata->>source=eq.${encodeURIComponent(source)}&select=content,metadata&order=metadata->chunk_index`,
    { headers: HEADERS }
  );
  const rows = await res.json();
  console.log(`\n📄 ${source} — ${rows.length} chunks:\n`);
  rows.forEach((r, i) => {
    console.log(`── Chunk ${i + 1} ──────────────────`);
    console.log(r.content.slice(0, 300));
    console.log();
  });
}

switch (cmd) {
  case "list":
    await list();
    break;

  case "delete":
    if (!arg) { console.log("Specify a file: node manage.js delete docs/file.md"); break; }
    await deleteFile(arg);
    break;

  case "delete-tag":
    if (!arg) { console.log("Specify a tag: node manage.js delete-tag nginx"); break; }
    await deleteTag(arg);
    break;

  case "show":
    if (!arg) { console.log("Specify a file: node manage.js show docs/file.md"); break; }
    await show(arg);
    break;

  default:
    console.log(`
Usage:
  node manage.js list                       — list all files in the database
  node manage.js delete docs/file.md        — delete a file
  node manage.js delete-tag nginx           — delete by tag
  node manage.js show docs/file.md          — show file chunks
    `);
}
