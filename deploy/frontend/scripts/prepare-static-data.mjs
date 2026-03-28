import { copyFileSync, existsSync, mkdirSync, readdirSync, rmSync, writeFileSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const frontendRoot = path.resolve(scriptDir, "..");
const sourceDataDir = path.resolve(frontendRoot, "..", "..", "data");
const publicDir = path.join(frontendRoot, "public");
const targetDataDir = path.join(publicDir, "data");

mkdirSync(publicDir, { recursive: true });
rmSync(targetDataDir, { recursive: true, force: true });
mkdirSync(targetDataDir, { recursive: true });
writeFileSync(path.join(targetDataDir, ".gitkeep"), "");

if (!existsSync(sourceDataDir)) {
  console.log("No repo data/ directory found; skipping static data copy.");
  process.exit(0);
}

const runFiles = readdirSync(sourceDataDir, { withFileTypes: true })
  .filter(
    (entry) => entry.isFile() && /(_results|_ensemble_results)\.json$/i.test(entry.name)
  )
  .map((entry) => entry.name)
  .sort();

for (const fileName of runFiles) {
  copyFileSync(path.join(sourceDataDir, fileName), path.join(targetDataDir, fileName));
}

console.log(`Copied ${runFiles.length} static run JSON files from ${sourceDataDir} to ${targetDataDir}.`);
