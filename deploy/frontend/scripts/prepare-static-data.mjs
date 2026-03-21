import { cpSync, existsSync, mkdirSync, rmSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const frontendRoot = path.resolve(scriptDir, "..");
const sourceDataDir = path.resolve(frontendRoot, "..", "..", "data");
const publicDir = path.join(frontendRoot, "public");
const targetDataDir = path.join(publicDir, "data");

mkdirSync(publicDir, { recursive: true });
rmSync(targetDataDir, { recursive: true, force: true });

if (!existsSync(sourceDataDir)) {
  console.log("No repo data/ directory found; skipping static data copy.");
  process.exit(0);
}

cpSync(sourceDataDir, targetDataDir, { recursive: true });
console.log(`Copied static run data from ${sourceDataDir} to ${targetDataDir}.`);
