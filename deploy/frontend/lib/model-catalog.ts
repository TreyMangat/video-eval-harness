import type { ModelCatalogItem } from "./types";

export const DEFAULT_MODEL_CATALOG: ModelCatalogItem[] = [
  {
    name: "gemini-3.1-pro",
    model_id: "google/gemini-3.1-pro-preview",
    provider: "openrouter",
    supports_images: true,
    notes: "Highest-accuracy option in the default set; best when label quality matters more than speed.",
  },
  {
    name: "gpt-5.4",
    model_id: "openai/gpt-5.4",
    provider: "openrouter",
    supports_images: true,
    notes: "Strong structured-output and reasoning model; useful as a second comparison anchor.",
  },
  {
    name: "qwen3.5-vl",
    model_id: "qwen/qwen3.5-397b-a17b",
    provider: "openrouter",
    supports_images: true,
    notes: "Lower-cost multimodal candidate that is useful for cost-versus-quality comparisons.",
  },
  {
    name: "claude-sonnet-4.6",
    model_id: "anthropic/claude-sonnet-4.6",
    provider: "openrouter",
    supports_images: true,
    notes: "Another strong frontier baseline for head-to-head agreement and latency comparisons.",
  },
];

export const DEFAULT_MODEL_SELECTION = DEFAULT_MODEL_CATALOG.map((model) => model.name);
