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
    name: "llama-4-maverick",
    model_id: "meta-llama/llama-4-maverick",
    provider: "openrouter",
    supports_images: true,
    notes: "Meta's multimodal MoE frontier model and the default fourth benchmark slot.",
  },
  {
    name: "gemini-3-flash",
    model_id: "google/gemini-3-flash-preview",
    provider: "openrouter",
    supports_images: true,
    notes: "Fast Gemini tier used for smoke tests and low-cost sweeps.",
  },
  {
    name: "gpt-5.4-mini",
    model_id: "openai/gpt-5.4-mini",
    provider: "openrouter",
    supports_images: true,
    notes: "Lower-cost GPT-5.4 variant used in the fast benchmark config.",
  },
  {
    name: "qwen3.5-27b",
    model_id: "qwen/qwen3.5-27b",
    provider: "openrouter",
    supports_images: true,
    notes: "Smaller Qwen variant for quick sweeps and throughput validation.",
  },
];

export const DEFAULT_MODEL_SELECTION = DEFAULT_MODEL_CATALOG.map((model) => model.name);
