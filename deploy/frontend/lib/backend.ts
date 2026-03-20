export function getBackendUrl(path: string): string {
  const baseUrl = process.env.MODAL_API_BASE_URL;
  if (!baseUrl) {
    throw new Error("MODAL_API_BASE_URL is not configured");
  }

  return new URL(path, baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`).toString();
}

export async function fetchBackend(path: string, init?: RequestInit): Promise<Response> {
  return fetch(getBackendUrl(path), {
    ...init,
    cache: "no-store",
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
}

export async function readBackendJson(response: Response): Promise<unknown> {
  const text = await response.text();
  return text ? JSON.parse(text) : {};
}
