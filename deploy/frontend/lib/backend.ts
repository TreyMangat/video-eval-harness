function normalizeBackendPath(path: string): string {
  const trimmedPath = path.replace(/^\/+/, "");
  if (trimmedPath.startsWith("api/")) {
    return trimmedPath;
  }
  return `api/${trimmedPath}`;
}

export function getBackendUrl(path: string): string {
  const baseUrl =
    process.env.NEXT_PUBLIC_API_URL ||
    process.env.MODAL_API_BASE_URL;
  if (!baseUrl) {
    throw new Error(
      "Neither NEXT_PUBLIC_API_URL nor MODAL_API_BASE_URL is configured"
    );
  }

  const normalizedBase = baseUrl.replace(/\/+$/, "");
  const normalizedPath = normalizeBackendPath(path);

  if (normalizedBase.endsWith("/api")) {
    return `${normalizedBase}/${normalizedPath.replace(/^api\//, "")}`;
  }

  return `${normalizedBase}/${normalizedPath}`;
}

export async function fetchBackend(path: string, init?: RequestInit): Promise<Response> {
  const hasFormDataBody =
    typeof FormData !== "undefined" && init?.body instanceof FormData;

  return fetch(getBackendUrl(path), {
    ...init,
    cache: "no-store",
    headers: hasFormDataBody
      ? init?.headers
      : {
          "Content-Type": "application/json",
          ...(init?.headers ?? {}),
        },
  });
}

export async function readBackendJson(response: Response): Promise<unknown> {
  const text = await response.text();
  return text ? JSON.parse(text) : {};
}
