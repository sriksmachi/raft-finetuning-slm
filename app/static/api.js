/** Shared HTTP contract; production and demo runs use isolated endpoints. */
export async function api(path, body, mode = "demo") {
  if (mode === "production" && (path === "/api/config" || path.startsWith("/api/runs"))) {
    path = path.replace("/api/", "/api/production/");
  }
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: { Accept: "application/json", ...(body === undefined ? {} : { "Content-Type": "application/json" }) },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  const result = await response.json();
  if (!response.ok) {
    const detail = result.detail;
    const message = typeof detail === "string" ? detail :
      Array.isArray(detail) ? detail.map((issue) => `${issue.loc.join(".")}: ${issue.msg}`).join("; ") :
      `API returned ${response.status}.`;
    throw new Error(message);
  }
  return result;
}
