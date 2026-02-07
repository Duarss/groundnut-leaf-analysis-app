// src/api/historyApi.js
import { getClientId } from "../utils/clientId";

async function _fetchJson(url, opts = {}) {
  const headers = {
    ...(opts.headers || {}),
    "X-Client-Id": getClientId(),
  };

  const res = await fetch(url, { ...opts, headers });

  // jangan crash kalau response bukan JSON
  const data = await res.json().catch(() => ({}));

  if (!res.ok) {
    const msg = data?.error || data?.detail || `Request failed (${res.status})`;
    throw new Error(msg);
  }
  return data;
}

export function fetchHistoryList({ limit = 10, offset = 0 } = {}, opts = {}) {
  return _fetchJson(`/api/history?limit=${limit}&offset=${offset}`, opts);
}

export function fetchHistoryDetail(analysisId, opts = {}) {
  const id = (analysisId ?? "").toString().trim();
  if (!id) {
    throw new Error("analysis_id tidak valid (undefined/kosong). Cek routing & parameter URL.");
  }
  return _fetchJson(`/api/history/${encodeURIComponent(id)}`, opts);
}

export function deleteHistoryItem(analysisId, opts = {}) {
  const id = (analysisId ?? "").toString().trim();
  if (!id) {
    throw new Error("analysis_id tidak valid (undefined/kosong).");
  }
  return _fetchJson(`/api/history/${encodeURIComponent(id)}`, { ...opts, method: "DELETE" });
}
