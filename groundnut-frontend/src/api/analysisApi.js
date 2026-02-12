// src/api/analysisApi.js
import { getClientId } from "../utils/clientId";

// Header untuk bypass halaman warning ngrok (free domain)
const NGROK_SKIP_HEADER = { "ngrok-skip-browser-warning": "true" };

// Helper fetch dengan default header ngrok-skip
async function _fetch(url, opts = {}) {
  const headers = {
    ...NGROK_SKIP_HEADER,
    ...(opts.headers || {}),
  };
  return fetch(url, { ...opts, headers });
}

async function _parseError(res, fallbackMsg) {
  let msg = fallbackMsg;
  try {
    const data = await res.json();
    if (data?.error) msg = data.error;
    if (data?.message) msg = data.message;
  } catch (e) {
    void e;
  }
  return msg;
}

// ================================
// CLASSIFICATION
// ================================
export async function classifyImage(file) {
  const formData = new FormData();
  formData.append("image", file);

  const res = await _fetch(`/api/classify`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error(await _parseError(res, "Gagal melakukan klasifikasi citra."));
  }

  const data = await res.json();
  const analysisId = data?.analysis_id ?? data?.id;

  return {
    analysis_id: analysisId,
    label: data?.label,
    confidence: data?.confidence,
    probs: data?.probs,
    segmentation_ready: data?.segmentation_ready || false,
    message: data?.message || "",
  };
}

// ================================
// SEGMENTATION & SEVERITY ESTIMATION
// ================================
export async function segmentImage(analysisId) {
  const res = await _fetch(`/api/segment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ analysis_id: analysisId }),
  });

  if (!res.ok) {
    throw new Error(await _parseError(res, "Gagal melakukan segmentasi citra."));
  }

  return res.json();
}

// ================================
// SAVE ANALYSIS
// ================================
export async function saveAnalysis(analysisId, body = {}) {
  const res = await _fetch(`/api/save`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Client-Id": getClientId(),
    },
    body: JSON.stringify({ analysis_id: analysisId, ...body }),
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data?.error || data?.detail || "Save gagal");
  return data;
}
