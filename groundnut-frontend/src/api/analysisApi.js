// src/api/analysisApi.js
import { getClientId } from "../utils/clientId";

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
// Expected backend response (tolerant):
//  - { analysis_id, label, confidence, probs }
//  - or { id, label, confidence, probs }
export async function classifyImage(file) {
  const formData = new FormData();
  formData.append("image", file);

  const res = await fetch(`/api/classify`, {
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
// Expected backend response (tolerant):
//  - { analysis_id, overlay_png_base64, mask_png_base64?, meta? }
export async function segmentImage(analysisId) {
  const res = await fetch(`/api/segment`, {
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
// Expected backend response:
//  - { saved: true, analysis_id, client_id, orig_image_path, seg_enabled, seg_overlay_path }
export async function saveAnalysis(analysisId, body = {}) {
  const res = await fetch(`/api/save`, {
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
