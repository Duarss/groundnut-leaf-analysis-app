// src/api/analysisApi.js

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

  const res = await fetch("/api/classify", {
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
// SEGMENTATION
// ================================
// Expected backend response (tolerant):
//  - { analysis_id, overlay_png_base64, mask_png_base64?, meta? }
export async function segmentImage(analysisId) {
  const res = await fetch("/api/segment", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ analysis_id: analysisId }),
  });

  if (!res.ok) {
    throw new Error(await _parseError(res, "Gagal melakukan segmentasi citra."));
  }

  return res.json();
}
