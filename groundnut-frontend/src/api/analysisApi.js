// src/api/analysisApi.js

export async function analyzeImage(file) {
  const formData = new FormData();
  formData.append("image", file);

  // With Vite proxy: this will be forwarded to http://localhost:5000/api/analyze
  const res = await fetch("/api/analyze", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    let msg = "Gagal menganalisis citra.";
    try {
      const errData = await res.json();
      if (errData?.error) msg = errData.error;
    } catch (err) {
      void err;
    }
    throw new Error(msg);
  }

  return res.json(); // { id, disease_label, confidence, probs }
}
