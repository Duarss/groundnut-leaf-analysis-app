// src/pages/HistoryDetailPage.jsx
import React, { useEffect, useMemo, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import Card from "../components/ui/Card";
import Button from "../components/ui/Button";
import { fetchHistoryDetail } from "../api/historyApi";
import { getClientId } from "../utils/clientId";

function _safeParse(jsonStr) {
  try { return JSON.parse(jsonStr); } catch { return null; }
}

function toDateString(created_at) {
  if (!created_at) return "-";
  if (typeof created_at === "number") return new Date(created_at * 1000).toLocaleString();
  const d = new Date(created_at);
  return Number.isNaN(d.getTime()) ? String(created_at) : d.toLocaleString();
}

export default function HistoryDetailPage() {
  const params = useParams();
  const nav = useNavigate();
  const [data, setData] = useState(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(true);

  const analysisId = (params.analysisId || params.id || "").toString().trim();

  useEffect(() => {
    let alive = true;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLoading(true);
    setErr("");

    fetchHistoryDetail(analysisId)
      .then((d) => {
        if (!alive) return;
        setData(d);
      })
      .catch((e) => {
        if (!alive) return;
        setErr(e?.message || "Gagal memuat detail");
        setData(null);
      })
      .finally(() => alive && setLoading(false));

    return () => { alive = false; };
  }, [analysisId]);

  const probsObj = useMemo(
    () => (data?.probs_json ? _safeParse(data.probs_json) : null),
    [data]
  );

  const cid = useMemo(() => getClientId(), []);

  // ✅ URL image via storage endpoint (pakai query client_id agar <img> bisa akses)
  const origImgUrl = useMemo(() => {
    const p = data?.orig_image_path;
    if (!p) return null;
    return `/api/storage?path=${encodeURIComponent(p)}&client_id=${encodeURIComponent(cid)}`;
  }, [data, cid]);

  const overlayImgUrl = useMemo(() => {
    const p = data?.seg_overlay_path;
    if (!p) return null;
    return `/api/storage?path=${encodeURIComponent(p)}&client_id=${encodeURIComponent(cid)}`;
  }, [data, cid]);

  return (
    <div className="p-4 max-w-4xl mx-auto">
      <div className="flex items-center gap-2 mb-3">
        <Button onClick={() => nav("/history")}>Kembali</Button>
        <h2 className="text-xl font-semibold">Detail Analisis</h2>
      </div>

      {loading && <div>Loading...</div>}
      {!loading && err && <div className="text-red-600">{err}</div>}

      {!loading && data && (
        <div className="space-y-3">
          <Card>
            <div className="text-sm text-gray-600">{toDateString(data.created_at)}</div>
            <div className="font-semibold">{data.label || "-"}</div>
            <div className="text-sm">
              Confidence: {typeof data.confidence === "number" ? data.confidence.toFixed(6) : "-"}
            </div>
            <div className="text-sm">
              Segmentasi: {data.seg_enabled ? "Ya" : "Tidak"}
            </div>

            {data.seg_enabled && typeof data.severity_pct === "number" && (
              <div className="text-sm">
                Severity: {data.severity_pct.toFixed(2)}% (FAO Level {data.severity_fao_level ?? "-"})
              </div>
            )}
          </Card>

          <Card>
            <div className="font-semibold mb-2">Gambar</div>

            {origImgUrl && (
              <div className="mb-3">
                <div className="text-sm mb-1">Original</div>
                <img
                  alt="original"
                  className="max-w-full rounded"
                  src={origImgUrl}
                />
              </div>
            )}

            {data.seg_enabled && overlayImgUrl && (
              <div>
                <div className="text-sm mb-1">Overlay Segmentasi</div>
                <img
                  alt="overlay"
                  className="max-w-full rounded"
                  src={overlayImgUrl}
                />
              </div>
            )}

            {!origImgUrl && (
              <div className="text-sm text-gray-600">Gambar original tidak tersedia.</div>
            )}
          </Card>

          <Card>
            <div className="font-semibold mb-2">Probabilitas Kelas</div>
            {probsObj ? (
              <pre className="text-xs whitespace-pre-wrap">{JSON.stringify(probsObj, null, 2)}</pre>
            ) : (
              <div className="text-sm text-gray-600">Tidak ada probs_json.</div>
            )}
          </Card>
        </div>
      )}
    </div>
  );
}
