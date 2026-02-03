// src/pages/HistoryPage.jsx
import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import Card from "../components/ui/Card";
import Button from "../components/ui/Button";
import { fetchHistoryList } from "../api/historyApi";

function toDateString(created_at) {
  if (!created_at) return "-";
  if (typeof created_at === "number") {
    // unix seconds
    return new Date(created_at * 1000).toLocaleString();
  }
  // string (misal "Tue, 03 Feb 2026 03:09:52 GMT")
  const d = new Date(created_at);
  return Number.isNaN(d.getTime()) ? String(created_at) : d.toLocaleString();
}

function fmtPct(x) {
  if (x === null || x === undefined) return "-";
  const n = Number(x);
  if (Number.isNaN(n)) return "-";
  return `${n.toFixed(2)}%`;
}

function fmtConf(x) {
  if (x === null || x === undefined) return "-";
  const n = Number(x);
  if (Number.isNaN(n)) return "-";
  return n.toFixed(6);
}

export default function HistoryPage() {
  const nav = useNavigate();
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false); // ✅ default false biar aman
  const [errMsg, setErrMsg] = useState("");

  useEffect(() => {
    const ctrl = new AbortController();
    let alive = true;

    (async () => {
      setLoading(true);
      setErrMsg("");

      try {
        const data = await fetchHistoryList(
          { limit: 50, offset: 0 },
          { signal: ctrl.signal }
        );

        // ✅ ambil items dari berbagai kemungkinan bentuk response
        const list =
          (Array.isArray(data?.items) && data.items) ||
          (Array.isArray(data?.data?.items) && data.data.items) ||
          (Array.isArray(data) && data) ||
          [];

        if (alive) setItems(list);
      } catch (e) {
        if (e?.name === "AbortError") return;
        if (alive) {
          setItems([]);
          setErrMsg(e?.message || "Gagal memuat history");
        }
      } finally {
        if (alive) setLoading(false);
      }
    })();

    return () => {
      alive = false;
      ctrl.abort();
    };
  }, []);

  const hasItems = useMemo(
    () => Array.isArray(items) && items.length > 0,
    [items]
  );

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto" }}>
      <Card title="Riwayat Analisis">
        {loading && <p>Memuat...</p>}

        {!loading && errMsg && (
          <div style={{ marginBottom: 12 }}>
            <p style={{ color: "crimson", margin: 0 }}>{errMsg}</p>
          </div>
        )}

        {!loading && !errMsg && !hasItems && (
          <p style={{ margin: 0 }}>Belum ada data tersimpan</p>
        )}

        {!loading && hasItems && (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <th style={th}>Waktu</th>
                  <th style={th}>Label</th>
                  <th style={th}>Confidence</th>
                  <th style={th}>Severity</th>
                  <th style={th}>Aksi</th>
                </tr>
              </thead>
              <tbody>
                {items.map((it) => (
                  <tr key={it.analysis_id}>
                    <td style={td}>{toDateString(it.created_at)}</td>
                    <td style={td}>{it.label ?? "-"}</td>
                    <td style={td}>{fmtConf(it.confidence)}</td>
                    <td style={td}>{fmtPct(it.severity_pct)}</td>
                    <td style={td}>
                      <Button onClick={() => nav(`/history/${it.analysis_id}`)}>
                        Detail
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}

const th = {
  textAlign: "left",
  padding: "10px 8px",
  borderBottom: "1px solid #ddd",
  whiteSpace: "nowrap",
};

const td = {
  padding: "10px 8px",
  borderBottom: "1px solid #eee",
  verticalAlign: "top",
};
