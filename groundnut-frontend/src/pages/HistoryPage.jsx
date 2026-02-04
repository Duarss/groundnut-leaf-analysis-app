// src/pages/HistoryPage.jsx
import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import Card from "../components/ui/Card";
import Button from "../components/ui/Button";
import { fetchHistoryList } from "../api/historyApi";
import { formatJakartaTime } from "../utils/dateTime";
import { useIsMobile } from "../utils/useIsMobile";

// Label keparahan (bahasa formal)
const FAO_SEVERITY_LABEL = {
  1: "Sangat Ringan",
  2: "Ringan",
  3: "Sedang",
  4: "Berat",
  5: "Sangat Berat",
};

function fmtConfPct(x) {
  if (x === null || x === undefined) return "-";
  const n = Number(x);
  if (Number.isNaN(n)) return "-";
  return `${(n * 100).toFixed(2)}%`;
}

function inferSeverityLevelFromPct(severity_pct) {
  const pct = Number(severity_pct);
  if (!Number.isFinite(pct)) return null;
  if (pct <= 10) return 1;
  if (pct <= 20) return 2;
  if (pct <= 40) return 3;
  if (pct <= 60) return 4;
  return 5;
}

function getSeverityLevel(it) {
  const lvl = Number(it?.severity_fao_level);
  if (Number.isFinite(lvl) && lvl >= 1 && lvl <= 5) return lvl;
  return inferSeverityLevelFromPct(it?.severity_pct);
}

function getSeverityLabel(it) {
  if (it?.seg_enabled === false) return "-";
  const lvl = getSeverityLevel(it);
  if (!lvl) return "-";
  return FAO_SEVERITY_LABEL[lvl] || "-";
}

function severityBadgeStyle(level) {
  if (!level) {
    return { background: "#f3f4f6", color: "#374151", border: "1px solid #e5e7eb" };
  }
  const map = {
    1: { background: "#ecfdf5", color: "#065f46", border: "1px solid #a7f3d0" },
    2: { background: "#f0fdf4", color: "#166534", border: "1px solid #bbf7d0" },
    3: { background: "#fffbeb", color: "#92400e", border: "1px solid #fde68a" },
    4: { background: "#fff7ed", color: "#9a3412", border: "1px solid #fdba74" },
    5: { background: "#fef2f2", color: "#991b1b", border: "1px solid #fecaca" },
  };
  return map[level] || map[3];
}

function SeverityBadge({ item }) {
  const label = getSeverityLabel(item);
  if (label === "-") return <span style={{ color: "#6b7280" }}>-</span>;

  const lvl = getSeverityLevel(item);
  const st = severityBadgeStyle(lvl);

  return (
    <span
      title="Kategori keparahan (acuan FAO)"
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 8,
        padding: "4px 10px",
        borderRadius: 999,
        fontSize: 13,
        fontWeight: 700,
        lineHeight: 1.2,
        ...st,
      }}
    >
      <span aria-hidden="true" style={{ width: 8, height: 8, borderRadius: 999, background: st.color }} />
      {label}
    </span>
  );
}

export default function HistoryPage() {
  const isMobile = useIsMobile();
  const nav = useNavigate();
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [errMsg, setErrMsg] = useState("");

  useEffect(() => {
    const ctrl = new AbortController();
    let alive = true;

    (async () => {
      setLoading(true);
      setErrMsg("");

      try {
        const data = await fetchHistoryList({ limit: 50, offset: 0 }, { signal: ctrl.signal });
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

  const hasItems = useMemo(() => Array.isArray(items) && items.length > 0, [items]);

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto" }}>
      <Card title="Riwayat Analisis">
        {loading && <p>Memuat...</p>}

        {!loading && errMsg && <p style={{ color: "crimson", margin: 0 }}>{errMsg}</p>}

        {!loading && !errMsg && !hasItems && <p style={{ margin: 0 }}>Belum ada data tersimpan</p>}

        {!loading && hasItems && (
          <>
            {/* Mobile: list cards */}
            {isMobile ? (
              <div style={{ display: "grid", gap: 12 }}>
                {items.map((it) => (
                  <div
                    key={it.analysis_id}
                    style={{
                      border: "1px solid #e5e7eb",
                      borderRadius: 14,
                      padding: 12,
                      background: "#fff",
                    }}
                  >
                    <div style={{ fontSize: 12, color: "#6b7280" }}>
                      {formatJakartaTime(it.created_at)} (WIB)
                    </div>

                    <div style={{ fontSize: 16, fontWeight: 900, marginTop: 4 }}>
                      {it.label ?? "-"}
                    </div>

                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 10 }}>
                      <span style={{ fontSize: 13, color: "#111827" }}>
                        Keyakinan: <b>{fmtConfPct(it.confidence)}</b>
                      </span>
                      <SeverityBadge item={it} />
                    </div>

                    <div style={{ marginTop: 12 }}>
                      <Button style={{ width: "100%" }} onClick={() => nav(`/history/${it.analysis_id}`)}>
                        Lihat Detail
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              /* Desktop: table */
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead>
                    <tr>
                      <th style={th}>Waktu (WIB)</th>
                      <th style={th}>Label</th>
                      <th style={th}>Confidence</th>
                      <th style={th}>Keparahan</th>
                      <th style={th}>Aksi</th>
                    </tr>
                  </thead>
                  <tbody>
                    {items.map((it) => (
                      <tr key={it.analysis_id}>
                        <td style={td}>{formatJakartaTime(it.created_at)}</td>
                        <td style={td}>{it.label ?? "-"}</td>
                        <td style={td}>{fmtConfPct(it.confidence)}</td>
                        <td style={td}>
                          <SeverityBadge item={it} />
                        </td>
                        <td style={td}>
                          <Button onClick={() => nav(`/history/${it.analysis_id}`)}>Detail</Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
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
