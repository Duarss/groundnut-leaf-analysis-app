// src/pages/HistoryPage.jsx
import React, { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import Card from "../components/ui/Card";
import Button from "../components/ui/Button";
import Toast from "../components/ui/Toast";
import ConfirmModal from "../components/ui/ConfirmModal";
import { fetchHistoryList, deleteHistoryItem } from "../api/historyApi";
import { formatJakartaTime } from "../utils/dateTime";
import { useIsMobile } from "../utils/useIsMobile";

const LABEL_ID = {
  "ALTERNARIA LEAF SPOT": "Bercak Daun Alternaria",
  "LEAF SPOT (EARLY AND LATE)": "Bercak Daun Tahap Awal & Akhir",
  ROSETTE: "Roset",
  RUST: "Karat Daun",
  HEALTHY: "Sehat",
};

function normLabel(label) {
  const key = String(label || "").trim().toUpperCase();
  if (!key) return "-";
  if (key === "LEAF SPOT" || key === "LEAFSPOT") return "LEAF SPOT (EARLY AND LATE)";
  return key;
}

function labelBilingual(label) {
  const k = normLabel(label);
  if (k === "-") return "-";
  const id = LABEL_ID[k];
  return id ? `${k} (${id})` : k;
}

function numOrNull(v) {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
}

function getSadClassIndex(it) {
  const a =
    it?.severity_sad_class_index ??
    it?.sad_class_index ??
    it?.sad_class ??
    it?.sad?.class_index ??
    it?.severity?.sad?.class_index ??
    null;

  const n = numOrNull(a);
  if (n == null) return null;

  const idx = Math.round(n);
  if (idx < 0 || idx > 11) return idx;
  return idx;
}

function severityBadgeStyleByPct(pct) {
  if (pct == null) return { background: "#f3f4f6", color: "#374151", border: "1px solid #e5e7eb" };
  if (pct >= 60) return { background: "#fef2f2", color: "#991b1b", border: "1px solid #fecaca" };
  if (pct >= 20) return { background: "#fffbeb", color: "#92400e", border: "1px solid #fde68a" };
  return { background: "#ecfdf5", color: "#065f46", border: "1px solid #a7f3d0" };
}

function SeverityBadge({ item }) {
  if (item?.seg_enabled === false) return <span style={{ color: "#6b7280" }}>-</span>;

  const cls = getSadClassIndex(item);
  const pct = numOrNull(item?.severity_pct);

  if (cls == null && pct == null) return <span style={{ color: "#6b7280" }}>-</span>;

  const st = severityBadgeStyleByPct(pct);
  const label = cls != null ? `SAD C${cls}` : "SAD";

  const titleParts = [];
  if (pct != null) titleParts.push(`Keparahan: ${pct.toFixed(2)}%`);
  if (cls != null) titleParts.push(`SAD class: ${cls}`);
  const title = titleParts.length ? titleParts.join(" • ") : "SAD";

  return (
    <span
      title={title}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 8,
        padding: "4px 10px",
        borderRadius: 999,
        fontSize: 13,
        fontWeight: 800,
        lineHeight: 1.2,
        ...st,
      }}
    >
      <span aria-hidden="true" style={{ width: 8, height: 8, borderRadius: 999, background: st.color }} />
      {label}
      {pct != null ? <span style={{ fontWeight: 900 }}>• {pct.toFixed(1)}%</span> : null}
    </span>
  );
}

const HistoryPage = () => {
  const isMobile = useIsMobile();
  const nav = useNavigate();
  const location = useLocation();

  const [toast, setToast] = useState({ open: false, type: "info", message: "" });

  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [errMsg, setErrMsg] = useState("");

  const [pageSize, setPageSize] = useState(5);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(null);
  const [hasNext, setHasNext] = useState(false);

  useEffect(() => {
    const t = location.state?.toast;
    if (t?.message) {
      setToast({ open: true, type: t.type || "info", message: t.message });
      nav(location.pathname, { replace: true, state: {} });
    }
  }, [location.pathname, location.state, nav]);

  useEffect(() => {
    const ctrl = new AbortController();
    let alive = true;

    (async () => {
      setLoading(true);
      setErrMsg("");

      try {
        const offset = (page - 1) * pageSize;
        const data = await fetchHistoryList({ limit: pageSize, offset }, { signal: ctrl.signal });

        const list =
          (Array.isArray(data?.items) && data.items) ||
          (Array.isArray(data?.data?.items) && data.data.items) ||
          (Array.isArray(data) && data) ||
          [];

        const totalFromApi =
          (Number.isFinite(Number(data?.total)) && Number(data.total)) ||
          (Number.isFinite(Number(data?.data?.total)) && Number(data.data.total)) ||
          (Number.isFinite(Number(data?.count)) && Number(data.count)) ||
          (Number.isFinite(Number(data?.data?.count)) && Number(data.data.count)) ||
          null;

        if (alive) {
          setItems(list);
          setTotal(totalFromApi);

          if (totalFromApi !== null) {
            setHasNext(offset + list.length < totalFromApi);
          } else {
            setHasNext(list.length === pageSize);
          }
        }
      } catch (e) {
        if (e?.name === "AbortError") return;
        if (alive) {
          setItems([]);
          setTotal(null);
          setHasNext(false);
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
  }, [page, pageSize]);

  const hasItems = useMemo(() => Array.isArray(items) && items.length > 0, [items]);

  const totalPages = useMemo(() => {
    if (total === null) return null;
    const n = Number(total);
    if (!Number.isFinite(n) || n <= 0) return 1;
    return Math.max(1, Math.ceil(n / pageSize));
  }, [total, pageSize]);

  const canPrev = page > 1;
  const canNext = totalPages ? page < totalPages : hasNext;

  function onChangePageSize(e) {
    const next = Number(e.target.value);
    if (!Number.isFinite(next)) return;
    setPageSize(next);
    setPage(1);
  }

  const [deletingId, setDeletingId] = useState(null);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [selectedItem, setSelectedItem] = useState(null);

  function openDeleteModal(it) {
    if (!it?.analysis_id) return;
    setSelectedItem(it);
    setConfirmOpen(true);
  }

  function closeDeleteModal() {
    if (deletingId) return;
    setConfirmOpen(false);
    setSelectedItem(null);
  }

  async function confirmDelete() {
    const id = selectedItem?.analysis_id;
    if (!id) return;

    setDeletingId(id);
    try {
      await deleteHistoryItem(id);

      setItems((prev) => prev.filter((x) => x.analysis_id !== id));
      setToast({ open: true, type: "success", message: "Data berhasil dihapus." });

      setConfirmOpen(false);
      setSelectedItem(null);

      setTimeout(() => {
        setItems((curr) => {
          if (curr.length === 0 && page > 1) {
            setPage((p) => Math.max(1, p - 1));
          }
          return curr;
        });
      }, 0);
    } catch (e) {
      setToast({ open: true, type: "error", message: e?.message || "Gagal menghapus data." });
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto" }}>
      <Toast
        open={toast.open}
        type={toast.type}
        message={toast.message}
        onClose={() => setToast((v) => ({ ...v, open: false }))}
      />

      <ConfirmModal
        open={confirmOpen}
        title="Hapus riwayat analisis?"
        description={
          selectedItem
            ? `Item dengan label "${labelBilingual(selectedItem?.label)} - ${(selectedItem?.analysis_id ?? "-")}" akan dihapus permanen. Tindakan ini tidak bisa dibatalkan.`
            : "Item akan dihapus permanen. Tindakan ini tidak bisa dibatalkan."
        }
        confirmText="Ya, hapus"
        cancelText="Batal"
        loading={!!deletingId}
        destructive
        onClose={closeDeleteModal}
        onConfirm={confirmDelete}
      />

      <Card title="Riwayat Analisis">
        {loading && <p>Memuat...</p>}
        {!loading && errMsg && <p style={{ color: "crimson", margin: 0 }}>{errMsg}</p>}
        {!loading && !errMsg && !hasItems && <p style={{ margin: 0 }}>Belum ada data tersimpan</p>}

        {!loading && hasItems && (
          <>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: 12,
                flexWrap: "wrap",
                marginBottom: 12,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontSize: 13, color: "#374151", fontWeight: 700 }}>Tampilkan</span>
                <select value={pageSize} onChange={onChangePageSize} style={select} aria-label="Jumlah data per halaman">
                  {[5, 10, 20, 50].map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </select>
                <span style={{ fontSize: 13, color: "#6b7280" }}>data / halaman</span>
              </div>

              <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                <span style={{ fontSize: 13, color: "#6b7280" }}>
                  Halaman <b style={{ color: "#111827" }}>{page}</b>
                  {totalPages ? (
                    <>
                      {" "}
                      / <b style={{ color: "#111827" }}>{totalPages}</b>
                    </>
                  ) : null}
                  {typeof total === "number" ? (
                    <>
                      {" "}
                      • Total <b style={{ color: "#111827" }}>{total}</b>
                    </>
                  ) : null}
                </span>
                <div style={{ display: "flex", gap: 8 }}>
                  <Button
                    disabled={!canPrev}
                    style={!canPrev ? { opacity: 0.6, cursor: "not-allowed" } : undefined}
                    onClick={() => setPage((p) => Math.max(1, p - 1))}
                  >
                    Sebelumnya
                  </Button>
                  <Button
                    disabled={!canNext}
                    style={!canNext ? { opacity: 0.6, cursor: "not-allowed" } : undefined}
                    onClick={() => setPage((p) => p + 1)}
                  >
                    Berikutnya
                  </Button>
                </div>
              </div>
            </div>

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
                      {labelBilingual(it.label)}
                    </div>

                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 10 }}>
                      <SeverityBadge item={it} />
                    </div>

                    <div style={{ marginTop: 12, display: "grid", gap: 8 }}>
                      <Button style={{ width: "100%" }} onClick={() => nav(`/history/${it.analysis_id}`)}>
                        Lihat Detail
                      </Button>

                      <Button
                        style={{ width: "100%", background: "#ef4444" }}
                        disabled={deletingId === it.analysis_id}
                        onClick={() => openDeleteModal(it)}
                      >
                        {deletingId === it.analysis_id ? "Menghapus..." : "Hapus"}
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead>
                    <tr>
                      <th style={th}>Waktu (WIB)</th>
                      <th style={th}>Label</th>
                      <th style={th}>Estimasi Keparahan</th>
                      <th style={th}>Aksi</th>
                    </tr>
                  </thead>
                  <tbody>
                    {items.map((it) => (
                      <tr key={it.analysis_id}>
                        <td style={td}>{formatJakartaTime(it.created_at)}</td>
                        <td style={td}>{labelBilingual(it.label)}</td>
                        <td style={td}>
                          <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                            <SeverityBadge item={it} />
                          </div>
                        </td>
                        <td style={td}>
                          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                            <Button onClick={() => nav(`/history/${it.analysis_id}`)}>Detail</Button>
                            <Button
                              style={{ background: "#ef4444" }}
                              disabled={deletingId === it.analysis_id}
                              onClick={() => openDeleteModal(it)}
                            >
                              {deletingId === it.analysis_id ? "Menghapus..." : "Hapus"}
                            </Button>
                          </div>
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
};

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

const select = {
  height: 34,
  padding: "0 10px",
  borderRadius: 10,
  border: "1px solid #e5e7eb",
  background: "#fff",
  color: "#111827",
  fontSize: 13,
  fontWeight: 700,
};

export default HistoryPage;