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

const HistoryPage = () => {
  const isMobile = useIsMobile();
  const nav = useNavigate();
  const location = useLocation();

  const [toast, setToast] = useState({ open: false, type: "info", message: "" });
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [errMsg, setErrMsg] = useState("");

  const [deletingId, setDeletingId] = useState(null);

  // Modal state
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [selectedItem, setSelectedItem] = useState(null);

  // Pagination
  const [pageSize, setPageSize] = useState(10);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(null);
  const [hasNext, setHasNext] = useState(false);

  useEffect(() => {
    const t = location.state?.toast;
    if (t?.message) {
      setToast({ open: true, type: t.type || "info", message: t.message });

      // ✅ bersihkan state agar toast tidak muncul lagi saat refresh/back
      nav(location.pathname, { replace: true, state: {} });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.state]);

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

        // Total bisa bervariasi tergantung bentuk response API
        const totalFromApi =
          (Number.isFinite(Number(data?.total)) && Number(data.total)) ||
          (Number.isFinite(Number(data?.data?.total)) && Number(data.data.total)) ||
          (Number.isFinite(Number(data?.count)) && Number(data.count)) ||
          (Number.isFinite(Number(data?.data?.count)) && Number(data.data.count)) ||
          null;

        if (alive) {
          setItems(list);
          setTotal(totalFromApi);

          // Jika total tidak tersedia, anggap masih ada halaman berikutnya bila list = pageSize
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

  // ===== Modal control =====
  function openDeleteModal(it) {
    if (!it?.analysis_id) return;
    setSelectedItem(it);
    setConfirmOpen(true);
  }

  function closeDeleteModal() {
    // saat sedang delete, jangan bisa close (biar state aman)
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

      // kalau page jadi kosong & page > 1, mundurkan page biar user gak lihat halaman kosong
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
      <Toast open={toast.open} type={toast.type} message={toast.message} onClose={() => setToast((v) => ({ ...v, open: false }))} />

      <ConfirmModal
        open={confirmOpen}
        title="Hapus riwayat analisis?"
        description={
          selectedItem
            ? `Item dengan label "${(selectedItem?.label ?? "-") + " - " + (selectedItem?.analysis_id ?? "-")}" akan dihapus permanen. Tindakan ini tidak bisa dibatalkan.`
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
            {/* Toolbar: page size + pagination */}
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
              {/* Kiri: page size */}
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

              {/* Kanan: pagination */}
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
                    <div style={{ fontSize: 12, color: "#6b7280" }}>{formatJakartaTime(it.created_at)} (WIB)</div>

                    <div style={{ fontSize: 16, fontWeight: 900, marginTop: 4 }}>{it.label ?? "-"}</div>

                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 10 }}>
                      <span style={{ fontSize: 13, color: "#111827" }}>
                        Keyakinan: <b>{fmtConfPct(it.confidence)}</b>
                      </span>
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
