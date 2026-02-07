// src/components/ui/ConfirmModal.jsx
import React, { useEffect } from "react";

export default function ConfirmModal({
  open,
  title = "Konfirmasi",
  description = "Apakah kamu yakin?",
  confirmText = "Ya, hapus",
  cancelText = "Batal",
  loading = false,
  destructive = true,
  onConfirm,
  onClose,
}) {
  useEffect(() => {
    if (!open) return;

    const onKeyDown = (e) => {
      if (e.key === "Escape") onClose?.();
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  const dangerBg = destructive ? "#ef4444" : "#111827";

  return (
    <div
      role="dialog"
      aria-modal="true"
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 9998,
        background: "rgba(0,0,0,0.45)",
        display: "grid",
        placeItems: "center",
        padding: 16,
      }}
      onMouseDown={(e) => {
        // klik di backdrop untuk close
        if (e.target === e.currentTarget && !loading) onClose?.();
      }}
    >
      <div
        style={{
          width: "100%",
          maxWidth: 420,
          background: "#fff",
          borderRadius: 16,
          boxShadow: "0 20px 50px rgba(0,0,0,0.25)",
          overflow: "hidden",
        }}
      >
        <div style={{ padding: 16, borderBottom: "1px solid #e5e7eb" }}>
          <div style={{ fontSize: 16, fontWeight: 900, color: "#111827" }}>{title}</div>
          <div style={{ marginTop: 6, fontSize: 14, color: "#4b5563", lineHeight: 1.4 }}>
            {description}
          </div>
        </div>

        <div style={{ padding: 16, display: "flex", gap: 10, justifyContent: "flex-end" }}>
          <button
            type="button"
            disabled={loading}
            onClick={onClose}
            style={{
              border: "1px solid #e5e7eb",
              background: "#fff",
              color: "#111827",
              padding: "10px 12px",
              borderRadius: 12,
              cursor: loading ? "not-allowed" : "pointer",
              fontWeight: 700,
            }}
          >
            {cancelText}
          </button>

          <button
            type="button"
            disabled={loading}
            onClick={onConfirm}
            style={{
              border: "none",
              background: dangerBg,
              color: "#fff",
              padding: "10px 12px",
              borderRadius: 12,
              cursor: loading ? "not-allowed" : "pointer",
              fontWeight: 800,
              minWidth: 110,
            }}
          >
            {loading ? "Menghapus..." : confirmText}
          </button>
        </div>
      </div>
    </div>
  );
}
