// src/components/ui/Toast.jsx
import React, { useEffect } from "react";

export default function Toast({
  open,
  type = "info", // "success" | "error" | "info"
  message,
  onClose,
  duration = 2500,
}) {
  useEffect(() => {
    if (!open) return;
    const t = setTimeout(() => onClose?.(), duration);
    return () => clearTimeout(t);
  }, [open, duration, onClose]);

  if (!open) return null;

  const isSuccess = type === "success";
  const isError = type === "error";

  const bg = isSuccess ? "#ECFDF5" : isError ? "#FEF2F2" : "#EFF6FF";
  const border = isSuccess ? "#10B981" : isError ? "#EF4444" : "#3B82F6";
  const text = isSuccess ? "#065F46" : isError ? "#991B1B" : "#1E3A8A";

  return (
    <div
      role="status"
      aria-live="polite"
      style={{
        position: "fixed",
        right: 16,
        top: 16,
        zIndex: 9999,
        maxWidth: 380,
        width: "calc(100% - 32px)",
      }}
    >
      <div
        style={{
          background: bg,
          border: `1px solid ${border}`,
          color: text,
          borderRadius: 14,
          padding: "12px 12px",
          boxShadow: "0 10px 25px rgba(0,0,0,0.12)",
          display: "flex",
          gap: 10,
          alignItems: "flex-start",
        }}
      >
        <div style={{ fontWeight: 900, lineHeight: 1.2 }}>
          {isSuccess ? "Berhasil" : isError ? "Gagal" : "Info"}
        </div>

        <div style={{ flex: 1, lineHeight: 1.35 }}>{message}</div>

        <button
          onClick={onClose}
          aria-label="Tutup"
          style={{
            border: "none",
            background: "transparent",
            cursor: "pointer",
            fontSize: 18,
            lineHeight: 1,
            padding: 2,
            color: text,
          }}
        >
          ×
        </button>
      </div>
    </div>
  );
}
