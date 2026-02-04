// src/components/ui/ImageBox.jsx
import React, { useMemo, useState } from "react";

export default function ImageBox({
  src,
  alt = "image",
  isMobile = false,
  maxHeightMobile = 260,
  maxHeightDesktop = 420,
  allowExpand = true,
  rounded = 12,
}) {
  const [expanded, setExpanded] = useState(false);

  const maxH = useMemo(() => {
    if (!src) return undefined;
    if (!allowExpand) return isMobile ? maxHeightMobile : maxHeightDesktop;
    if (expanded) return "none";
    return isMobile ? maxHeightMobile : maxHeightDesktop;
  }, [src, allowExpand, expanded, isMobile, maxHeightMobile, maxHeightDesktop]);

  if (!src) return null;

  return (
    <div style={{ display: "grid", gap: 10 }}>
      <div
        style={{
          border: "1px solid #e5e7eb",
          borderRadius: rounded,
          padding: 10,
          background: "#fff",
        }}
      >
        <img
          src={src}
          alt={alt}
          loading="lazy"
          style={{
            width: "100%",
            maxHeight: maxH === "none" ? undefined : maxH,
            height: "auto",
            display: "block",
            borderRadius: Math.max(8, rounded - 2),
            objectFit: "contain",
            background: "#fafafa",
          }}
        />
      </div>

      {allowExpand && isMobile && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          style={{
            width: "100%",
            border: "1px solid #e5e7eb",
            background: "#fff",
            borderRadius: 999,
            padding: "10px 12px",
            fontWeight: 700,
            fontSize: 13,
            cursor: "pointer",
          }}
        >
          {expanded ? "Ringkas" : "Perbesar"}
        </button>
      )}
    </div>
  );
}
