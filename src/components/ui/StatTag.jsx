// src/components/ui/StatTag.jsx
import React from "react";

const severityClass = (level) => {
  if (!level) return "tag-neutral";
  const v = level.toLowerCase();
  if (v.includes("ringan") || v.includes("low")) return "tag-low";
  if (v.includes("sedang") || v.includes("medium")) return "tag-medium";
  return "tag-high";
};

const StatTag = ({ label, value }) => {
  return (
    <div className="stat-tag">
      <span className="stat-label">{label}</span>
      <span className={`stat-value ${severityClass(value)}`}>{value}</span>
    </div>
  );
};

export default StatTag;
