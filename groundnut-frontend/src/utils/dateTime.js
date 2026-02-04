// src/utils/dateTime.js

export function formatJakartaTime(value, options = {}) {
  if (!value) return "-";

  const date =
    typeof value === "number"
      ? new Date(value * 1000) // unix seconds
      : new Date(value);

  if (Number.isNaN(date.getTime())) return "-";

  return new Intl.DateTimeFormat("id-ID", {
    timeZone: "Asia/Jakarta",
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    ...options,
  }).format(date);
}
