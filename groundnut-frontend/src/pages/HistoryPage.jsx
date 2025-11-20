// src/pages/HistoryPage.jsx
import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import Card from "../components/ui/Card";
// import { fetchHistory } from "../api/analysisApi";

const dummyHistory = [
  {
    id: "dummy-123",
    disease_label: "Leaf Spot (Early)",
    severity_level: "Sedang",
    severity_percent: 37.5,
    created_at: "2025-11-19 10:15",
  },
  {
    id: "dummy-124",
    disease_label: "Rust",
    severity_level: "Berat",
    severity_percent: 61.2,
    created_at: "2025-11-18 16:02",
  },
];

const HistoryPage = () => {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        setError("");
        // const data = await fetchHistory();
        // setItems(data);
        setTimeout(() => {
          setItems(dummyHistory);
          setLoading(false);
        }, 400);
      } catch (err) {
        setError(err.message || "Gagal memuat riwayat.");
        setLoading(false);
      }
    }
    load();
  }, []);

  return (
    <div className="page page-history">
      <h2>Riwayat Penggunaan Sistem</h2>
      <p className="page-description">
        Halaman ini menyimpan hasil klasifikasi dan analisis yang telah kamu
        lakukan sebelumnya. Kamu dapat membuka kembali detail setiap analisis
        termasuk segmentasi dan tingkat keparahan penyakit.
      </p>

      <Card title="Daftar Riwayat">
        {loading && <p>Memuat riwayat...</p>}
        {error && <p className="error-text">{error}</p>}
        {!loading && !error && items.length === 0 && (
          <p className="placeholder">Belum ada riwayat penggunaan.</p>
        )}

        {!loading && !error && items.length > 0 && (
          <div className="history-table-wrapper">
            <table className="history-table">
              <thead>
                <tr>
                  <th>Waktu</th>
                  <th>Penyakit</th>
                  <th>Keparahan</th>
                  <th>Persentase</th>
                  <th>Aksi</th>
                </tr>
              </thead>
              <tbody>
                {items.map((row) => (
                  <tr key={row.id}>
                    <td>{row.created_at}</td>
                    <td>{row.disease_label}</td>
                    <td>{row.severity_level}</td>
                    <td>{row.severity_percent.toFixed(1)}%</td>
                    <td>
                      <Link to={`/history/${row.id}`} className="link">
                        Lihat Detail
                      </Link>
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
};

export default HistoryPage;
