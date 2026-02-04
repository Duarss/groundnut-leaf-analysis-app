// src/pages/HomePage.jsx
import React from "react";
import { Link } from "react-router-dom";
import Button from "../components/ui/Button";
import Card from "../components/ui/Card";
import { useIsMobile } from "../utils/useIsMobile";

const HomePage = () => {
  const isMobile = useIsMobile();

  return (
    <div className="page page-home">
      <section
        className="hero"
        style={{
          display: "grid",
          gridTemplateColumns: isMobile ? "1fr" : "1.2fr 0.8fr",
          gap: 16,
          alignItems: "start",
        }}
      >
        <div className="hero-text">
          <h1 style={{ lineHeight: 1.2, marginTop: 0 }}>
            Sistem Visi Komputer untuk <br />
            <span className="accent">Penyakit Daun Kacang Tanah</span>
          </h1>

          <p>
            Unggah foto daun kacang tanah, sistem akan melakukan klasifikasi
            penyakit, segmentasi area terinfeksi, dan estimasi tingkat
            keparahan berdasarkan model CNN (EfficientNet-B4) dan U-Net.
          </p>

          <div
            className="hero-actions"
            style={{
              display: "flex",
              gap: 10,
              flexWrap: "wrap",
              alignItems: "center",
            }}
          >
            <Link to="/classify" style={{ flex: isMobile ? "1 1 100%" : "0 0 auto" }}>
              <Button style={isMobile ? { width: "100%" } : undefined}>
                Mulai Analisis
              </Button>
            </Link>

            <Link to="/guide" style={{ flex: isMobile ? "1 1 100%" : "0 0 auto" }}>
              <Button
                variant="ghost"
                style={isMobile ? { width: "100%" } : undefined}
              >
                Lihat Panduan Penggunaan
              </Button>
            </Link>
          </div>
        </div>

        <div className="hero-visual">
          <Card title="Alur Sistem">
            <ol className="hero-flow" style={{ paddingLeft: 18, margin: 0 }}>
              <li>Unggah citra daun kacang tanah</li>
              <li>Preprocessing & klasifikasi dengan EfficientNet-B4</li>
              <li>Segmentasi area terinfeksi dengan U-Net</li>
              <li>Estimasi tingkat keparahan (skala FAO)</li>
              <li>Tinjau hasil & simpan ke riwayat</li>
            </ol>
          </Card>
        </div>
      </section>
    </div>
  );
};

export default HomePage;
