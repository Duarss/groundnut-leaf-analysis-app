// src/pages/HomePage.jsx
import React from "react";
import { Link } from "react-router-dom";
import Button from "../components/ui/Button";
import Card from "../components/ui/Card";

const HomePage = () => {
  return (
    <div className="page page-home">
      <section className="hero">
        <div className="hero-text">
          <h1>
            Sistem Visi Komputer untuk <br />
            <span className="accent">Penyakit Daun Kacang Tanah</span>
          </h1>
          <p>
            Unggah foto daun kacang tanah, sistem akan melakukan klasifikasi
            penyakit, segmentasi area terinfeksi, dan estimasi tingkat
            keparahan berdasarkan model CNN (EfficientNet-B4) dan U-Net.
          </p>
          <div className="hero-actions">
            <Link to="/classify">
              <Button>Mulai Klasifikasi</Button>
            </Link>
            <Link to="/guide">
              <Button variant="ghost">Lihat Panduan Penggunaan</Button>
            </Link>
          </div>
        </div>
        <div className="hero-visual">
          <Card
            title="Alur Sistem"
          >
            <ol className="hero-flow">
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
