// src/pages/GuidePage.jsx
import React, { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import Card from "../components/ui/Card";
import Button from "../components/ui/Button";
import { useIsMobile } from "../utils/useIsMobile";

function AccordionSection({ id, title, isOpen, onToggle, children, isMobile }) {
  return (
    <Card title={title}>
      <div
        style={{
          display: "flex",
          gap: 10,
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          marginBottom: isOpen ? 10 : 0,
        }}
      >
        <div style={{ fontSize: 13, color: "#6b7280" }}>
          {isOpen ? "Klik untuk menutup" : "Klik untuk membuka"}
        </div>

        <Button
          variant="secondary"
          onClick={() => onToggle(id)}
          style={isMobile ? { width: "100%" } : undefined}
        >
          {isOpen ? "Tutup" : "Buka"}
        </Button>
      </div>

      {isOpen ? <div>{children}</div> : null}
    </Card>
  );
}

const GuidePage = () => {
  const nav = useNavigate();
  const isMobile = useIsMobile();

  const sectionGap = isMobile ? 14 : 18;
  const lineHeight = isMobile ? 1.7 : 1.6;

  const items = useMemo(
    () => [
      {
        id: "tujuan",
        title: "Panduan Penggunaan Sistem",
        content: (
          <>
            <p style={{ lineHeight, marginTop: 0 }}>
              Panduan ini membantu Anda menggunakan sistem analisis daun dengan benar, mulai dari
              cara mengambil foto sampai memahami hasil yang ditampilkan.
            </p>
            <p style={{ lineHeight, marginBottom: 0 }}>
              Agar hasil lebih optimal, ikuti checklist dan tips foto di bawah sebelum melakukan
              analisis.
            </p>
          </>
        ),
      },
      {
        id: "checklist",
        title: "Checklist Cepat Sebelum Mengambil Foto",
        content: (
          <>
            <ul style={{ lineHeight: 1.8, paddingLeft: 18, marginTop: 0 }}>
              <li>✅ Objek adalah foto daun/tanaman kacang tanah (objek utama jelas)</li>
              <li>✅ Daun terlihat jelas (tidak buram) dan cukup terang</li>
              <li>✅ Objek utama mengisi sebagian besar area foto</li>
              <li>✅ Usahakan hanya satu tanaman/objek utama dalam satu foto</li>
              <li>✅ Hindari objek lain yang dominan atau menutupi area gejala</li>
              <li>✅ Latar belakang polos/kontras, tidak ramai</li>
              <li>✅ Hindari bayangan keras dan pantulan cahaya</li>
            </ul>
            <p style={{ color: "#6b7280", marginTop: 10, marginBottom: 0 }}>
              Checklist ini sangat berpengaruh terhadap akurasi klasifikasi, segmentasi, dan estimasi
              keparahan.
            </p>
          </>
        ),
      },
      {
        id: "alur",
        title: "Alur Singkat Penggunaan",
        content: (
          <ol style={{ lineHeight: 1.8, paddingLeft: 18, marginTop: 0, marginBottom: 0 }}>
            <li>Unggah atau ambil foto daun kacang tanah.</li>
            <li>Sistem memprediksi label penyakit (hasil klasifikasi) beserta nilai keyakinan.</li>
            <li>Jika terindikasi penyakit tertentu, lanjutkan segmentasi untuk menandai area terinfeksi.</li>
            <li>Sistem menghitung keparahan sebagai persentase (area terinfeksi dibanding area daun).</li>
            <li>Sistem memetakan persentase ke kelas SAD untuk membantu interpretasi.</li>
            <li>
              Tekan <b>Simpan Hasil</b> untuk menyimpan ke Riwayat Analisis.
            </li>
          </ol>
        ),
      },
      {
        id: "tips",
        title: "Tips Mengambil Foto yang Disarankan",
        content: (
          <ul style={{ lineHeight: 1.8, paddingLeft: 18, marginTop: 0, marginBottom: 0 }}>
            <li>Ambil foto dari jarak sedang, jangan terlalu jauh.</li>
            <li>Pastikan fokus kamera tajam pada daun (bukan background).</li>
            <li>Gunakan cahaya alami bila memungkinkan.</li>
            <li>Usahakan background kontras agar daun mudah dipisahkan dari latar.</li>
            <li>Ambil lebih dari 1 foto bila perlu (pilih yang paling jelas).</li>
          </ul>
        ),
      },
      {
        id: "sad",
        title: "Apa itu SAD (Standard Area Diagrams)?",
        content: (
          <>
            <p style={{ lineHeight, marginTop: 0 }}>
              <b>SAD (Standard Area Diagrams)</b> adalah pendekatan visual yang membantu menafsirkan
              <b> persentase keparahan</b> (area terinfeksi dibanding area daun) dengan membandingkannya
              terhadap <i>diagram/kelas referensi</i> yang digunakan luas dalam penilaian penyakit tanaman.
            </p>
            <p style={{ lineHeight, marginBottom: 0 }}>
              Di sistem ini, keparahan ditampilkan sebagai <b>persentase</b> dan <b>kelas SAD</b>{" "}
              (misalnya skema Horsfall–Barratt 12 kelas) agar interpretasi lebih konsisten dan transparan.
            </p>

            <div style={{ marginTop: 10, fontSize: 13, color: "#6b7280", lineHeight: 1.6 }}>
              Referensi bacaan:
              <ul style={{ marginTop: 6, paddingLeft: 18, marginBottom: 0 }}>
                <li>
                  Ringkasan ilmiah tentang SAD (Phytopathology, 2017):{" "}
                  <a
                    href="https://doi.org/10.1094/PHYTO-02-17-0069-FI"
                    target="_blank"
                    rel="noreferrer"
                  >
                    DOI: 10.1094/PHYTO-02-17-0069-FI
                  </a>
                </li>
                <li>
                  Halaman publikasi (USDA ARS):{" "}
                  <a
                    href="https://www.ars.usda.gov/research/publications/publication/?seqNo115=342880"
                    target="_blank"
                    rel="noreferrer"
                  >
                    USDA ARS Publication
                  </a>
                </li>
              </ul>
            </div>
          </>
        ),
      },
      {
        id: "baca",
        title: "Cara Membaca Hasil Analisis",
        content: (
          <ul style={{ lineHeight: 1.8, paddingLeft: 18, marginTop: 0, marginBottom: 0 }}>
            <li>
              <b>Label Penyakit</b>
              <br />
              Hasil klasifikasi jenis penyakit dari foto daun.
            </li>
            <li>
              <b>Keyakinan (Confidence)</b>
              <br />
              Seberapa yakin model terhadap label yang diprediksi.
            </li>
            <li>
              <b>Segmentasi Area Terinfeksi</b>
              <br />
              Visualisasi area yang diprediksi terinfeksi pada daun (overlay).
            </li>
            <li>
              <b>Estimasi Tingkat Keparahan</b>
              <br />
              Ditampilkan sebagai <b>persentase</b> (area terinfeksi dibanding area daun) dan <b>kelas SAD</b>{" "}
              untuk membantu interpretasi tingkat keparahan secara visual.
            </li>
          </ul>
        ),
      },
      {
        id: "limitasi",
        title: "Kapan Hasil Bisa Kurang Akurat?",
        content: (
          <>
            <ul style={{ lineHeight: 1.8, paddingLeft: 18, marginTop: 0 }}>
              <li>Foto terlalu gelap atau buram.</li>
              <li>Objek utama terlalu kecil di foto.</li>
              <li>Background terlalu ramai atau warnanya mirip daun.</li>
              <li>Ada pantulan cahaya kuat/bayangan keras.</li>
              <li>
                Dalam satu foto terdapat lebih dari satu jenis penyakit atau gejala campuran pada objek
                utama (model dilatih dengan satu label per gambar).
              </li>
              <li>
                Terdapat banyak objek/tanaman dalam satu frame sehingga area gejala sulit dipisahkan.
              </li>
            </ul>
            <p style={{ color: "#6b7280", marginBottom: 0 }}>
              Jika hasil kurang sesuai, ulangi dengan foto yang lebih jelas dan mengikuti checklist.
            </p>
          </>
        ),
      },
    ],
    [lineHeight]
  );

  const allIds = useMemo(() => items.map((x) => x.id), [items]);

  // Desktop initial: open all
  const [openAll, setOpenAll] = useState(!isMobile);

  // Mobile initial: buka "tujuan"
  const [openIds, setOpenIds] = useState(isMobile ? ["tujuan"] : []);

  const isSectionOpen = (id) => (openAll ? true : openIds.includes(id));

  const onToggle = (id) => {
    if (!isMobile && openAll) {
      setOpenAll(false);
      setOpenIds(allIds.filter((x) => x !== id));
      return;
    }

    setOpenIds((prev) => {
      if (isMobile) {
        return prev[0] === id ? [] : [id];
      }
      return prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id];
    });
  };

  const toggleOpenAllDesktop = () => {
    setOpenAll((prev) => {
      const next = !prev;

      if (next) {
        return true;
      }

      setOpenIds(["tujuan"]);
      return false;
    });
  };

  {!isMobile && (
    <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 12 }}>
      <Button variant="secondary" onClick={toggleOpenAllDesktop}>
        {openAll ? "Tutup semua" : "Buka semua"}
      </Button>
    </div>
  )}

  return (
    <div style={{ maxWidth: 900, margin: "0 auto" }}>
      {/* Desktop-only: Buka semua / Tutup semua */}
      {!isMobile && (
        <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 12 }}>
          <Button variant="secondary" onClick={toggleOpenAllDesktop}>
            {openAll ? "Tutup semua" : "Buka semua"}
          </Button>
        </div>
      )}

      {items.map((it, idx) => (
        <React.Fragment key={it.id}>
          <AccordionSection
            id={it.id}
            title={it.title}
            isOpen={isSectionOpen(it.id)}
            onToggle={onToggle}
            isMobile={isMobile}
          >
            {it.content}
          </AccordionSection>

          {idx < items.length - 1 ? <div style={{ height: sectionGap }} /> : null}
        </React.Fragment>
      ))}

      <div style={{ height: sectionGap }} />

      {/* CTA (selalu tampil) */}
      <Card>
        <div
          style={{
            display: "flex",
            flexDirection: isMobile ? "column" : "row",
            gap: 12,
          }}
        >
          <Button onClick={() => nav("/classify")}>Mulai Analisis</Button>
          <Button
            onClick={() => nav("/history")}
            style={{
              background: "#f3f4f6",
              color: "#111827",
            }}
          >
            Lihat Riwayat Analisis
          </Button>
        </div>
      </Card>
    </div>
  );
};

export default GuidePage;
