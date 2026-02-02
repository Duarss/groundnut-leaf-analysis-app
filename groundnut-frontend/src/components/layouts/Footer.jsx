// src/components/layouts/Footer.jsx
import React from "react";

const Footer = () => {
  const year = new Date().getFullYear();

  return (
    <footer className="footer">
      <div className="footer-inner">
        <span>© {year} Groundnut Leaf Analyzer</span>
        <span>Universitas Surabaya</span>
        <span>TA - 160422035</span>
      </div>
    </footer>
  );
};

export default Footer;
