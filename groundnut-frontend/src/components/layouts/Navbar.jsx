// src/components/layouts/Navbar.jsx
import React, { useState } from "react";
import { NavLink } from "react-router-dom";

const Navbar = () => {
  const [isMobileOpen, setIsMobileOpen] = useState(false);

  const toggleMenu = () => {
    setIsMobileOpen((prev) => !prev);
  };

  const closeMenu = () => {
    setIsMobileOpen(false);
  };

  return (
    <header className="navbar">
      <div className="navbar-inner">
        <div className="navbar-brand">
          <span className="brand-logo">
            <img src="/src/assets/app-logo.svg" alt="App Logo" width="30" height="30" />
          </span>
          <div className="brand-text">
            <div className="brand-title">Groundnut Leaf Analyzer</div>
            <div className="brand-subtitle">Classify + Segment + Estimate Severity + Monitor</div>
          </div>
        </div>

        {/* Hamburger button (visible on mobile) */}
        <button
          className="navbar-toggle"
          type="button"
          onClick={toggleMenu}
          aria-label="Toggle navigation"
        >
          <span className="bar" />
          <span className="bar" />
          <span className="bar" />
        </button>

        {/* Links */}
        <nav
          className={`navbar-links ${
            isMobileOpen ? "navbar-links-open" : ""
          }`}
        >
          <NavLink to="/" end onClick={closeMenu}>
            Beranda
          </NavLink>
          <NavLink to="/classify" onClick={closeMenu}>
            Klasifikasi
          </NavLink>
          <NavLink to="/history" onClick={closeMenu}>
            Riwayat
          </NavLink>
          <NavLink to="/guide" onClick={closeMenu}>
            Panduan
          </NavLink>
        </nav>
      </div>
    </header>
  );
};

export default Navbar;
