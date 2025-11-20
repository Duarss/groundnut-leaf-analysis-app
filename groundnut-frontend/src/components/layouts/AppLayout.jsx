// src/components/layouts/AppLayout.jsx
import React from "react";
import Navbar from "./Navbar";
import Footer from "./Footer";

const AppLayout = ({ children }) => {
  return (
    <div className="app-root">
      <Navbar />
      <main className="app-main">{children}</main>
      <Footer />
    </div>
  );
};

export default AppLayout;