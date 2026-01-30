// src/App.jsx
import React from "react";
import { Routes, Route } from "react-router-dom";
import AppLayout from "./components/layouts/AppLayout";

import HomePage from "./pages/HomePage";
import ClassifyPage from "./pages/ClassifyPage";
import SegmentPage from "./pages/SegmentPage";
import AnalysisPage from "./pages/AnalysisPage";
import GuidePage from "./pages/GuidePage";
import HistoryPage from "./pages/HistoryPage";
import HistoryDetailPage from "./pages/HistoryDetailPage";

const App = () => {
  return (
    <AppLayout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/classify" element={<ClassifyPage />} />
        <Route path="/segment/:id" element={<SegmentPage />} />
        {/* detail analisis (segmentasi + keparahan) untuk 1 hasil */}
        <Route path="/analysis/:id" element={<AnalysisPage />} />
        <Route path="/guide" element={<GuidePage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/history/:id" element={<HistoryDetailPage />} />
      </Routes>
    </AppLayout>
  );
};

export default App;