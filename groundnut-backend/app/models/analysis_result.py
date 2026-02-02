# app/models/analysis_result.py
from sqlalchemy import Column, String, Integer, Float, Text, Boolean
from app.database.db import Base

class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    analysis_id = Column(String(64), primary_key=True, index=True)
    created_at = Column(Integer, nullable=False)

    # image refs
    orig_image_path = Column(Text, nullable=False)

    # classification
    label = Column(String(128), nullable=False)
    confidence = Column(Float, nullable=True)
    probs_json = Column(Text, nullable=True)

    # segmentation (optional)
    seg_enabled = Column(Boolean, nullable=False, default=False)
    seg_overlay_path = Column(Text, nullable=True)

    # severity (optional, hanya kalau segmentasi jalan)
    severity_pct = Column(Float, nullable=True)
    severity_fao_level = Column(Integer, nullable=True)
