# app/database/init_db.py
from app.database.db import Base, engine
from app.models.analysis_result import AnalysisResult

def main():
    print("Creating tables...")
    Base.metadata.create_all(bind=engine)
    print("Done.")

if __name__ == "__main__":
    main()
