# tools/ (CLI pipeline)

Semua script di folder ini diasumsikan dijalankan dari folder `groundnut-backend/` (working directory).
Default path:
- Raw dataset: `datasets/raw/cleaned_ori_dataset`
- Output split: `datasets/processed/<task>_dataset`
- Orphans report: `datasets/results/orphans_report`

Catatan penting:
- `tools/segmentation_preprocessing.py` akan FAIL (raise RuntimeError) jika menemukan orphans pada `datasets/results/orphans_report`
  (default `--fail_on_orphans 1`), supaya dataset konsisten sebelum preprocessing.
