# GSS Stuffing Report System (fully local)

Everything runs on your own computer: no AI service, no API key, no internet connection at run time.
Documents are read with Tesseract OCR plus rules in `extract.py`. Reports are built with reportlab.

## Install and run (macOS)

```bash
brew install tesseract
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```
Open http://127.0.0.1:8000. (Windows/Linux: install Tesseract from your package manager, then the same pip steps.)

## What is read, and how well

| Document | How it is read | What to expect |
|---|---|---|
| **Loading order** (printed, photo or PDF) | OCR several ways, table rebuilt row by row, rows merged by vote, then **checked against the order's own total line** | Lots, bags, net and gross, container (ISO 6346 check digit), booking, transporter, driver, truck, border, port. A struck-through or smudged weight is recalculated from the bag count and confirmed against the total; anything inferred is flagged in the Note column. Rows that do not add up to the stated total are flagged, never silently dropped. |
| **Tally sheet** (handwritten) | Printed labels are located; each handwritten value is **cropped and shown beside its input** | Tesseract cannot read pen handwriting reliably, so these values are typed by the surveyor while looking at the crop. The container and date are pre-filled only if they pass validation. Seal numbers are never guessed. The six row totals must add up to the loading order's bag count (checked live). |
| **Inspection checklist** (handwritten) | Pass/Fail ticks are **measured from the ink**; container, seal and date are cropped for typing | The seven ticks are read automatically. |

Photos: "Upload all photos" puts a batch into the shoot-list order using the **camera capture time (EXIF)**, falling back to file date. Check every slot before generating the report.

Reports: **Download PDF** builds the stuffing report (details, lots table, seals, photo record, tally and checklist scans, optional e-stamp). Photos are shrunk so the PDF stays small enough to email.

## Complete an existing report

For reports whose text is already written (typed or exported elsewhere): open **Complete a report** in the top bar.

1. Upload the stuffing report PDF. The app shows its page count and picks up the container number from its text.
2. Add the photos (bulk upload sorts them by camera time into the shoot-list slots) and the signed tally sheet and checklist (photo or PDF).
3. Tick whether to apply the e-stamp to the tally and checklist pages, then **Insert and export PDF**.

The photographic record, tally sheet and checklist are added after the report, each page with the company header and footer. Your original pages are untouched. If the PDF already contains old or blank photo pages, set "Keep only the first N pages" so the new pages replace them. The job is added to the tracker as "Report ready". These documents are inserted as they are and are not read.

## Tuning for your business

Edit `reference.json`: bag weight (default 60 kg), tare per bag (1 kg), known container owner codes, ports, borders and locations. OCR output is snapped to these lists.

## Notes

- Data is stored in `gss.db` (SQLite) in this folder; set `DB_PATH` to move it.
- There is no login. Do not expose the server beyond your own network without adding access control.
- Only the first page of a PDF is read.
- Sending email from the app is not built yet; download the PDF and send it as you do today.
