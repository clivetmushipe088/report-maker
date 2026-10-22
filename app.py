"""GSS Stuffing Report System - prototype backend (FastAPI + SQLite + reportlab)."""
import os, io, re, json, base64, sqlite3
from xml.sax.saxutils import escape
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from PIL import Image as PILImage, ImageOps
import pytesseract
import extract
from fastapi.staticfiles import StaticFiles
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, Flowable

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.environ.get("DB_PATH", os.path.join(HERE, "gss.db"))
GOLD, NAVY = colors.HexColor("#c9a24b"), colors.HexColor("#0f1f33")
LAB = {"empty": "Empty container, doors open", "closed": "Doors closed, container no.", "lin1": "Paper lining being installed",
       "lin2": "Lining done, doors closed", "start": "Loading started", "full": "Fully loaded", "door1": "One door closed, container no.",
       "closedfull": "Both doors closed", "seal": "Door with seals", "sealcu": "Seal close-up"}
app = FastAPI(title="GSS Reports")

def db():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
    c.execute("create table if not exists jobs(id text primary key, container text, client text, surveyor text, status text, updated text default current_timestamp)")
    return c

@app.get("/api/jobs")
def list_jobs():
    with db() as c:
        return [dict(r) for r in c.execute("select * from jobs order by updated desc")]

@app.post("/api/jobs")
def save_job(j: dict):
    with db() as c:
        c.execute("insert into jobs(id,container,client,surveyor,status) values(?,?,?,?,?) on conflict(id) do update set "
                  "container=excluded.container, client=excluded.client, surveyor=excluded.surveyor, status=excluded.status, updated=current_timestamp",
                  (str(j["id"]), j.get("container", ""), j.get("client", ""), j.get("surveyor", ""), j.get("status", "Draft")))
    return {"ok": True}

class Stamp(Flowable):
    def __init__(s, d): super().__init__(); s.d = d; s.width = s.height = 36 * mm
    def draw(s):
        c = s.canv; c.saveState(); c.translate(18 * mm, 18 * mm); c.rotate(12)
        c.setStrokeColor(colors.HexColor("#2f5d8c")); c.setFillColor(colors.HexColor("#2f5d8c")); c.setLineWidth(1.4)
        c.circle(0, 0, 16 * mm); c.circle(0, 0, 14 * mm); c.setFont("Helvetica-Bold", 7.5)
        c.drawCentredString(0, 5, "GLOBAL SAFEZONE"); c.drawCentredString(0, -3, "E-STAMP"); c.drawCentredString(0, -11, s.d); c.restoreState()

def pic(b, w, h):
    iw, ih = ImageReader(io.BytesIO(b)).getSize(); k = min(w / iw, h / ih)
    return Image(io.BytesIO(b), iw * k, ih * k)

ASSETS = os.path.join(HERE, "static", "assets")
HDR, FTR = ImageReader(os.path.join(ASSETS, "header.png")), ImageReader(os.path.join(ASSETS, "footer.png"))
HDR_H, FTR_H, BODY_W = 180 * mm * 169 / 1055, 180 * mm * 94 / 1053, 180 * mm

def page_plain(c, d):
    """Company header and footer banners only (used for pages added to an existing report)."""
    c.saveState()
    c.drawImage(HDR, 15 * mm, 297 * mm - 6 * mm - HDR_H, BODY_W, HDR_H, mask="auto")
    c.drawImage(FTR, 15 * mm, 6 * mm, BODY_W, FTR_H, mask="auto")
    c.restoreState()

def page(c, d):
    """Company header and footer banners on every page, with a page number."""
    page_plain(c, d)
    c.saveState(); c.setFont("Helvetica", 8); c.setFillColor(colors.grey); c.drawRightString(195 * mm, 6 * mm + FTR_H + 2 * mm, f"Page {d.page}")
    c.restoreState()

@app.post("/api/extract/{kind}")
def extract_doc(kind: str, file: UploadFile = File(...), expected_bags: int = Form(0)):
    """Read a document locally (Tesseract + rules). kind: order | tally | insp. Photos and PDFs both work."""
    if kind not in ("order", "tally", "insp"):
        raise HTTPException(404, "Unknown document type.")
    raw = file.file.read()
    if not raw:
        raise HTTPException(400, "The uploaded file is empty. Re-upload the document.")
    try:
        r = extract.run(kind, raw, file.filename or "", {"expected_bags": expected_bags or None})
    except pytesseract.TesseractNotFoundError as exc:
        raise HTTPException(503, "Tesseract is not installed or not on PATH. On macOS run: brew install tesseract, then restart the server.") from exc
    except Exception as exc:
        raise HTTPException(400, f"This file could not be read ({type(exc).__name__}: {str(exc)[:160]}). Try a clear JPG, PNG or PDF.") from exc
    r["preview"] = base64.b64encode(r["preview"]).decode()
    return r

@app.post("/api/exif")
def photo_times(files: list[UploadFile] = File(...)):
    """Camera capture time of each photo (EXIF), used to put a batch of photos in shoot-list order."""
    out = []
    for f in files:
        ts = ""
        try:
            ex = PILImage.open(f.file).getexif()
            ts = str(ex.get_ifd(0x8769).get(36867) or ex.get(306) or "")
        except Exception:
            pass
        out.append({"name": f.filename, "ts": ts})
    return out

def shrink(b, name=""):
    """Re-encode a photo or scan as a bounded JPEG so the PDF stays small enough to email."""
    try:
        im = ImageOps.exif_transpose(PILImage.open(io.BytesIO(b))).convert("RGB")
        im.thumbnail((1800, 1800)); o = io.BytesIO(); im.save(o, "JPEG", quality=82); return o.getvalue()
    except Exception:
        return extract.jpeg(extract.load_image(b, name))

def evidence(F, stamp, date):
    """Pages that follow the report text: photographic record, signed tally sheet, inspection checklist."""
    st = getSampleStyleSheet(); N = ParagraphStyle("n", parent=st["Normal"], fontSize=9, leading=12)
    B = ParagraphStyle("b", parent=N, fontName="Helvetica-Bold"); H = ParagraphStyle("h", parent=st["Title"], fontSize=20, textColor=NAVY)
    P = lambda t, s=N: Paragraph(escape(str(t or "")), s)
    secs = []
    items = [[pic(F["photo_" + k], 84 * mm, 62 * mm), P(LAB[k], B)] for k in LAB if "photo_" + k in F]
    if items:
        rows = [[[x[0], x[1]] for x in items[i:i + 2]] + [""] * (2 - len(items[i:i + 2])) for i in range(0, len(items), 2)]
        ph = Table(rows, colWidths=[90 * mm, 90 * mm]); ph.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
        secs.append([P("PHOTOGRAPHIC RECORD", H), ph])
    for k, title in (("scan_tally", "LOADING TALLY SHEET"), ("scan_insp", "CONTAINER INSPECTION CHECKLIST")):
        if k in F:
            sec = [P(title, H), pic(F[k], 175 * mm, 158 * mm)]
            if stamp: sec += [Spacer(1, 3 * mm), Stamp(date)]
            secs.append(sec)
    out = []
    for n, sec in enumerate(secs): out += ([PageBreak()] if n else []) + sec
    return out

@app.post("/api/complete/inspect")
def complete_inspect(file: UploadFile = File(...)):
    """Open an existing stuffing report PDF: page count, container number from its text, first-page preview."""
    raw = file.file.read()
    if raw[:4] != b"%PDF": raise HTTPException(400, "Please upload the stuffing report as a PDF.")
    try:
        import pymupdf
        doc = pymupdf.open(stream=raw, filetype="pdf")
        text = "\n".join(pg.get_text() for pg in doc)
        m = [x.replace(" ", "") for x in re.findall(r"[A-Z]{4}\s?\d{7}", text)]
        cont = next((x for x in m if extract.iso_ok(x)), m[0] if m else "")
        prev = base64.b64encode(doc[0].get_pixmap(dpi=60).tobytes("jpeg")).decode()
        return {"pages": len(doc), "container": cont, "preview": prev}
    except Exception as exc:
        raise HTTPException(400, f"That PDF could not be opened ({type(exc).__name__}). Check the file is not damaged or password protected.") from exc

@app.post("/api/complete")
def complete(report: UploadFile = File(...), files: list[UploadFile] = File(default=[]), stamp: int = Form(0), date: str = Form(""), keep: int = Form(0)):
    """Insert photos, tally sheet and checklist after an existing stuffing report and return one PDF."""
    import pymupdf
    raw = report.file.read()
    if raw[:4] != b"%PDF": raise HTTPException(400, "Please upload the stuffing report as a PDF.")
    F = {f.filename: shrink(f.file.read(), f.filename) for f in files}
    ev = evidence(F, bool(stamp), date)
    if not ev: raise HTTPException(400, "Add at least one photo, the tally sheet or the checklist.")
    buf = io.BytesIO(); SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=42 * mm, bottomMargin=30 * mm,
                                         title="Inserted pages").build(ev, onFirstPage=page_plain, onLaterPages=page_plain)
    doc = pymupdf.open(stream=raw, filetype="pdf")
    if 0 < keep < len(doc): doc.select(list(range(keep)))
    doc.insert_pdf(pymupdf.open(stream=buf.getvalue(), filetype="pdf"))
    return Response(doc.tobytes(garbage=3, deflate=True), media_type="application/pdf")

@app.post("/api/report")
def report(job: str = Form(...), files: list[UploadFile] = File(default=[])):
    J = json.loads(job); F = {f.filename: shrink(f.file.read(), f.filename) for f in files}
    o, R, T, I = J["order"], J.get("rep", {}), J.get("tally", {}), J.get("insp", {})
    lots = o.get("lots", []); ob = sum(int(l[2]) for l in lots); tb = sum(int(x) for x in T.get("rows", [])) or ob
    kg = int(T.get("bagKg") or 60); net = sum(int(l[3]) for l in lots) or tb * kg
    gross = int(T.get("gross") or 0) or sum(int(l[4]) for l in lots)
    st = getSampleStyleSheet(); N = ParagraphStyle("n", parent=st["Normal"], fontSize=9, leading=12)
    B = ParagraphStyle("b", parent=N, fontName="Helvetica-Bold"); H = ParagraphStyle("h", parent=st["Title"], fontSize=20, textColor=NAVY)
    P = lambda t, s=N: Paragraph(escape(str(t or "")), s)
    grid = TableStyle([("GRID", (0, 0), (-1, -1), .5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3ecd9")),
                       ("VALIGN", (0, 0), (-1, -1), "TOP"), ("FONTSIZE", (0, 0), (-1, -1), 9)])
    s = [P("STUFFING REPORT", H)]
    desc = f"1x20ST of {tb} bags of {kg}kg each of {R.get('goods', '')}. Lots: {', '.join(l[0] for l in lots)}. Gross weight {gross:,} kgs, net weight {net:,} kgs."
    kv = [("VESSEL", R.get("vessel")), ("DESCRIPTION OF GOODS", desc), ("QUANTITY", f"{tb} bags"), ("PACKING", "In bags"), ("COUNTRY OF ORIGIN", "Rwanda"),
          ("SHIPPER", R.get("shipper")), ("CONSIGNEE", R.get("consignee")), ("NOTIFY PARTY", R.get("notify")), ("BILL OF LADING NO.", R.get("bl")),
          ("FROM / TO", f"{R.get('from', '')} / {R.get('to', '')}")]
    t = Table([[P(a, B), P(b)] for a, b in kv], colWidths=[48 * mm, 132 * mm]); t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")])); s += [t, Spacer(1, 4 * mm)]
    s.append(P("We supervised stuffing through tallying and weighing, and below are our findings:"))
    tt = Table([[P(x, B) for x in ("No", "Container", "Type", "Customs seal", "Shipper seal", "Lot numbers", "Bags")],
                [P(1), P(o.get("container")), P("20FT"), P(T.get("customs")), P(T.get("shipperSeal")), P("\n".join(l[0] for l in lots)), P(ob)]],
               colWidths=[10 * mm, 30 * mm, 14 * mm, 28 * mm, 28 * mm, 50 * mm, 20 * mm]); tt.setStyle(grid); s += [Spacer(1, 2 * mm), tt, Spacer(1, 3 * mm)]
    lt = Table([[P(x, B) for x in ("Lot", "Client", "Bags", "Net kg", "Gross kg")]] + [[P(l[0]), P(l[1]), P(l[2]), P(l[3]), P(l[4])] for l in lots] +
               [[P("Total", B), "", P(ob, B), P(sum(int(l[3]) for l in lots), B), P(sum(int(l[4]) for l in lots), B)]],
               colWidths=[40 * mm, 70 * mm, 20 * mm, 25 * mm, 25 * mm]); lt.setStyle(grid); s += [lt, Spacer(1, 3 * mm)]
    s.append(P(f"Remarks: All bags were stuffed in apparently good and sound condition. Stuffing took place on {T.get('date', '')}, {T.get('time', '')}."))
    s += [Spacer(1, 4 * mm), P(f"Place of intervention: Kigali, Rwanda. Surveyor: {J.get('surveyor', '')}. Date of issue: {T.get('date', '')}."), Spacer(1, 4 * mm)]
    if J.get("stamp"): s.append(Stamp(T.get("date", "")))
    ev = evidence(F, J.get("stamp"), T.get("date", ""))
    if ev: s += [PageBreak()] + ev
    buf = io.BytesIO(); SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=42 * mm, bottomMargin=30 * mm,
                                         title="Stuffing Report").build(s, onFirstPage=page, onLaterPages=page)
    return Response(buf.getvalue(), media_type="application/pdf")

app.mount("/", StaticFiles(directory=os.path.join(HERE, "static"), html=True), name="static")
