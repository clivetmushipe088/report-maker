"""Offline document reading for the GSS report system.

Tesseract does the character recognition. The logic around it (table rebuilding, consistency
checks, check-digit repair, inference from business rules) lives here so that a misread cell is
repaired or flagged instead of silently dropping a lot. No network or AI service is used.
"""
import io, os, re, json, difflib
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np, cv2, pytesseract
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
REF = json.load(open(os.path.join(HERE, "reference.json"), encoding="utf-8"))
BAG_KG, TARE = int(REF.get("bag_kg", 60)), int(REF.get("tare_kg_per_bag", 1))

# ---------------------------------------------------------------- images
def load_image(raw: bytes, name: str = "") -> np.ndarray:
    """Return an RGB array. PDFs are rendered locally (first page); photos are rotated upright."""
    if raw[:4] == b"%PDF" or name.lower().endswith(".pdf"):
        import pymupdf
        doc = pymupdf.open(stream=raw, filetype="pdf")
        pm = doc[0].get_pixmap(dpi=250)
        im = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
    else:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    a = np.array(im)
    long = max(a.shape[:2])
    if long > 3200:
        a = cv2.resize(a, None, fx=3200 / long, fy=3200 / long, interpolation=cv2.INTER_AREA)
    elif long < 2400:
        k = 2400 / long
        a = cv2.resize(a, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    return deskew(a)

def deskew(a):
    """Straighten a slightly rotated page by maximising the sharpness of the text-line profile."""
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY); k = 900 / max(g.shape)
    sm = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else g
    bw = cv2.threshold(sm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]; h, w = bw.shape
    def score(ang):
        r = cv2.warpAffine(bw, cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1), (w, h)); p = r.sum(1).astype(float)
        return float(((p[1:] - p[:-1]) ** 2).sum())
    best = max(np.arange(-10, 10.01, 0.5), key=score)
    best = max(np.arange(best - 0.5, best + 0.51, 0.1), key=score)
    if abs(best) < 0.2: return a
    H, W = a.shape[:2]
    return cv2.warpAffine(a, cv2.getRotationMatrix2D((W / 2, H / 2), best, 1), (W, H), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))

def jpeg(a: np.ndarray, mx=1800, q=82) -> bytes:
    im = Image.fromarray(a)
    im.thumbnail((mx, mx))
    b = io.BytesIO(); im.save(b, "JPEG", quality=q); return b.getvalue()

def ink_mask(a):
    """Boolean mask of coloured (blue / purple / red) ink: pens, stamps. Printed text is neutral."""
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    return (hsv[..., 1] > 70) & (hsv[..., 2] > 40)

def printed_layer(a):
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    m = cv2.dilate(ink_mask(a).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    g = g.copy(); g[m] = 255
    return g

def pen_layer(a):
    """Handwriting-only image (dark on white) built from coloured ink."""
    m = ink_mask(a).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return 255 - m

def ocr(img, psm=6, wl=None):
    cfg = f"--psm {psm}" + (f" -c tessedit_char_whitelist={wl}" if wl else "")
    return pytesseract.image_to_string(img, config=cfg)

# ---------------------------------------------------------------- helpers
def num(s):
    s = str(s).strip()
    if re.fullmatch(r"\d{1,3}(?:[,\.]\d{3})+", s): return int(re.sub(r"[,\.]", "", s))
    return int(s) if s.isdigit() else None

NUM = r"\d{1,3}(?:[,\.]\d{3})+|\d+"
CONF_L = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G", "4": "A"}
CONF_D = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2", "G": "6", "T": "7", "A": "4"}

def iso_ok(c):
    c = re.sub(r"\s", "", c or "").upper()
    if not re.fullmatch(r"[A-Z]{4}\d{7}", c): return False
    v, m = 10, {}
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if v % 11 == 0: v += 1
        m[ch] = v; v += 1
    s = sum((m[c[i]] if i < 4 else int(c[i])) * 2 ** i for i in range(10))
    return (s % 11) % 10 == int(c[10])

def fix_container(raw):
    """-> (value, status). status: ok | corrected | suggested | unverified | ''."""
    s = re.sub(r"[^A-Z0-9]", "", (raw or "").upper())
    if len(s) < 8: return "", ""
    best = None
    for i in range(0, max(1, len(s) - 10)):
        w = s[i:i + 11]
        if len(w) < 11: break
        cand = "".join(CONF_L.get(ch, ch) if not ch.isalpha() else ch for ch in w[:4]) + "".join(CONF_D.get(ch, ch) for ch in w[4:])
        if re.fullmatch(r"[A-Z]{4}\d{7}", cand):
            if iso_ok(cand): return cand, ("ok" if cand == w else "corrected")
            best = best or cand
    if best:  # one confusable substitution that makes the check digit valid
        alt = {v: k for k, v in CONF_D.items()}
        for p in range(11):
            pool = {best[p]} | ({alt.get(best[p])} if p >= 4 and alt.get(best[p]) else set()) | ({CONF_D.get(best[p])} if p >= 4 and CONF_D.get(best[p]) else set())
            for ch in pool - {None}:
                c2 = best[:p] + ch + best[p + 1:]
                if c2 != best and iso_ok(c2): return c2, "corrected"
        return best, "unverified"
    digits = re.findall(r"\d{7}", s)  # letters unreadable: find an owner code that makes the digits valid
    if digits:
        letters = re.sub(r"[^A-Z]", "", s)[:4]
        c = [p + digits[0] for p in REF["container_owners"] if iso_ok(p + digits[0])]
        if c: return max(c, key=lambda x: difflib.SequenceMatcher(None, letters, x[:4]).ratio()), "suggested"
    return "", ""

def snap(val, options, cutoff=0.72):
    v = re.sub(r"\s+", " ", (val or "").upper()).strip(" .,:;|-")
    if not v or not options: return v
    m = difflib.get_close_matches(v, [o.upper() for o in options], 1, cutoff)
    return m[0] if m else v

def vote(vals):
    vals = [v for v in vals if v]
    return Counter(vals).most_common(1)[0][0] if vals else ""

def norm_date(s):
    m = re.search(r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{4})", s or "")
    return f"{int(m[1]):02d}/{int(m[2]):02d}/{m[3]}" if m else ""

# ---------------------------------------------------------------- loading order
LOT_RE = re.compile(r"([0-9OQ]\d{2})\s*[/\\|]\s*([0-9OQ]{3})\s*[/\\|]\s*([0-9OQ]{4,6})")
SEP = r"[\s|_\-\.\[\]\(\)]*"
TOT_RE = re.compile(rf"({NUM})\s*BAGS?{SEP}({NUM})\s*K\s*G\s*S?{SEP}({NUM})\s*K\s*G\s*S?", re.I)

def _lot(m): return "/".join(x.replace("O", "0").replace("Q", "0") for x in m.groups())

def parse_row(txt, bag_kg):
    t = txt.replace("|", " ")
    mb = re.search(rf"({NUM})\s*BAGS?", t, re.I)
    bags = num(mb[1]) if mb else None
    client = re.sub(r"[^A-Za-z ]", " ", t[:mb.start()] if mb else re.split(r"\d", t)[0])
    client = re.sub(r"\s+", " ", client).strip().upper()
    vals = [num(m[1]) for m in re.finditer(rf"({NUM})\s*K\s*G\s*S?", t, re.I)]
    if bags:  # weights that cannot belong to this row (e.g. a total from the next line) are ignored
        vals = [v for v in vals if v and 30 * bags <= v <= 110 * bags]
    n_kg = len(re.findall(r"K\s*G\s*S", t, re.I))
    net = gross = None; flags = []
    for v in vals:
        if bags and v == bags * bag_kg and net is None: net = v
        elif bags and v == bags * (bag_kg + TARE) and gross is None: gross = v
    rest = [v for v in vals if v not in (net, gross)]
    if net is None and gross is None and len(rest) >= 2: net, gross = rest[0], rest[1]
    elif rest and (net is None or gross is None):
        v = rest[0]
        if net is None and gross is None:
            if bags and abs(v - bags * bag_kg) <= abs(v - bags * (bag_kg + TARE)) and n_kg < 2: net = v
            elif n_kg >= 2: gross = v      # one of two weights unreadable: keep the one we see as gross if it is the larger
            else: net = v
        elif net is None: net = v
        else: gross = v
    if bags is None and net and net % bag_kg == 0: bags = net // bag_kg; flags.append("bags")
    if net is None and bags: net = bags * bag_kg; flags.append("net")
    if gross is None and bags and net: gross = net + bags * TARE; flags.append("gross")
    return {"client": client, "bags": bags, "net": net, "gross": gross, "flags": flags}

def parse_order_text(text, bag_kg=BAG_KG):
    lines = [re.sub(r"\s+", " ", l.replace(":", " ")).strip() for l in text.splitlines()]
    lines = [l for l in lines if l]
    rows, totals, cur = [], None, None
    for l in lines:
        mt = TOT_RE.search(l)
        if mt and not LOT_RE.search(l):
            totals = (num(mt[1]), num(mt[2]), num(mt[3])); cur = None; continue
        if not LOT_RE.search(l) and re.search(r"\d\s*BAGS", l, re.I) and len(re.findall(rf"({NUM})\s*K\s*G\s*S", l, re.I)) >= 2:
            cur = None; continue
        ml = LOT_RE.search(l)
        if ml:
            cur = {"lot": _lot(ml), "txt": l[ml.end():]}; rows.append(cur); continue
        if cur is not None and re.search(r"K\s*G\s*S|BAGS?", l, re.I) and not re.match(r"(transporter|driver|truck|container|booking|border|port|tel)", l, re.I):
            cur["txt"] += " " + l
        else: cur = None
    lots = []
    for r in rows:
        p = parse_row(r["txt"], bag_kg); p["lot"] = r["lot"]; lots.append(p)
    return lots, totals

def order_fields(texts):
    def grab(pat, post=lambda x: x):
        out = []
        for t in texts:
            m = re.search(pat, t, re.I)
            if m: out.append(post(re.sub(r"\s+", " ", m[1]).strip(" .,:;|_-")))
        return vote(out)
    cont = vote([fix_container(m[1])[0] for t in texts for m in [re.search(r"Container\s*No\.?\s*[:\-]?\s*([A-Z0-9 ]{9,16})", t, re.I)] if m])
    status = fix_container(cont)[1] if cont else ""
    return {
        "container": cont, "container_status": status,
        "booking": grab(r"Booking\s*[:\-]?\s*([A-Z0-9]{8,})", str.upper),
        "transporter": grab(r"Transporter\s*[:\-]?\s*([^\n]+)", str.upper),
        "driver": grab(r"Driver\s*Name\s*[:\-]?\s*([^\n]+)", str.upper),
        "truck": grab(r"Truck\s*No\.?\s*[:\-]?\s*([A-Z0-9 /]+)", lambda x: x.upper()),
        "border": snap(grab(r"BORDER\s*[:\-]?\s*([^\n]+)"), REF["borders"]),
        "pod": snap(grab(r"Port\s*Of\s*Discharge\s*[:\-]?\s*([^\n]+)"), REF["ports"], 0.6),
        "date": norm_date(grab(r"Date\s*[:\-]?\s*([0-9/ .\-]{8,12})")),
    }

def extract_order(a, hints=None):
    pr, gray = printed_layer(a), cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    jobs = [(pr, 6), (pr, 4), (gray, 6), (gray, 4)]
    with ThreadPoolExecutor(4) as ex: texts = list(ex.map(lambda j: ocr(j[0], j[1]), jobs))
    parsed = [parse_order_text(t) for t in texts]
    tvote = Counter(tt for _, tt in parsed if tt)
    totals = tvote.most_common(1)[0][0] if tvote else None
    # merge per lot number: take the candidate with the fewest inferred fields, then the majority
    clusters = []   # a lot number misread in one pass ("028/695/26016") joins its true cluster
    for vi, (lots, _) in enumerate(parsed):
        for l in lots:
            l["v"] = vi
            for c in clusters:   # same lot = near-identical number, same bag count, and never two rows from one pass
                if (all(x["v"] != vi for x in c) and difflib.SequenceMatcher(None, c[0]["lot"], l["lot"]).ratio() >= 0.9
                        and (c[0]["bags"] == l["bags"] or c[0]["bags"] is None or l["bags"] is None or "bags" in c[0]["flags"] or "bags" in l["flags"])):
                    c.append(l); break
            else: clusters.append([l])
    merged = []
    for cands in clusters:
        lot = vote([c["lot"] for c in cands])
        best = min(cands, key=lambda c: (sum(c[k] is None for k in ("bags", "net", "gross")), len(c["flags"]), -sum(1 for d in cands if (d["bags"], d["net"], d["gross"]) == (c["bags"], c["net"], c["gross"]))))
        l = dict(best); l["lot"] = lot; l["client"] = vote([c["client"] for c in cands]) or best["client"]
        merged.append(l)
    pre = Counter(l["lot"].rsplit("/", 1)[0] for lots, _ in parsed for l in lots if l["lot"].count("/") == 2)
    if pre:   # lots on one order share their prefix, so a one-character misread in it is repaired
        mp = pre.most_common(1)[0][0]
        for l in merged:
            p, _, suf = l["lot"].rpartition("/")
            if p != mp and difflib.SequenceMatcher(None, p, mp).ratio() >= 0.85: l["lot"] = mp + "/" + suf
    merged.sort(key=lambda l: l["lot"])
    for l in merged:
        for k in ("bags", "net", "gross"):
            if l[k] is None: l[k] = 0; l["flags"] = l["flags"] + ["unreadable"]
    warnings = []
    if totals and sum(l["bags"] for l in merged) > totals[0]:   # an extra row (a misread lot number) that the total disproves
        over = sum(l["bags"] for l in merged) - totals[0]
        cand = [l for l in merged if l["bags"] == over and (l["flags"] or any(o is not l and difflib.SequenceMatcher(None, o["lot"], l["lot"]).ratio() >= 0.75 for o in merged))]
        if len(cand) == 1:
            merged.remove(cand[0]); warnings.append(f"A row for lot {cand[0]['lot']} ({over} bags) looked like a misread duplicate and was removed, because the lots would otherwise exceed the order total.")
    if totals:
        tb, tn, tg = totals
        unk = [l for l in merged if l["flags"]]
        if len(unk) == 1:  # one doubtful row: solve it from the stated totals
            l = unk[0]; ob = sum(x["bags"] for x in merged if x is not l); on = sum(x["net"] for x in merged if x is not l); og = sum(x["gross"] for x in merged if x is not l)
            if tb - ob > 0 and tn - on == (tb - ob) * BAG_KG and tg - og == (tn - on) + (tb - ob) * TARE:
                same = (l["bags"], l["net"], l["gross"]) == (tb - ob, tn - on, tg - og)
                l["bags"], l["net"], l["gross"] = tb - ob, tn - on, tg - og
                l["flags"] = ["calculated, matches order total"] if same else ["from order total"]
        sb = sum(l["bags"] for l in merged)
        if sb < tb:
            miss = tb - sb
            merged.append({"lot": "", "client": merged[-1]["client"] if merged else "", "bags": miss, "net": miss * BAG_KG, "gross": miss * (BAG_KG + TARE), "flags": ["lot missing"]})
            warnings.append(f"Lot rows added up to {sb} bags but the order total is {tb}. A row of {miss} bags was added: enter its lot number.")
        elif sb > tb:
            warnings.append(f"Lot rows add up to {sb} bags but the order total says {tb}. Check the lots.")
    elif merged: warnings.append("The order total line was not found, so the lot rows could not be cross-checked.")
    for l in merged:
        if l["flags"] == ["calculated, matches order total"]:
            warnings.append(f"Lot {l['lot']}: a weight was unreadable (struck through or overwritten?). It was calculated from {l['bags']} bags and confirmed by the order total.")
        elif l["flags"] and l["flags"] != ["from order total"] and l["flags"] != ["lot missing"]:
            warnings.append(f"Lot {l['lot']}: {', '.join(l['flags'])} could not be read and was calculated from the bag count. Check it against the paper.")
        elif l["flags"] == ["from order total"]:
            warnings.append(f"Lot {l['lot']}: figures were unreadable and were solved from the order total. Check it against the paper.")
    for l in merged:
        if l["bags"] and l["net"] and l["net"] != l["bags"] * BAG_KG and not l["flags"]:
            warnings.append(f"Lot {l['lot']}: net weight {l['net']} kg is not {l['bags']} bags x {BAG_KG} kg. Check the paper.")
    f = order_fields(texts)
    if f["container"] and not iso_ok(f["container"]): warnings.append(f"Container {f['container']} does not pass the ISO 6346 check digit. Check it against the paper.")
    f["lots"] = [[l["lot"], l["client"], l["bags"], l["net"], l["gross"], ",".join(l["flags"])] for l in merged]
    f["totals"] = {"bags": totals[0], "net": totals[1], "gross": totals[2]} if totals else None
    return f, warnings, "\n\n".join(texts[:2]), {}

# ---------------------------------------------------------------- handwritten forms (tally / checklist)
# Tesseract cannot read pen handwriting reliably, so these forms are handled differently from the printed loading
# order: the printed labels are located, ticks are measured from ink, and every handwritten value is cropped out
# and returned as an image so it can be shown right beside its input. A handwritten value is only pre-filled when
# it passes a real validation (container check digit, valid date). Seal numbers have no check digit, so they are
# never guessed.
def _words(img, scale=2):
    big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    d = pytesseract.image_to_data(big, config="--psm 6", output_type=pytesseract.Output.DICT)
    out = []
    for i in range(len(d["text"])):
        raw = d["text"][i].strip()
        if not raw: continue
        x, y, w, h = (d[k][i] // scale for k in ("left", "top", "width", "height"))
        m = re.match(r"([A-Za-z']{3,})[.\u2026_:]{2,}", raw)   # "DATE......" runs the label into its dotted leader
        if m: w = int(w * len(m[1]) / len(raw)); raw = m[1]
        out.append((raw.upper().strip(":.\u2026_"), x, y, w, h))
    return out

def _anchor(words, label, nth=0):
    hits = [w for w in words if difflib.SequenceMatcher(None, w[0], label).ratio() >= 0.8]
    return hits[nth] if len(hits) > nth else None

def _crop(a, x0, y0, x1, y1):
    H, W = a.shape[:2]; x0, x1, y0, y1 = max(0, int(x0)), min(W, int(x1)), max(0, int(y0)), min(H, int(y1))
    return (x0, y0, x1, y1) if x1 - x0 > 20 and y1 - y0 > 8 else None

def _b64(a, box, mx=600):
    import base64
    if not box: return ""
    return "data:image/jpeg;base64," + base64.b64encode(jpeg(a[box[1]:box[3], box[0]:box[2]], mx, 80)).decode()

def _read(pen, gray, box, wl=None):
    out = []
    if not box: return out
    for src in (pen, gray):
        c = cv2.copyMakeBorder(cv2.resize(src[box[1]:box[3], box[0]:box[2]], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC), 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        for p in (7, 13):
            t = ocr(c, p, wl).strip().replace("\n", " ")
            if t: out.append(t)
    return out

def _valid_date(s):
    d = norm_date(s)
    if not d: return ""
    dd, mm, yy = map(int, d.split("/"))
    return d if 1 <= dd <= 31 and 1 <= mm <= 12 and 2024 <= yy <= 2035 else ""

def extract_tally(a, hints=None):
    hints = hints or {}
    pr, pen, gray = printed_layer(a), pen_layer(a), cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    H, W = pen.shape; w = _words(pr); warnings, crops = [], {}
    def right(lab, x_end, dy=(-30, 16)):
        an = _anchor(w, lab)
        return _crop(a, an[1] + an[3] + 4, an[2] + dy[0], x_end, an[2] + an[4] + dy[1]) if an else None
    boxes = {"container": right("CONTAINER", W * 0.44), "customs": right("CUSTOMS", W * 0.44), "shipperSeal": right("CARRIER'S", W),
             "date": right("DATE", W * 0.44), "time": right("TIME", W)}
    for k, b in boxes.items(): crops[k] = _b64(a, b)
    container = ""; cstat = ""
    for t in _read(pen, gray, boxes["container"]):
        c, st = fix_container(t)
        if c and iso_ok(c): container, cstat = c, st; break
    date = ""
    for t in _read(pen, gray, boxes["date"], "0123456789-/. "):
        date = _valid_date(t)
        if date: break
    rows_crops, rows = [], []
    tot, wn = _anchor(w, "TOTAL"), _anchor(w, "WEIGHT")
    if tot and wn and wn[2] > tot[2]:
        x0, x1, y0, y1 = tot[1] - int(tot[3] * 0.85), tot[1] + int(tot[3] * 0.4), tot[2] + tot[4] + 6, wn[2] - 6
        col = pen[y0:y1, max(0, x0):x1]
        prof = (col < 128).sum(1); bands, inb, st = [], False, 0
        for i, v in enumerate(prof):
            if v > 3 and not inb: st, inb = i, True
            elif v <= 3 and inb:
                inb = False
                if i - st > 18: bands.append((st, i))
        if len(bands) != 6:
            hgt = (y1 - y0) / 6; bands = [(int(i * hgt), int((i + 1) * hgt)) for i in range(6)]
            warnings.append("The six row-total entries could not be separated automatically. Read them from the paper.")
        for s0, e0 in bands: rows_crops.append(_b64(a, _crop(a, x0, y0 + s0 - 8, x1, y0 + e0 + 8), 300))
    crops["rows"] = rows_crops
    res = {"container": container, "container_status": cstat, "customs": "", "shipperSeal": "", "date": date, "time": "", "rows": rows, "gross": 0}
    exp = hints.get("expected_bags")
    warnings.append("The tally is handwritten and is not read automatically. Type the highlighted values while looking at the cropped handwriting shown beside each field." +
                    (f" The six row totals must add up to {exp} bags." if exp else ""))
    return res, warnings, "", crops

def extract_insp(a, hints=None):
    pr, pen, gray = printed_layer(a), pen_layer(a), cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    H, W = pen.shape; w = _words(pr); warnings, crops = [], {}
    res = {"container": "", "container_status": "", "seal": "", "shipper": "", "inspector": "", "date": "", "results": {}}
    def row_end(word, limit):   # right edge of the printed label on this row (the label may be split into several words)
        return max(q[1] + q[3] for q in w if abs(q[2] - word[2]) < 18 and word[1] <= q[1] < limit)
    cw = min((q for q in w if q[0] == "NUMBER" and q[2] < H * 0.6), key=lambda q: q[2], default=None)   # "Container Number:" label
    if cw:
        gap = int(H * 0.0245)   # the OCR finds the "Seal Number" label; the container row is one form row above it
        bc = _crop(a, cw[1] + cw[3] + 4, cw[2] - 30 - gap, W * 0.5, cw[2] + cw[4] + 24 - gap); crops["container"] = _b64(a, bc)
        for t in _read(pen, gray, bc):
            c, st = fix_container(t)
            if c and iso_ok(c): res["container"], res["container_status"] = c, st; break
        crops["seal"] = _b64(a, _crop(a, cw[1] + cw[3] + 4, cw[2] - 30, W * 0.5, cw[2] + cw[4] + 24))
    names = ["outside", "doors", "right", "left", "front", "ceiling", "floor"]
    passes = sorted([q for q in w if q[0] == "PASS"], key=lambda q: q[2]); fails = sorted([q for q in w if q[0].startswith("FAIL")], key=lambda q: q[2])
    ink, dark = ink_mask(a), (gray < 120)
    if len(passes) == 7:
        for i, (_, px, py, pw, ph) in enumerate(passes):
            fl = next((f for f in fails if abs(f[2] - py) < 25), None)
            def score(bx, by, bw):
                y0, y1, x0, x1 = max(0, by - 44), max(0, by - 12), max(0, bx + 2), bx + bw - 2
                return float(ink[y0:y1, x0:x1].mean()) + 0.35 * float(dark[y0:y1, x0:x1].mean()) if y1 > y0 and x1 > x0 else 0
            sp = score(px, py, pw); sf = score(fl[1], fl[2], fl[3]) if fl else 0
            res["results"][names[i]] = "pass" if sp > sf * 1.4 + 0.004 else ("fail" if sf > sp * 1.4 + 0.004 else "")
        if not all(res["results"].values()): warnings.append("Some Pass/Fail ticks were unclear. Set them manually.")
    else: warnings.append(f"Found {len(passes)} of 7 Pass/Fail rows, so the ticks were not read. Set them manually.")
    dn = max((q for q in w if q[0].startswith("DATE") and q[2] > H * 0.6), key=lambda q: q[2], default=None)
    if dn:
        bd = _crop(a, dn[1] + dn[3] + 10, dn[2] - 26, W, dn[2] + dn[4] + 20); crops["date"] = _b64(a, bd)
        for t in _read(pen, gray, bd, "0123456789-/. "):
            res["date"] = _valid_date(t)
            if res["date"]: break
    warnings.append("Pass/Fail ticks are measured from the ink. The handwritten container, seal and date are not read automatically: type them from the cropped handwriting shown beside each field.")
    return res, warnings, "", crops

def run(kind, raw, name="", hints=None):
    a = load_image(raw, name)
    fn = {"order": extract_order, "tally": extract_tally, "insp": extract_insp}[kind]
    fields, warnings, text, crops = fn(a, hints or {})
    return {"fields": fields, "warnings": warnings, "text": text, "crops": crops, "preview": jpeg(a)}
