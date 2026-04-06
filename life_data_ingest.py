#!/usr/bin/env python3
"""
life_data_ingest.py — Personal Life Data Ingestion
====================================================
Ingests every digital trace of your life into brain/raw/ so the
LLM compiler can build a rich, interconnected personal knowledge base.

PATHWAY REPAIR THEORY:
Each data type creates a DIFFERENT KIND of retrieval pathway to the same memories.
Your iMessages contain the social context. Your GPS tracks contain the spatial
context. Your health data contains the physiological context. Your browser history
contains your attention and interest at the time. Your Spotify contains the emotional
context. When these are ALL linked to the same event in the brain, that event becomes
retrievable from any direction -- even if some pathways are damaged or forgotten.
This is the same mechanism physical therapists use to help memory-impaired patients:
build MORE routes to the same destination.

Supported data sources:
  Apple Health XML         → steps, heart rate, sleep, workouts, weight
  Google Takeout JSON      → location history, activity
  Browser history          → Chrome / Firefox / Safari (SQLite)
  iMessage chat.db         → conversations, contacts
  Email (.mbox / .eml)     → sent/received messages
  Calendar (.ics)          → events, appointments
  Contacts (.vcf)          → people directory
  Spotify history          → listening patterns, moods
  Twitter/X archive        → posts, timeline
  Documents (.docx/.xlsx/.pptx/.epub) → content extraction
  Code files               → structure and purpose (not full source)
  GPS tracks (.gpx)        → where you went

Usage:
    python life_data_ingest.py --apple-health ~/Downloads/export.xml
    python life_data_ingest.py --browser chrome
    python life_data_ingest.py --spotify ~/Downloads/my_spotify_data/
    python life_data_ingest.py --all ~/Downloads/my_data_export/
"""

import os, re, sys, json, sqlite3, logging, argparse, email, mailbox
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("life_data")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

ROOT = Path(__file__).parent / "brain"
RAW  = ROOT / "raw"

def _write_raw(name: str, content: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RAW / f"{name}_{ts}.txt"
    path.write_text(content)
    log.info("Wrote: %s (%d chars)", path.name, len(content))
    return path

# ---------------------------------------------------------------------------
# Apple Health
# ---------------------------------------------------------------------------

def ingest_apple_health(xml_path: Path):
    """Parse Apple Health export.xml and write summaries to raw/."""
    try:
        import xml.etree.ElementTree as ET
    except ImportError:
        log.error("xml.etree.ElementTree not available")
        return

    log.info("Parsing Apple Health export (this may take a minute for large files)...")
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    records = {}
    for rec in root.findall("Record"):
        rtype = rec.get("type", "").replace("HKQuantityTypeIdentifier", "").replace("HKCategoryTypeIdentifier", "")
        date  = rec.get("startDate", "")[:10]
        value = rec.get("value", "")
        unit  = rec.get("unit", "")
        records.setdefault(rtype, []).append({"date": date, "value": value, "unit": unit})

    summary_lines = [
        "Source: Apple Health Export",
        f"Export date: {datetime.now().strftime('%Y-%m-%d')}",
        f"Total record types: {len(records)}",
        ""
    ]
    for rtype, entries in sorted(records.items()):
        summary_lines.append(f"## {rtype} ({len(entries)} records)")
        for e in entries[-5:]:  # last 5 of each type
            summary_lines.append(f"  {e['date']}: {e['value']} {e['unit']}")
        summary_lines.append("")

    # Workouts
    for wo in root.findall("Workout"):
        wtype    = wo.get("workoutActivityType", "").replace("HKWorkoutActivityType", "")
        duration = wo.get("duration", "?")
        date     = wo.get("startDate", "")[:10]
        summary_lines.append(f"Workout: {date} {wtype} {duration}min")

    _write_raw("apple_health", "\n".join(summary_lines))
    log.info("Apple Health: %d record types processed", len(records))

# ---------------------------------------------------------------------------
# Google Takeout Location
# ---------------------------------------------------------------------------

def ingest_google_takeout(folder: Path):
    """Process Google Takeout data folder."""
    folder = Path(folder)
    processed = 0

    # Location history
    for loc_file in folder.rglob("Records.json"):
        log.info("Processing location history: %s", loc_file)
        try:
            data = json.loads(loc_file.read_text(errors="replace"))
            locations = data.get("locations", [])
            lines = [f"Source: Google Location History\nTotal points: {len(locations)}\n"]
            for loc in locations[-200:]:  # last 200 points
                ts  = loc.get("timestamp", "")[:10]
                lat = loc.get("latitudeE7", 0) / 1e7
                lon = loc.get("longitudeE7", 0) / 1e7
                lines.append(f"{ts}: ({lat:.4f}, {lon:.4f})")
            _write_raw("google_location", "\n".join(lines))
            processed += 1
        except Exception as e:
            log.error("Location parse failed: %s", e)

    if processed == 0:
        log.warning("No recognized Google Takeout files found in %s", folder)

# ---------------------------------------------------------------------------
# Browser history
# ---------------------------------------------------------------------------

def ingest_browser_history(browser: str):
    """Extract browser history from Chrome, Firefox, or Safari SQLite DBs."""
    home = Path.home()

    if browser == "chrome":
        candidates = [
            home / "Library/Application Support/Google/Chrome/Default/History",
            home / ".config/google-chrome/Default/History",
            home / "AppData/Local/Google/Chrome/User Data/Default/History",
        ]
    elif browser == "firefox":
        ff_dir = home / "Library/Application Support/Firefox/Profiles"
        candidates = list(ff_dir.glob("*/places.sqlite")) if ff_dir.exists() else []
    elif browser == "safari":
        candidates = [home / "Library/Safari/History.db"]
    else:
        log.error("Unknown browser: %s. Use chrome, firefox, or safari", browser)
        return

    import shutil, tempfile
    for db_path in candidates:
        if not db_path.exists():
            continue
        log.info("Reading browser history: %s", db_path)

        # Copy DB (browser may have it locked)
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        shutil.copy2(db_path, tmp_path)

        try:
            conn = sqlite3.connect(str(tmp_path))
            conn.row_factory = sqlite3.Row

            if browser in ("chrome",):
                rows = conn.execute("""
                    SELECT url, title, visit_count,
                           datetime(last_visit_time/1000000-11644473600, 'unixepoch') AS last_visit
                    FROM urls
                    ORDER BY last_visit_time DESC LIMIT 2000
                """).fetchall()
            elif browser == "firefox":
                rows = conn.execute("""
                    SELECT url, title, visit_count,
                           datetime(last_visit_date/1000000, 'unixepoch') AS last_visit
                    FROM moz_places WHERE visit_count > 0
                    ORDER BY last_visit_date DESC LIMIT 2000
                """).fetchall()
            elif browser == "safari":
                rows = conn.execute("""
                    SELECT hi.url, hv.title,
                           datetime(hv.visit_time + 978307200, 'unixepoch') AS last_visit
                    FROM history_visits hv
                    JOIN history_items hi ON hv.history_item = hi.id
                    ORDER BY hv.visit_time DESC LIMIT 2000
                """).fetchall()

            lines = [f"Source: {browser.title()} Browser History\nExtracted: {datetime.now().strftime('%Y-%m-%d')}\n"]
            for r in rows:
                lines.append(f"{r['last_visit'] or '?'}: {r.get('title','') or ''} | {r['url'][:100]}")

            _write_raw(f"browser_{browser}", "\n".join(lines))
            conn.close()
        except Exception as e:
            log.error("Browser history read failed: %s", e)
        finally:
            tmp_path.unlink(missing_ok=True)
        return

    log.warning("No %s history database found", browser)

# ---------------------------------------------------------------------------
# iMessage
# ---------------------------------------------------------------------------

def ingest_imessage(chat_db: Path):
    """Extract iMessage conversations from iPhone backup chat.db."""
    try:
        conn = sqlite3.connect(str(chat_db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT m.text, m.is_from_me,
                   datetime(m.date/1000000000 + 978307200, 'unixepoch') AS msg_date,
                   h.id AS contact
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.rowid
            WHERE m.text IS NOT NULL AND m.text != ''
            ORDER BY m.date DESC LIMIT 5000
        """).fetchall()

        lines = [f"Source: iMessage\nExtracted: {datetime.now().strftime('%Y-%m-%d')}\n"]
        for r in rows:
            direction = "Me" if r["is_from_me"] else (r["contact"] or "?")
            lines.append(f"[{r['msg_date']}] {direction}: {r['text'][:200]}")

        _write_raw("imessage", "\n".join(lines))
        log.info("iMessage: %d messages extracted", len(rows))
    except Exception as e:
        log.error("iMessage read failed: %s", e)

# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

def ingest_email(path: Path):
    """Process .mbox file or folder of .eml files."""
    path = Path(path)
    lines = [f"Source: Email\nExtracted: {datetime.now().strftime('%Y-%m-%d')}\n"]
    count = 0

    def _parse_msg(msg):
        subject = msg.get("Subject", "")
        from_   = msg.get("From", "")
        date    = msg.get("Date", "")
        body    = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        body = part.get_payload(decode=True).decode("utf-8", errors="replace")[:500]
                    except Exception:
                        pass
                    break
        else:
            try:
                body = msg.get_payload(decode=True).decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
        return f"[{date}] From: {from_} | Subject: {subject}\n{body}\n"

    if path.is_file() and path.suffix == ".mbox":
        mbox = mailbox.mbox(str(path))
        for msg in mbox:
            lines.append(_parse_msg(msg))
            count += 1
    elif path.is_dir():
        for eml in path.glob("*.eml"):
            with eml.open("rb") as f:
                msg = email.message_from_binary_file(f)
                lines.append(_parse_msg(msg))
                count += 1

    _write_raw("email", "\n".join(lines))
    log.info("Email: %d messages extracted", count)

# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------

def ingest_calendar(ics_path: Path):
    """Parse .ics calendar file into text entries."""
    content = Path(ics_path).read_text(errors="replace")
    lines   = [f"Source: Calendar ({ics_path.name})\n"]
    events  = []

    current = {}
    for line in content.splitlines():
        if line.startswith("BEGIN:VEVENT"):
            current = {}
        elif line.startswith("END:VEVENT"):
            events.append(current.copy())
        elif ":" in line:
            key, _, val = line.partition(":")
            key = key.split(";")[0]
            current[key] = val.strip()

    for e in events:
        summary  = e.get("SUMMARY", "")
        dtstart  = e.get("DTSTART", "")[:8]
        location = e.get("LOCATION", "")
        desc     = e.get("DESCRIPTION", "")[:200]
        lines.append(f"Event: {dtstart} | {summary} | {location}\n{desc}\n")

    _write_raw("calendar", "\n".join(lines))
    log.info("Calendar: %d events extracted", len(events))

# ---------------------------------------------------------------------------
# Contacts
# ---------------------------------------------------------------------------

def ingest_contacts(vcf_path: Path):
    """Parse .vcf vCard file."""
    content = Path(vcf_path).read_text(errors="replace")
    lines   = [f"Source: Contacts ({vcf_path.name})\n"]
    count   = 0

    card = {}
    for line in content.splitlines():
        if line.startswith("BEGIN:VCARD"):
            card = {}
        elif line.startswith("END:VCARD"):
            name   = card.get("FN", card.get("N", "Unknown"))
            phone  = card.get("TEL", "")
            email_ = card.get("EMAIL", "")
            org    = card.get("ORG", "")
            lines.append(f"Contact: {name} | {phone} | {email_} | {org}")
            count += 1
        elif ":" in line:
            key, _, val = line.partition(":")
            key = key.split(";")[0]
            card[key] = val.strip()

    _write_raw("contacts", "\n".join(lines))
    log.info("Contacts: %d cards extracted", count)

# ---------------------------------------------------------------------------
# Spotify
# ---------------------------------------------------------------------------

def ingest_spotify(folder: Path):
    """Process Spotify StreamingHistory*.json files."""
    folder = Path(folder)
    lines  = [f"Source: Spotify Listening History\n"]
    count  = 0

    for hist_file in sorted(folder.glob("StreamingHistory*.json")):
        try:
            data = json.loads(hist_file.read_text(errors="replace"))
            for entry in data:
                ts     = entry.get("endTime", "")
                artist = entry.get("artistName", "")
                track  = entry.get("trackName", "")
                ms     = entry.get("msPlayed", 0)
                if ms > 30000:  # only songs played >30s
                    lines.append(f"{ts}: {artist} — {track} ({ms//1000}s)")
                    count += 1
        except Exception as e:
            log.error("Spotify parse error %s: %s", hist_file.name, e)

    _write_raw("spotify", "\n".join(lines))
    log.info("Spotify: %d tracks extracted", count)

# ---------------------------------------------------------------------------
# Twitter/X archive
# ---------------------------------------------------------------------------

def ingest_twitter(folder: Path):
    """Process Twitter/X data export archive."""
    folder   = Path(folder)
    tweets_f = folder / "data" / "tweets.js"
    if not tweets_f.exists():
        tweets_f = folder / "tweets.js"
    if not tweets_f.exists():
        log.error("tweets.js not found in %s", folder)
        return

    # Twitter JS starts with "window.YTD.tweets.part0 = "
    raw = tweets_f.read_text(errors="replace")
    raw = re.sub(r"^window\.[^=]+=\s*", "", raw).strip()
    try:
        data   = json.loads(raw)
        tweets = [t.get("tweet", t) for t in data]
        lines  = [f"Source: Twitter/X Archive\n"]
        for t in tweets:
            created = t.get("created_at", "")
            text    = t.get("full_text", t.get("text", ""))
            likes   = t.get("favorite_count", "0")
            rts     = t.get("retweet_count", "0")
            lines.append(f"[{created}] {text[:280]}  ❤{likes} 🔁{rts}")
        _write_raw("twitter", "\n".join(lines))
        log.info("Twitter: %d tweets extracted", len(tweets))
    except Exception as e:
        log.error("Twitter parse failed: %s", e)

# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

def ingest_document(file_path: Path):
    """Extract text from .docx, .xlsx, .pptx, or .epub."""
    file_path = Path(file_path)
    suffix    = file_path.suffix.lower()
    text      = ""

    if suffix == ".docx":
        try:
            from docx import Document
            doc  = Document(str(file_path))
            text = "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            log.error("pip install python-docx")
            return

    elif suffix == ".xlsx":
        try:
            import openpyxl
            wb    = openpyxl.load_workbook(str(file_path), read_only=True, data_only=True)
            lines = []
            for ws in wb.worksheets:
                lines.append(f"Sheet: {ws.title}")
                for row in ws.iter_rows(values_only=True):
                    lines.append("\t".join(str(c or "") for c in row))
            text = "\n".join(lines)
        except ImportError:
            log.error("pip install openpyxl")
            return

    elif suffix == ".pptx":
        try:
            from pptx import Presentation
            prs   = Presentation(str(file_path))
            lines = []
            for i, slide in enumerate(prs.slides):
                lines.append(f"\n--- Slide {i+1} ---")
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        lines.append(shape.text)
            text = "\n".join(lines)
        except ImportError:
            log.error("pip install python-pptx")
            return

    elif suffix == ".epub":
        try:
            import ebooklib
            from ebooklib import epub
            from html.parser import HTMLParser

            class _Strip(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.parts = []
                def handle_data(self, d):
                    self.parts.append(d)
                def get_text(self):
                    return " ".join(self.parts)

            book  = epub.read_epub(str(file_path))
            parts = []
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                p = _Strip()
                p.feed(item.get_content().decode("utf-8", errors="replace"))
                parts.append(p.get_text())
            text = "\n".join(parts)
        except ImportError:
            log.error("pip install ebooklib")
            return

    else:
        log.error("Unsupported document type: %s", suffix)
        return

    if text.strip():
        _write_raw(f"doc_{file_path.stem[:30]}", f"Source: {file_path}\n\n{text[:20000]}")
        log.info("Document: %d chars extracted from %s", len(text), file_path.name)

# ---------------------------------------------------------------------------
# Code files
# ---------------------------------------------------------------------------

def ingest_code(file_path: Path):
    """Extract structure from code files (not full source — just purpose/shape)."""
    file_path = Path(file_path)
    source    = file_path.read_text(errors="replace")

    # Extract functions, classes, imports, top-level docstring
    lines      = source.splitlines()
    imports    = [l.strip() for l in lines if l.startswith(("import ", "from "))][:20]
    funcs      = [l.strip() for l in lines if re.match(r"^(def |async def |func |function )", l)][:30]
    classes    = [l.strip() for l in lines if re.match(r"^class ", l)][:20]
    docstring  = ""
    if source.startswith('"""') or source.startswith("'''"):
        end = source.find('"""', 3) if source.startswith('"""') else source.find("'''", 3)
        if end > 0:
            docstring = source[3:end].strip()[:500]

    content = (
        f"Source: {file_path}\nLanguage: {file_path.suffix}\nLines: {len(lines)}\n\n"
        f"Purpose:\n{docstring}\n\n"
        f"Imports:\n" + "\n".join(imports) + "\n\n"
        f"Classes:\n" + "\n".join(classes) + "\n\n"
        f"Functions:\n" + "\n".join(funcs)
    )
    _write_raw(f"code_{file_path.stem[:30]}", content)

# ---------------------------------------------------------------------------
# GPS tracks
# ---------------------------------------------------------------------------

def ingest_gpx(gpx_path: Path):
    """Parse .gpx GPS track file."""
    import xml.etree.ElementTree as ET
    try:
        tree   = ET.parse(str(gpx_path))
        root   = tree.getroot()
        ns     = {"gpx": "http://www.topografix.com/GPX/1/1"}
        points = root.findall(".//gpx:trkpt", ns) or root.findall(".//trkpt")

        lines = [f"Source: GPS Track ({gpx_path.name})\nPoints: {len(points)}\n"]
        for pt in points[::max(1, len(points)//100)]:  # sample max 100 points
            lat  = pt.get("lat", "")
            lon  = pt.get("lon", "")
            time = pt.findtext("{http://www.topografix.com/GPX/1/1}time", pt.findtext("time", ""))
            ele  = pt.findtext("{http://www.topografix.com/GPX/1/1}ele", pt.findtext("ele", ""))
            lines.append(f"{time[:16] if time else '?'}: ({lat}, {lon}) ele={ele}m")

        _write_raw(f"gps_{gpx_path.stem[:30]}", "\n".join(lines))
        log.info("GPX: %d track points extracted", len(points))
    except Exception as e:
        log.error("GPX parse failed: %s", e)

# ---------------------------------------------------------------------------
# Auto-detect from folder (--all)
# ---------------------------------------------------------------------------

def ingest_all(folder: Path):
    """Walk a folder and try to auto-detect and process all known data types."""
    folder = Path(folder)
    log.info("Auto-detecting data types in %s", folder)

    for f in folder.rglob("export.xml"):
        if f.stat().st_size > 10000:
            log.info("Found Apple Health export: %s", f)
            ingest_apple_health(f)

    for f in folder.rglob("StreamingHistory*.json"):
        log.info("Found Spotify history: %s", f.parent)
        ingest_spotify(f.parent)
        break

    for f in folder.rglob("tweets.js"):
        log.info("Found Twitter archive: %s", f.parent.parent)
        ingest_twitter(f.parent.parent)

    for f in folder.rglob("Records.json"):
        if "Location" in str(f) or "location" in str(f):
            log.info("Found Google location history: %s", f.parent)
            ingest_google_takeout(f.parent)

    for f in folder.rglob("*.ics"):
        log.info("Found calendar: %s", f)
        ingest_calendar(f)

    for f in folder.rglob("*.vcf"):
        log.info("Found contacts: %s", f)
        ingest_contacts(f)

    for f in folder.rglob("*.gpx"):
        log.info("Found GPS track: %s", f)
        ingest_gpx(f)

    for f in folder.rglob("*.mbox"):
        log.info("Found email mbox: %s", f)
        ingest_email(f)

    for f in folder.rglob("*.eml"):
        log.info("Found email folder: %s", f.parent)
        ingest_email(f.parent)
        break

    for ext in (".docx", ".epub", ".pptx"):
        for f in folder.rglob(f"*{ext}"):
            ingest_document(f)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Life data ingestion — turns your digital life into brain pathways"
    )
    parser.add_argument("--apple-health",  metavar="PATH", help="Apple Health export.xml")
    parser.add_argument("--google-takeout",metavar="PATH", help="Google Takeout folder")
    parser.add_argument("--browser",       metavar="NAME", help="chrome | firefox | safari")
    parser.add_argument("--imessage",      metavar="PATH", help="iPhone backup chat.db")
    parser.add_argument("--email",         metavar="PATH", help=".mbox file or .eml folder")
    parser.add_argument("--calendar",      metavar="PATH", help=".ics calendar file")
    parser.add_argument("--contacts",      metavar="PATH", help=".vcf contacts file")
    parser.add_argument("--spotify",       metavar="PATH", help="Spotify data download folder")
    parser.add_argument("--twitter",       metavar="PATH", help="Twitter archive folder")
    parser.add_argument("--document",      metavar="PATH", help=".docx/.xlsx/.pptx/.epub file")
    parser.add_argument("--code",          metavar="PATH", help="Code file to index structure")
    parser.add_argument("--gpx",           metavar="PATH", help=".gpx GPS track file")
    parser.add_argument("--all",           metavar="PATH", help="Auto-detect everything in folder")
    args = parser.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)

    if args.apple_health:
        ingest_apple_health(Path(args.apple_health))
    if args.google_takeout:
        ingest_google_takeout(Path(args.google_takeout))
    if args.browser:
        ingest_browser_history(args.browser)
    if args.imessage:
        ingest_imessage(Path(args.imessage))
    if args.email:
        ingest_email(Path(args.email))
    if args.calendar:
        ingest_calendar(Path(args.calendar))
    if args.contacts:
        ingest_contacts(Path(args.contacts))
    if args.spotify:
        ingest_spotify(Path(args.spotify))
    if args.twitter:
        ingest_twitter(Path(args.twitter))
    if args.document:
        ingest_document(Path(args.document))
    if args.code:
        ingest_code(Path(args.code))
    if args.gpx:
        ingest_gpx(Path(args.gpx))
    if args.all:
        ingest_all(Path(args.all))

    if not any(vars(args).values()):
        parser.print_help()
        print("\nAfter ingesting, run: python compile.py")

if __name__ == "__main__":
    main()
