"""
Download all raw data for Marine Hydrodynamics SLM project.
Sources:
  1. MIT OCW 2.20 - Marine Hydrodynamics (lecture notes, problem sets, exams)
  2. MIT OCW 2.29 - Numerical Marine Hydrodynamics (lecture notes)
  3. Open-access fluid mechanics textbooks
"""

import os
import re
import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup

BASE = Path("c:/Users/raghu/Desktop/IIT/Sem 10/MTP/raw_data")
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research use)"}

OCW_BASE = "https://ocw.mit.edu"


def extract_pdf_url(resource_page_url: str) -> str | None:
    """Fetch an OCW resource page and return the direct PDF download URL."""
    try:
        r = requests.get(resource_page_url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # OCW embeds the PDF URL in an <a> tag ending with .pdf
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            if href.endswith(".pdf"):
                if href.startswith("http"):
                    return href
                return OCW_BASE + href

        # Fallback: look inside script/meta tags for a pdf path
        match = re.search(r'(\/courses\/[^"\']+\.pdf)', r.text)
        if match:
            return OCW_BASE + match.group(1)

    except Exception as e:
        print(f"  [ERROR] Could not fetch {resource_page_url}: {e}")
    return None


def download_pdf(url: str, dest: Path, label: str) -> bool:
    """Download a PDF to dest. Returns True on success."""
    if dest.exists():
        print(f"  [SKIP]  {label} (already exists)")
        return True
    try:
        r = requests.get(url, headers=HEADERS, timeout=30, stream=True)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        size_kb = dest.stat().st_size // 1024
        print(f"  [OK]    {label}  ({size_kb} KB)")
        return True
    except Exception as e:
        print(f"  [FAIL]  {label}: {e}")
        return False


def fetch_and_download(resource_slug: str, dest: Path, label: str) -> bool:
    """Resolve OCW resource page -> PDF URL -> download."""
    page_url = f"{OCW_BASE}/courses/2-20-marine-hydrodynamics-13-021-spring-2005/resources/{resource_slug}/"
    pdf_url = extract_pdf_url(page_url)
    if pdf_url:
        return download_pdf(pdf_url, dest, label)
    print(f"  [MISS]  Could not resolve PDF for {label}")
    return False


def fetch_and_download_229(resource_slug: str, dest: Path, label: str) -> bool:
    """Resolve OCW 2.29 resource page -> PDF URL -> download."""
    page_url = f"{OCW_BASE}/courses/2-29-numerical-marine-hydrodynamics-13-024-spring-2003/resources/{resource_slug}/"
    pdf_url = extract_pdf_url(page_url)
    if pdf_url:
        return download_pdf(pdf_url, dest, label)
    print(f"  [MISS]  Could not resolve PDF for {label}")
    return False


# ─────────────────────────────────────────────
# 1. MIT OCW 2.20 — Lecture Notes (L1–L24)
# ─────────────────────────────────────────────
print("\n=== MIT OCW 2.20 — Lecture Notes ===")
lecture_dir = BASE / "mit_ocw_2.20" / "lecture_notes"

for i in range(1, 25):
    slug = f"lecture{i}"
    dest = lecture_dir / f"lecture{i:02d}.pdf"
    fetch_and_download(slug, dest, f"Lecture {i:02d}")
    time.sleep(0.5)  # polite crawl delay


# ─────────────────────────────────────────────
# 2. MIT OCW 2.20 — Problem Sets
# ─────────────────────────────────────────────
print("\n=== MIT OCW 2.20 — Problem Sets ===")
ps_dir = BASE / "mit_ocw_2.20" / "problem_sets"

ps_slugs = [
    ("supp04",  "supplemental_problems"),
    ("ps_1",    "ps_01"),
    ("ps_2a",   "ps_02a"),
    ("ps_2b",   "ps_02b"),
    ("ps_3a",   "ps_03a"),
    ("ps_3b",   "ps_03b"),
    ("ps_4a",   "ps_04a"),
    ("ps_4b",   "ps_04b"),
    ("ps_5a",   "ps_05a"),
    ("ps_5b",   "ps_05b"),
    ("ps_6",    "ps_06"),
    ("ps_7a",   "ps_07a"),
    ("ps_7b",   "ps_07b"),
    ("ps_8a",   "ps_08a"),
    ("ps_8b",   "ps_08b"),
    ("ps_9a",   "ps_09a"),
    ("ps_9b",   "ps_09b"),
    ("ps_10a",  "ps_10a"),
    ("ps_10b",  "ps_10b"),
    ("ps_11a",  "ps_11a"),
    ("ps_11b",  "ps_11b"),
    ("ps_12a",  "ps_12a"),
    ("ps_13a",  "ps_13a"),
    ("ps_14a",  "ps_14a"),
    ("chal_1",  "challenge_01"),
    ("chall_2", "challenge_02"),
    ("chall_3", "challenge_03"),
]

for slug, name in ps_slugs:
    dest = ps_dir / f"{name}.pdf"
    fetch_and_download(slug, dest, name)
    time.sleep(0.5)


# ─────────────────────────────────────────────
# 3. MIT OCW 2.20 — Exams
# ─────────────────────────────────────────────
print("\n=== MIT OCW 2.20 — Exams ===")
exam_dir = BASE / "mit_ocw_2.20" / "exams"

exam_slugs = [
    ("final_solutions", "final_exam_with_solutions"),
]
for slug, name in exam_slugs:
    dest = exam_dir / f"{name}.pdf"
    fetch_and_download(slug, dest, name)
    time.sleep(0.5)


# ─────────────────────────────────────────────
# 4. MIT OCW 2.29 — Numerical Marine Hydrodynamics
# ─────────────────────────────────────────────
print("\n=== MIT OCW 2.29 — Numerical Marine Hydrodynamics ===")
dir_229 = BASE / "mit_ocw_2.29" / "lecture_notes"

# Fetch the lecture notes index page to find slugs
print("  Scanning 2.29 lecture notes index...")
try:
    r = requests.get(
        f"{OCW_BASE}/courses/2-29-numerical-marine-hydrodynamics-13-024-spring-2003/pages/lecture-notes/",
        headers=HEADERS, timeout=15
    )
    soup = BeautifulSoup(r.text, "html.parser")
    slugs_229 = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = re.search(r"/resources/([\w_]+)/?$", href)
        if m:
            slug = m.group(1)
            if slug not in slugs_229:
                slugs_229.append(slug)
    print(f"  Found {len(slugs_229)} resource slugs")
    for slug in slugs_229:
        dest = dir_229 / f"{slug}.pdf"
        fetch_and_download_229(slug, dest, slug)
        time.sleep(0.5)
except Exception as e:
    print(f"  [ERROR] Could not scan 2.29 index: {e}")


# ─────────────────────────────────────────────
# 5. Open-access textbooks
# ─────────────────────────────────────────────
print("\n=== Open-Access Textbooks ===")
tb_dir = BASE / "open_textbooks"

open_books = [
    (
        "https://open.umn.edu/opentextbooks/formats/1141",
        "basics_of_fluid_mechanics.pdf",
        "Basics of Fluid Mechanics (Potto Project)"
    ),
]

for url, filename, label in open_books:
    dest = tb_dir / filename
    download_pdf(url, dest, label)
    time.sleep(1)


print("\n=== Download complete ===")
print(f"Files saved to: {BASE}")

# Print summary
total_files = sum(1 for _ in BASE.rglob("*.pdf"))
total_size  = sum(f.stat().st_size for f in BASE.rglob("*.pdf"))
print(f"Total PDFs downloaded: {total_files}")
print(f"Total size: {total_size // (1024*1024)} MB")
