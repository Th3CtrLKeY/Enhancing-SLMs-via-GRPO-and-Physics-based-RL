"""
OCR extraction for scanned PDFs using pytesseract + PyMuPDF.
Handles:
  - Faltinsen textbook (scanned book)
  - MIT OCW 2.29 lecture notes (scanned)
  - IIT exam papers (scanned)
"""

import json
import re
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

# Point pytesseract to the installed Tesseract binary
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

RAW = Path("c:/Users/raghu/Desktop/IIT/Sem 10/MTP/raw_data")
OUT = Path("c:/Users/raghu/Desktop/IIT/Sem 10/MTP/extracted_text")
CHUNK_TOKENS = 500


# ── helpers (same as extract_text.py) ──────────────────────────────────────

def rough_token_count(text: str) -> int:
    return int(len(text.split()) * 1.3)


def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'-\n(\w)', r'\1', text)
    return text.strip()


def ocr_pdf(pdf_path: Path, dpi: int = 250) -> list[dict]:
    """
    Render each PDF page as an image and run Tesseract OCR.
    Returns list of {page, text} dicts.
    dpi=250 is a good balance of accuracy vs speed for textbooks.
    """
    doc = fitz.open(str(pdf_path))
    total = len(doc)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        print(f"    page {page_num:3d}/{total}", end="\r", flush=True)

        # Render page to image at target DPI
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)  # grayscale = faster OCR
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        # OCR with Tesseract (--oem 3 = LSTM, --psm 3 = auto page segmentation)
        text = pytesseract.image_to_string(img, config="--oem 3 --psm 3")
        text = clean_text(text)
        if text:
            pages.append({"page": page_num, "text": text})

    doc.close()
    print()
    return pages


def chunk_pages(pages: list[dict], source: str, chunk_size: int = CHUNK_TOKENS) -> list[dict]:
    chunks = []
    if not pages:
        return chunks
    buffer = ""
    start_page = pages[0]["page"]

    for p in pages:
        buffer += "\n\n" + p["text"]
        if rough_token_count(buffer) >= chunk_size:
            chunks.append({
                "source": source,
                "start_page": start_page,
                "end_page": p["page"],
                "text": buffer.strip(),
                "token_estimate": rough_token_count(buffer),
            })
            words = buffer.split()
            buffer = " ".join(words[-80:])
            start_page = p["page"]

    if buffer.strip():
        chunks.append({
            "source": source,
            "start_page": start_page,
            "end_page": pages[-1]["page"],
            "text": buffer.strip(),
            "token_estimate": rough_token_count(buffer),
        })
    return chunks


def save_txt(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    kb = len(text) // 1024
    print(f"  [OK]  {path.name}  ({kb} KB)")


def save_chunks(out_dir: Path, chunks: list[dict], stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, chunk in enumerate(chunks):
        txt_path = out_dir / f"{stem}_chunk_{i:04d}.txt"
        txt_path.write_text(chunk["text"], encoding="utf-8")
        manifest.append({**chunk, "file": txt_path.name, "text": None})
    (out_dir / f"{stem}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"  [OK]  {out_dir.name}/ — {len(chunks)} chunks saved")


def process_scanned_textbook(pdf_path: Path, out_dir: Path, stem: str):
    print(f"\n  OCR: {pdf_path.name} ...")
    pages = ocr_pdf(pdf_path, dpi=250)
    print(f"  Extracted {len(pages)} pages with text")
    chunks = chunk_pages(pages, source=stem)
    save_chunks(out_dir, chunks, stem)


def process_scanned_single(pdf_path: Path, out_path: Path):
    print(f"\n  OCR: {pdf_path.name} ...")
    pages = ocr_pdf(pdf_path, dpi=300)  # higher DPI for exam papers
    print(f"  Extracted {len(pages)} pages with text")
    if not pages:
        print(f"  [SKIP] No text found")
        return
    full_text = "\n\n---PAGE BREAK---\n\n".join(
        f"[Page {p['page']}]\n{p['text']}" for p in pages
    )
    save_txt(out_path, full_text)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print(" OCR Extraction for Scanned PDFs")
    print("=" * 55)

    # 1. Faltinsen textbook
    print("\n=== 1. Faltinsen Textbook ===")
    faltinsen_pdf = RAW / "Fatinsen_Sea_Loads_on_Ships_and_Offshore.pdf"
    if faltinsen_pdf.exists():
        process_scanned_textbook(
            faltinsen_pdf,
            OUT / "textbooks" / "faltinsen_chunks",
            "faltinsen"
        )
    else:
        print(f"  [MISS] {faltinsen_pdf.name} not found")

    # 2. MIT OCW 2.29 lecture notes
    print("\n=== 2. MIT OCW 2.29 — Lecture Notes ===")
    dir_229 = RAW / "mit_ocw_2.29" / "lecture_notes"
    for pdf in sorted(dir_229.glob("*.pdf")):
        out = OUT / "mit_ocw_2.29" / "lecture_notes" / (pdf.stem + ".txt")
        if out.exists() and out.stat().st_size > 100:
            print(f"  [SKIP] {pdf.stem} (already extracted)")
            continue
        process_scanned_single(pdf, out)

    # 3. IIT exam papers
    print("\n=== 3. IIT Exam Papers ===")
    for pdf in sorted((RAW / "exam_papers").glob("*.pdf")):
        out = OUT / "exam_papers" / (pdf.stem + ".txt")
        if out.exists() and out.stat().st_size > 100:
            print(f"  [SKIP] {pdf.stem} (already extracted)")
            continue
        process_scanned_single(pdf, out)

    # Summary
    print("\n=== OCR Complete ===")
    all_txt  = list(OUT.rglob("*.txt"))
    total_kb = sum(f.stat().st_size for f in all_txt) // 1024
    print(f"Total .txt files : {len(all_txt)}")
    print(f"Total text size  : {total_kb} KB")


if __name__ == "__main__":
    main()
