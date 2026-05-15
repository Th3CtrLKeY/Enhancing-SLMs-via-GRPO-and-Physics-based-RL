"""
Step 2: PDF Text Extraction
- Digital PDFs  → PyMuPDF (fast, clean)
- Scanned PDFs  → EasyOCR (no external binary needed)

Output structure:
  extracted_text/
    textbooks/
      newman_chunks/      ← ~500-token chunks with metadata
      faltinsen_chunks/
    mit_ocw_2.20/
      lecture_notes/      ← one .txt per lecture
      problem_sets/       ← one .txt per problem set
      exams/
    mit_ocw_2.29/
      lecture_notes/
    exam_papers/          ← OCR output, one .txt per paper
"""

import json
import re
from pathlib import Path

import fitz  # PyMuPDF

RAW   = Path("c:/Users/raghu/Desktop/IIT/Sem 10/MTP/raw_data")
OUT   = Path("c:/Users/raghu/Desktop/IIT/Sem 10/MTP/extracted_text")
CHUNK_TOKENS = 500   # approximate target chunk size for textbooks


# ── helpers ────────────────────────────────────────────────────────────────

def rough_token_count(text: str) -> int:
    """Rough word-based token estimate (1 word ≈ 1.3 tokens)."""
    return int(len(text.split()) * 1.3)


def clean_text(text: str) -> str:
    """Remove noise from extracted PDF text."""
    # Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove page headers/footers that are just numbers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # Remove hyphenation at end of lines
    text = re.sub(r'-\n(\w)', r'\1', text)
    return text.strip()


def extract_digital_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract text from a digital (non-scanned) PDF.
    Returns list of {page, text} dicts.
    """
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        text = clean_text(text)
        if text:
            pages.append({"page": page_num, "text": text})
    doc.close()
    return pages


def chunk_pages(pages: list[dict], source: str, chunk_size: int = CHUNK_TOKENS) -> list[dict]:
    """
    Merge page texts into ~chunk_size token chunks with overlap.
    Returns list of chunk dicts ready for QA generation.
    """
    chunks = []
    buffer = ""
    start_page = pages[0]["page"] if pages else 1

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
            # 20% overlap: keep last ~100 tokens worth of text
            words = buffer.split()
            buffer = " ".join(words[-80:])
            start_page = p["page"]

    if buffer.strip():
        chunks.append({
            "source": source,
            "start_page": start_page,
            "end_page": pages[-1]["page"] if pages else start_page,
            "text": buffer.strip(),
            "token_estimate": rough_token_count(buffer),
        })
    return chunks


def save_txt(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"  [OK]  {path.relative_to(OUT)}  ({len(text):,} chars)")


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
    print(f"  [OK]  {out_dir.relative_to(OUT)} — {len(chunks)} chunks")


# ── OCR for scanned PDFs ───────────────────────────────────────────────────

def extract_scanned_pdf_ocr(pdf_path: Path) -> list[dict]:
    """
    Extract text from scanned PDF using EasyOCR.
    Converts each page to an image, runs OCR, returns [{page, text}].
    """
    try:
        import easyocr
        import numpy as np
    except ImportError:
        print("  [WARN] easyocr not available, skipping OCR extraction.")
        return []

    print(f"  Loading EasyOCR reader (first run downloads ~100 MB model)...")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num, page in enumerate(doc, start=1):
        print(f"    OCR page {page_num}/{len(doc)}...", end="\r")
        # Render at 300 DPI for better OCR accuracy
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )
        results = reader.readtext(img_array, detail=0, paragraph=True)
        text = clean_text("\n".join(results))
        if text:
            pages.append({"page": page_num, "text": text})
    doc.close()
    print()
    return pages


# ── extraction tasks ───────────────────────────────────────────────────────

def process_digital_single(pdf_path: Path, out_path: Path, label: str):
    """Extract full text from a digital PDF, save as single .txt."""
    pages = extract_digital_pdf(pdf_path)
    full_text = "\n\n---PAGE BREAK---\n\n".join(
        f"[Page {p['page']}]\n{p['text']}" for p in pages
    )
    save_txt(out_path, full_text)


def process_textbook_chunks(pdf_path: Path, out_dir: Path, stem: str):
    """Extract + chunk a textbook PDF for QA generation."""
    pages = extract_digital_pdf(pdf_path)
    chunks = chunk_pages(pages, source=stem)
    save_chunks(out_dir, chunks, stem)


def process_scanned_single(pdf_path: Path, out_path: Path):
    """OCR a scanned PDF and save as single .txt."""
    pages = extract_scanned_pdf_ocr(pdf_path)
    if not pages:
        print(f"  [SKIP] No OCR output for {pdf_path.name}")
        return
    full_text = "\n\n---PAGE BREAK---\n\n".join(
        f"[Page {p['page']}]\n{p['text']}" for p in pages
    )
    save_txt(out_path, full_text)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print("\n=== 1. Textbooks (chunked) ===")
    process_textbook_chunks(
        RAW / "Marine_Hydrodynamics_Newman_2018.pdf",
        OUT / "textbooks" / "newman_chunks",
        "newman"
    )
    process_textbook_chunks(
        RAW / "Fatinsen_Sea_Loads_on_Ships_and_Offshore.pdf",
        OUT / "textbooks" / "faltinsen_chunks",
        "faltinsen"
    )

    print("\n=== 2. MIT OCW 2.20 — Lecture Notes ===")
    for pdf in sorted((RAW / "mit_ocw_2.20" / "lecture_notes").glob("*.pdf")):
        out = OUT / "mit_ocw_2.20" / "lecture_notes" / (pdf.stem + ".txt")
        process_digital_single(pdf, out, pdf.stem)

    print("\n=== 3. MIT OCW 2.20 — Problem Sets ===")
    for pdf in sorted((RAW / "mit_ocw_2.20" / "problem_sets").glob("*.pdf")):
        out = OUT / "mit_ocw_2.20" / "problem_sets" / (pdf.stem + ".txt")
        process_digital_single(pdf, out, pdf.stem)

    print("\n=== 4. MIT OCW 2.20 — Exams ===")
    for pdf in sorted((RAW / "mit_ocw_2.20" / "exams").glob("*.pdf")):
        out = OUT / "mit_ocw_2.20" / "exams" / (pdf.stem + ".txt")
        process_digital_single(pdf, out, pdf.stem)

    print("\n=== 5. MIT OCW 2.29 — Lecture Notes ===")
    for pdf in sorted((RAW / "mit_ocw_2.29" / "lecture_notes").glob("*.pdf")):
        out = OUT / "mit_ocw_2.29" / "lecture_notes" / (pdf.stem + ".txt")
        process_digital_single(pdf, out, pdf.stem)

    print("\n=== 6. IIT Exam Papers (OCR — deferred) ===")
    print("  Scanned papers will be processed separately via ocr_exam_papers.py")
    exam_list = sorted((RAW / "exam_papers").glob("*.pdf"))
    for pdf in exam_list:
        print(f"  [PENDING] {pdf.name}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n=== Extraction Complete ===")
    all_txt = list(OUT.rglob("*.txt"))
    all_json = list(OUT.rglob("*_manifest.json"))
    total_chars = sum(f.stat().st_size for f in all_txt)

    print(f"Text files  : {len(all_txt)}")
    print(f"Manifests   : {len(all_json)}")
    print(f"Total size  : {total_chars // 1024} KB of extracted text")
    print(f"Output dir  : {OUT}")


if __name__ == "__main__":
    main()
