# src/legalrag/ingest.py
import os
import re
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .config import settings
from .logging_utils import setup_logging

log = logging.getLogger("legalrag.ingest")

import re
from typing import Optional, List, Dict, Any

LAW_HINT_RE = re.compile(
    r"(?:§\s*\d+[a-zA-Z]*|Art\.?\s*\d+[a-zA-Z]*)\s+(?P<law>[A-Za-zÄÖÜäöüß\-]{2,20})\b"
)

def parse_law_hint(question: str) -> Optional[str]:
    m = LAW_HINT_RE.search(question)
    if not m:
        return None
    return m.group("law").strip()

# Matches headings like:
# "##### § 1 Name"
# "### § 433 Vertragstypische Pflichten beim Kaufvertrag"
# "#### Art. 1 ..."
NORM_RE = re.compile(
    r"^\s*#{1,6}\s*(?P<kind>§|Art\.?|Artikel)\s*(?P<num>\d+[a-zA-Z]*)\s*(?P<title>.*)$"
)

# Absatz lines like "(1) ..." "(2) ..."
ABS_RE = re.compile(r"^\s*\(\s*(?P<abs>\d+[a-zA-Z]?)\s*\)\s*(?P<rest>.*)$")

H1_RE = re.compile(r"^\s*#\s+(.+?)\s*$")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_law_title_and_abbr(lines: List[str], fallback_name: str) -> Tuple[str, Optional[str]]:
    title = None
    for line in lines[:80]:
        m = H1_RE.match(line)
        if m:
            title = m.group(1).strip()
            break
    if not title:
        title = fallback_name

    # Parse abbreviation in parentheses at end: "(BGB)"
    abbr = None
    m2 = re.search(r"\(([^()]{1,30})\)\s*$", title)
    if m2:
        abbr = m2.group(1).strip()

    return title, abbr


def infer_short_name(abbr: Optional[str], md_path: str) -> str:
    if abbr and 2 <= len(abbr) <= 15:
        return abbr

    base = os.path.splitext(os.path.basename(md_path))[0]
    base_norm = base.lower()

    # gesetze repo: many laws are stored as ".../<LAW>/index.md"
    if base_norm == "index":
        parent = os.path.basename(os.path.dirname(md_path))
        parent = re.sub(r"[^A-Za-z0-9ÄÖÜäöüß]+", "_", parent).strip("_")
        return parent.upper() if parent else "GESETZ"

    base = re.sub(r"[^A-Za-z0-9ÄÖÜäöüß]+", "_", base).strip("_")
    return base.upper() if base else "GESETZ"



def split_by_norm(lines: List[str]) -> List[Tuple[Dict[str, str], List[str]]]:
    blocks: List[Tuple[Dict[str, str], List[str]]] = []
    current_hdr: Optional[Dict[str, str]] = None
    current_body: List[str] = []

    def flush():
        nonlocal current_hdr, current_body
        if current_hdr is not None:
            blocks.append((current_hdr, current_body))
        current_hdr, current_body = None, []

    for line in lines:
        m = NORM_RE.match(line)
        if m:
            flush()
            kind = m.group("kind").strip()
            if kind.lower().startswith("artikel"):
                kind = "Art."
            if kind == "Art":
                kind = "Art."
            num = m.group("num").strip()
            ttl = (m.group("title") or "").strip()
            current_hdr = {"kind": kind, "num": num, "title": ttl}
            current_body = []
        else:
            if current_hdr is not None:
                current_body.append(line)

    flush()
    return blocks


def split_absatz(body_lines: List[str]) -> List[Tuple[Optional[str], str]]:
    chunks: List[Tuple[Optional[str], str]] = []
    cur_abs: Optional[str] = None
    cur_lines: List[str] = []
    found = False

    def flush():
        nonlocal cur_lines
        txt = "\n".join(cur_lines).strip()
        if txt:
            chunks.append((cur_abs, txt))
        cur_lines = []

    for line in body_lines:
        m = ABS_RE.match(line)
        if m:
            found = True
            flush()
            cur_abs = m.group("abs").strip()
            rest = (m.group("rest") or "").strip()
            cur_lines = [rest] if rest else []
        else:
            cur_lines.append(line)

    flush()

    if not found:
        txt = "\n".join(body_lines).strip()
        return [(None, txt)] if txt else []
    return chunks


def make_snippets(text: str, chunk_id: str) -> List[Dict[str, Any]]:
    # Simple snippet strategy: first ~3 paragraphs or sentence-like segments
    snips: List[Dict[str, Any]] = []
    parts = re.split(r"\n{2,}|(?<=[\.\!\?])\s+", text)
    cursor = 0
    sid = 1
    for p in parts:
        p = p.strip()
        if not p:
            continue
        idx = text.find(p, cursor)
        if idx < 0:
            continue
        snips.append({
            "snippet_id": f"{chunk_id}::S{sid}",
            "text": p,
            "span": {"start": idx, "end": idx + len(p)}
        })
        cursor = idx + len(p)
        sid += 1
        if len(snips) >= 3:
            break
    return snips


def build_or_load_md_filelist(root: Path, filelist_path: Path) -> List[str]:
    """
    Deterministic file discovery on Windows:
    - Prefer precomputed file list
    - Otherwise generate it once using suffix filter (case-insensitive)
    """
    if filelist_path.exists():
        lines = filelist_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        files = [ln.strip() for ln in lines if ln.strip()]
        return files

    files: List[str] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".md":
            files.append(str(p))

    filelist_path.parent.mkdir(parents=True, exist_ok=True)
    filelist_path.write_text("\n".join(files), encoding="utf-8")
    return files


def ingest_all() -> None:
    setup_logging()
    log.info("RUNNING FILE: %s", __file__)

    os.makedirs(settings.PROCESSED_DIR, exist_ok=True)

    # Atomic write outputs
    chunks_tmp = os.path.join(settings.PROCESSED_DIR, "statute_chunks.jsonl.tmp")
    sources_tmp = os.path.join(settings.PROCESSED_DIR, "sources.jsonl.tmp")
    chunks_out = os.path.join(settings.PROCESSED_DIR, "statute_chunks.jsonl")
    sources_out = os.path.join(settings.PROCESSED_DIR, "sources.jsonl")

    raw_root = settings.RAW_MD_DIR
    root = Path(raw_root).expanduser().resolve()
    log.info("RAW_MD_DIR(raw)=%r | abs=%s | exists=%s", raw_root, root, root.exists())

    if not root.exists():
        log.warning("RAW_MD_DIR does not exist. Writing empty outputs.")
        Path(chunks_tmp).write_text("", encoding="utf-8")
        Path(sources_tmp).write_text("", encoding="utf-8")
        os.replace(chunks_tmp, chunks_out)
        os.replace(sources_tmp, sources_out)
        return

    filelist_path = Path(settings.PROCESSED_DIR) / "md_filelist.txt"
    md_files = build_or_load_md_filelist(root, filelist_path)
    log.info("Loaded md_files from filelist: %s | count=%d", str(filelist_path), len(md_files))
    if md_files:
        log.info("Sample md: %s", md_files[0])

    # Optional small test run
    

# Optional: also drop anything under folders named "index" or "readme" (defensive)
    md_files = [p for p in md_files if "/index/" not in p.replace("\\", "/").lower()]

    if md_files:
        log.info("After exclude sample md: %s", md_files[0])

# Optional small test run AFTER filtering
    max_files = os.environ.get("MAX_MD_FILES")
    if max_files:
        try:
            md_files = md_files[: int(max_files)]
        except Exception:
            pass

    log.info("RAW_MD_DIR=%s | md_files=%d", raw_root, len(md_files))
    if not md_files:
        log.warning("No .md files after filtering. Writing empty outputs.")
        Path(chunks_tmp).write_text("", encoding="utf-8")
        Path(sources_tmp).write_text("", encoding="utf-8")
        os.replace(chunks_tmp, chunks_out)
        os.replace(sources_tmp, sources_out)
        return


    # Exclude non-law pages (filter AFTER filelist load)
    excluded = {"readme.md", "lizenz.md", "license.md"}
    md_files = [p for p in md_files if os.path.basename(p).lower() not in excluded]

    log.info("After exclude filter: md_files=%d", len(md_files))
    if md_files:
        log.info("After exclude sample md: %s", md_files[0])


    sources = 0
    chunks = 0
    t0 = time.time()

    with open(chunks_tmp, "w", encoding="utf-8") as ch_out, open(sources_tmp, "w", encoding="utf-8") as src_out:
        for idx, md_path in enumerate(md_files, start=1):
            try:
                with open(md_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.read().splitlines()
            except Exception:
                continue

            title, abbr = extract_law_title_and_abbr(lines, fallback_name=os.path.basename(md_path))
            short = infer_short_name(abbr, md_path)

            # Compute source hash
            try:
                sh = sha256_file(md_path)
            except Exception:
                continue

            rel = md_path.replace("\\", "/")
            source_id = f"SRC::gesetze::{os.path.basename(md_path)}::{sh[:12]}"

            src_out.write(json.dumps({
                "source_id": source_id,
                "kind": "bundesrecht_markdown",
                "provider": "bundestag/gesetze",
                "path": rel,
                "sha256": sh,
                "law_title": title,
                "law_short": short,
            }, ensure_ascii=False) + "\n")
            sources += 1

            norm_blocks = split_by_norm(lines)
            if not norm_blocks:
                # Skip pages that do not contain §/Art headings (index pages, metadata)
                continue

            for hdr, body in norm_blocks:
                abs_chunks = split_absatz(body)
                for abs_num, abs_text in abs_chunks:
                    abs_text = abs_text.strip()
                    if not abs_text:
                        continue

                    kind = hdr["kind"]
                    num = hdr["num"]
                    norm_title = hdr["title"]

                    if kind.startswith("§"):
                        canonical = f"{short} § {num}" + (f" Abs. {abs_num}" if abs_num else "")
                        locator = {"gesetz": short, "paragraf": num, "absatz": abs_num, "satz": None}
                    else:
                        canonical = f"{short} Art. {num}" + (f" Abs. {abs_num}" if abs_num else "")
                        locator = {"gesetz": short, "artikel": num, "absatz": abs_num, "satz": None}

                    chunk_id = f"STATUTE::{short}::{kind}{num}::Abs{abs_num or 'NA'}::{sh[:8]}::{chunks+1}"

                    obj = {
                        "chunk_id": chunk_id,
                        "doc_type": "statute",
                        "law": {"title": title, "short": short, "jurisdiction": "DE-BUND"},
                        "provision": {"kind": kind, "num": num, "norm_title": norm_title, "absatz": abs_num, "satz": None},
                        "citation": {"canonical": canonical, "locator": locator},
                        "text": abs_text,
                        "snippets": make_snippets(abs_text, chunk_id),
                        "source": {"source_id": source_id, "path": rel, "sha256": sh},
                    }
                    ch_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    chunks += 1

            if idx % 200 == 0:
                log.info("Progress: %d/%d files | sources=%d chunks=%d", idx, len(md_files), sources, chunks)

    os.replace(chunks_tmp, chunks_out)
    os.replace(sources_tmp, sources_out)

    log.info("Ingestion complete | sources=%d chunks=%d | seconds=%.1f", sources, chunks, time.time() - t0)


if __name__ == "__main__":
    ingest_all()
