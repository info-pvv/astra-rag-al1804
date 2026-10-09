#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сборка RAG-базы знаний по материалам курса AL-1803 (Astra Linux SE 1.8).

Из PDF-руководств извлекается текст (PyMuPDF), страницы нарезаются на чанки
с привязкой к документу / страницам / разделу и индексируются в SQLite FTS5.
Для устойчивости к формам слов текст нормализуется стеммером Snowball
(отдельная колонка stems, по ней строится FTS-индекс).

Опционально (--embed) чанки дополнительно кодируются семантическими
векторами через fastembed (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) — тогда поиск
может работать в гибридном режиме (BM25 + косинусная близость, слияние RRF).

Запуск:
    python build_kb.py            # только полнотекстовый индекс
    python build_kb.py --embed    # + семантические векторы

Результат: rag/kb.sqlite
"""
import argparse
import re
import sqlite3
import statistics
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import pymupdf as fitz
except ImportError:
    import fitz
import snowballstemmer

BASE = Path(__file__).resolve().parent.parent   # папка AL-1803 с PDF
DB_PATH = Path(__file__).resolve().parent / "kb.sqlite"

# Короткие названия документов для цитирования в ответах
DOCS = {
    "Программа_курса_AL-1804.pdf": "Программа курса AL-1804",
    "Opisanie_primeneniya.pdf": "Описание применения SE 1.8",
    "Rukovodstvo_administratora_ch1.pdf": "Руководство администратора ч.1",
    "Rukovodstvo_administratora_ch2_ustanovka_i_obnovlenie.pdf":
        "Руководство администратора ч.2 (установка и обновление)",
    "Rukovodstvo_KSZ_ch1_KSZ_PARSEC.pdf": "Руководство по КСЗ ч.1 (PARSEC)",
    "Rukovodstvo_KSZ_ch2.pdf": "Руководство по КСЗ ч.2",
    "Rukovodstvo_KSZ_ch3_zaschischennaya_SUBD.pdf": "Руководство по КСЗ ч.3 (защищённая СУБД)",
    "Rukovodstvo_polzovatelya.pdf": "Руководство пользователя",
    "Vedomost_ED_1.8.6.pdf": "Ведомость ЭД 1.8.6",
}

CHUNK_TARGET = 1100   # целевой размер чанка, символов
CHUNK_MAX = 1400      # жёсткий предел
OVERLAP_MAX = 350     # максимум текста, переносимого в следующий чанк

TOP_RE = re.compile(r"^(ГЛАВА|ЧАСТЬ|РАЗДЕЛ|ПРИЛОЖЕНИЕ|Приложение)\b")
BULLET_RE = re.compile(r"^(?:[•▪●‣◦·–—-]\s+|\d{1,3}[.)]\s+|[a-zа-я]\)\s+)")
TOKEN_RE = re.compile(r"[0-9a-zа-яё][0-9a-zа-яё.\-]*", re.IGNORECASE)
TOC_LINE_RE = re.compile(r"(?:[.·…]\s*){4,}\s*\d+\s*$")

_rus = snowballstemmer.stemmer("russian")
_eng = snowballstemmer.stemmer("english")


def stem(word: str) -> str:
    s = _rus.stemWord(word)
    if s and s != word:
        return s
    return _eng.stemWord(word) or word


STOP = {
    "и", "в", "во", "не", "на", "с", "со", "по", "для", "как", "это", "или",
    "то", "же", "от", "до", "при", "из", "за", "к", "у", "о", "а", "но",
    "если", "что", "чтобы", "его", "их", "ее", "её", "так", "также", "об",
    "бы", "ли", "ты", "он", "она", "они", "мы", "вы", "the", "and", "for",
    "not", "you", "with",
}


def tokens_for_index(text: str) -> list:
    """Нормализованные токены для колонки stems (индексируются FTS5)."""
    out = []
    for t in TOKEN_RE.findall(text.lower()):
        if re.fullmatch(r"[0-9][0-9.\-]*", t):
            clean = re.sub(r"[^0-9]+", " ", t).strip()
            if " " in clean:
                out.extend(clean.split())          # версия «1.8» -> токены «1», «8» рядом
            elif len(clean) >= 2:
                out.append(clean)
            continue
        if len(t) < 3 or t in STOP:
            continue
        out.append(stem(t))
    return out


def fts_expr(query: str) -> str:
    """Собирает выражение для MATCH из пользовательского запроса."""
    parts = []
    seen = set()
    for t in TOKEN_RE.findall(query.lower()):
        if re.fullmatch(r"[0-9][0-9.\-]*", t):
            clean = re.sub(r"[^0-9]+", " ", t).strip()
            if " " in clean:
                part = '"%s"' % clean              # фраза: 1.8 -> "1 8"
            elif len(clean) >= 2:
                part = clean
            else:
                continue
        elif re.search(r"[.\-]", t):               # русб.10015-01, al-1802 и т.п.
            part = '"%s"' % re.sub(r"[^0-9a-zа-яё]+", " ", t).strip()
        elif len(t) < 3 or t in STOP:
            continue
        else:
            s = stem(t)
            part = s + ("*" if len(s) >= 5 else "")
        if part not in seen:
            seen.add(part)
            parts.append(part)
    return " OR ".join(parts)


def block_to_par(raw: str) -> str:
    """Блок PyMuPDF -> абзац: склейка строк переноса, сохранение пунктов списков."""
    t = raw.replace("\u00ad", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t\xa0]+", " ", ln).strip() for ln in t.split("\n")]
    lines = [ln for ln in lines if ln]
    out = []
    for ln in lines:
        if out and not (BULLET_RE.match(ln) or TOC_LINE_RE.search(ln)):
            out[-1] += " " + ln
        else:
            out.append(ln)
    return "\n".join(out).strip()


def pack_sentences(text: str, limit: int) -> list:
    """Длинный абзац режется по предложениям на куски <= limit."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    parts, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > limit:
            parts.append(cur)
            cur = s
        else:
            cur = (cur + " " + s) if cur else s
    if cur:
        parts.append(cur)
    return parts


def build_chunks(paras: list) -> list:
    """paras: [(page_no, section, text)] -> [(p_start, p_end, section, text)]"""
    chunks = []
    cur, cur_len = [], 0

    def flush():
        nonlocal cur, cur_len
        if not cur:
            return
        pages = [x[0] for x in cur]
        text = "\n\n".join(x[2] for x in cur)
        sec = next((x[1] for x in cur if x[1]), "")
        chunks.append((min(pages), max(pages), sec, text))
        tail, tl = [], 0
        for item in reversed(cur):
            if tl + len(item[2]) > OVERLAP_MAX:
                break
            tail.insert(0, item)
            tl += len(item[2])
        cur, cur_len = tail, tl

    for pno, sec, p in paras:
        parts = pack_sentences(p, CHUNK_MAX) if len(p) > CHUNK_MAX else [p]
        for part in parts:
            if cur and cur_len + len(part) > CHUNK_MAX:
                # после flush() в cur остаётся перекрытие (<= OVERLAP_MAX);
                # крупный кусок может слегка превысить CHUNK_MAX — это не критично
                flush()
            cur.append((pno, sec, part))
            cur_len += len(part)
    flush()
    return chunks


def page_data(page):
    """(метка страницы, базовый кегль, заголовки [(y,size,text)], блоки [(y,text)])"""
    label = ""
    try:
        label = page.get_label() or ""
    except Exception:
        pass
    d = page.get_text("dict")
    sizes, heads = [], []
    for b in d.get("blocks", []):
        if b.get("type", 0) != 0:
            continue
        for l in b.get("lines", []):
            spans = l.get("spans", [])
            if not spans:
                continue
            text = "".join(sp["text"] for sp in spans).strip()
            size = max(sp["size"] for sp in spans)
            bold = any((sp.get("flags", 0) & 16) or ("Bold" in sp.get("font", "")) for sp in spans)
            for sp in spans:
                sizes.extend([sp["size"]] * max(1, len(sp["text"])))
            if text and len(text) <= 130:
                heads.append((l["bbox"][1], size, bold, text))
    body = statistics.median(sizes) if sizes else 0.0
    heads.sort(key=lambda x: x[0])

    blocks = []
    for b in page.get_text("blocks"):
        if len(b) >= 7 and b[6] == 0 and b[4].strip():
            blocks.append((round(b[1]), b[0], b[4]))
    blocks.sort(key=lambda x: (x[0], x[1]))
    return label, body, heads, blocks


def is_toc_page(ptext: str) -> bool:
    lines = [ln for ln in ptext.split("\n") if ln.strip()]
    if len(lines) < 8:
        return False
    hits = sum(1 for ln in lines if TOC_LINE_RE.search(ln))
    return hits / len(lines) > 0.5


def is_heading_line(htext: str, size: float, bold: bool, body: float) -> bool:
    """Заголовок: крупнее основного текста, либо жирный/нумерованный короткий заголовок."""
    if re.fullmatch(r"[0-9\s.\-–—]+", htext):
        return False
    if body > 0 and size >= body * 1.18:
        return True
    if len(htext) <= 100 and (bold or re.match(r"^\d+\.\d+", htext)):
        return True
    return False


def extract_all(con: sqlite3.Connection):
    """Извлечение текста из всех PDF: таблицы pages и chunks (+stems)."""
    stats = []
    for fname, title in DOCS.items():
        path = BASE / fname
        if not path.exists():
            print(f"  !! {fname}: файл не найден, пропуск")
            continue
        doc = fitz.open(path)
        paras, doc_chars = [], 0
        chapter, section = "", ""

        # 1-й проход: данные страниц + частоты строк-заголовков
        page_datas, head_counts = [], {}
        for pno in range(len(doc)):
            pd = page_data(doc[pno])
            page_datas.append(pd)
            label, body, heads, _blocks = pd
            for _y, size, bold, htext in heads:
                if is_heading_line(htext, size, bold, body):
                    key = htext.strip().lower()
                    head_counts[key] = head_counts.get(key, 0) + 1
        # колонтитулы: строка-«заголовок», повторяющаяся на многих страницах
        header_limit = max(3, int(len(doc) * 0.15))

        for pno, (label, body, heads, blocks) in enumerate(page_datas):
            for _y, size, bold, htext in heads:
                if is_heading_line(htext, size, bold, body):
                    if head_counts.get(htext.strip().lower(), 0) > header_limit:
                        continue                      # колонтитул, не заголовок
                    if TOP_RE.match(htext):
                        chapter = htext
                        section = htext
                    else:
                        section = htext
            par_texts = []
            for _y, _x, raw in blocks:
                par = block_to_par(raw)
                if len(par) >= 25:
                    par_texts.append(par)
            ptext = "\n\n".join(par_texts)
            doc_chars += len(ptext)
            sec_label = f"{chapter} / {section}" if chapter and section and chapter != section else (chapter or section)
            con.execute(
                "INSERT INTO pages(doc, page, page_label, text) VALUES (?,?,?,?)",
                (title, pno + 1, label, ptext),
            )
            if is_toc_page(ptext):
                continue
            for par in par_texts:
                paras.append((pno + 1, sec_label, par))
        doc.close()

        chunks = build_chunks(paras)
        for p_start, p_end, sec, text in chunks:
            stems = " ".join(tokens_for_index(text))
            cur = con.execute(
                "INSERT INTO chunks(doc, page_start, page_end, section, text, stems) VALUES (?,?,?,?,?,?)",
                (title, p_start, p_end, sec, text, stems),
            )
            con.execute("INSERT INTO fts(rowid, stems) VALUES (?,?)", (cur.lastrowid, stems))
        n_pages = pno + 1
        stats.append((title, n_pages, doc_chars, len(chunks)))
        flag = "  !!! мало текста (нет слоя?) —" if doc_chars < 500 and n_pages > 2 else ""
        print(f"  {title}: {n_pages} стр., {doc_chars} симв., {len(chunks)} чанков{flag}")
    return stats


def embed_all(con: sqlite3.Connection, model_name: str):
    """Семантические векторы для всех чанков через fastembed."""
    import numpy as np
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=model_name)
    rows = con.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()
    t0 = time.time()
    batch, ids = [], []
    n = 0
    for cid, text in rows:
        batch.append(text)
        ids.append(cid)
        if len(batch) >= 64:
            for cid2, vec in zip(ids, model.embed(batch)):
                con.execute("UPDATE chunks SET vec=? WHERE id=?",
                            (np.asarray(vec, dtype=np.float32).tobytes(), cid2))
            n += len(batch)
            print(f"  ... {n}/{len(rows)} чанков ({time.time() - t0:.0f} с)")
            batch, ids = [], []
    if batch:
        for cid2, vec in zip(ids, model.embed(batch)):
            con.execute("UPDATE chunks SET vec=? WHERE id=?",
                        (np.asarray(vec, dtype=np.float32).tobytes(), cid2))
        n += len(batch)
    print(f"  векторы посчитаны для {n} чанков за {time.time() - t0:.0f} с")


def main():
    ap = argparse.ArgumentParser(description="Сборка RAG-базы AL-1803")
    ap.add_argument("--embed", action="store_true",
                    help="добавить семантические векторы (нужен fastembed)")
    ap.add_argument("--embed-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    args = ap.parse_args()

    con = sqlite3.connect(DB_PATH)
    con.executescript("""
        DROP TABLE IF EXISTS chunks;
        DROP TABLE IF EXISTS pages;
        DROP TABLE IF EXISTS fts;
        DROP TABLE IF EXISTS meta;
        CREATE TABLE chunks(
            id INTEGER PRIMARY KEY,
            doc TEXT NOT NULL,
            page_start INTEGER, page_end INTEGER,
            section TEXT,
            text TEXT NOT NULL,
            stems TEXT NOT NULL,
            vec BLOB
        );
        CREATE TABLE pages(
            doc TEXT, page INTEGER, page_label TEXT, text TEXT,
            PRIMARY KEY(doc, page)
        );
        CREATE VIRTUAL TABLE fts USING fts5(stems);
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
    """)

    print("Извлечение текста и построение индекса...")
    stats = extract_all(con)
    con.commit()

    total_chunks = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    con.executemany("INSERT INTO meta(key, value) VALUES (?,?)", [
        ("built", time.strftime("%Y-%m-%d %H:%M")),
        ("docs", str(len(stats))),
        ("chunks", str(total_chunks)),
        ("embed_model", args.embed_model if args.embed else ""),
    ])
    con.commit()

    if args.embed:
        print(f"Подсчёт эмбеддингов ({args.embed_model})...")
        try:
            embed_all(con, args.embed_model)
        except Exception as e:
            print(f"  !! эмбеддинги не удалось построить: {e}")
            print("  база сохранена в режиме полнотекстового поиска")
    con.commit()

    n_vec = con.execute("SELECT COUNT(*) FROM chunks WHERE vec IS NOT NULL").fetchone()[0]
    size_mb = DB_PATH.stat().st_size / 1e6
    print(f"\nГотово: {DB_PATH}")
    print(f"  документов: {len(stats)}, чанков: {total_chunks}, с векторами: {n_vec}")
    print(f"  размер базы: {size_mb:.1f} МБ")
    con.close()


if __name__ == "__main__":
    main()
