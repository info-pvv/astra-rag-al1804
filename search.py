#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Поиск по RAG-базе знаний AL-1803 (Astra Linux SE 1.8).

Примеры:
    python search.py "мандатный контроль целостности"
    python search.py "как изменить права доступа к файлу" -k 10
    python search.py "PARSEC привилегии" --full            # полные тексты чанков
    python search.py --id 512                              # чанк 512 и соседи целиком
    python search.py --page "Руководство по КСЗ ч.1 (PARSEC)" 210   # страница целиком

Режимы поиска: auto (по умолчанию — гибрид, если есть векторы), fts, vec.
"""
import argparse
import re
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import snowballstemmer

DB_PATH = Path(__file__).resolve().parent / "kb.sqlite"
_rus = snowballstemmer.stemmer("russian")
_eng = snowballstemmer.stemmer("english")

STOP = {
    "и", "в", "во", "не", "на", "с", "со", "по", "для", "как", "это", "или",
    "то", "же", "от", "до", "при", "из", "за", "к", "у", "о", "а", "но",
    "если", "что", "чтобы", "его", "их", "ее", "её", "так", "также", "об",
    "бы", "ли", "ты", "он", "она", "они", "мы", "вы", "the", "and", "for",
    "not", "you", "with",
}
TOKEN_RE = re.compile(r"[0-9a-zа-яё][0-9a-zа-яё.\-]*", re.IGNORECASE)


def stem(word: str) -> str:
    s = _rus.stemWord(word)
    if s and s != word:
        return s
    return _eng.stemWord(word) or word


def fts_expr(query: str) -> str:
    parts, seen = [], set()
    for t in TOKEN_RE.findall(query.lower()):
        if re.fullmatch(r"[0-9][0-9.\-]*", t):
            clean = re.sub(r"[^0-9]+", " ", t).strip()
            if " " in clean:
                part = '"%s"' % clean
            elif len(clean) >= 2:
                part = clean
            else:
                continue
        elif re.search(r"[.\-]", t):
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


def plain_terms(query: str) -> list:
    """Слова запроса для поиска окна сниппета."""
    terms = set()
    for t in TOKEN_RE.findall(query.lower()):
        if len(t) >= 3 and t not in STOP:
            terms.add(t)
            s = stem(t)
            if len(s) >= 4:
                terms.add(s[:6])
    return sorted(terms)


def make_snippet(text: str, terms: list, width: int = 600) -> str:
    flat = text.replace("\n", " ")
    hits = []
    for t in terms:
        try:
            hits.extend(m.start() for m in re.finditer(re.escape(t), flat, re.IGNORECASE))
        except re.error:
            pass
    if hits:
        c = sorted(hits)[len(hits) // 2]
        a = max(0, c - width // 2)
        b = min(len(flat), a + width)
        a = max(0, b - width)
        return ("…" if a > 0 else "") + flat[a:b] + ("…" if b < len(flat) else "")
    return flat[:width] + ("…" if len(flat) > width else "")


def load_vectors(con):
    import numpy as np
    rows = con.execute("SELECT id, vec FROM chunks WHERE vec IS NOT NULL ORDER BY id").fetchall()
    if not rows:
        return None, None
    ids = [r[0] for r in rows]
    dim = len(rows[0][1]) // 4
    mat = np.zeros((len(rows), dim), dtype=np.float32)
    for i, (_cid, blob) in enumerate(rows):
        mat[i] = np.frombuffer(blob, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return ids, mat / norms


_model = None


def embed_query(query: str, model_name: str):
    """Вектор запроса (кэшируется модель между вызовами)."""
    import numpy as np
    from fastembed import TextEmbedding
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=model_name)
    if hasattr(_model, "query_embed"):
        vec = next(iter(_model.query_embed([query])))
    else:
        vec = next(iter(_model.embed(["query: " + query])))
    v = np.asarray(vec, dtype=np.float32)
    return v / (float(np.linalg.norm(v)) or 1.0)


def print_hit(rank, score, row, terms, full=False):
    cid, doc, p1, p2, section, text = row
    pages = f"стр. {p1}" if p1 == p2 else f"стр. {p1}–{p2}"
    sec = (section or "").strip()
    if len(sec) > 90:
        sec = sec[:87] + "..."
    print(f"{rank}) [{score}] {doc} — {pages} [id {cid}]")
    if sec:
        print(f"   Раздел: {sec}")
    if full:
        bar = "=" * 70
        print("   " + bar)
        for line in text.split("\n"):
            print("   " + line)
        print("   " + bar)
    else:
        snip = make_snippet(text, terms)
        for chunk in [snip[i:i + 96] for i in range(0, len(snip), 96)]:
            print("   " + chunk)
    print()


def main():
    ap = argparse.ArgumentParser(description="Поиск по RAG-базе AL-1803")
    ap.add_argument("query", nargs="*", help="поисковый запрос")
    ap.add_argument("-k", type=int, default=8, help="сколько результатов (по умолчанию 8)")
    ap.add_argument("--full", action="store_true", help="печатать полные тексты чанков")
    ap.add_argument("--mode", choices=["auto", "fts", "vec"], default="auto")
    ap.add_argument("--doc", default="", help="фильтр: подстрока названия документа")
    ap.add_argument("--id", type=int, help="показать чанк по id и его соседей")
    ap.add_argument("--page", nargs=2, metavar=("ДОК", "СТР"),
                    help="показать страницу целиком")
    args = ap.parse_args()
    query = " ".join(args.query).strip()

    if not DB_PATH.exists():
        sys.exit("База не найдена. Сначала запустите: python build_kb.py")
    con = sqlite3.connect(DB_PATH)

    if args.id:
        for offset in (-1, 0, 1):
            row = con.execute(
                "SELECT id, doc, page_start, page_end, section, text FROM chunks WHERE id=?",
                (args.id + offset,),
            ).fetchone()
            if row:
                if offset:
                    print(f"--- сосед (id {row[0]}) ---")
                print_hit(row[0], "", row, [], full=True)
        return

    if args.page:
        doc, page = args.page[0], int(args.page[1])
        rows = con.execute(
            "SELECT page, page_label, text FROM pages WHERE doc LIKE ? AND page=?",
            (f"%{doc}%", page),
        ).fetchall()
        if not rows:
            sys.exit(f"Страница не найдена: {doc}, стр. {page}")
        for p, label, text in rows:
            lbl = f" (метка «{label}»)" if label else ""
            print(f"=== {doc}, стр. {p}{lbl} ===\n")
            print(text)
        return

    if not query:
        sys.exit('Укажите запрос, например: python search.py "мандатный контроль целостности"')

    doc_filter = f"%{args.doc}%" if args.doc else None

    # --- полнотекстовый поиск (BM25 по стеммам) ---
    fts_hits = []
    expr = fts_expr(query)
    if expr:
        sql = ("SELECT c.id, c.doc, c.page_start, c.page_end, c.section, c.text "
               "FROM fts JOIN chunks c ON c.id = fts.rowid WHERE fts MATCH ?")
        params = [expr]
        if doc_filter:
            sql += " AND c.doc LIKE ?"
            params.append(doc_filter)
        sql += " ORDER BY bm25(fts) LIMIT ?"
        params.append(max(40, args.k * 3))
        fts_hits = con.execute(sql, params).fetchall()

    # --- семантический поиск, если в базе есть векторы ---
    vec_hits, vec_err = [], None
    n_vec = con.execute("SELECT COUNT(*) FROM chunks WHERE vec IS NOT NULL").fetchone()[0]
    if args.mode in ("auto", "vec") and n_vec > 0:
        try:
            ids, mat = load_vectors(con)
            meta = con.execute("SELECT value FROM meta WHERE key='embed_model'").fetchone()
            model_name = (meta[0] if meta and meta[0] else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            qv = embed_query(query, model_name)
            sims = mat @ qv
            order = sims.argsort()[::-1]
            for i in order:
                row = con.execute(
                    "SELECT id, doc, page_start, page_end, section, text FROM chunks WHERE id=?",
                    (int(ids[i]),),
                ).fetchone()
                if doc_filter and args.doc.lower() not in row[1].lower():
                    continue
                vec_hits.append(row)
                if len(vec_hits) >= max(40, args.k * 3):
                    break
        except Exception as e:
            vec_err = f"{type(e).__name__}: {e}"

    # --- выбор и слияние результатов ---
    if args.mode == "vec":
        ranked = [(r[0], "vec", r) for r in vec_hits]
    elif args.mode == "fts" or not vec_hits:
        ranked = [(r[0], "bm25", r) for r in fts_hits]
    else:
        rrf, by_id = {}, {}
        for lst in (fts_hits, vec_hits):
            for r, row in enumerate(lst, start=1):
                cid = row[0]
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (60.0 + r)
                by_id[cid] = row
        ranked = [(cid, f"rrf={s:.4f}", by_id[cid])
                  for cid, s in sorted(rrf.items(), key=lambda x: -x[1])]

    if not ranked:
        print("Ничего не найдено. Попробуйте переформулировать запрос.")
        if vec_err:
            print(f"(семантический поиск недоступен: {vec_err})")
        return

    terms = plain_terms(query)
    print(f"Запрос: «{query}»  |  показано {min(len(ranked), args.k)} из {len(ranked)}\n")
    for i, (_cid, score, row) in enumerate(ranked[:args.k], start=1):
        print_hit(i, score, row, terms, full=args.full)

    if vec_err and args.mode == "auto":
        print(f"(семантический поиск недоступен, использован полнотекстовый: {vec_err})")


if __name__ == "__main__":
    main()
