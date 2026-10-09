#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Веб-сервер RAG-базы по курсу AL-1803 (Astra Linux SE 1.8).

Только стандартная библиотека Python + модуль search.py (тот же каталог).
Слушает 127.0.0.1 — доступен исключительно с локальной машины.

Запуск:
    python server.py                 # http://127.0.0.1:8180
    python server.py --port 9000

API:
    GET  /                    страница-интерфейс (index.html)
    GET  /api/health          состояние базы
    POST /api/ask             {"query": "...", "k": 5}      — вопрос по курсу
    POST /api/quiz            {"question": "...", "options": ["...", ...]}
                              — выбор правильного варианта
    GET  /api/page?doc=&page= полный текст страницы документа
"""
import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import sqlite3
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import search as S  # noqa: E402  (модуль поиска рядом: kb.sqlite, стемминг, RRF)

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "kb.sqlite"
PORT = 8184
# Адрес привязки: локально 127.0.0.1; в контейнере (Render/Docker) задаётся RAG_HOST=0.0.0.0
BIND_HOST = os.environ.get("RAG_HOST", "127.0.0.1").strip()

# Ключ доступа читается только из переменной окружения RAG_ACCESS_KEY
# (задаётся в настройках Space/хостинга). Если не установлен — доступ без пароля.
ACCESS_KEY = os.environ.get("RAG_ACCESS_KEY", "").strip()

# Нейросеть (Groq, бесплатный тариф): ключ и модель — только через переменные окружения.
# Если GROQ_API_KEY не задан — запрос к нейросети вернёт подсказку о настройке.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b").strip()

# Карта «название документа → файл PDF» (pdf_map.json рядом с сервером).
# Файл опционален: без него ссылки на PDF просто не показываются.
# Поддерживаются алиасы (aliases) — короткие названия документов из базы.
PDF_MAP = {}
PDF_ALIAS = {}
_pdf_map_path = HERE / "pdf_map.json"
if _pdf_map_path.exists():
    try:
        PDF_MAP = json.loads(_pdf_map_path.read_text(encoding="utf-8"))
        for _fname, _info in PDF_MAP.items():
            PDF_ALIAS[_fname] = _fname
            for _a in (_info.get("aliases") or []):
                PDF_ALIAS[_a] = _fname
    except Exception:                                       # noqa: BLE001
        PDF_MAP, PDF_ALIAS = {}, {}


def check_access(req_key: str) -> bool:
    """Постоянное (constant-time) сравнение введённого ключа с заданным."""
    if not ACCESS_KEY:
        return True
    given = hashlib.sha256((req_key or "").encode("utf-8")).digest()
    expected = hashlib.sha256(ACCESS_KEY.encode("utf-8")).digest()
    return hmac.compare_digest(given, expected)

# ---------------------------------------------------------------- состояние
_lock = threading.Lock()
VEC_IDS = None          # список id чанков с векторами
VEC_MAT = None          # нормированная матрица векторов (numpy), или None
VEC_ERR = ""            # причина недоступности векторов
META = {}


def load_once():
    """Однократная загрузка векторов и метаданных базы в память."""
    global VEC_IDS, VEC_MAT, VEC_ERR, META
    con = sqlite3.connect(DB_PATH)
    try:
        META = dict(con.execute("SELECT key, value FROM meta").fetchall())
    except sqlite3.Error:
        META = {}
    # На хостинге с малым объёмом RAM (RAG_DISABLE_VEC=1) семантика отключается:
    # остаётся BM25-поиск, который даёт основное качество.
    if os.environ.get("RAG_DISABLE_VEC", "").strip() == "1":
        VEC_IDS, VEC_MAT, VEC_ERR = None, None, "отключён переменной RAG_DISABLE_VEC=1"
        con.close()
        return
    try:
        VEC_IDS, VEC_MAT = S.load_vectors(con)
    except Exception as e:                                  # noqa: BLE001
        VEC_IDS, VEC_MAT, VEC_ERR = None, None, f"{type(e).__name__}: {e}"
    con.close()


def db():
    return sqlite3.connect(DB_PATH)


def pdf_ref(doc: str):
    """Ссылка на PDF документа по его названию в базе (через алиасы), иначе None."""
    fname = PDF_ALIAS.get(doc)
    if fname:
        from urllib.parse import quote
        return "/pdf?f=" + quote(fname)
    return None


# ------------------------------------------------------------------ поиск
def fts_hits(con, query, limit, doc_filter=None):
    expr = S.fts_expr(query)
    if not expr:
        return []
    sql = ("SELECT c.id, c.doc, c.page_start, c.page_end, c.section, c.text, "
           "bm25(fts) AS b FROM fts JOIN chunks c ON c.id = fts.rowid "
           "WHERE fts MATCH ?")
    params = [expr]
    if doc_filter:
        sql += " AND c.doc LIKE ?"
        params.append(f"%{doc_filter}%")
    sql += " ORDER BY bm25(fts) LIMIT ?"
    params.append(limit)
    try:
        return con.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return []


def vec_hits(con, query, limit, doc_filter=None):
    """Топ чанков по косинусной близости (если векторы доступны)."""
    if VEC_MAT is None or VEC_IDS is None:
        return [], VEC_ERR or "векторы не построены"
    model = META.get("embed_model") or \
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    try:
        qv = S.embed_query(query, model)
    except Exception as e:                                  # noqa: BLE001
        return [], f"{type(e).__name__}: {e}"
    sims = VEC_MAT @ qv
    order = sims.argsort()[::-1]
    out = []
    for i in order:
        cid = int(VEC_IDS[i])
        row = con.execute(
            "SELECT id, doc, page_start, page_end, section, text FROM chunks WHERE id=?",
            (cid,)).fetchone()
        if not row:
            continue
        if doc_filter and doc_filter.lower() not in row[1].lower():
            continue
        out.append((row, float(sims[i])))
        if len(out) >= limit:
            break
    return out, ""


def retrieve(con, query, k=6, doc_filter=None):
    """Гибрид: BM25 + векторы, слияние по RRF. Возвращает список словарей."""
    pool = max(40, k * 4)
    fh = fts_hits(con, query, pool, doc_filter)
    vh, verr = vec_hits(con, query, pool, doc_filter)

    rrf, by_id = {}, {}
    for rank, row in enumerate(fh, start=1):
        cid = row[0]
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (60.0 + rank)
        by_id[cid] = row[:6]
    for rank, (row, _sim) in enumerate(vh, start=1):
        cid = row[0]
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (60.0 + rank)
        by_id[cid] = row
    ranked = sorted(rrf.items(), key=lambda x: -x[1])[:k]

    hits = []
    for cid, score in ranked:
        r = by_id[cid]
        hits.append({
            "id": r[0], "doc": r[1], "page_start": r[2], "page_end": r[3],
            "section": (r[4] or "").strip(), "text": r[5], "score": round(score, 5),
        })
    return hits, verr


# --------------------------------------------------- сборка ответа на вопрос
def q_stems(q):
    out = set()
    for t in S.TOKEN_RE.findall(q.lower()):
        if re.fullmatch(r"[0-9][0-9.\-]*", t):
            out.update(p for p in re.sub(r"[^0-9]+", " ", t).split() if p)
            continue
        if len(t) < 3 or t in S.STOP:
            continue
        out.add(S.stem(t))
    return out


SENT_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-ZА-ЯЁ0-9a-zа-яё«(])|\n+")


def sentences(text):
    parts = [s.strip() for s in SENT_RE.split(text) if s and s.strip()]
    return [p for p in parts if len(p) >= 20]


def score_sentence(sent, stems):
    toks = {S.stem(t) for t in S.TOKEN_RE.findall(sent.lower())
            if len(t) >= 3 and t not in S.STOP}
    if not toks:
        return 0.0
    ov = len(toks & stems)
    if not ov:
        return 0.0
    base = ov / (len(toks) ** 0.5)
    if ov >= 3:
        base *= 1.25
    if re.search(r"\b(команда|выполнить|параметр|командой|следует|необходимо)\b",
                 sent, re.IGNORECASE):
        base *= 1.08
    if re.search(r"[a-zа-яё_\-]+\s+[-–—]\s|\bman\b|/etc/|/var/|/usr/", sent,
                 re.IGNORECASE):
        base *= 1.08
    return base


def build_answer(query, k):
    con = db()
    try:
        hits, verr = retrieve(con, query, k=k)
        if not hits:
            return {"query": query, "answer": [], "sources": [],
                    "note": "Ничего не найдено — попробуйте переформулировать запрос."}
        stems = q_stems(query)
        for h in hits:
            h["pdf_url"] = pdf_ref(h["doc"])

        # кандидаты-предложения по всем найденным чанкам
        cand = []
        for h in hits:
            for idx, s in enumerate(sentences(h["text"])):
                sc = score_sentence(s, stems)
                if sc > 0:
                    cand.append((sc, h, idx, s))
        cand.sort(key=lambda x: -x[0])

        # отбор с дедупликацией почти одинаковых предложений
        picked, seen = [], set()
        for sc, h, idx, s in cand:
            key = re.sub(r"\W+", " ", s.lower())[:90]
            if key in seen:
                continue
            seen.add(key)
            picked.append({
                "text": s, "doc": h["doc"], "page_start": h["page_start"],
                "page_end": h["page_end"], "section": h["section"],
                "pdf_url": h.get("pdf_url"),
                "score": round(sc, 3),
            })
            if len(picked) >= min(7, max(3, k)):
                break

        # источники: чанки с наиболее релевантным сниппетом
        terms = S.plain_terms(query)
        sources = []
        for h in hits[:k]:
            sources.append({
                "id": h["id"], "doc": h["doc"],
                "page_start": h["page_start"], "page_end": h["page_end"],
                "section": h["section"], "score": h["score"],
                "pdf_url": h.get("pdf_url"),
                "snippet": S.make_snippet(h["text"], terms or list(stems), width=700),
                "text": h["text"],
            })
        out = {"query": query, "answer": picked, "sources": sources}
        if verr:
            out["note"] = f"Семантический поиск недоступен ({verr}) — использован полнотекстовый."
        return out
    finally:
        con.close()


# ------------------------------------------------- оценка вариантов ответа
def option_support(con, question, option):
    """Поддержка варианта документацией: BM25 + косинусная близость."""
    q = f"{question} {option}".strip()
    fh = fts_hits(con, q, 8)
    # BM25 в SQLite FTS5: чем меньше — тем лучше; берём сумму по топ-3
    fts_score = 0.0
    for _cid, _doc, _p1, _p2, _sec, _txt, b in fh[:3]:
        fts_score += 1.0 / (1.0 + abs(float(b)))

    vscore = 0.0
    best_row = None
    if VEC_MAT is not None:
        model = META.get("embed_model") or \
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        try:
            qv = S.embed_query(q, model)
            sims = VEC_MAT @ qv
            idx = sims.argsort()[::-1][:3]
            vscore = float(sum(sims[i] for i in idx) / len(idx))
            best_i = int(sims.argmax())
            best_row = con.execute(
                "SELECT id, doc, page_start, page_end, section, text FROM chunks WHERE id=?",
                (int(VEC_IDS[best_i]),)).fetchone()
        except Exception:                                   # noqa: BLE001
            vscore = 0.0
    if best_row is None and fh:
        best_row = fh[0][:6]
    return {"fts": fts_score, "vec": vscore, "row": best_row}


def build_quiz(question, options):
    con = db()
    try:
        stats = []
        for i, opt in enumerate(options):
            s = option_support(con, question, opt)
            stats.append({"index": i, "option": opt, **s})

        # нормализация двух метрик в [0,1] и объединение
        def norm(key):
            vals = [x[key] for x in stats]
            lo, hi = min(vals), max(vals)
            span = (hi - lo) or 1.0
            return [(v - lo) / span for v in vals]

        fn, vn = norm("fts"), norm("vec")
        for s, f, v in zip(stats, fn, vn):
            s["combined"] = round(0.5 * f + 0.5 * v, 4)

        order = sorted(stats, key=lambda x: -x["combined"])
        best, runner = order[0], (order[1] if len(order) > 1 else None)
        margin = best["combined"] - (runner["combined"] if runner else 0.0)

        terms = S.plain_terms(question + " " + best["option"])
        row = best["row"]
        evidence = []
        if row:
            evidence = [{
                "id": row[0], "doc": row[1],
                "page_start": row[2], "page_end": row[3],
                "section": (row[4] or "").strip(),
                "snippet": S.make_snippet(row[5], terms, width=650),
                "text": row[5],
                "pdf_url": pdf_ref(row[1]),
            }]
        # дополнительно: чанки по запросу «вопрос + правильный вариант»
        for h2 in retrieve(con, f"{question} {best['option']}", k=3)[0]:
            if row and h2["id"] == row[0]:
                continue
            evidence.append({
                "id": h2["id"], "doc": h2["doc"],
                "page_start": h2["page_start"], "page_end": h2["page_end"],
                "section": h2["section"],
                "pdf_url": pdf_ref(h2["doc"]),
                "snippet": S.make_snippet(h2["text"], terms, width=650),
                "text": h2["text"],
            })

        # цитаты: предложения, подтверждающие выбранный вариант
        stems = q_stems(question + " " + best["option"])
        quotes = []
        for ev in evidence:
            for s in sorted(sentences(ev["text"]),
                            key=lambda x: -score_sentence(x, stems)):
                sc = score_sentence(s, stems)
                if sc <= 0:
                    continue
                quotes.append({"text": s, "doc": ev["doc"],
                               "page_start": ev["page_start"],
                               "page_end": ev["page_end"],
                               "section": ev["section"],
                               "pdf_url": ev.get("pdf_url")})
                break
            if len(quotes) >= 3:
                break

        # уверенность по отрыву лидера
        if margin >= 0.35:
            conf = "высокая"
        elif margin >= 0.12:
            conf = "средняя"
        else:
            conf = "низкая"

        return {
            "question": question,
            "options": [
                {"index": s["index"], "text": s["option"],
                 "combined": s["combined"], "fts": round(s["fts"], 4),
                 "vec": round(s["vec"], 4),
                 "is_answer": s["index"] == best["index"]}
                for s in stats
            ],
            "answer_index": best["index"],
            "answer_text": best["option"],
            "confidence": conf,
            "margin": round(margin, 4),
            "quotes": quotes,
            "evidence": evidence,
            "ranking": [s["index"] for s in order],
        }
    finally:
        con.close()


def groq_chat(prompt):
    """Запрос к Groq (OpenAI-совместимый API). Возвращает (ответ, модель, ошибка)."""
    import urllib.request
    import urllib.error
    body = json.dumps({
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 2048,
    }).encode("utf-8")
    auth_header = "Bearer " + GROQ_API_KEY
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=body, method="POST",
        headers={"Authorization": auth_header,
                 "Content-Type": "application/json",
                 "User-Agent": "astra-rag/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=110) as r:
            d = json.loads(r.read().decode("utf-8"))
        return d["choices"][0]["message"]["content"].strip(), d.get("model", GROQ_MODEL), ""
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode("utf-8")).get("error", {}).get("message", "")
        except Exception:                                   # noqa: BLE001
            detail = ""
        return "", GROQ_MODEL, f"Ошибка Groq API (HTTP {e.code}): {detail or 'нет деталей'}"
    except Exception as e:                                  # noqa: BLE001
        return "", GROQ_MODEL, f"Не удалось обратиться к нейросети: {type(e).__name__}: {e}"


# ------------------------------------------------------------------ сервер
class Handler(BaseHTTPRequestHandler):
    server_version = "AL1804-RAG/1.0"

    def log_message(self, fmt, *args):                      # тихий лог
        pass

    def _provided_key(self) -> str:
        """Ключ из HTTP-заголовка Basic-аутентификации (браузер спрашивает его сам)."""
        header = self.headers.get("Authorization") or ""
        if not header.startswith("Basic "):
            return ""
        try:
            decoded = base64.b64decode(header[6:].strip()).decode("utf-8")
        except Exception:                                   # noqa: BLE001
            return ""
        return decoded.split(":", 1)[1] if ":" in decoded else decoded

    def _require_access(self) -> bool:
        """True — доступ разрешён; иначе отправляет 401 и возвращает False."""
        if check_access(self._provided_key()):
            return True
        body = '{"error": "требуется ключ доступа"}'.encode("utf-8")
        try:
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="Astra RAG"')
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception:                                   # noqa: BLE001
            pass
        return False

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def do_GET(self):                                       # noqa: N802
        from urllib.parse import urlparse, parse_qs
        if not self._require_access():
            return
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            f = HERE / "index.html"
            if not f.exists():
                return self._send(404, b"index.html not found",
                                  "text/plain; charset=utf-8")
            return self._send(200, f.read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/health":
            con = db()
            try:
                chunks = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                docs = con.execute("SELECT COUNT(DISTINCT doc) FROM chunks").fetchone()[0]
            finally:
                con.close()
            return self._json({"ok": True, "chunks": chunks, "docs": docs,
                               "vector_search": VEC_MAT is not None,
                               "pdf_docs": sorted(PDF_MAP.keys()),
                               "meta": META})
        if u.path == "/api/page":
            q = parse_qs(u.query)
            doc = (q.get("doc") or [""])[0]
            try:
                page = int((q.get("page") or ["0"])[0])
            except ValueError:
                return self._json({"error": "bad page"}, 400)
            con = db()
            try:
                rows = con.execute(
                    "SELECT page, page_label, text FROM pages "
                    "WHERE doc LIKE ? AND page=?", (f"%{doc}%", page)).fetchall()
            finally:
                con.close()
            if not rows:
                return self._json({"error": "not found"}, 404)
            return self._json({"doc": doc, "page": page,
                               "label": rows[0][1] or "", "text": rows[0][2]})
        if u.path == "/api/pdfs":
            return self._json({"pdfs": PDF_MAP})
        if u.path == "/pdf":
            q = parse_qs(u.query)
            fname = (q.get("f") or [""])[0]
            if fname not in PDF_MAP:
                return self._json({"error": "unknown document"}, 404)
            pdf_path = HERE / "pdf" / fname
            if not pdf_path.exists():
                return self._json({"error": "PDF file not found on server"}, 404)
            data = pdf_path.read_bytes()
            from urllib.parse import quote
            ascii_name = fname.encode("ascii", "ignore").decode() or "document.pdf"
            utf8_name = quote(fname)
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Disposition",
                             f"inline; filename=\"{ascii_name}\"; filename*=UTF-8''{utf8_name}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        return self._json({"error": "not found"}, 404)

    def do_POST(self):                                      # noqa: N802
        from urllib.parse import urlparse
        if not self._require_access():
            return
        path = urlparse(self.path).path
        try:
            data = self._read_json()
        except Exception:                                   # noqa: BLE001
            return self._json({"error": "invalid JSON"}, 400)

        if path == "/api/ask":
            q = (data.get("query") or "").strip()
            if not q:
                return self._json({"error": "Пустой запрос"}, 400)
            try:
                k = int(data.get("k") or 5)
            except (TypeError, ValueError):
                k = 5
            k = max(1, min(15, k))
            with _lock:
                res = build_answer(q, k)
            return self._json(res)

        if path == "/api/quiz":
            q = (data.get("question") or "").strip()
            opts = [str(o).strip() for o in (data.get("options") or [])]
            opts = [o for o in opts if o]
            if not q or len(opts) < 2:
                return self._json(
                    {"error": "Нужны вопрос и минимум два варианта ответа"}, 400)
            if len(opts) > 10:
                return self._json({"error": "Слишком много вариантов (макс. 10)"}, 400)
            with _lock:
                res = build_quiz(q, opts)
            return self._json(res)

        if path == "/api/llm":
            prompt = (data.get("prompt") or "").strip()
            if not prompt:
                return self._json({"error": "Пустой промпт"}, 400)
            if not GROQ_API_KEY:
                return self._json({
                    "error": "Нейросеть не настроена на сервере: не задана переменная окружения GROQ_API_KEY.",
                }, 503)
            with _lock:
                answer, model, err = groq_chat(prompt)
            if err:
                return self._json({"error": err}, 502)
            return self._json({"answer": answer, "model": model})

        return self._json({"error": "not found"}, 404)


def main():
    global PORT
    ap = argparse.ArgumentParser(description="Веб-сервер RAG-базы AL-1803")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    PORT = args.port

    if not DB_PATH.exists():
        sys.exit("База не найдена. Сначала выполните: python build_kb.py --embed")

    print("Загрузка индекса в память...")
    load_once()
    print(f"  база: {DB_PATH}")
    print(f"  чанков: {META.get('chunks', '?')}, собрана: {META.get('built', '?')}")
    if VEC_MAT is None:
        print(f"  ! семантический поиск недоступен: {VEC_ERR}")
    else:
        print(f"  семантический поиск: включён ({VEC_MAT.shape[0]} векторов)")
    if ACCESS_KEY:
        print("  ключ доступа: установлен (переменная RAG_ACCESS_KEY)")
    else:
        print("  ключ доступа: НЕ установлен — страница открыта всем")
    if GROQ_API_KEY:
        print(f"  нейросеть (Groq): включена, модель {GROQ_MODEL}")
    else:
        print("  нейросеть (Groq): не настроена (нет GROQ_API_KEY)")

    srv = ThreadingHTTPServer((BIND_HOST, PORT), Handler)
    url = f"http://{BIND_HOST}:{PORT}/"
    print(f"\nОткройте в браузере: {url}\n(остановить — Ctrl+C)")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:                                   # noqa: BLE001
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено.")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
