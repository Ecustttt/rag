import json
import logging
import time
import uuid
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _build_logger():
    logger = logging.getLogger("rag")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    fh = TimedRotatingFileHandler(
        LOG_DIR / "app.jsonl", when="midnight", backupCount=14, encoding="utf-8",
    )
    fh.setFormatter(JsonFormatter())
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)
    logger.propagate = False
    return logger


log = _build_logger()


class Trace:
    def __init__(self, session_id, question, mode, stream, user_id=None):
        self.trace_id = uuid.uuid4().hex[:16]
        self.t0 = time.perf_counter()
        self._marks = {}
        self._done = False
        self.data = {
            "trace_id": self.trace_id,
            "session_id": session_id,
            "user_id": user_id,
            "question": question,
            "mode": mode,
            "stream": stream,
            "status": "ok",
        }

    def mark(self, name):
        self._marks[name] = time.perf_counter()

    def set(self, **kw):
        self.data.update(kw)

    def _compute_durations(self):
        pairs = [
            ("intent_ms", "intent_start", "intent_end"),
            ("rewrite_ms", "rewrite_start", "rewrite_end"),
            ("retrieve_ms", "retrieve_start", "retrieve_end"),
            ("gen_first_ms", "gen_start", "gen_first_token"),
            ("gen_ms", "gen_start", "gen_end"),
        ]
        for name, a, b in pairs:
            if a in self._marks and b in self._marks:
                self.data[name] = int((self._marks[b] - self._marks[a]) * 1000)

    def emit(self, error=None, status=None):
        if self._done:
            return
        self._done = True
        self._compute_durations()
        if error:
            self.data["status"] = "error"
            self.data["error"] = str(error)[:500]
        elif status:
            self.data["status"] = status
        self.data["total_ms"] = int((time.perf_counter() - self.t0) * 1000)
        log.info("trace", extra={"fields": self.data}) #写入文件
        return dict(self.data) #返回数据给后端