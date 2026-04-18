from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def num_tag(value: float, decimals: int = 4) -> str:
    s = f"{float(value):.{decimals}f}"
    return s.replace('-', 'm').replace('.', 'p')


def task_id_for_value(name: str, value: float, decimals: int = 4) -> str:
    return f"{name}_{num_tag(value, decimals=decimals)}"


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return _jsonify(obj.tolist())
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        val = float(obj)
        return None if math.isnan(val) else val
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _fsync_parent_dir(path: Path) -> None:
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def atomic_write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(_jsonify(data), f, indent=indent)
        f.write('\n')
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)
    _fsync_parent_dir(path)


def append_jsonl(path: str | Path, data: Any, *, fsync: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        json.dump(_jsonify(data), f)
        f.write('\n')
        if fsync:
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
    if fsync:
        _fsync_parent_dir(path)


def load_jsonl(path: str | Path) -> list[Any]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[Any] = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return out


def load_json(path: str | Path) -> Any | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def strip_internal_keys(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not str(k).startswith('_')}


def load_task_row(*paths: str | Path) -> dict[str, Any] | None:
    """
    Load a per-task row from one of the candidate JSON files.

    Supported formats:
      - {"row": {...}, ...}
      - a plain row dict itself
    """
    for path in paths:
        data = load_json(path)
        if not isinstance(data, dict):
            continue
        row = data.get('row', data)
        if isinstance(row, dict):
            return strip_internal_keys(dict(row))
    return None


def save_task_row(
    row: dict[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
    task_result_path: str | Path | None = None,
    task_id: str | None = None,
    task_meta: dict[str, Any] | None = None,
) -> None:
    payload = {
        'task_id': task_id,
        'saved_at': utc_now_iso(),
        'row': strip_internal_keys(row),
    }
    if task_meta:
        payload.update(task_meta)
    if checkpoint_path is not None:
        atomic_write_json(checkpoint_path, payload)
    if task_result_path is not None:
        atomic_write_json(task_result_path, payload)


def save_error_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(text)
        if not text.endswith('\n'):
            f.write('\n')
    os.replace(tmp, path)


def normalise_results(rows: Iterable[dict[str, Any]], sort_key: str | None = None) -> list[dict[str, Any]]:
    out = [strip_internal_keys(dict(r)) for r in rows]
    if sort_key is not None:
        out.sort(key=lambda r: r.get(sort_key, float('inf')))
    return _jsonify(out)
