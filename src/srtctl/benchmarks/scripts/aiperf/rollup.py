#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate benchmark-rollup.json and benchmark-rollup.csv from aiperf concurrency-sweep results.

Emits the SAME schema as sa-bench's rollup so IBDB ingests aiperf runs identically. One row per
concurrency, read from ``<log_dir>/artifacts/conc_<c>/**/profile_export_aiperf.json``.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OUTPUT_FIELDS = [
    "Config",
    "Total GPU Count",
    "Decode GPU Count",
    "Concurrency",
    "Total Token Throughput",
    "Output Token Throughput",
    "Median TTFT",
    "Median TPOT",
    "Median ITL",
    "P90 Decode Running Requests",
    "Output Token Throughput per User",
    "Total Token Throughput per GPU",
]

CONC_DIR_RE = re.compile(r"conc_(\d+)")


def _read_job_metadata(log_dir: Path) -> dict[str, Any] | None:
    for metadata_path in sorted(log_dir.parent.glob("*.json")):
        try:
            data = json.loads(metadata_path.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to parse {metadata_path}: {exc}", file=sys.stderr)
            continue
        if data and "resources" in data:
            return data
    return None


def _as_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _compute_gpu_counts(resources: dict[str, Any]) -> tuple[int | None, int | None]:
    gpus_per_node = _as_int(resources.get("gpus_per_node"))
    prefill_nodes = _as_int(resources.get("prefill_nodes"))
    decode_nodes = _as_int(resources.get("decode_nodes"))
    agg_nodes = _as_int(resources.get("agg_nodes"))
    if gpus_per_node <= 0:
        return None, None
    if prefill_nodes > 0 or decode_nodes > 0:
        total = (prefill_nodes + decode_nodes) * gpus_per_node
    elif agg_nodes > 0:
        total = agg_nodes * gpus_per_node
    else:
        total = gpus_per_node
    if decode_nodes > 0:
        return total, decode_nodes * gpus_per_node
    dw, gpd = _as_int(resources.get("decode_workers")), _as_int(resources.get("gpus_per_decode"))
    if dw > 0 and gpd > 0:
        return total, dw * gpd
    aw, gpa = _as_int(resources.get("agg_workers")), _as_int(resources.get("gpus_per_agg"))
    if aw > 0 and gpa > 0:
        return total, aw * gpa
    if agg_nodes > 0:
        return total, total
    return total, None


def _scalar(metric: Any) -> float | None:
    """aiperf metrics are dicts with 'avg'/'p50'/... ; extract a scalar mean."""
    if isinstance(metric, dict):
        return metric.get("avg")
    if isinstance(metric, (int, float)):
        return float(metric)
    return None


def _pctl(metric: Any, key: str) -> float | None:
    return metric.get(key) if isinstance(metric, dict) else None


def _safe_ratio(n: float | int | None, d: float | int | None) -> float | None:
    if n is None or d in (None, 0):
        return None
    return float(n) / float(d)


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def main(log_dir: Path) -> None:
    artifacts = log_dir / "artifacts"
    files = sorted(artifacts.glob("conc_*/**/profile_export_aiperf.json")) if artifacts.exists() else []
    if not files:
        print("No aiperf results found", file=sys.stderr)
        return

    metadata = _read_job_metadata(log_dir)
    config_name = metadata.get("job_name") if metadata else None
    resources = metadata.get("resources") if metadata else None
    total_gpu, decode_gpu = _compute_gpu_counts(resources) if resources else (None, None)
    bench = (metadata or {}).get("benchmark", {})

    runs: list[dict[str, Any]] = []
    csv_rows: list[dict[str, object]] = []
    config = {"model": (metadata or {}).get("model", {}).get("path"),
              "isl": bench.get("isl"), "osl": bench.get("osl")}

    # de-dupe by concurrency, keeping the newest export
    by_conc: dict[int, Path] = {}
    for f in files:
        m = CONC_DIR_RE.search(str(f))
        if not m:
            continue
        c = int(m.group(1))
        if c not in by_conc or f.stat().st_mtime > by_conc[c].stat().st_mtime:
            by_conc[c] = f

    for conc in sorted(by_conc):
        try:
            d = json.loads(by_conc[conc].read_text())
        except json.JSONDecodeError as exc:
            print(f"Failed to parse {by_conc[conc]}: {exc}", file=sys.stderr)
            continue
        out_tps = _scalar(d.get("output_token_throughput"))
        tot_tps = _scalar(d.get("total_token_throughput"))
        ttft = d.get("time_to_first_token", {})
        itl = d.get("inter_token_latency", {})
        e2el = d.get("request_latency", {})
        tpu = _scalar(d.get("output_token_throughput_per_user"))

        runs.append({
            "concurrency": conc,
            "throughput_toks": out_tps,
            "request_throughput": _scalar(d.get("request_throughput")),
            "ttft_mean_ms": _scalar(ttft),
            "ttft_p99_ms": _pctl(ttft, "p99"),
            "tpot_mean_ms": _scalar(itl),
            "tpot_p99_ms": _pctl(itl, "p99"),
            "itl_mean_ms": _scalar(itl),
            "itl_p99_ms": _pctl(itl, "p99"),
            "e2el_mean_ms": _scalar(e2el),
            "completed_requests": None,
            "total_input_tokens": _scalar(d.get("total_usage_prompt_tokens")),
            "total_output_tokens": _scalar(d.get("total_output_tokens")),
        })

        csv_rows.append({k: _fmt(v) for k, v in {
            "Config": config_name or str(config.get("model") or "aiperf"),
            "Total GPU Count": total_gpu,
            "Decode GPU Count": decode_gpu,
            "Concurrency": conc,
            "Total Token Throughput": tot_tps,
            "Output Token Throughput": out_tps,
            "Median TTFT": _pctl(ttft, "p50"),
            "Median TPOT": _pctl(itl, "p50"),
            "Median ITL": _pctl(itl, "p50"),
            "P90 Decode Running Requests": None,
            "Output Token Throughput per User": tpu,
            "Total Token Throughput per GPU": _safe_ratio(tot_tps, total_gpu),
        }.items()})

    if not runs:
        print("No valid aiperf results found", file=sys.stderr)
        return

    rollup = {
        "benchmark_type": "aiperf",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": config,
        "runs": runs,
    }
    (log_dir / "benchmark-rollup.json").write_text(json.dumps(rollup, indent=2))
    print(f"Wrote {log_dir / 'benchmark-rollup.json'}")

    csv_rows.sort(key=lambda r: int(r["Concurrency"]) if r["Concurrency"] else -1)
    with (log_dir / "benchmark-rollup.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"Wrote {log_dir / 'benchmark-rollup.csv'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/logs"))
