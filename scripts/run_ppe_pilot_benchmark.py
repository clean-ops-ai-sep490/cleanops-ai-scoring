from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from urllib import error, request


DEFAULT_FIELDS = [
    "case_id",
    "image_url",
    "required_objects",
    "expected_status",
    "predicted_status",
    "expected_missing_items",
    "predicted_missing_items",
    "latency_ms",
    "notes",
]


def _split_items(raw: str) -> list[str]:
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


def _post_json(url: str, payload: dict, timeout_sec: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_sec) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _clean_error_message(exc: Exception) -> str:
    if isinstance(exc, error.HTTPError):
        try:
            payload = exc.read().decode("utf-8")
        except Exception:  # noqa: BLE001
            payload = str(exc)
        return f"HTTP {exc.code}: {payload}"
    return str(exc)


def run_benchmark(input_csv: Path, output_csv: Path, api_base_url: str, timeout_sec: int) -> None:
    endpoint = f"{api_base_url.rstrip('/')}/ppe/evaluate"
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"Input CSV contains no benchmark rows: {input_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys([*DEFAULT_FIELDS, *rows[0].keys()]))

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            image_url = (row.get("image_url") or "").strip()
            required_objects = _split_items(row.get("required_objects", ""))
            if not image_url or not required_objects:
                raise ValueError(f"Row {row.get('case_id', '<missing>')} is missing image_url or required_objects.")

            started = time.perf_counter()
            try:
                payload = _post_json(
                    endpoint,
                    {"image_urls": [image_url], "required_objects": required_objects},
                    timeout_sec,
                )
                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                row["predicted_status"] = str(payload.get("status", "")).upper()
                row["predicted_missing_items"] = ",".join(payload.get("missing_items") or [])
                row["latency_ms"] = elapsed_ms
            except Exception as exc:  # noqa: BLE001
                row["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                row["notes"] = f"{row.get('notes', '').strip()} | inference_error={_clean_error_message(exc)}".strip()

            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the real PPE pilot benchmark against a live scoring API.")
    parser.add_argument("--input-csv", default="benchmarks/ppe/pilot_benchmark.csv")
    parser.add_argument("--output-csv", default="benchmarks/reports/ppe_pilot_evaluated.csv")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-sec", type=int, default=60)
    args = parser.parse_args()

    run_benchmark(
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
        api_base_url=args.api_base_url,
        timeout_sec=args.timeout_sec,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
