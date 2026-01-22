from __future__ import annotations

import argparse
import io
import os
import statistics
import sys
import time
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from fastapi.testclient import TestClient


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BACKEND_DIR))

from main import app  # noqa: E402


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = max(0, min(len(values_sorted) - 1, int(round((p / 100.0) * (len(values_sorted) - 1)))))
    return float(values_sorted[k])


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "count": float(len(values)),
        "avg_ms": statistics.mean(values) * 1000 if values else 0.0,
        "p50_ms": _pct(values, 50) * 1000,
        "p95_ms": _pct(values, 95) * 1000,
        "max_ms": (max(values) * 1000) if values else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Dataset API benchmark (in-process TestClient).")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--user", type=str, default="bench")
    parser.add_argument("--api-key", type=str, default="bench-key")
    args = parser.parse_args()

    results: dict[str, list[float]] = {
        "create": [],
        "list": [],
        "upload": [],
        "download_zip": [],
        "delete": [],
    }
    failures: list[str] = []

    with TemporaryDirectory() as tmpdir:
        os.environ["AIPT_STORAGE_DIR"] = str(Path(tmpdir) / "storage")
        os.environ["AIPT_API_KEY"] = args.api_key
        os.environ.setdefault("AIPT_DATABASE_URL", "sqlite:///:memory:")

        headers = {"X-User": args.user, "X-API-Key": args.api_key}

        with TestClient(app) as client:
            resp = client.post("/projects", json={"name": f"bench-project-{uuid4().hex[:8]}", "type": "目标检测"})
            resp.raise_for_status()
            project_id = resp.json()["id"]

            for i in range(args.iterations):
                name = f"bench-{uuid4().hex[:8]}"

                t0 = time.perf_counter()
                resp = client.post("/datasets", json={"name": name, "project_id": project_id}, headers=headers)
                results["create"].append(time.perf_counter() - t0)
                if resp.status_code != 201:
                    failures.append(f"create[{i}]={resp.status_code}:{resp.text}")
                    continue
                dataset_id = resp.json()["data"]["id"]

                t0 = time.perf_counter()
                resp = client.get("/datasets", headers={"X-User": args.user})
                results["list"].append(time.perf_counter() - t0)
                if resp.status_code != 200:
                    failures.append(f"list[{i}]={resp.status_code}:{resp.text}")

                t0 = time.perf_counter()
                resp = client.post(
                    f"/datasets/{dataset_id}/files",
                    headers=headers,
                    files=[("files", ("tiny.txt", b"x" * 1024, "text/plain"))],
                )
                results["upload"].append(time.perf_counter() - t0)
                if resp.status_code != 201:
                    failures.append(f"upload[{i}]={resp.status_code}:{resp.text}")

                t0 = time.perf_counter()
                resp = client.get(f"/datasets/{dataset_id}/download", headers={"X-User": args.user})
                results["download_zip"].append(time.perf_counter() - t0)
                if resp.status_code != 200:
                    failures.append(f"download[{i}]={resp.status_code}:{resp.text}")
                else:
                    # sanity check zip
                    with zipfile.ZipFile(io.BytesIO(resp.content), "r") as zf:
                        _ = zf.namelist()

                t0 = time.perf_counter()
                resp = client.delete(f"/datasets/{dataset_id}", headers=headers)
                results["delete"].append(time.perf_counter() - t0)
                if resp.status_code != 200:
                    failures.append(f"delete[{i}]={resp.status_code}:{resp.text}")

    print("Dataset benchmark (TestClient, in-process)")
    print(f"iterations: {args.iterations}")
    print(f"failures: {len(failures)}")
    for key, vals in results.items():
        s = _summary(vals)
        print(f"- {key}: avg={s['avg_ms']:.2f}ms p50={s['p50_ms']:.2f}ms p95={s['p95_ms']:.2f}ms max={s['max_ms']:.2f}ms")
    if failures:
        print("\nFailure samples:")
        for item in failures[:10]:
            print("-", item)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
