from __future__ import annotations

import unittest
from unittest.mock import patch

from src.api import retrain_api


def _metrics(
    candidate: float = 0.324761,
    baseline: float = 0.361106,
    required: float = 0.366106,
    passed: bool = False,
):
    return {
        "unet": {"miou": 0.0},
        "benchmark": {
            "candidate": {"mean_iou": candidate},
            "baseline": {"mean_iou": baseline},
            "gate": {
                "required_mean_iou": required,
                "passed": passed,
            },
        },
    }


class RetrainApiBenchmarkArtifactTests(unittest.TestCase):
    def test_verify_candidate_metrics_blob_accepts_matching_benchmark(self) -> None:
        local = _metrics()
        blob = _metrics()

        verified = retrain_api._verify_candidate_metrics_blob(  # noqa: SLF001
            local,
            blob,
            "scoring/external/latest/metrics/metrics.json",
        )

        self.assertEqual(blob, verified)

    def test_verify_candidate_metrics_blob_rejects_missing_benchmark_field(self) -> None:
        local = _metrics()
        blob = {
            "unet": {"miou": 0.0},
            "benchmark": {
                "candidate": {"mean_iou": 0.324761},
                "baseline": {"mean_iou": 0.361106},
            },
        }

        with self.assertRaisesRegex(RuntimeError, "Candidate metrics blob verification failed"):
            retrain_api._verify_candidate_metrics_blob(  # noqa: SLF001
                local,
                blob,
                "scoring/external/latest/metrics/metrics.json",
            )

    def test_read_metric_does_not_treat_boolean_as_numeric_metric(self) -> None:
        metrics = _metrics(passed=True)

        self.assertIsNone(retrain_api._read_metric(metrics, "benchmark.gate.passed"))  # noqa: SLF001
        self.assertTrue(retrain_api._read_benchmark_gate_passed(metrics))  # noqa: SLF001

    def test_remote_mode_uses_published_blob_and_does_not_upload_local_candidate(self) -> None:
        job_id = "job-remote-ok"
        try:
            retrain_api._jobs[job_id] = {"jobId": job_id, "status": "queued"}  # noqa: SLF001

            with (
                patch.object(retrain_api, "RETRAIN_USE_REMOTE_TRAINER", True),
                patch.object(retrain_api, "RETRAIN_STORAGE_CONNECTION_STRING", "UseDevelopmentStorage=true"),
                patch.object(retrain_api, "_invoke_remote_trainer", return_value="[BENCHMARK_GATE] candidate=0.3283"),
                patch.object(retrain_api, "_download_existing_metrics", return_value=_metrics()),
                patch.object(retrain_api, "_upload_blob_file") as upload_blob,
            ):
                retrain_api._run_retrain_job(  # noqa: SLF001
                    job_id,
                    retrain_api.RetrainJobCreateRequest(batchId="batch-remote-ok"),
                )

            job = retrain_api._jobs[job_id]  # noqa: SLF001
            self.assertEqual(job["status"], "completed")
            self.assertEqual(job["benchmarkCandidateMiou"], 0.324761)
            upload_blob.assert_not_called()
        finally:
            retrain_api._jobs.pop(job_id, None)  # noqa: SLF001

    def test_remote_mode_fails_when_benchmark_logs_but_blob_lacks_benchmark(self) -> None:
        job_id = "job-remote-missing-benchmark"
        try:
            retrain_api._jobs[job_id] = {"jobId": job_id, "status": "queued"}  # noqa: SLF001

            with (
                patch.object(retrain_api, "RETRAIN_USE_REMOTE_TRAINER", True),
                patch.object(retrain_api, "RETRAIN_STORAGE_CONNECTION_STRING", "UseDevelopmentStorage=true"),
                patch.object(retrain_api, "_invoke_remote_trainer", return_value="[BENCHMARK_GATE] candidate=0.3283"),
                patch.object(retrain_api, "_download_existing_metrics", return_value={"unet": {"miou": 0.0}}),
            ):
                retrain_api._run_retrain_job(  # noqa: SLF001
                    job_id,
                    retrain_api.RetrainJobCreateRequest(batchId="batch-remote-missing"),
                )

            job = retrain_api._jobs[job_id]  # noqa: SLF001
            self.assertEqual(job["status"], "failed")
            self.assertIn("Candidate metrics blob verification failed", job["message"])
        finally:
            retrain_api._jobs.pop(job_id, None)  # noqa: SLF001


if __name__ == "__main__":
    unittest.main()
