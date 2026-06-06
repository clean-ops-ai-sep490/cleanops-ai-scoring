from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
    from src import trainer_api
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
    if exc.name == "fastapi":
        TestClient = None
        trainer_api = None
        FASTAPI_AVAILABLE = False
    else:
        raise
else:
    FASTAPI_AVAILABLE = True


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed in this Python environment")
class TrainerApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_command = trainer_api.TRAINER_COMMAND
        self.original_workdir = trainer_api.TRAINER_WORKDIR
        self.original_api_key = trainer_api.TRAINER_API_KEY
        self.original_enabled = trainer_api.TRAINER_API_ENABLED

    def tearDown(self) -> None:
        trainer_api.TRAINER_COMMAND = self.original_command
        trainer_api.TRAINER_WORKDIR = self.original_workdir
        trainer_api.TRAINER_API_KEY = self.original_api_key
        trainer_api.TRAINER_API_ENABLED = self.original_enabled

    def test_trainer_job_runs_command_and_sets_job_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer_api.TRAINER_API_ENABLED = True
            trainer_api.TRAINER_API_KEY = ""
            trainer_api.TRAINER_COMMAND = "python scripts/run_retrain_pipeline.py"
            trainer_api.TRAINER_WORKDIR = tmp

            captured_env = {}

            def fake_run(*args, **kwargs):
                captured_env.update(kwargs["env"])
                return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="ok\n", stderr="")

            with patch("src.trainer_api.subprocess.run", side_effect=fake_run):
                response = TestClient(trainer_api.app).post(
                    "/trainer/jobs",
                    json={
                        "jobId": "job-1",
                        "batchId": "batch-1",
                        "sourceWindowFromUtc": "2026-06-01T00:00:00Z",
                        "reviewedSampleCount": 2,
                        "samples": [],
                    },
                )

            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["status"], "completed")
            self.assertEqual(body["exitCode"], 0)
            self.assertEqual(body["stdoutTail"], "ok")
            self.assertEqual(captured_env["TRAINER_JOB_ID"], "job-1")
            self.assertEqual(captured_env["TRAINER_BATCH_ID"], "batch-1")
            self.assertEqual(captured_env["TRAINER_REVIEWED_SAMPLE_COUNT"], "2")

    def test_trainer_job_returns_failed_when_command_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer_api.TRAINER_API_ENABLED = True
            trainer_api.TRAINER_API_KEY = ""
            trainer_api.TRAINER_COMMAND = "python scripts/run_retrain_pipeline.py"
            trainer_api.TRAINER_WORKDIR = tmp

            with patch(
                "src.trainer_api.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=trainer_api.TRAINER_COMMAND,
                    returncode=7,
                    stdout="partial",
                    stderr="boom",
                ),
            ):
                response = TestClient(trainer_api.app).post(
                    "/trainer/jobs",
                    json={
                        "jobId": "job-2",
                        "batchId": "batch-2",
                        "reviewedSampleCount": 0,
                        "samples": [],
                    },
                )

            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["status"], "failed")
            self.assertEqual(body["exitCode"], 7)
            self.assertEqual(body["stdoutTail"], "partial")
            self.assertEqual(body["stderrTail"], "boom")

    def test_trainer_job_requires_existing_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer_api.TRAINER_API_ENABLED = True
            trainer_api.TRAINER_API_KEY = ""
            trainer_api.TRAINER_COMMAND = "python scripts/run_retrain_pipeline.py"
            trainer_api.TRAINER_WORKDIR = str(Path(tmp) / "missing")

            response = TestClient(trainer_api.app).post(
                "/trainer/jobs",
                json={"jobId": "job-3", "batchId": "batch-3"},
            )

        self.assertEqual(response.status_code, 500)
        self.assertIn("TRAINER_WORKDIR", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
