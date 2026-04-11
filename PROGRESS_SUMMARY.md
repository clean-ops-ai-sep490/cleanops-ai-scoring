# Progress Summary - Human Feedback Retraining and RabbitMQ Architecture

Date: 2026-04-11
Project: cleaning_ai_poc

## 1. Current State (Done)

- Existing scoring flow is implemented in `src/api/main.py` with `PASS/PENDING/FAIL` based on quality score.
- Environment-driven configuration has already been applied (`.env`, `.env.example`, config module).
- Security baseline improved: secrets moved out of source, `.gitignore` updated.
- Documentation already available:
  - `STATUS_UPDATE.md`
  - `API_USAGE_AND_SCORING.md`

## 2. Core Product Direction (Confirmed)

Goal: Build a Human-in-the-Loop system that learns from reviewer corrections and supports retraining over time.

Key concern clarified:
- If reviewers only click `PASS/FAIL`, the system can retrain a decision/calibration layer effectively.
- `PASS/FAIL` alone is not enough for deep improvement of YOLO/U-Net perception quality.
- To improve core detection/segmentation quality, we still need annotation-level data (bbox/mask) for a subset of hard cases.

## 3. Runtime Architecture Decision (Confirmed)

Use RabbitMQ as the central processing backbone.

Design principles:
- API requests for scoring should go through RabbitMQ.
- Images are not stored in the queue; only blob URIs + metadata are sent.
- Workers fetch images from blob storage, run inference, write results to Redis/Postgres.
- API returns result via sync facade (wait short timeout), and can fallback to async (`202 + job_id`) when needed.

## 4. Recommended Service Topology

- `backend-api`: orchestration/business API (client-facing)
- `ai-scoring`: model inference logic (or worker-executed component)
- `evaluator-worker`: consumes scoring jobs from RabbitMQ
- `retrainer-worker`: consumes retrain jobs/events
- `rabbitmq`: messaging backbone
- `redis`: short-lived result/job state cache
- `postgres`: source of truth for inference/feedback/retrain metadata
- `blob storage`: image source of truth

## 5. Proposed Monorepo Wrapper (New Parent Folder)

Confirmed direction: wrap current project into a new parent folder to keep context clean and scaling easier.

Target structure:
- `apps/backend-api`
- `apps/ai-scoring`
- `apps/workers/evaluator-worker`
- `apps/workers/retrainer-worker`
- `infra/docker`
- `infra/migrations`
- `packages/contracts`
- `packages/common`
- `docs/architecture`

## 6. Data/Feedback and Retrain Strategy

Data flow:
1. Inference event logged.
2. Reviewer submits correction (`PASS/PENDING/FAIL` + reason).
3. Feedback event stored and queued.
4. Retrain pipeline consumes validated feedback.

Two-layer retraining strategy:
- Layer A (fast): decision-layer recalibration from human verdicts.
- Layer B (deep): periodic YOLO/U-Net retrain using annotation-level data only.

## 7. Quality and Safety Gates

Must-have controls:
- Idempotency key to prevent duplicate submissions.
- Retry + DLQ policy for failed queue messages.
- Promotion gate before model deployment (compare against baseline KPI).
- Rollback strategy if new model underperforms.
- Prioritize minimizing false-pass in sensitive environments.

## 8. Milestones

- Milestone A: RabbitMQ scoring path + compatibility with current response schema.
- Milestone B: Feedback endpoints + review queue + decision-layer retrain.
- Milestone C: Annotation pipeline + full YOLO/U-Net retrain + automated model promotion/rollback.

## 9. Immediate Next Actions

1. Create backend API contract and RabbitMQ message contract (`evaluation.request`, `evaluation.result`, `evaluation.dlq`).
2. Add worker skeletons (`evaluator-worker`, `retrainer-worker`).
3. Add initial Docker Compose for API, RabbitMQ, Redis, Postgres, workers.
4. Define DB schema/migrations for:
   - inference_events
   - feedback_events
   - review_queue
   - retrain_jobs
   - model_registry
   - eval_reports

## 10. Notes for Non-AI Stakeholders

- Human feedback is valuable immediately, even without full annotation.
- Fast improvements come from better decision logic (less wrong verdicts, less pending overload).
- Strong long-term accuracy gains require some curated annotation data.
- This plan is phased to deliver early operational value while building long-term model quality.
