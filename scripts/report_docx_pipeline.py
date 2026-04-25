from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


LEGACY_SCRIPTS = {
    "refresh": "refresh_complete_report_docx.py",
    "enrich": "enrich_final_report_docx.py",
    "fix-metrics": "check_and_fix_report_metrics.py",
    "polish": "make_report_concise.py",
}


DEFAULTS = {
    "source": Path(r"C:\Users\phong\Downloads\cleanops_report_complete_refreshed.docx"),
    "verified": Path(r"C:\Users\phong\Downloads\cleanops_report_complete_verified.docx"),
    "final": Path(r"C:\Users\phong\Downloads\cleanops_report_complete_final.docx"),
    "checked": Path(r"C:\Users\phong\Downloads\cleanops_report_complete_final_checked.docx"),
    "presentation": Path(r"C:\Users\phong\Downloads\cleanops_report_complete_final_presentation_ready.docx"),
}


def run_script(script_name: str, args: list[str]) -> None:
    script_path = SCRIPTS / script_name
    if not script_path.exists():
        raise FileNotFoundError(
            f"Required local report helper is missing: {script_path}. "
            "Restore the helper script or run the matching mode before cleaning local report tooling."
        )
    command = [sys.executable, str(script_path), *args]
    subprocess.run(command, cwd=ROOT, check=True)


def run_refresh(source: Path, output: Path) -> None:
    run_script(LEGACY_SCRIPTS["refresh"], ["--source", str(source), "--output", str(output)])


def run_enrich(source: Path, output: Path) -> None:
    run_script(LEGACY_SCRIPTS["enrich"], ["--source", str(source), "--output", str(output)])


def run_fix_metrics(source: Path, output: Path) -> None:
    run_script(LEGACY_SCRIPTS["fix-metrics"], ["--source", str(source), "--output", str(output)])


def run_polish(source: Path, output: Path, style: str) -> None:
    run_script(LEGACY_SCRIPTS["polish"], ["--style", style, "--source", str(source), "--output", str(output)])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Single entrypoint for the report DOCX pipeline. The heavy one-off helper scripts "
            "are kept local/ignored; this file documents and orchestrates the supported flow."
        )
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_io(subparser: argparse.ArgumentParser, default_source: Path, default_output: Path) -> None:
        subparser.add_argument("--source", default=str(default_source), help="Source DOCX path.")
        subparser.add_argument("--output", default=str(default_output), help="Output DOCX path.")

    add_io(
        subparsers.add_parser("refresh", help="Patch stale scoring formulas and raw report text."),
        DEFAULTS["source"],
        DEFAULTS["verified"],
    )
    add_io(
        subparsers.add_parser("enrich", help="Add diagrams, dataset/training/deployment sections, images, and references."),
        DEFAULTS["verified"],
        DEFAULTS["final"],
    )
    add_io(
        subparsers.add_parser("fix-metrics", help="Replace ambiguous charts and add checked benchmark notes."),
        DEFAULTS["final"],
        DEFAULTS["checked"],
    )
    polish_parser = subparsers.add_parser("polish", help="Create concise or presentation-ready wording.")
    add_io(polish_parser, DEFAULTS["checked"], DEFAULTS["presentation"])
    polish_parser.add_argument(
        "--style",
        choices=["concise", "presentation-ready"],
        default="presentation-ready",
        help="Polish style.",
    )

    full_parser = subparsers.add_parser("full", help="Run refresh -> enrich -> fix-metrics -> polish.")
    full_parser.add_argument("--source", default=str(DEFAULTS["source"]), help="Initial source DOCX path.")
    full_parser.add_argument("--verified", default=str(DEFAULTS["verified"]), help="Intermediate verified DOCX path.")
    full_parser.add_argument("--final", default=str(DEFAULTS["final"]), help="Intermediate final DOCX path.")
    full_parser.add_argument("--checked", default=str(DEFAULTS["checked"]), help="Intermediate checked DOCX path.")
    full_parser.add_argument("--output", default=str(DEFAULTS["presentation"]), help="Final polished DOCX path.")
    full_parser.add_argument(
        "--style",
        choices=["concise", "presentation-ready"],
        default="presentation-ready",
        help="Polish style for the final output.",
    )

    args = parser.parse_args()

    if args.mode == "refresh":
        run_refresh(Path(args.source), Path(args.output))
    elif args.mode == "enrich":
        run_enrich(Path(args.source), Path(args.output))
    elif args.mode == "fix-metrics":
        run_fix_metrics(Path(args.source), Path(args.output))
    elif args.mode == "polish":
        run_polish(Path(args.source), Path(args.output), args.style)
    elif args.mode == "full":
        run_refresh(Path(args.source), Path(args.verified))
        run_enrich(Path(args.verified), Path(args.final))
        run_fix_metrics(Path(args.final), Path(args.checked))
        run_polish(Path(args.checked), Path(args.output), args.style)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
