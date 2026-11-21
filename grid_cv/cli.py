"""
cli.py – Typer CLI for batch drone image inspection.

Usage
-----
    python -m grid_cv inspect ./sample_images --output ./results

Outputs per-run:
  - Annotated copies of each image in <output>/annotated/
  - <output>/inspections.geojson
  - <output>/summary.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from grid_cv.geo import save_csv, save_geojson
from grid_cv.pipeline import InspectionPipeline, InspectionResult, annotate_image
from grid_cv.telemetry import extract_telemetry

app = typer.Typer(
    name="grid-cv",
    help="Detect electric distribution asset failures from drone imagery.",
    add_completion=False,
)
console = Console()
log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(
        p for p in source.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command("inspect")
def inspect_cmd(
    source: Path = typer.Argument(..., help="Image file or directory of drone images"),
    output: Path = typer.Option(Path("results"), "--output", "-o", help="Output directory"),
    detector_weights: Optional[Path] = typer.Option(None, "--detector", help="Path to detector .pt weights"),
    classifier_weights: Optional[Path] = typer.Option(None, "--classifier", help="Path to classifier .pt weights"),
    no_annotate: bool = typer.Option(False, "--no-annotate", help="Skip saving annotated images"),
    no_sahi: bool = typer.Option(False, "--no-sahi", help="Disable SAHI slicing (faster on small images)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Batch-inspect drone images for distribution asset failures."""
    _setup_logging(verbose)

    images = _collect_images(source)
    if not images:
        console.print(f"[red]No supported images found in {source}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[bold green]grid-cv[/bold green] — found [cyan]{len(images)}[/cyan] image(s)")

    # Prepare output directories
    output.mkdir(parents=True, exist_ok=True)
    annotated_dir = output / "annotated"
    if not no_annotate:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    # Initialise pipeline
    pipeline = InspectionPipeline(
        detector_weights=detector_weights,
        classifier_weights=classifier_weights,
        use_sahi=not no_sahi,
    )
    console.print(
        f"Pipeline: detector=[magenta]{pipeline._detector_mode}[/magenta]  "
        f"classifier=[magenta]{pipeline._classifier_mode}[/magenta]"
    )

    results: list[InspectionResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Inspecting…", total=len(images))

        for img_path in images:
            progress.update(task, description=f"[cyan]{img_path.name}[/cyan]")
            telemetry = extract_telemetry(img_path)
            try:
                result = pipeline.inspect(img_path, telemetry=telemetry)
            except Exception as exc:
                console.print(f"[red]FAILED[/red] {img_path.name}: {exc}")
                progress.advance(task)
                continue

            results.append(result)

            # Save annotated image
            if not no_annotate and result.detections:
                import cv2 as _cv2
                img_bgr = _cv2.imread(str(img_path))
                if img_bgr is not None:
                    annotated = annotate_image(img_bgr, result.detections)
                    out_name = img_path.stem + "_annotated" + img_path.suffix
                    _cv2.imwrite(str(annotated_dir / out_name), annotated)

            progress.advance(task)

    if not results:
        console.print("[red]No results produced.[/red]")
        raise typer.Exit(code=1)

    # Persist outputs
    geojson_path = save_geojson(results, output / "inspections.geojson")
    csv_path = save_csv(results, output / "summary.csv")

    # Summary table
    table = Table(title="Inspection Summary", show_lines=True)
    table.add_column("Image", style="cyan", no_wrap=True)
    table.add_column("Detections", justify="right")
    table.add_column("Worst Severity", justify="center")
    table.add_column("ms", justify="right")

    severity_style = {
        "NONE": "green", "LOW": "yellow", "MEDIUM": "orange1", "CRITICAL": "red"
    }

    for r in results:
        ws = r.worst_severity
        style = severity_style.get(ws, "white")
        table.add_row(
            Path(r.image_path).name,
            str(len(r.detections)),
            f"[{style}]{ws}[/{style}]",
            f"{r.inference_ms:.0f}",
        )

    console.print(table)
    console.print(f"\n[green]✓[/green] GeoJSON → {geojson_path}")
    console.print(f"[green]✓[/green] CSV     → {csv_path}")
    if not no_annotate:
        console.print(f"[green]✓[/green] Annotated images → {annotated_dir}/")


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start the FastAPI inspection service."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn not installed. Run: pip install uvicorn[standard][/red]")
        raise typer.Exit(1)
    uvicorn.run("grid_cv.api:app", host=host, port=port, reload=reload)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
