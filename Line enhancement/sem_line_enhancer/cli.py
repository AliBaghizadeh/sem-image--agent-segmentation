"""
Command-line helpers for SEM preprocessing workflows.

Two presets are available:
1. `lines` (default) — optimized for ferroelastic line enhancement.
2. `boundaries` — tuned for grain-boundary-only SEM images.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import cv2
import numpy as np

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None

from .loader import SEMImageLoader
from .pipeline import SEMPreprocessor
from .presets import PIPELINE_PRESETS, PREPROCESSOR_PRESETS, DEFAULT_PRESET

def maybe_start_run(enabled: bool, run_name: str):
    if not enabled or mlflow is None:
        with nullcontext():
            yield
        return
    with mlflow.start_run(run_name=run_name):
        yield


def log_parameters(params: Dict[str, object], prefix: str = "") -> None:
    if mlflow is None:
        return
    for key, value in params.items():
        mlflow.log_param(f"{prefix}{key}", value)


def run_preprocess(args: argparse.Namespace) -> None:
    input_dir = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_params = dict(PIPELINE_PRESETS[args.preset])
    preprocessor_params = dict(PREPROCESSOR_PRESETS[args.preset])
    prefer_gpu = not args.cpu_only

    loader = SEMImageLoader(input_dir=input_dir)
    preprocessor = SEMPreprocessor(prefer_gpu=prefer_gpu, **preprocessor_params)

    image_paths = loader.scan_directory()
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    if not image_paths:
        raise SystemExit(f"No images found under {input_dir}")

    print(f"[CLI] Processing {len(image_paths)} images from {input_dir}")
    print(f"[CLI] Saving outputs to {output_dir}")

    context = (
        nullcontext()
        if not args.mlflow or mlflow is None
        else mlflow.start_run(run_name=args.mlflow_run_name)
    )
    with context:
        if args.mlflow and mlflow is not None:
            log_parameters({"preset": args.preset}, prefix="")
            log_parameters(pipeline_params, prefix="pipeline_")
            log_parameters(preprocessor_params, prefix="preprocessor_")
            mlflow.log_param("prefer_gpu", prefer_gpu)
            mlflow.log_param("num_images", len(image_paths))
            mlflow.log_param("input_dir", str(input_dir))
            mlflow.log_param("output_dir", str(output_dir))

        for path in image_paths:
            image, metadata = loader.load_image(path)
            i_lines, i_base, i_fused, _ = preprocessor.preprocess_dual(
                image, **pipeline_params
            )
            stem = Path(metadata["filename"]).stem
            np.save(output_dir / f"{stem}_lines.npy", i_lines)
            np.save(output_dir / f"{stem}_base.npy", i_base)
            np.save(output_dir / f"{stem}_fused.npy", i_fused)

    print("[CLI] Completed preprocessing run.")


def export_preprocessed(args: argparse.Namespace) -> None:
    input_dir = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    suffixes: List[str] = [f"_{kind}.npy" for kind in args.types]

    files = [
        path
        for path in sorted(input_dir.glob("*.npy"))
        if any(path.name.endswith(suffix) for suffix in suffixes)
    ]
    if not files:
        raise SystemExit(f"No matching .npy files found in {input_dir}")

    for path in files:
        arr = np.load(path)
        arr = np.nan_to_num(arr)
        arr = np.clip(arr, 0, 1)
        img_u8 = (arr * 255).astype(np.uint8)
        out_path = output_dir / f"{path.stem}.png"
        cv2.imwrite(str(out_path), img_u8)
        if args.verbose:
            print(f"[EXPORT] {path.name} -> {out_path.name}")

    print(f"[EXPORT] Saved {len(files)} PNGs to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sem-line-enhancer",
        description="CLI tools for SEM preprocessing",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess", help="Run dual-path preprocessing")
    preprocess.add_argument("--input", type=str, default="data/raw")
    preprocess.add_argument("--output", type=str, default="data/preprocessed")
    preprocess.add_argument(
        "--preset",
        choices=list(PIPELINE_PRESETS.keys()),
        default=DEFAULT_PRESET,
        help="Choose the pipeline preset (lines vs boundaries).",
    )
    preprocess.add_argument("--limit", type=int, default=None)
    preprocess.add_argument("--mlflow", action="store_true")
    preprocess.add_argument(
        "--mlflow-run-name",
        default="preprocess_images",
    )
    preprocess.add_argument("--cpu-only", action="store_true")
    preprocess.set_defaults(func=run_preprocess)

    export = subparsers.add_parser("export", help="Convert *.npy outputs to PNGs")
    export.add_argument("--input", type=str, default="data/preprocessed")
    export.add_argument("--output", type=str, default="data/preprocessed_png")
    export.add_argument(
        "--types",
        nargs="+",
        choices=["lines", "base", "fused"],
        default=["lines", "base", "fused"],
    )
    export.add_argument("--verbose", action="store_true")
    export.set_defaults(func=export_preprocessed)

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
