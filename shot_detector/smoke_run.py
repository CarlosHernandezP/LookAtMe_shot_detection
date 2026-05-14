"""Smoke-run wrapper for autoresearch.

Runs `shot_detector.train_shot_model` as a subprocess, then emits a
TensorBoard event file + `hparams.yaml` from the JSON report so the
autoresearch contract (TB scalars + hparams) is satisfied without
modifying the trainer itself.

Usage (identical flags to train_shot_model, plus output-dir is required):

    uv run python -m shot_detector.smoke_run \
        --output-dir shot_detector/runs/smoke \
        --val-fraction 0.2 \
        --no-save-model

After the trainer exits 0, this script writes:
    <output-dir>/tb_logs/events.out.tfevents.*
    <output-dir>/hparams.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _peek_output_dir(argv: list[str]) -> tuple[Path, list[str]]:
    """Return (output_dir, argv_with_output_dir_ensured). Defaults to a unique
    `shot_detector/runs/auto_<utc-ts>_<pid>` when not supplied — required for
    autoresearch sweeps where multiple trials run from the same launch_cmd."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--output-dir", type=str, default=None)
    known, rest = p.parse_known_args(argv)
    if known.output_dir is None:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        default = Path("shot_detector/runs") / f"auto_{ts}_{os.getpid()}"
        return default, [*rest, "--output-dir", str(default)]
    return Path(known.output_dir), argv


def _load_report(out_dir: Path) -> tuple[dict, str]:
    cv_path = out_dir / "flat_wall_shot_cv_report.json"
    single_path = out_dir / "flat_wall_shot_report.json"
    if cv_path.exists():
        return json.loads(cv_path.read_text()), "cv"
    if single_path.exists():
        return json.loads(single_path.read_text()), "single"
    raise FileNotFoundError(
        f"No report JSON found under {out_dir}. Trainer may have failed before writing metrics."
    )


def _flatten_scalars(d: dict, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_scalars(v, key))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def _argv_to_hparams(argv: list[str]) -> dict[str, object]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--data-dir", type=str, default="shot_detector/data_csv_only")
    p.add_argument("--include-lists-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str)
    p.add_argument("--cv-folds", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--min-samples", type=int, default=5)
    p.add_argument("--serve-weight-mult", type=float, default=1.0)
    p.add_argument("--non-idle-weight-mult", type=float, default=1.0)
    p.add_argument("--idle-min-prob", type=float, default=None)
    p.add_argument(
        "--weight-scheme", type=str, choices=["max_inverse", "sklearn_balanced"], default="max_inverse"
    )
    p.add_argument("--export-counts", action="store_true")
    p.add_argument("--no-save-model", action="store_false", dest="save_model")
    p.add_argument("--select-top-k-features", type=int, default=None)
    p.add_argument("--selection-train-fraction", type=float, default=0.8)
    p.add_argument("--auto-select-features", action="store_true")
    p.add_argument("--feature-select-grid", type=str, default="80,120,160,200,240,280,320,360,400,440,480,533")
    p.add_argument("--feature-select-cv-folds", type=int, default=3)
    p.add_argument("--feature-select-method", type=str, choices=["xgb_importance", "mrmr"], default="xgb_importance")
    p.add_argument("--feature-indices-json", type=str, default=None)
    p.add_argument("--xgb-n-estimators", type=int, default=220)
    p.add_argument("--xgb-max-depth", type=int, default=6)
    p.add_argument("--xgb-learning-rate", type=float, default=0.1)
    p.add_argument("--xgb-subsample", type=float, default=0.8)
    p.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    ns, _unknown = p.parse_known_args(argv)
    return vars(ns)


def _write_tb_events(out_dir: Path, scalars: dict[str, float]) -> Path:
    import socket
    import time as _t

    import tensorflow as tf  # heavy; import only when needed
    from tensorflow.core.framework import summary_pb2
    from tensorflow.core.util import event_pb2

    tb_dir = out_dir / "tb_logs"
    tb_dir.mkdir(parents=True, exist_ok=True)
    path = tb_dir / f"events.out.tfevents.{int(_t.time())}.{socket.gethostname()}"
    writer = tf.io.TFRecordWriter(str(path))
    try:
        for name, value in scalars.items():
            ev = event_pb2.Event(
                wall_time=_t.time(),
                step=0,
                summary=summary_pb2.Summary(
                    value=[summary_pb2.Summary.Value(tag=name, simple_value=float(value))]
                ),
            )
            writer.write(ev.SerializeToString())
    finally:
        writer.close()
    return tb_dir


def _write_hparams_yaml(out_dir: Path, hparams: dict[str, object]) -> Path:
    path = out_dir / "hparams.yaml"
    lines: list[str] = []
    for k in sorted(hparams.keys()):
        v = hparams[k]
        if v is None:
            lines.append(f"{k}: null")
        elif isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            lines.append(f"{k}: {v}")
        else:
            esc = str(v).replace("'", "''")
            lines.append(f"{k}: '{esc}'")
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out_dir, argv = _peek_output_dir(argv)
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    cmd = ["uv", "run", "python", "-m", "shot_detector.train_shot_model", *argv]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(repo_root))
    wall = time.time() - t0
    if proc.returncode != 0:
        print(f"[smoke_run] trainer exited {proc.returncode} after {wall:.1f}s", file=sys.stderr)
        return proc.returncode

    report, kind = _load_report(out_dir)
    if kind == "single":
        scalars = _flatten_scalars(report.get("validation", {}), prefix="val")
    else:
        cv = report.get("cv", {})
        scalars = _flatten_scalars(cv.get("oof_aggregate", {}), prefix="oof")
        scalars.update(_flatten_scalars(cv.get("per_fold_mean_std", {}), prefix="cv"))
    if not scalars:
        print(f"[smoke_run] no scalar metrics found in report ({kind})", file=sys.stderr)
        return 2

    tb_dir = _write_tb_events(out_dir, scalars)
    hp_path = _write_hparams_yaml(out_dir, _argv_to_hparams(argv))
    print(f"[smoke_run] wall={wall:.1f}s tb={tb_dir} hparams={hp_path} scalars={len(scalars)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
