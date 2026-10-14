"""Run selected environmental modules against one protected workbook input."""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import time

from environment_core import MODULES, Observations, dump_json, sha256


def code_identity():
    return {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def run(config_path, modules, resume=False):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output = Path(config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output / "execution.log", encoding="utf-8"),
        ],
        force=True,
    )
    ctx = Observations(config)
    identity = code_identity()
    checkpoint = output / "checkpoint.json"
    if resume and checkpoint.exists():
        prior = json.loads(checkpoint.read_text(encoding="utf-8"))
        if (
            prior["input_sha256"] != ctx.input_hash
            or prior["code"] != identity
            or prior["config"] != config
        ):
            raise ValueError(
                "Checkpoint does not match current code, inputs and settings"
            )
        saved = json.loads(
            (output / "environment_results.json").read_text(encoding="utf-8")
        )
        for r in saved["records"]:
            if r["previous_value"] is None and r["value"] is not None:
                ctx.set(
                    r["excel_row"] - 2,
                    r["variable"],
                    r["value"],
                    r["source"],
                    **r["detail"],
                )
        ctx.completed = saved["completed"]
        ctx.products = saved["products"]
    for name in modules:
        if name in ctx.completed:
            continue
        start = time.monotonic()
        logging.info("START module %s %s", name, MODULES[name].__name__)
        MODULES[name](ctx)
        ctx.completed.append(name)
        summary = ctx.save()
        dump_json(
            checkpoint,
            {
                "input_sha256": ctx.input_hash,
                "code": identity,
                "config": config,
                "completed": ctx.completed,
            },
        )
        logging.info("DONE module %s elapsed %.1fs", name, time.monotonic() - start)
    if sha256(ctx.path) != ctx.input_hash:
        raise RuntimeError("Workbook changed during extraction; no writeback performed")
    logging.info(
        "Extraction finished; results are staged for verified writeback to %s", ctx.path
    )
    return ctx


def main(default_modules=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--modules",
        nargs="+",
        choices=list(MODULES),
        default=default_modules or list(MODULES),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--writeback",
        action="store_true",
        help="Back up and write verified environmental values to the configured DATA workbook",
    )
    args = parser.parse_args()
    run(args.config, args.modules, args.resume)
    if args.writeback:
        from workbook_update import publish_result

        receipt = publish_result(json.loads(args.config.read_text(encoding="utf-8")))
        logging.info("Writeback complete: %s", receipt["workbook"])


if __name__ == "__main__":
    main()
