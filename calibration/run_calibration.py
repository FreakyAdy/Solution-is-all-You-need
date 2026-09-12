#!/usr/bin/env python3
"""
PHANTOM CORE — Master Calibration Runner
==========================================
Command-line entry point for the calibration pipeline.

Usage:
    python run_calibration.py --model /path/to/model --output profiles/llama70b/
    python run_calibration.py --model meta-llama/Meta-Llama-3-70B --samples 50
    python run_calibration.py --model ./model.gguf --output profiles/ --device cpu
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import structlog

# Configure structured logging for CLI
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="phantom-calibrate",
        description=(
            "PHANTOM CORE — Calibration Pipeline\n"
            "Generates the .phantom profile that unlocks all 7 innovations "
            "for a specific model on your hardware."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --model meta-llama/Meta-Llama-3-70B\n"
            "  %(prog)s --model ./models/llama-70b.gguf --output ./profiles\n"
            "  %(prog)s --model Qwen/Qwen2-72B --samples 100 --device auto\n"
        ),
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to model weights (GGUF, safetensors, HuggingFace dir, or HF model ID)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for calibration profile (default: ~/.phantom/profiles/<model_name>/)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of calibration prompts to use (default: 50)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Computation device (default: auto)",
    )
    parser.add_argument(
        "--prompts",
        default=None,
        help="Path to custom calibration prompts JSONL file",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["spectral", "kv_ae", "gates", "wraith"],
        help="Skip specific calibration steps",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )

    return parser.parse_args()


def print_banner():
    """Print the calibration banner."""
    print(r"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██████╗ ██╗  ██╗ █████╗ ███╗   ██╗████████╗ ██████╗ ███╗  ║
║   ██╔══██╗██║  ██║██╔══██╗████╗  ██║╚══██╔══╝██╔═══██╗████╗ ║
║   ██████╔╝███████║███████║██╔██╗ ██║   ██║   ██║   ██║██╔██╗║
║   ██╔═══╝ ██╔══██║██╔══██║██║╚██╗██║   ██║   ██║   ██║██║╚█║
║   ██║     ██║  ██║██║  ██║██║ ╚████║   ██║   ╚██████╔╝██║ ╚║
║   ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝  ║
║                    CALIBRATION                               ║
║            "Run the Unreachable."                            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


def main():
    """Main calibration entry point."""
    args = parse_args()

    if args.verbose:
        structlog.configure(
            processors=[
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
                structlog.dev.ConsoleRenderer(),
            ],
        )

    print_banner()

    # Resolve output directory
    if args.output:
        output_dir = args.output
    else:
        model_name = Path(args.model).stem.replace("/", "_").replace("\\", "_")
        output_dir = str(Path.home() / ".phantom" / "profiles" / model_name)

    print(f"[PHANTOM CORE] Model:    {args.model}")
    print(f"[PHANTOM CORE] Output:   {output_dir}")
    print(f"[PHANTOM CORE] Samples:  {args.samples}")
    print(f"[PHANTOM CORE] Device:   {args.device}")
    if args.skip:
        print(f"[PHANTOM CORE] Skipping: {', '.join(args.skip)}")
    print()

    # Import here to avoid slow startup for --help
    try:
        from phantom.calibrate import CalibrationPipeline
    except ImportError as e:
        print(f"[ERROR] Failed to import PHANTOM CORE Python package: {e}")
        print("[ERROR] Make sure you've installed the package: pip install -e .")
        sys.exit(1)

    # Run the pipeline
    t_start = time.time()

    try:
        pipeline = CalibrationPipeline(
            model_path=args.model,
            output_dir=output_dir,
            num_samples=args.samples,
            device=args.device,
            prompts_path=args.prompts,
        )

        profile = pipeline.run()

        t_total = time.time() - t_start
        print(f"\n{'='*62}")
        print(f"[PHANTOM CORE] CALIBRATION SUCCESSFUL")
        print(f"  Total time:    {t_total/60:.1f} minutes")
        print(f"  Profile hash:  {profile['model_hash']}")
        print(f"  Profile path:  {output_dir}/")
        print(f"  Hardware tier: {profile.get('hardware_tier', 'UNKNOWN')}")
        print(f"{'='*62}")
        print()
        print("Next steps:")
        print(f"  phantom-core serve --model {args.model} \\")
        print(f"      --profile {output_dir}/{profile['model_hash']}.phantom")
        print()

    except KeyboardInterrupt:
        print("\n[PHANTOM CORE] Calibration interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.error("calibration_failed", error=str(e))
        print(f"\n[ERROR] Calibration failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
