"""Console entry point for pwdstrength CLI ('pwd')."""

import argparse
import sys
from pwdstrength import __version__
from pwdstrength.utils import load_config, setup_logger


def main():
    parser = argparse.ArgumentParser(
        prog="pwd",
        description="pwdstrength: AI-Based Password Strength Prediction CLI",
    )
    parser.add_argument("--version", action="version", version=f"pwdstrength {__version__}")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML configuration")
    parser.add_argument("--info", action="store_true", help="Print loaded configuration and exit")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Score command
    score_p = subparsers.add_parser("score", help="Score a single password via CLI")
    score_p.add_argument("password", help="Password string to analyze")
    score_p.add_argument("--details", action="store_true", help="Show 22-feature and component breakdown")

    args = parser.parse_args()
    logger = setup_logger("pwdstrength.cli")

    if args.info or args.command is None and len(sys.argv) == 1:
        cfg = load_config(args.config)
        logger.info(f"Loaded config from {args.config} (max_len={cfg.get('data', {}).get('max_len', 32)})")
        print(f"pwdstrength v{__version__} ready. Use 'pwd score <password>' or make targets.")
        return 0

    if args.command == "score":
        from pwdstrength.features import extract_features, get_feature_reasons
        from pwdstrength.baselines import CompositionBaseline
        pw = args.password
        feats = extract_features(pw)
        reasons = get_feature_reasons(pw, feats)
        comp = CompositionBaseline()
        y_est = comp.predict_y([pw])[0]
        cls_est = comp.predict_class([pw])[0]
        level_names = ["Very Weak", "Weak", "Fair", "Strong", "Very Strong"]
        print(f"\nPassword Analysis:")
        print(f"  Length: {len(pw)}")
        print(f"  Estimated Guess log10(G*): {y_est:.2f} (~10^{y_est:.1f} guesses)")
        print(f"  Strength Level: {cls_est} ({level_names[cls_est]})")
        print(f"  Top Structural Reasons: {', '.join(reasons)}")
        if args.details:
            print(f"  Features vector (dim={len(feats)}): {feats}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
