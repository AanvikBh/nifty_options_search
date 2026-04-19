"""
orchestrator.py – Main search loop CLI.

End-to-end workflow:
  1. Generate / load frozen dataset
  2. Compute features
  3. For each round:
     a. Allocate budget via trust controller
     b. Generate candidates from each proposer
     c. Validate candidates
     d. Optionally tune parameters
     e. Backtest each candidate
     f. Score and store results
     g. Update trust
  4. Output ranked shortlist + experiment log
"""

import argparse
import json
import logging
import time
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import config
from data.downloader import generate_index_data, generate_vix_data, generate_option_data
from data.cleaner import clean_index, clean_vix, clean_options
from data.store import save_dataset, load_index, load_vix, load_options, dataset_exists
from features.engine import compute_features
from strategy.grammar import StrategyRule
from strategy.validator import is_valid, ValidationError
from proposers import llm_proposer, mutation_proposer, diversity_proposer
from tuner.param_tuner import tune as tune_params
from backtest.engine import run_backtest, BacktestResult
from backtest.metrics import BacktestMetrics
from controller.scorer import composite_score, rank_candidates
from controller.experiment_store import ExperimentStore
from controller.trust import TrustController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orchestrator")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Data generation / loading
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dataset():
    """Generate synthetic dataset if it doesn't exist yet."""
    if dataset_exists():
        logger.info("Dataset already exists – loading from disk.")
        return

    logger.info("Generating synthetic NIFTY dataset (this takes ~30s) ...")
    t0 = time.time()

    idx = generate_index_data()
    vix = generate_vix_data()
    opts = generate_option_data(idx, vix)

    idx = clean_index(idx)
    vix = clean_vix(vix)
    opts = clean_options(opts, idx)

    save_dataset(idx, vix, opts)
    logger.info(f"Dataset generated in {time.time() - t0:.1f}s")
    logger.info(f"  Index rows : {len(idx)}")
    logger.info(f"  VIX rows   : {len(vix)}")
    logger.info(f"  Option rows: {len(opts):,}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Feature computation
# ─────────────────────────────────────────────────────────────────────────────

def build_features():
    """Compute the full feature matrix from the frozen dataset."""
    logger.info("Computing features ...")
    idx = load_index()
    vix = load_vix()
    opts = load_options()

    # Use a subset of dates for feature computation to limit time
    # Full computation uses ALL option data which is slow due to IV solving
    t0 = time.time()
    feats = compute_features(idx, vix, opts)
    logger.info(f"Features computed in {time.time() - t0:.1f}s – {len(feats)} rows")
    return feats, idx, opts


def build_features_fast(idx, vix):
    """
    Fast feature computation that skips IV-solving (uses VIX as proxy).
    Used for the main search loop to avoid ~10min feature computation.
    """
    import numpy as np
    import pandas as pd

    idx = idx.copy().set_index("date").sort_index()
    log_ret = np.log(idx["close"] / idx["close"].shift(1))

    idx["nifty_ret1"] = idx["close"].pct_change(1)
    idx["nifty_ret5"] = idx["close"].pct_change(5)
    idx["rv5"] = log_ret.rolling(5).std() * np.sqrt(252)
    idx["rv20"] = log_ret.rolling(20).std() * np.sqrt(252)

    vix_s = vix.set_index("date")["vix"]
    idx["india_vix"] = vix_s
    idx["india_vix_change"] = vix_s.pct_change()

    # Use VIX/100 as ATM IV proxy (avoids expensive IV solve)
    idx["atm_iv"] = idx["india_vix"] / 100.0
    idx["iv_minus_rv20"] = idx["atm_iv"] - idx["rv20"]
    idx["skew_proxy"] = 0.01  # placeholder
    idx["term_slope"] = 0.005  # placeholder
    idx["oi_change"] = 0
    idx["volume"] = 5000

    feature_cols = [
        "nifty_ret1", "nifty_ret5", "rv5", "rv20",
        "atm_iv", "iv_minus_rv20", "skew_proxy", "term_slope",
        "oi_change", "volume", "india_vix", "india_vix_change",
    ]
    result = idx[feature_cols].copy().reset_index().rename(columns={"index": "date"})
    result = result.ffill()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Search round
# ─────────────────────────────────────────────────────────────────────────────

def run_search_round(
    round_num: int,
    budget_alloc: dict,
    features_df,
    index_df,
    options_df,
    store: ExperimentStore,
    trust: TrustController,
    top_strategies: list,
    do_tune: bool = False,
) -> list:
    """
    Execute one search round.

    Returns list of (BacktestResult, score, source) for this round.
    """
    results = []
    existing_fps = store.all_fingerprints()

    for source, n_candidates in budget_alloc.items():
        if n_candidates <= 0:
            continue

        logger.info(f"  [{source}] Generating {n_candidates} candidates ...")
        seed = round_num * 1000 + hash(source) % 1000

        # Generate candidates
        if source == "llm":
            top_dicts = [s["rule_json"] for s in top_strategies[:5]] if top_strategies else None
            if top_dicts:
                top_dicts = [json.loads(d) if isinstance(d, str) else d for d in top_dicts]
            candidates = llm_proposer.propose(n_candidates, top_dicts, seed=seed)
        elif source == "mutation":
            # Get top rules as StrategyRule objects
            top_rules = []
            for s in top_strategies[:10]:
                try:
                    rule_data = s["rule_json"]
                    if isinstance(rule_data, str):
                        rule_data = json.loads(rule_data)
                    top_rules.append(StrategyRule.model_validate(rule_data))
                except Exception:
                    continue
            if top_rules:
                candidates = mutation_proposer.propose(n_candidates, top_rules, seed=seed)
            else:
                # Fall back to diversity if no history
                candidates = diversity_proposer.propose(n_candidates, existing_fps, seed=seed)
        elif source == "diversity":
            candidates = diversity_proposer.propose(n_candidates, existing_fps, seed=seed)
        else:
            continue

        # Validate and backtest
        for candidate in candidates:
            if not is_valid(candidate):
                continue

            fp = candidate.fingerprint()
            if fp in existing_fps:
                continue
            existing_fps.add(fp)

            # Optional parameter tuning
            if do_tune:
                def eval_fn(r):
                    bt = run_backtest(
                        r, features_df, index_df, options_df,
                        start_date=config.TRAIN_START, end_date=config.TRAIN_END,
                    )
                    return composite_score(bt.metrics, r)

                try:
                    candidate, _ = tune_params(candidate, eval_fn, seed=seed)
                except Exception as e:
                    logger.warning(f"  Tuning failed: {e}")

            # Backtest on training period
            try:
                bt_result = run_backtest(
                    candidate, features_df, index_df, options_df,
                    start_date=config.TRAIN_START, end_date=config.TRAIN_END,
                )
            except Exception as e:
                logger.warning(f"  Backtest failed: {e}")
                continue

            score = composite_score(bt_result.metrics, candidate)

            # Store experiment
            parent_fp = None
            if source == "mutation" and top_strategies:
                parent_fp = top_strategies[0].get("fingerprint", None)

            store.log_experiment(
                rule=candidate,
                metrics=bt_result.metrics,
                score=score,
                source=source,
                round_num=round_num,
                parent_fingerprint=parent_fp,
            )

            # Update trust
            trust.update(source, score, fp, existing_fps)

            results.append((bt_result, score, source))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Final evaluation & output
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_holdout(top_rules, features_df, index_df, options_df):
    """Run the top strategies on the holdout period (2025)."""
    logger.info("\n" + "=" * 60)
    logger.info("HOLDOUT EVALUATION (2025)")
    logger.info("=" * 60)

    holdout_results = []
    for rule_data in top_rules[:5]:
        try:
            if isinstance(rule_data, str):
                rule_data = json.loads(rule_data)
            rule = StrategyRule.model_validate(rule_data)
        except Exception:
            continue

        bt = run_backtest(
            rule, features_df, index_df, options_df,
            start_date=config.HOLDOUT_START, end_date=config.HOLDOUT_END,
        )
        score = composite_score(bt.metrics, rule)
        holdout_results.append((rule, bt, score))

        logger.info(f"\n  Strategy: {rule.to_readable()}")
        logger.info(f"  Holdout Sharpe: {bt.metrics.sharpe_ratio:.4f}")
        logger.info(f"  Holdout Return: {bt.metrics.total_return:.2f}")
        logger.info(f"  Holdout MaxDD:  {bt.metrics.max_drawdown:.2f}")
        logger.info(f"  Holdout Score:  {score:.4f}")
        logger.info(f"  Trades: {bt.metrics.n_trades}")

    return holdout_results


def print_final_summary(store: ExperimentStore, trust: TrustController):
    """Print the final search summary."""
    logger.info("\n" + "=" * 60)
    logger.info("SEARCH COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total experiments: {store.total_experiments()}")
    logger.info(f"\n{trust.summary()}")

    top = store.get_top_strategies(10)
    logger.info(f"\n{'='*60}")
    logger.info("TOP 10 STRATEGIES (by composite score)")
    logger.info(f"{'='*60}")
    for i, s in enumerate(top, 1):
        rule = json.loads(s["rule_json"]) if isinstance(s["rule_json"], str) else s["rule_json"]
        try:
            r = StrategyRule.model_validate(rule)
            logger.info(
                f"\n  #{i} [{s['source']}] Score={s['composite_score']:.4f} "
                f"Sharpe={s['sharpe_ratio']:.4f} "
                f"MaxDD={s['max_drawdown']:.2f} "
                f"Trades={s['n_trades']}"
            )
            logger.info(f"     {r.to_readable()}")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Trust-Calibrated Strategy Search for NIFTY Weekly Options"
    )
    parser.add_argument("--budget", type=int, default=config.DEFAULT_BUDGET,
                        help=f"Total candidate evaluations (default: {config.DEFAULT_BUDGET})")
    parser.add_argument("--rounds", type=int, default=config.DEFAULT_ROUNDS,
                        help=f"Number of search rounds (default: {config.DEFAULT_ROUNDS})")
    parser.add_argument("--tune", action="store_true",
                        help="Enable parameter tuning (slower)")
    parser.add_argument("--holdout", action="store_true", default=True,
                        help="Run holdout evaluation after search")
    parser.add_argument("--export-csv", type=str, default="experiment_log.csv",
                        help="Export experiment log to CSV")
    parser.add_argument("--regenerate-data", action="store_true",
                        help="Force regeneration of synthetic dataset")
    parser.add_argument("--fast-features", action="store_true", default=True,
                        help="Use fast feature computation (VIX proxy for IV)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Trust-Calibrated Strategy Search for NIFTY Weekly Options")
    logger.info("=" * 60)
    logger.info(f"Budget: {args.budget} candidates over {args.rounds} rounds")
    logger.info(f"Parameter tuning: {'ON' if args.tune else 'OFF'}")
    logger.info(f"Holdout evaluation: {'ON' if args.holdout else 'OFF'}")

    # Step 1: Data
    if args.regenerate_data:
        import shutil
        if config.DATA_DIR.exists():
            shutil.rmtree(config.DATA_DIR)
    ensure_dataset()

    # Step 2: Features
    idx = load_index()
    vix = load_vix()
    opts = load_options()

    if args.fast_features:
        logger.info("Using fast feature computation (VIX proxy) ...")
        features_df = build_features_fast(idx, vix)
    else:
        features_df = compute_features(idx, vix, opts)

    logger.info(f"Feature matrix: {len(features_df)} rows × {len(features_df.columns)} cols")

    # Step 3: Search
    store = ExperimentStore()
    trust = TrustController()
    candidates_per_round = args.budget // args.rounds

    t_start = time.time()

    for round_num in range(1, args.rounds + 1):
        logger.info(f"\n{'─'*60}")
        logger.info(f"ROUND {round_num}/{args.rounds}")
        logger.info(f"{'─'*60}")

        # Allocate budget
        alloc = trust.allocate_budget(candidates_per_round)
        logger.info(f"Budget allocation: {alloc}")

        # Get top strategies for proposers
        top = store.get_top_strategies(10)

        # Run the round
        round_results = run_search_round(
            round_num=round_num,
            budget_alloc=alloc,
            features_df=features_df,
            index_df=idx,
            options_df=opts,
            store=store,
            trust=trust,
            top_strategies=top,
            do_tune=args.tune,
        )

        # Round summary
        n_positive = sum(1 for _, s, _ in round_results if s > 0)
        avg_score = (
            sum(s for _, s, _ in round_results) / max(len(round_results), 1)
        )
        logger.info(f"\n  Round {round_num} results: {len(round_results)} evaluated, "
                     f"{n_positive} positive, avg_score={avg_score:.4f}")
        logger.info(trust.summary())

    elapsed = time.time() - t_start
    logger.info(f"\nSearch completed in {elapsed:.1f}s")

    # Step 4: Final output
    print_final_summary(store, trust)

    # Step 5: Holdout evaluation
    if args.holdout:
        top = store.get_top_strategies(5)
        top_rules = [s["rule_json"] for s in top]
        evaluate_holdout(top_rules, features_df, idx, opts)

    # Export
    if args.export_csv:
        csv_path = config.PROJECT_ROOT / args.export_csv
        store.export_csv(str(csv_path))
        logger.info(f"\nExperiment log exported to {csv_path}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
