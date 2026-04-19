"""
controller/experiment_store.py – SQLite-backed experiment log.

Stores every trial:
  • Candidate structure + parameters (JSON)
  • Backtest metrics
  • Proposal source (llm / mutation / diversity)
  • Lineage (parent fingerprint if mutation)
  • Composite score
  • Round number, timestamp
"""

import json
import sqlite3
import time
from typing import List, Optional, Dict

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config
from strategy.grammar import StrategyRule
from backtest.metrics import BacktestMetrics


CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint TEXT NOT NULL,
    round_num INTEGER NOT NULL,
    source TEXT NOT NULL,
    parent_fingerprint TEXT,
    rule_json TEXT NOT NULL,
    template TEXT NOT NULL,
    n_conditions INTEGER NOT NULL,
    dte INTEGER NOT NULL,
    hold_period INTEGER NOT NULL,
    total_return REAL,
    avg_return REAL,
    sharpe_ratio REAL,
    max_drawdown REAL,
    hit_rate REAL,
    n_trades INTEGER,
    turnover REAL,
    stability REAL,
    composite_score REAL,
    created_at REAL NOT NULL
)
"""


class ExperimentStore:
    """SQLite experiment log."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(config.EXPERIMENT_DB)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute(CREATE_TABLE)
        conn.commit()
        conn.close()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def log_experiment(self, rule: StrategyRule, metrics: BacktestMetrics,
                       score: float, source: str, round_num: int,
                       parent_fingerprint: str = None) -> int:
        """
        Log one experiment trial.

        Returns the row ID.
        """
        conn = self._conn()
        cursor = conn.execute(
            """INSERT INTO experiments
            (fingerprint, round_num, source, parent_fingerprint,
             rule_json, template, n_conditions, dte, hold_period,
             total_return, avg_return, sharpe_ratio, max_drawdown,
             hit_rate, n_trades, turnover, stability, composite_score,
             created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rule.fingerprint(),
                round_num,
                source,
                parent_fingerprint,
                rule.model_dump_json(),
                rule.template,
                len(rule.conditions),
                rule.dte,
                rule.hold_period,
                metrics.total_return,
                metrics.avg_return,
                metrics.sharpe_ratio,
                metrics.max_drawdown,
                metrics.hit_rate,
                metrics.n_trades,
                metrics.turnover,
                metrics.stability,
                score,
                time.time(),
            ),
        )
        conn.commit()
        row_id = cursor.lastrowid
        conn.close()
        return row_id

    def get_top_strategies(self, n: int = 10,
                           source: str = None) -> List[dict]:
        """Return the top-n strategies by composite score."""
        conn = self._conn()
        conn.row_factory = sqlite3.Row

        query = "SELECT * FROM experiments"
        params = []
        if source:
            query += " WHERE source = ?"
            params.append(source)
        query += " ORDER BY composite_score DESC LIMIT ?"
        params.append(n)

        rows = conn.execute(query, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_round_stats(self, round_num: int) -> dict:
        """Return aggregate stats for a given round."""
        conn = self._conn()
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """SELECT
                COUNT(*) as n,
                AVG(composite_score) as avg_score,
                MAX(composite_score) as max_score,
                AVG(sharpe_ratio) as avg_sharpe,
                source
            FROM experiments
            WHERE round_num = ?
            GROUP BY source""",
            (round_num,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in row]

    def get_source_hit_rates(self) -> Dict[str, float]:
        """
        Compute hit rate per source: fraction of experiments with
        positive composite score.
        """
        conn = self._conn()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT
                source,
                COUNT(*) as total,
                SUM(CASE WHEN composite_score > 0 THEN 1 ELSE 0 END) as hits
            FROM experiments
            GROUP BY source"""
        ).fetchall()
        conn.close()

        result = {}
        for r in rows:
            total = r["total"]
            hits = r["hits"]
            result[r["source"]] = hits / total if total > 0 else 0.0
        return result

    def total_experiments(self) -> int:
        conn = self._conn()
        count = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        conn.close()
        return count

    def all_fingerprints(self) -> set:
        """Return all stored fingerprints for de-duplication."""
        conn = self._conn()
        rows = conn.execute("SELECT DISTINCT fingerprint FROM experiments").fetchall()
        conn.close()
        return {r[0] for r in rows}

    def export_csv(self, path: str):
        """Export the experiment log to CSV."""
        import pandas as pd
        conn = self._conn()
        df = pd.read_sql_query("SELECT * FROM experiments ORDER BY composite_score DESC", conn)
        conn.close()
        df.to_csv(path, index=False)
        return path
