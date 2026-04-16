from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
import sqlite3
from typing import cast

from biocompute.models import (
    Evidence,
    EvidenceMaturity,
    FitnessScores,
    PriorKnowledge,
    StrategyPriorArt,
    TherapeuticHypothesis,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS hypotheses (
    id TEXT PRIMARY KEY,
    generation INTEGER,
    target_gene TEXT,
    modality TEXT,
    delivery TEXT,
    duration TEXT,
    tissue_context TEXT,
    fitness_total REAL,
    parent_id TEXT,
    mutation_type TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS scores (
    hypothesis_id TEXT,
    dimension TEXT,
    score REAL,
    source TEXT,
    raw_data JSON,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(id)
);

CREATE TABLE IF NOT EXISTS critiques (
    hypothesis_id TEXT,
    critique_text TEXT,
    model_used TEXT,
    created_at TEXT,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(id)
);

CREATE TABLE IF NOT EXISTS evidence (
    hypothesis_id TEXT,
    source_type TEXT,
    source_id TEXT,
    summary TEXT,
    relevance_score REAL,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(id)
);

CREATE TABLE IF NOT EXISTS prior_knowledge (
    hypothesis_id TEXT PRIMARY KEY,
    gene TEXT,
    disease TEXT,
    maturity INTEGER,
    known_facts TEXT,
    attempted_approaches TEXT,
    gaps TEXT,
    key_papers TEXT,
    summary TEXT,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(id)
);

CREATE TABLE IF NOT EXISTS strategy_prior_art (
    hypothesis_id TEXT PRIMARY KEY,
    strategy TEXT,
    disease_class TEXT,
    prior_studies TEXT,
    modality_status TEXT,
    our_differentiation TEXT,
    key_papers TEXT,
    summary TEXT,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(id)
);
"""


def _serialize_string_list(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=True, separators=(",", ":"))


def _deserialize_string_list(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    try:
        parsed = cast(object, json.loads(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    parsed_list = cast(list[object], parsed)
    return [item for item in parsed_list if isinstance(item, str)]


def _serialize_string_mapping(values: Mapping[str, str]) -> str:
    return json.dumps(dict(values), ensure_ascii=True, separators=(",", ":"))


def _deserialize_string_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, str):
        return {}
    try:
        parsed = cast(object, json.loads(value))
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}

    normalized: dict[str, str] = {}
    for key, item in cast(dict[object, object], parsed).items():
        if not isinstance(key, str) or not isinstance(item, str):
            continue
        normalized[key] = item
    return normalized


class ArchiveStore:
    def __init__(self, db_path: str):
        self.conn: sqlite3.Connection
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        _ = self.conn.executescript(SCHEMA)

    def save_hypothesis(
        self,
        hypothesis: TherapeuticHypothesis,
        scores: FitnessScores,
        fitness_total: float,
        dimension_sources: Mapping[str, str] | None = None,
        dimension_raw_data: Mapping[str, object] | None = None,
    ) -> None:
        _ = self.conn.execute(
            """INSERT OR REPLACE INTO hypotheses
               (id, generation, target_gene, modality, delivery, duration,
                tissue_context, fitness_total, parent_id, mutation_type, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                hypothesis.id,
                hypothesis.generation,
                hypothesis.target_gene,
                hypothesis.modality,
                hypothesis.delivery,
                hypothesis.duration,
                hypothesis.tissue_context,
                fitness_total,
                hypothesis.parent_id,
                hypothesis.mutation_type,
                datetime.now().isoformat(),
            ),
        )
        _ = self.conn.execute(
            "DELETE FROM scores WHERE hypothesis_id = ?",
            (hypothesis.id,),
        )
        sources = dimension_sources or {}
        raw = dimension_raw_data or {}
        for dimension, score in scores.dimensions().items():
            source = sources.get(dimension)
            raw_json = json.dumps(raw[dimension]) if dimension in raw else None
            _ = self.conn.execute(
                "INSERT INTO scores (hypothesis_id, dimension, score, source, raw_data) VALUES (?, ?, ?, ?, ?)",
                (hypothesis.id, dimension, score, source, raw_json),
            )
        self.conn.commit()

    def save_evidence(self, hypothesis_id: str, evidence: Evidence) -> None:
        _ = self.conn.execute(
            """INSERT INTO evidence
               (hypothesis_id, source_type, source_id, summary, relevance_score)
               VALUES (?, ?, ?, ?, ?)""",
            (
                hypothesis_id,
                evidence.source_type,
                evidence.source_id,
                evidence.summary,
                evidence.relevance_score,
            ),
        )
        self.conn.commit()

    def save_critique(self, hypothesis_id: str, text: str, model: str) -> None:
        _ = self.conn.execute(
            """INSERT INTO critiques (hypothesis_id, critique_text, model_used, created_at)
               VALUES (?, ?, ?, ?)""",
            (hypothesis_id, text, model, datetime.now().isoformat()),
        )
        self.conn.commit()

    def save_prior_knowledge(
        self,
        hypothesis_id: str,
        prior_knowledge: PriorKnowledge,
    ) -> None:
        _ = self.conn.execute(
            """INSERT OR REPLACE INTO prior_knowledge
               (hypothesis_id, gene, disease, maturity, known_facts,
                attempted_approaches, gaps, key_papers, summary)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                hypothesis_id,
                prior_knowledge.gene,
                prior_knowledge.disease,
                int(prior_knowledge.maturity),
                _serialize_string_list(prior_knowledge.known_facts),
                _serialize_string_list(prior_knowledge.attempted_approaches),
                _serialize_string_list(prior_knowledge.gaps),
                _serialize_string_list(prior_knowledge.key_papers),
                prior_knowledge.summary,
            ),
        )
        self.conn.commit()

    def save_strategy_prior_art(
        self,
        hypothesis_id: str,
        strategy_prior_art: StrategyPriorArt,
    ) -> None:
        _ = self.conn.execute(
            """INSERT OR REPLACE INTO strategy_prior_art
               (hypothesis_id, strategy, disease_class, prior_studies,
                modality_status, our_differentiation, key_papers, summary)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                hypothesis_id,
                strategy_prior_art.strategy,
                strategy_prior_art.disease_class,
                _serialize_string_list(strategy_prior_art.prior_studies),
                _serialize_string_mapping(strategy_prior_art.modality_status),
                _serialize_string_list(strategy_prior_art.our_differentiation),
                _serialize_string_list(strategy_prior_art.key_papers),
                strategy_prior_art.summary,
            ),
        )
        self.conn.commit()

    def get_hypothesis(self, hypothesis_id: str) -> dict[str, object] | None:
        row = cast(
            sqlite3.Row | None,
            self.conn.execute(
                "SELECT * FROM hypotheses WHERE id = ?",
                (hypothesis_id,),
            ).fetchone(),
        )
        return dict(row) if row else None

    def get_evidence(self, hypothesis_id: str) -> list[dict[str, object]]:
        rows = cast(
            list[sqlite3.Row],
            self.conn.execute(
                "SELECT * FROM evidence WHERE hypothesis_id = ?",
                (hypothesis_id,),
            ).fetchall(),
        )
        return [dict(row) for row in rows]

    def get_prior_knowledge(self, hypothesis_id: str) -> PriorKnowledge | None:
        if not self._table_exists("prior_knowledge"):
            return None

        row = cast(
            sqlite3.Row | None,
            self.conn.execute(
                "SELECT * FROM prior_knowledge WHERE hypothesis_id = ?",
                (hypothesis_id,),
            ).fetchone(),
        )
        if row is None:
            return None

        expected_columns = {
            "gene",
            "disease",
            "maturity",
            "known_facts",
            "attempted_approaches",
            "gaps",
            "key_papers",
            "summary",
        }
        if not expected_columns.issubset(set(row.keys())):
            return None

        gene = cast(object, row["gene"])
        disease = cast(object, row["disease"])
        summary = cast(object, row["summary"])
        maturity_value = cast(object, row["maturity"])
        if not isinstance(gene, str) or not isinstance(disease, str):
            return None
        if not isinstance(summary, str) or not isinstance(maturity_value, int):
            return None

        try:
            maturity = EvidenceMaturity(maturity_value)
        except ValueError:
            return None

        return PriorKnowledge(
            gene=gene,
            disease=disease,
            maturity=maturity,
            known_facts=_deserialize_string_list(cast(object, row["known_facts"])),
            attempted_approaches=_deserialize_string_list(
                cast(object, row["attempted_approaches"])
            ),
            gaps=_deserialize_string_list(cast(object, row["gaps"])),
            key_papers=_deserialize_string_list(cast(object, row["key_papers"])),
            summary=summary,
        )

    def get_strategy_prior_art(self, hypothesis_id: str) -> StrategyPriorArt | None:
        if not self._table_exists("strategy_prior_art"):
            return None

        row = cast(
            sqlite3.Row | None,
            self.conn.execute(
                "SELECT * FROM strategy_prior_art WHERE hypothesis_id = ?",
                (hypothesis_id,),
            ).fetchone(),
        )
        if row is None:
            return None

        expected_columns = {
            "strategy",
            "disease_class",
            "prior_studies",
            "modality_status",
            "our_differentiation",
            "key_papers",
            "summary",
        }
        if not expected_columns.issubset(set(row.keys())):
            return None

        strategy = cast(object, row["strategy"])
        disease_class = cast(object, row["disease_class"])
        summary = cast(object, row["summary"])
        if not isinstance(strategy, str) or not isinstance(disease_class, str):
            return None
        if not isinstance(summary, str):
            return None

        return StrategyPriorArt(
            strategy=strategy,
            disease_class=disease_class,
            prior_studies=_deserialize_string_list(cast(object, row["prior_studies"])),
            modality_status=_deserialize_string_mapping(
                cast(object, row["modality_status"])
            ),
            our_differentiation=_deserialize_string_list(
                cast(object, row["our_differentiation"])
            ),
            key_papers=_deserialize_string_list(cast(object, row["key_papers"])),
            summary=summary,
        )

    def get_scores(self, hypothesis_id: str) -> FitnessScores | None:
        rows = cast(
            list[sqlite3.Row],
            self.conn.execute(
                "SELECT dimension, score, source, raw_data FROM scores WHERE hypothesis_id = ?",
                (hypothesis_id,),
            ).fetchall(),
        )
        if not rows:
            return None

        values: dict[str, float] = {}
        valid_dimensions = set(FitnessScores().dimensions())
        for row in rows:
            dimension = cast(object, row["dimension"])
            score = cast(object, row["score"])
            if not isinstance(dimension, str) or dimension not in valid_dimensions:
                continue
            if not isinstance(score, int | float):
                continue
            values[dimension] = float(score)

        return FitnessScores(**values)

    def get_scores_with_metadata(self, hypothesis_id: str) -> list[dict[str, object]]:
        rows = cast(
            list[sqlite3.Row],
            self.conn.execute(
                "SELECT dimension, score, source, raw_data FROM scores WHERE hypothesis_id = ?",
                (hypothesis_id,),
            ).fetchall(),
        )
        result: list[dict[str, object]] = []
        for row in rows:
            raw_data_str = cast(object, row["raw_data"])
            parsed_raw: object = None
            if isinstance(raw_data_str, str):
                try:
                    parsed_raw = cast(object, json.loads(raw_data_str))
                except json.JSONDecodeError:
                    parsed_raw = raw_data_str
            result.append(
                {
                    "dimension": row["dimension"],
                    "score": row["score"],
                    "source": row["source"],
                    "raw_data": parsed_raw,
                }
            )
        return result

    def get_critiques(self, hypothesis_id: str) -> list[dict[str, object]]:
        rows = cast(
            list[sqlite3.Row],
            self.conn.execute(
                "SELECT * FROM critiques WHERE hypothesis_id = ?",
                (hypothesis_id,),
            ).fetchall(),
        )
        return [dict(row) for row in rows]

    def get_top_hypotheses(self, n: int = 10) -> list[dict[str, object]]:
        rows = cast(
            list[sqlite3.Row],
            self.conn.execute(
                "SELECT * FROM hypotheses ORDER BY fitness_total DESC LIMIT ?",
                (n,),
            ).fetchall(),
        )
        return [dict(row) for row in rows]

    def get_generation(self, generation: int) -> list[dict[str, object]]:
        rows = cast(
            list[sqlite3.Row],
            self.conn.execute(
                "SELECT * FROM hypotheses WHERE generation = ? ORDER BY fitness_total DESC",
                (generation,),
            ).fetchall(),
        )
        return [dict(row) for row in rows]

    def _table_exists(self, table_name: str) -> bool:
        row = cast(
            sqlite3.Row | None,
            self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table_name,),
            ).fetchone(),
        )
        return row is not None

    def close(self) -> None:
        self.conn.close()
