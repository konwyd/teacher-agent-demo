"""Isolated persistent shadow-evidence ledger for Memory v2 Phase 2A."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from app.memory.admission_normalization import NormalizedCandidate
from app.memory.contracts import (
    AdmissionDecision,
    EvidenceClusterSnapshot,
    MemoryCandidate,
    MemorySourceKind,
    ProvenanceReference,
)


class ShadowEvidenceLedger:
    """SQLite evidence repository isolated from active learner-memory storage."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS shadow_evidence_clusters (
                    student_key TEXT NOT NULL,
                    semantic_key TEXT NOT NULL,
                    cluster_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    normalized_payload_json TEXT NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    current_evidence_maturity REAL NOT NULL,
                    last_proposed_decision TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (student_key, semantic_key)
                );

                CREATE TABLE IF NOT EXISTS shadow_evidence_events (
                    student_key TEXT NOT NULL,
                    semantic_key TEXT NOT NULL,
                    provenance_fingerprint TEXT NOT NULL,
                    polarity TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    turn_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (
                        student_key, semantic_key, provenance_fingerprint
                    )
                );

                CREATE INDEX IF NOT EXISTS idx_shadow_evidence_student
                    ON shadow_evidence_clusters (student_key);
                CREATE INDEX IF NOT EXISTS idx_shadow_events_cluster
                    ON shadow_evidence_events (student_key, semantic_key);
                """
            )

    @staticmethod
    def _cluster_id(student_key: str, semantic_key: str) -> str:
        digest = hashlib.sha256(
            f"{student_key}\n{semantic_key}".encode("utf-8")
        ).hexdigest()[:24]
        return f"evidence_{digest}"

    @staticmethod
    def _fingerprint(provenance: ProvenanceReference) -> str:
        canonical = provenance.model_dump_json(exclude={"schema_version"})
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def append(
        self,
        *,
        student_key: str,
        candidate: MemoryCandidate,
        normalized: NormalizedCandidate,
        decision: AdmissionDecision,
        evidence_maturity: float,
    ) -> None:
        if not normalized.semantic_key:
            return
        payload_json = json.dumps(
            normalized.normalized_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._connect() as connection:
            row = connection.execute(
                "SELECT normalized_payload_json FROM shadow_evidence_clusters "
                "WHERE student_key = ? AND semantic_key = ?",
                (student_key, normalized.semantic_key),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO shadow_evidence_clusters ("
                    "student_key, semantic_key, cluster_id, memory_type, "
                    "normalized_payload_json, first_seen, last_seen, "
                    "current_evidence_maturity, last_proposed_decision"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        student_key,
                        normalized.semantic_key,
                        self._cluster_id(student_key, normalized.semantic_key),
                        candidate.memory_type.value,
                        payload_json,
                        candidate.created_at.isoformat(),
                        candidate.created_at.isoformat(),
                        0.0,
                        "",
                    ),
                )
                polarity = "support"
            else:
                polarity = (
                    "support"
                    if row["normalized_payload_json"] == payload_json
                    else "contradiction"
                )

            inserted = 0
            for provenance in candidate.provenance:
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO shadow_evidence_events ("
                    "student_key, semantic_key, provenance_fingerprint, polarity, "
                    "provenance_json, source_kind, conversation_id, turn_id, created_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        student_key,
                        normalized.semantic_key,
                        self._fingerprint(provenance),
                        polarity,
                        provenance.model_dump_json(),
                        provenance.source_kind.value,
                        provenance.conversation_id,
                        provenance.turn_id,
                        candidate.created_at.isoformat(),
                    ),
                )
                inserted += cursor.rowcount
            if inserted:
                connection.execute(
                    "UPDATE shadow_evidence_clusters SET last_seen = ?, "
                    "current_evidence_maturity = ?, last_proposed_decision = ? "
                    "WHERE student_key = ? AND semantic_key = ?",
                    (
                        candidate.created_at.isoformat(),
                        evidence_maturity,
                        decision.decision.value,
                        student_key,
                        normalized.semantic_key,
                    ),
                )

    def snapshot(
        self,
        student_key: str,
        *,
        semantic_keys: tuple[str, ...] | None = None,
    ) -> tuple[EvidenceClusterSnapshot, ...]:
        query = (
            "SELECT * FROM shadow_evidence_clusters WHERE student_key = ?"
        )
        parameters: list[object] = [student_key]
        if semantic_keys is not None:
            if not semantic_keys:
                return ()
            placeholders = ",".join("?" for _ in semantic_keys)
            query += f" AND semantic_key IN ({placeholders})"
            parameters.extend(semantic_keys)
        query += " ORDER BY semantic_key"
        with self._connect() as connection:
            clusters = connection.execute(query, parameters).fetchall()
            return tuple(
                self._snapshot_row(connection, row)
                for row in clusters
            )

    def inspect(self, student_key: str) -> tuple[EvidenceClusterSnapshot, ...]:
        return self.snapshot(student_key)

    def reset(self, student_key: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM shadow_evidence_events WHERE student_key = ?",
                (student_key,),
            )
            connection.execute(
                "DELETE FROM shadow_evidence_clusters WHERE student_key = ?",
                (student_key,),
            )

    @staticmethod
    def _snapshot_row(
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> EvidenceClusterSnapshot:
        events = connection.execute(
            "SELECT * FROM shadow_evidence_events "
            "WHERE student_key = ? AND semantic_key = ? "
            "ORDER BY created_at, provenance_fingerprint",
            (row["student_key"], row["semantic_key"]),
        ).fetchall()
        supports = tuple(
            ProvenanceReference.model_validate_json(event["provenance_json"])
            for event in events
            if event["polarity"] == "support"
        )
        contradictions = tuple(
            ProvenanceReference.model_validate_json(event["provenance_json"])
            for event in events
            if event["polarity"] == "contradiction"
        )
        distinct_turns = {
            (event["conversation_id"], event["turn_id"])
            for event in events
        }
        conversations = {event["conversation_id"] for event in events}
        source_kinds = tuple(
            sorted(
                {MemorySourceKind(event["source_kind"]) for event in events},
                key=lambda item: item.value,
            )
        )
        last_decision = row["last_proposed_decision"] or None
        return EvidenceClusterSnapshot(
            cluster_id=row["cluster_id"],
            semantic_key=row["semantic_key"],
            normalized_payload=json.loads(row["normalized_payload_json"]),
            memory_type=row["memory_type"],
            supporting_provenance=supports,
            contradicting_provenance=contradictions,
            support_count=len(supports),
            contradiction_count=len(contradictions),
            distinct_turn_count=len(distinct_turns),
            distinct_conversation_count=len(conversations),
            verified_support_count=sum(
                item.source_kind is MemorySourceKind.VERIFIED_RESULT
                for item in supports
            ),
            source_kinds=source_kinds,
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            current_evidence_maturity=row["current_evidence_maturity"],
            last_proposed_decision=last_decision,
        )
