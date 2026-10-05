"""Controlled matches and evaluation reports."""

from chess_ai.arena.external import (
    ExternalAgentMoveError,
    ExternalOpponentSession,
    ExternalSessionResult,
    report_external_benchmarks,
    run_external_session,
)
from chess_ai.arena.gameplay_selection import (
    GAMEPLAY_SELECTION_FORMAT,
    GAMEPLAY_SELECTION_VERSION,
    GameplaySelectionConfig,
    GameplaySelectionError,
    GameplaySelectionSummary,
    run_gameplay_selection,
)
from chess_ai.arena.match import ArenaAgent, MatchResult, MoveRecord, play_match, run_match
from chess_ai.arena.paired_audit import (
    PairedAuditError,
    PairedAuditSummary,
    run_paired_audit,
)
from chess_ai.arena.ratings import (
    EloEstimate,
    approximate_elo,
    build_elo_estimate,
    estimate_elo,
    estimate_elo_difference,
)
from chess_ai.arena.tournament import (
    CandidateChampionReport,
    ScoreStats,
    TournamentResult,
    candidate_champion_report,
    evaluate_candidate,
    run_tournament,
)

__all__ = [
    "GAMEPLAY_SELECTION_FORMAT",
    "GAMEPLAY_SELECTION_VERSION",
    "ArenaAgent",
    "CandidateChampionReport",
    "EloEstimate",
    "ExternalAgentMoveError",
    "ExternalOpponentSession",
    "ExternalSessionResult",
    "GameplaySelectionConfig",
    "GameplaySelectionError",
    "GameplaySelectionSummary",
    "MatchResult",
    "MoveRecord",
    "PairedAuditError",
    "PairedAuditSummary",
    "ScoreStats",
    "TournamentResult",
    "approximate_elo",
    "build_elo_estimate",
    "candidate_champion_report",
    "estimate_elo",
    "estimate_elo_difference",
    "evaluate_candidate",
    "play_match",
    "report_external_benchmarks",
    "run_external_session",
    "run_gameplay_selection",
    "run_match",
    "run_paired_audit",
    "run_tournament",
]
