"""Controlled student/teacher curriculum experiments."""

from chess_ai.curriculum.gated_refinement import (
    GATED_FORMAT,
    GATED_VERSION,
    GatedHoldoutAuditSummary,
    GatedRefinementConfig,
    GatedRefinementError,
    GatedRefinementSummary,
    initialize_gated_refinement,
    run_gated_holdout_audit,
    run_gated_refinement,
)
from chess_ai.curriculum.mixed_batch import (
    MIXED_BATCH_FORMAT,
    MIXED_BATCH_VERSION,
    MixedBatchConfig,
    MixedBatchError,
    MixedBatchSummary,
    finalize_mixed_batch,
    run_mixed_batch,
)
from chess_ai.curriculum.teacher_batch import (
    TEACHER_BATCH_FORMAT,
    TEACHER_BATCH_VERSION,
    TeacherBatchConfig,
    TeacherBatchError,
    TeacherBatchSummary,
    run_teacher_batch,
)
from chess_ai.curriculum.teacher_cycle import (
    CURRICULUM_FORMAT,
    CURRICULUM_VERSION,
    TeacherCycleConfig,
    TeacherCycleError,
    TeacherCycleSummary,
    run_teacher_cycle,
)

__all__ = [
    "CURRICULUM_FORMAT",
    "CURRICULUM_VERSION",
    "GATED_FORMAT",
    "GATED_VERSION",
    "MIXED_BATCH_FORMAT",
    "MIXED_BATCH_VERSION",
    "TEACHER_BATCH_FORMAT",
    "TEACHER_BATCH_VERSION",
    "GatedHoldoutAuditSummary",
    "GatedRefinementConfig",
    "GatedRefinementError",
    "GatedRefinementSummary",
    "MixedBatchConfig",
    "MixedBatchError",
    "MixedBatchSummary",
    "TeacherBatchConfig",
    "TeacherBatchError",
    "TeacherBatchSummary",
    "TeacherCycleConfig",
    "TeacherCycleError",
    "TeacherCycleSummary",
    "finalize_mixed_batch",
    "initialize_gated_refinement",
    "run_gated_holdout_audit",
    "run_gated_refinement",
    "run_mixed_batch",
    "run_teacher_batch",
    "run_teacher_cycle",
]
