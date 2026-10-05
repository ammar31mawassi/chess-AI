"""Local Human-vs-Neural graphical workbench and training-data boundary."""

from chess_ai.gui.app import (
    DEFAULT_DATASET_PATH,
    DEFAULT_PGN_DIR,
    NeuralChessApp,
    discover_default_checkpoint,
    grid_to_square,
    launch_gui,
    square_to_grid,
)
from chess_ai.gui.controller import HumanNeuralGame
from chess_ai.gui.session_training import (
    SESSION_PLAN_FORMAT,
    SESSION_PLAN_VERSION,
    SessionTrainingError,
    SessionTrainingPlan,
    SessionTrainingResult,
    create_session_training_plan,
    run_session_training,
    update_session_dataset_path,
)
from chess_ai.gui.training_data import (
    DEFAULT_HUMAN_GUI_DATASET_PATH,
    ExplicitTrainingOptInRequired,
    HumanGuiAppendResult,
    HumanGuiDatasetError,
    append_human_gui_game,
    load_human_gui_dataset,
)

__all__ = [
    "DEFAULT_DATASET_PATH",
    "DEFAULT_HUMAN_GUI_DATASET_PATH",
    "DEFAULT_PGN_DIR",
    "SESSION_PLAN_FORMAT",
    "SESSION_PLAN_VERSION",
    "ExplicitTrainingOptInRequired",
    "HumanGuiAppendResult",
    "HumanGuiDatasetError",
    "HumanNeuralGame",
    "NeuralChessApp",
    "SessionTrainingError",
    "SessionTrainingPlan",
    "SessionTrainingResult",
    "append_human_gui_game",
    "create_session_training_plan",
    "discover_default_checkpoint",
    "grid_to_square",
    "launch_gui",
    "load_human_gui_dataset",
    "run_session_training",
    "square_to_grid",
    "update_session_dataset_path",
]
