"""Tkinter workbench for playing a saved neural checkpoint by clicking moves.

The window deliberately contains no chess-rule implementation.  It asks
``HumanNeuralGame`` for selectable squares and applies only moves accepted by
that controller.  Neural inference runs on a daemon worker so Tk's event loop
continues repainting while the model thinks.
"""

from __future__ import annotations

import gc
import math
import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import chess
import torch

from chess_ai.agents.neural_agent import NeuralAgent
from chess_ai.agents.neural_mcts_agent import NeuralMCTSAgent
from chess_ai.agents.opening_book_agent import OpeningBookAgent
from chess_ai.gui.controller import HumanNeuralGame
from chess_ai.gui.session_training import (
    SessionTrainingPlan,
    SessionTrainingResult,
    create_session_training_plan,
    run_session_training,
    update_session_dataset_path,
)

DEFAULT_DATASET_PATH = Path("data/datasets/human_gui.pt")
DEFAULT_PGN_DIR = Path("data/games/human_gui")

PIECE_SYMBOLS: dict[tuple[chess.Color, chess.PieceType], str] = {
    (chess.WHITE, chess.KING): "♔",
    (chess.WHITE, chess.QUEEN): "♕",
    (chess.WHITE, chess.ROOK): "♖",
    (chess.WHITE, chess.BISHOP): "♗",
    (chess.WHITE, chess.KNIGHT): "♘",
    (chess.WHITE, chess.PAWN): "♙",
    (chess.BLACK, chess.KING): "♚",
    (chess.BLACK, chess.QUEEN): "♛",
    (chess.BLACK, chess.ROOK): "♜",
    (chess.BLACK, chess.BISHOP): "♝",
    (chess.BLACK, chess.KNIGHT): "♞",
    (chess.BLACK, chess.PAWN): "♟",
}

LIGHT_SQUARE = "#E9DDC7"
DARK_SQUARE = "#7A5C52"
SELECTED_SQUARE = "#F4D35E"
LAST_MOVE_SQUARE = "#DDB967"
CHECK_SQUARE = "#DF6C63"
LEGAL_MARKER = "#187B73"


def grid_to_square(row: int, column: int, *, flipped: bool) -> chess.Square:
    """Map a displayed board cell to a python-chess square."""

    if not 0 <= row < 8 or not 0 <= column < 8:
        raise ValueError("board row and column must be between 0 and 7")
    if flipped:
        file_index = 7 - column
        rank_index = row
    else:
        file_index = column
        rank_index = 7 - row
    return chess.square(file_index, rank_index)


def square_to_grid(square: chess.Square, *, flipped: bool) -> tuple[int, int]:
    """Map a python-chess square to its displayed row and column."""

    if isinstance(square, bool) or square not in chess.SQUARES:
        raise ValueError("square must be between 0 and 63")
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)
    if flipped:
        return rank_index, 7 - file_index
    return 7 - rank_index, file_index


def discover_default_checkpoint(root: str | Path = "checkpoints") -> Path | None:
    """Find the most useful local checkpoint without downloading anything."""

    checkpoint_root = Path(root)
    human_session_root = checkpoint_root / "human_sessions"
    if human_session_root.is_dir():
        human_candidates = sorted(
            human_session_root.rglob("best.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if human_candidates:
            return human_candidates[0]
    preferred = (
        checkpoint_root / "gpu_first" / "best.pt",
        checkpoint_root / "dev" / "best.pt",
    )
    for candidate in preferred:
        if candidate.is_file():
            return candidate
    if not checkpoint_root.is_dir():
        return None
    best_candidates = sorted(
        checkpoint_root.rglob("best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if best_candidates:
        return best_candidates[0]
    latest_candidates = sorted(
        checkpoint_root.rglob("last.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return latest_candidates[0] if latest_candidates else None


@dataclass(frozen=True, slots=True)
class _AiResult:
    token: int
    controller: HumanNeuralGame
    move: chess.Move | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _TrainingUiResult:
    token: int
    plan: SessionTrainingPlan
    result: SessionTrainingResult | None = None
    error: str | None = None


class NeuralChessApp:
    """Single-window Human-vs-Neural chess workbench."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        checkpoint: str | Path | None = None,
        human_color: str = "white",
        device: str = "auto",
        opening_book: str | Path | None = None,
        search_simulations: int = 0,
        c_puct: float = 1.5,
        dataset_path: str | Path = DEFAULT_DATASET_PATH,
        pgn_dir: str | Path = DEFAULT_PGN_DIR,
        training_enabled: bool = False,
    ) -> None:
        self.root = root
        self.root.title("Neural Chess Workbench")
        self.root.geometry("1080x720")
        self.root.minsize(880, 600)

        discovered = Path(checkpoint) if checkpoint is not None else discover_default_checkpoint()
        self.checkpoint_var = tk.StringVar(value=str(discovered or ""))
        self.human_color_var = tk.StringVar(value=human_color)
        self.device_var = tk.StringVar(value=device)
        self.training_var = tk.BooleanVar(value=training_enabled)
        self.opening_book = Path(opening_book) if opening_book is not None else None
        if isinstance(search_simulations, bool) or not isinstance(search_simulations, int):
            raise ValueError("search_simulations must be a non-negative integer")
        if search_simulations < 0:
            raise ValueError("search_simulations must be a non-negative integer")
        if not math.isfinite(c_puct) or c_puct <= 0.0:
            raise ValueError("c_puct must be finite and positive")
        self.search_simulations = search_simulations
        self.c_puct = c_puct
        self.dataset_var = tk.StringVar(value=str(dataset_path))
        self.status_var = tk.StringVar(value="Choose a checkpoint, then start a new game.")
        self.session_var = tk.StringVar(value="Preparing automatic training files...")

        self.pgn_dir = Path(pgn_dir)
        self.controller: HumanNeuralGame | None = None
        self._agent_cache: tuple[Path, str, int, float, NeuralAgent] | None = None
        self._selected_square: chess.Square | None = None
        self._legal_destinations: set[chess.Square] = set()
        self._flipped = human_color == "black"
        self._ai_busy = False
        self._training_busy = False
        self._worker_token = 0
        self._training_token = 0
        self._closing = False
        self._finalized_game_id: str | None = None
        self._ai_results: queue.Queue[_AiResult] = queue.Queue()
        self._training_results: queue.Queue[_TrainingUiResult] = queue.Queue()
        self._session_confirmed_games = 0
        self._session_confirmed_examples = 0
        self._session_game_number = 0
        self._session_source_checkpoint: Path | None = None
        self._session_dataset_path: Path | None = None
        self._session_device: str | None = None
        self._training_session: SessionTrainingPlan | None = None
        self._session_setup_error: str | None = None
        self._training_runner = run_session_training
        self._board_size = 0.0
        self._board_left = 0.0
        self._board_top = 0.0

        try:
            self._training_session = create_session_training_plan(dataset_path=Path(dataset_path))
        except (OSError, RuntimeError, ValueError) as exc:
            self._session_setup_error = str(exc)
            self.status_var.set("Games are available, but automatic training setup failed.")
        self._refresh_session_label()

        self._configure_style()
        self._build_layout()
        self._bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(50, self._poll_ai_results)
        self.root.after(50, self._poll_training_results)
        self.root.after(30, self._start_discovered_checkpoint)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure(".", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 16))
        style.configure("Status.TLabel", font=("Segoe UI Semibold", 11))
        style.configure("Train.TButton", font=("Segoe UI Semibold", 10), padding=(10, 8))

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=0)
        container.rowconfigure(0, weight=1)

        board_frame = ttk.Frame(container)
        board_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        board_frame.rowconfigure(0, weight=1)
        board_frame.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(board_frame, highlightthickness=0, background="#202020")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<Button-1>", self._on_board_click)

        panel = ttk.Frame(container, width=330)
        panel.grid(row=0, column=1, sticky="ns")
        panel.grid_propagate(False)
        panel.columnconfigure(0, weight=1)

        ttk.Label(panel, text="Neural Chess Workbench", style="Title.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )

        setup = ttk.LabelFrame(panel, text="Game setup", padding=10)
        setup.grid(row=1, column=0, sticky="ew")
        setup.columnconfigure(0, weight=1)
        self.checkpoint_entry = ttk.Entry(setup, textvariable=self.checkpoint_var)
        self.checkpoint_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.browse_button = ttk.Button(setup, text="Browse...", command=self.browse_checkpoint)
        self.browse_button.grid(row=0, column=1)

        options = ttk.Frame(setup)
        options.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        options.columnconfigure(1, weight=1)
        ttk.Label(options, text="Play as").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.color_combo = ttk.Combobox(
            options,
            textvariable=self.human_color_var,
            values=("white", "black"),
            state="readonly",
            width=10,
        )
        self.color_combo.grid(row=0, column=1, sticky="ew")
        ttk.Label(options, text="Device").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=(6, 0)
        )
        self.device_combo = ttk.Combobox(
            options,
            textvariable=self.device_var,
            values=("auto", "cpu", "cuda"),
            state="readonly",
            width=10,
        )
        self.device_combo.grid(row=1, column=1, sticky="ew", pady=(6, 0))

        self.training_check = ttk.Checkbutton(
            setup,
            text="Learn from this game (confirm at end)",
            variable=self.training_var,
        )
        self.training_check.grid(row=2, column=0, columnspan=2, sticky="w", pady=(9, 0))
        ttk.Label(setup, text="Training dataset").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(8, 2)
        )
        self.dataset_entry = ttk.Entry(setup, textvariable=self.dataset_var)
        self.dataset_entry.grid(row=4, column=0, columnspan=2, sticky="ew")
        ttk.Label(
            setup,
            text="Setup changes apply to the next game.",
            foreground="#555555",
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(5, 0))

        controls = ttk.LabelFrame(panel, text="Game controls", padding=10)
        controls.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        controls.columnconfigure((0, 1), weight=1)
        self.new_button = ttk.Button(controls, text="New Game", command=self.new_game)
        self.new_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.undo_button = ttk.Button(controls, text="Undo Turn", command=self.undo_turn)
        self.undo_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.resign_button = ttk.Button(controls, text="Resign", command=self.resign)
        self.resign_button.grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(8, 0))
        self.flip_button = ttk.Button(controls, text="Flip Board", command=self.flip_board)
        self.flip_button.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(8, 0))

        session = ttk.LabelFrame(panel, text="Automatic training session", padding=10)
        session.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        session.columnconfigure(0, weight=1)
        ttk.Label(
            session,
            textvariable=self.session_var,
            wraplength=285,
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky="ew")
        self.done_button = ttk.Button(
            session,
            text="I'm Done — Train AI",
            style="Train.TButton",
            command=self.finish_session_and_train,
        )
        self.done_button.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        status = ttk.LabelFrame(panel, text="Status", padding=10)
        status.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        self.status_label = ttk.Label(
            status,
            textvariable=self.status_var,
            style="Status.TLabel",
            wraplength=285,
            justify=tk.LEFT,
        )
        self.status_label.pack(fill=tk.X)

        history = ttk.LabelFrame(panel, text="Move history", padding=8)
        history.grid(row=5, column=0, sticky="nsew", pady=(10, 0))
        panel.rowconfigure(5, weight=1)
        self.history_text = tk.Text(
            history,
            height=10,
            width=36,
            state=tk.DISABLED,
            wrap=tk.WORD,
            font=("Consolas", 10),
            relief=tk.FLAT,
            borderwidth=0,
        )
        scrollbar = ttk.Scrollbar(history, orient=tk.VERTICAL, command=self.history_text.yview)
        self.history_text.configure(yscrollcommand=scrollbar.set)
        self.history_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._update_controls()

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-n>", lambda _event: self.new_game())
        self.root.bind("<Control-o>", lambda _event: self.browse_checkpoint())
        self.root.bind("<Control-z>", lambda _event: self.undo_turn())
        self.root.bind("<Control-f>", lambda _event: self.flip_board())
        self.root.bind("<Control-Return>", lambda _event: self.finish_session_and_train())
        self.root.bind("<Escape>", lambda _event: self.clear_selection())

    def _refresh_session_label(self) -> None:
        plan = self._training_session
        if plan is None:
            detail = self._session_setup_error or "automatic training files are unavailable"
            self.session_var.set(f"Automatic training unavailable:\n{detail}")
            return
        game_word = "game" if self._session_confirmed_games == 1 else "games"
        self.session_var.set(
            f"Run {plan.session_id}\n"
            f"{self._session_confirmed_games} confirmed {game_word} - "
            f"{self._session_confirmed_examples} human moves"
        )

    def _sync_session_dataset(self, dataset_path: Path) -> bool:
        plan = self._training_session
        if plan is None:
            return False
        try:
            self._training_session = update_session_dataset_path(plan, dataset_path)
        except (OSError, RuntimeError, ValueError) as exc:
            self._session_setup_error = str(exc)
            self._refresh_session_label()
            return False
        self._session_setup_error = None
        self._refresh_session_label()
        return True

    def _start_discovered_checkpoint(self) -> None:
        if self._closing:
            return
        if self.checkpoint_var.get().strip():
            self.new_game(skip_discard_prompt=True)
        else:
            self.render_board()

    def browse_checkpoint(self) -> None:
        if self._ai_busy or self._training_busy or self._session_confirmed_games:
            return
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Choose a policy-value checkpoint",
            filetypes=(("PyTorch checkpoints", "*.pt"), ("All files", "*.*")),
        )
        if selected:
            self.checkpoint_var.set(selected)

    def _load_agent(self, checkpoint: Path, device: str) -> NeuralAgent:
        resolved = checkpoint.resolve()
        if self._agent_cache is not None:
            cached_path, cached_device, cached_simulations, cached_c_puct, cached_agent = (
                self._agent_cache
            )
            if (
                cached_path == resolved
                and cached_device == device
                and cached_simulations == self.search_simulations
                and cached_c_puct == self.c_puct
            ):
                return cached_agent
        agent = (
            NeuralMCTSAgent(
                resolved,
                device=device,
                simulations=self.search_simulations,
                c_puct=self.c_puct,
                seed=0,
            )
            if self.search_simulations > 0
            else NeuralAgent(
                resolved,
                device=device,
                deterministic=True,
                temperature=0.0,
                seed=0,
            )
        )
        self._agent_cache = (
            resolved,
            device,
            self.search_simulations,
            self.c_puct,
            agent,
        )
        return agent

    def new_game(self, *, skip_discard_prompt: bool = False) -> None:
        if self._ai_busy or self._training_busy:
            return
        if (
            not skip_discard_prompt
            and self.controller is not None
            and self.controller.move_history
            and not self.controller.game_over
            and not messagebox.askyesno(
                "Start a new game?",
                "The current unfinished game will not become training data. Start a new game?",
                parent=self.root,
                default=messagebox.NO,
            )
        ):
            return

        checkpoint_text = self.checkpoint_var.get().strip()
        if not checkpoint_text:
            messagebox.showerror("Checkpoint required", "Choose a neural checkpoint first.")
            return
        checkpoint = Path(checkpoint_text)
        if not checkpoint.is_file():
            messagebox.showerror(
                "Checkpoint not found", f"Checkpoint does not exist:\n{checkpoint}"
            )
            return
        dataset_text = self.dataset_var.get().strip()
        if not dataset_text:
            messagebox.showerror("Dataset path required", "Enter a separate human dataset path.")
            return
        resolved_checkpoint = checkpoint.resolve()
        resolved_dataset = Path(dataset_text).resolve()
        if self._session_confirmed_games:
            if (
                resolved_checkpoint != self._session_source_checkpoint
                or resolved_dataset != self._session_dataset_path
                or self.device_var.get() != self._session_device
            ):
                messagebox.showerror(
                    "Training session settings are locked",
                    (
                        "Confirmed games in this session must use one source checkpoint, device, "
                        "and dataset. Train this session first, then the workbench will reserve "
                        "fresh files for the next cycle."
                    ),
                    parent=self.root,
                )
                return
        elif self._training_session is not None and not self._sync_session_dataset(
            resolved_dataset
        ):
            messagebox.showerror(
                "Could not prepare automatic training",
                self._session_setup_error or "The session configuration could not be updated.",
                parent=self.root,
            )
            return

        self.status_var.set("Loading neural checkpoint...")
        self.root.update_idletasks()
        try:
            base_agent = self._load_agent(checkpoint, self.device_var.get())
            agent = (
                OpeningBookAgent(
                    base_agent,
                    self.opening_book,
                    seed=self._session_game_number,
                )
                if self.opening_book is not None
                else base_agent
            )
            controller = HumanNeuralGame(
                agent,
                human_color=self.human_color_var.get(),
                checkpoint_label=str(checkpoint.resolve()),
                training_enabled=bool(self.training_var.get()),
                session_seed=0,
                pgn_dir=self.pgn_dir,
                dataset_path=resolved_dataset,
                session_id=(
                    f"{self._training_session.session_id}-game-{self._session_game_number + 1:04d}"
                    if self._training_session is not None
                    else None
                ),
            )
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            self.status_var.set("Could not start the game.")
            messagebox.showerror("Could not start game", str(exc), parent=self.root)
            return

        self._worker_token += 1
        self._session_game_number += 1
        self.controller = controller
        self._selected_square = None
        self._legal_destinations.clear()
        self._flipped = controller.human_color == chess.BLACK
        self._finalized_game_id = None
        self.status_var.set(controller.status)
        self.render_board()
        self._render_history()
        self._update_controls()
        if not controller.is_human_turn:
            self._start_ai_turn()

    def clear_selection(self) -> None:
        self._selected_square = None
        self._legal_destinations.clear()
        self.render_board()

    def _on_canvas_resize(self, _event: tk.Event[tk.Misc]) -> None:
        self.render_board()

    def _canvas_square(self, x: float, y: float) -> chess.Square | None:
        if self._board_size <= 0:
            return None
        relative_x = x - self._board_left
        relative_y = y - self._board_top
        if not 0 <= relative_x < self._board_size or not 0 <= relative_y < self._board_size:
            return None
        square_size = self._board_size / 8
        column = int(relative_x // square_size)
        row = int(relative_y // square_size)
        return grid_to_square(row, column, flipped=self._flipped)

    def _on_board_click(self, event: tk.Event[tk.Misc]) -> None:
        controller = self.controller
        if (
            controller is None
            or self._ai_busy
            or self._training_busy
            or controller.game_over
            or not controller.is_human_turn
        ):
            return
        square = self._canvas_square(float(event.x), float(event.y))
        if square is None:
            return

        legal_sources = set(controller.legal_sources())
        if self._selected_square is None:
            if square in legal_sources:
                self._select_square(square)
            return
        if square in self._legal_destinations:
            self._play_selected_move(square)
            return
        if square in legal_sources:
            self._select_square(square)
        else:
            self.clear_selection()

    def _select_square(self, square: chess.Square) -> None:
        if self.controller is None:
            return
        self._selected_square = square
        self._legal_destinations = set(self.controller.legal_destinations(square))
        self.render_board()

    def _promotion_choice(
        self,
        source: chess.Square,
        destination: chess.Square,
    ) -> chess.PieceType | None:
        if self.controller is None:
            return None
        choices = self.controller.promotion_choices(source, destination)
        if not choices:
            return None
        while True:
            response = simpledialog.askstring(
                "Pawn promotion",
                "Promote to queen, rook, bishop, or knight (q/r/b/n):",
                parent=self.root,
                initialvalue="q",
            )
            if response is None:
                return None
            selected = response.strip().lower()
            mapping = {
                "q": chess.QUEEN,
                "queen": chess.QUEEN,
                "r": chess.ROOK,
                "rook": chess.ROOK,
                "b": chess.BISHOP,
                "bishop": chess.BISHOP,
                "n": chess.KNIGHT,
                "knight": chess.KNIGHT,
            }
            piece = mapping.get(selected)
            if piece in choices:
                return piece
            messagebox.showwarning(
                "Invalid promotion",
                "Choose q, r, b, or n.",
                parent=self.root,
            )

    def _play_selected_move(self, destination: chess.Square) -> None:
        controller = self.controller
        source = self._selected_square
        if controller is None or source is None:
            return
        promotion = self._promotion_choice(source, destination)
        if controller.promotion_choices(source, destination) and promotion is None:
            return
        try:
            controller.play_human_move(source, destination, promotion)
        except (RuntimeError, ValueError, TypeError) as exc:
            messagebox.showerror("Move rejected", str(exc), parent=self.root)
            return
        self._selected_square = None
        self._legal_destinations.clear()
        self.status_var.set(controller.status)
        self.render_board()
        self._render_history()
        self._update_controls()
        if controller.game_over:
            self._finalize_game()
        else:
            self._start_ai_turn()

    def _start_ai_turn(self) -> None:
        controller = self.controller
        if (
            controller is None
            or self._training_busy
            or controller.game_over
            or controller.is_human_turn
        ):
            return
        self._worker_token += 1
        token = self._worker_token
        self._ai_busy = True
        self.status_var.set("Neural AI is thinking...")
        self._update_controls()

        def worker() -> None:
            try:
                move = controller.play_ai_turn()
                result = _AiResult(token=token, controller=controller, move=move)
            # This is the boundary of a daemon thread. Always report a
            # failure to Tk's main thread so the workbench cannot remain
            # permanently stuck in the "thinking" state.
            except Exception as exc:
                result = _AiResult(token=token, controller=controller, error=str(exc))
            self._ai_results.put(result)

        threading.Thread(target=worker, name="chess-ai-inference", daemon=True).start()

    def _poll_ai_results(self) -> None:
        if self._closing:
            return
        while True:
            try:
                result = self._ai_results.get_nowait()
            except queue.Empty:
                break
            if result.token != self._worker_token or result.controller is not self.controller:
                continue
            self._ai_busy = False
            if result.error is not None:
                self.status_var.set("Neural move failed.")
                messagebox.showerror("Neural move failed", result.error, parent=self.root)
            else:
                self.status_var.set(result.controller.status)
                self.render_board()
                self._render_history()
                if result.controller.game_over:
                    self._finalize_game()
            self._update_controls()
        self.root.after(50, self._poll_ai_results)

    def _finalize_game(self) -> None:
        controller = self.controller
        if controller is None or not controller.game_over:
            return
        if self._finalized_game_id == controller.game_id:
            return

        saved_message = ""
        try:
            saved_path = controller.save_pgn()
            saved_message = f"\nPGN saved to {saved_path}"
        except (OSError, RuntimeError, ValueError) as exc:
            messagebox.showerror("Could not save PGN", str(exc), parent=self.root)
            self.status_var.set(
                controller.status + "\nPGN was not saved. Use Retry Save to try again."
            )
            self._update_controls()
            return

        self.status_var.set(controller.status + saved_message)
        self.render_board()
        self._render_history()
        self._update_controls()

        if not controller.training_enabled:
            self._finalized_game_id = controller.game_id
            self._update_controls()
            return
        confirmed = messagebox.askyesno(
            "Add this game to training data?",
            (
                f"Result: {controller.result}\n\n"
                "Add only your moves from this completed game to the human training dataset?\n\n"
                "Mistakes and losing moves can be poor policy labels. The default is No."
            ),
            parent=self.root,
            default=messagebox.NO,
        )
        if not confirmed:
            self._finalized_game_id = controller.game_id
            self._update_controls()
            return
        try:
            append_result = controller.append_training_examples(confirm_training=True)
        except (OSError, RuntimeError, ValueError) as exc:
            messagebox.showerror("Could not add training data", str(exc), parent=self.root)
            self.status_var.set(
                controller.status
                + saved_message
                + "\nTraining data was not added. Use Retry Save to try again."
            )
            self._update_controls()
            return
        self._record_confirmed_game(controller, append_result.added_examples)
        self._finalized_game_id = controller.game_id
        self.status_var.set(
            controller.status
            + saved_message
            + f"\nAdded {append_result.added_examples} human examples "
            + f"({append_result.total_examples} total)."
        )
        self._update_controls()

    def _record_confirmed_game(
        self,
        controller: HumanNeuralGame,
        added_examples: int,
    ) -> None:
        source_checkpoint = Path(controller.checkpoint_label).resolve()
        dataset_path = controller.dataset_path.resolve()
        device = (
            str(controller.agent.device)
            if isinstance(controller.agent, NeuralAgent)
            else self.device_var.get()
        )
        if self._session_source_checkpoint is None:
            self._session_source_checkpoint = source_checkpoint
            self._session_dataset_path = dataset_path
            self._session_device = device
            self.checkpoint_var.set(str(source_checkpoint))
            self.dataset_var.set(str(dataset_path))
            self.device_var.set(device)
            self._sync_session_dataset(dataset_path)
        elif (
            source_checkpoint != self._session_source_checkpoint
            or dataset_path != self._session_dataset_path
            or device != self._session_device
        ):
            self._session_setup_error = (
                "A confirmed game used settings that differ from this training session. "
                "Restart the workbench before training these datasets separately."
            )
        self._session_confirmed_games += 1
        self._session_confirmed_examples += added_examples
        self._refresh_session_label()

    def finish_session_and_train(self) -> None:
        if self._ai_busy or self._training_busy:
            return
        if self.controller is not None and (
            not self.controller.game_over or self._finalized_game_id != self.controller.game_id
        ):
            messagebox.showinfo(
                "Finish the current game first",
                "Finish or resign the current game and complete its save decision before training.",
                parent=self.root,
            )
            return
        plan = self._training_session
        source_checkpoint = self._session_source_checkpoint
        dataset_path = self._session_dataset_path
        device = self._session_device
        if self._session_confirmed_games == 0:
            messagebox.showinfo(
                "No confirmed games yet",
                (
                    "Finish at least one game with collection enabled, then confirm adding its "
                    "human moves before training."
                ),
                parent=self.root,
            )
            return
        if plan is None or source_checkpoint is None or dataset_path is None or device is None:
            messagebox.showerror(
                "Automatic training is unavailable",
                self._session_setup_error or "The session training files are incomplete.",
                parent=self.root,
            )
            return
        if not self._sync_session_dataset(dataset_path):
            messagebox.showerror(
                "Could not prepare training",
                self._session_setup_error or "The session configuration could not be updated.",
                parent=self.root,
            )
            return
        plan = self._training_session
        if plan is None:  # pragma: no cover - guarded by _sync_session_dataset
            return

        if not messagebox.askyesno(
            "Finish this session and train?",
            (
                f"This session added {self._session_confirmed_games} confirmed game(s) and "
                f"{self._session_confirmed_examples} human moves.\n\n"
                f"Training uses the cumulative confirmed dataset:\n{dataset_path}\n\n"
                f"Source checkpoint:\n{source_checkpoint}\n\n"
                f"New output:\n{plan.checkpoint_dir}"
            ),
            parent=self.root,
            default=messagebox.NO,
        ):
            return

        self._training_token += 1
        token = self._training_token
        self._training_busy = True
        self._worker_token += 1
        self.controller = None
        self._agent_cache = None
        self._selected_square = None
        self._legal_destinations.clear()
        self._finalized_game_id = None
        self.status_var.set(
            f"Training on {device}... The window will unlock when run {plan.session_id} finishes."
        )
        self.render_board()
        self._render_history()
        self._update_controls()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        def worker() -> None:
            try:
                training_result = self._training_runner(
                    plan,
                    source_checkpoint=source_checkpoint,
                    device=device,
                )
                result = _TrainingUiResult(
                    token=token,
                    plan=plan,
                    result=training_result,
                )
            except Exception as exc:
                result = _TrainingUiResult(token=token, plan=plan, error=str(exc))
            self._training_results.put(result)

        threading.Thread(target=worker, name="chess-ai-training", daemon=True).start()

    def _poll_training_results(self) -> None:
        if self._closing:
            return
        while True:
            try:
                queued = self._training_results.get_nowait()
            except queue.Empty:
                break
            if queued.token != self._training_token or queued.plan != self._training_session:
                continue
            self._training_busy = False
            if queued.error is not None or queued.result is None:
                detail = queued.error or "Training ended without a result."
                failed_session = queued.plan.session_id
                dataset_path = self._session_dataset_path
                if dataset_path is not None and self._reserve_training_plan(dataset_path):
                    self.status_var.set(
                        f"Training run {failed_session} failed. Confirmed data is safe and a "
                        "fresh run was reserved; use I'm Done — Train AI to retry."
                    )
                else:
                    self.status_var.set(
                        f"Training run {failed_session} failed. Confirmed data is safe, but "
                        "fresh retry files could not be reserved. Restart the workbench."
                    )
                messagebox.showerror("Automatic training failed", detail, parent=self.root)
            else:
                self._complete_training_cycle(queued.result)
            self._update_controls()
        self.root.after(50, self._poll_training_results)

    def _reserve_training_plan(self, dataset_path: Path) -> bool:
        try:
            self._training_session = create_session_training_plan(dataset_path=dataset_path)
        except (OSError, RuntimeError, ValueError) as exc:
            self._training_session = None
            self._session_setup_error = str(exc)
            self._refresh_session_label()
            return False
        self._session_setup_error = None
        self._refresh_session_label()
        return True

    def _complete_training_cycle(self, result: SessionTrainingResult) -> None:
        completed_plan = self._training_session
        dataset_path = self._session_dataset_path or Path(self.dataset_var.get()).resolve()
        best_checkpoint = result.best_checkpoint.resolve()
        self.checkpoint_var.set(str(best_checkpoint))
        self.controller = None
        self._agent_cache = None
        self._session_confirmed_games = 0
        self._session_confirmed_examples = 0
        self._session_game_number = 0
        self._session_source_checkpoint = None
        self._session_dataset_path = None
        self._session_device = None
        self._reserve_training_plan(dataset_path)
        output = (
            completed_plan.checkpoint_dir if completed_plan is not None else best_checkpoint.parent
        )
        self.status_var.set(
            f"Training complete. Selected {best_checkpoint}.\nSaved this run under {output}. "
            "Start New Game to play the updated AI."
        )
        self.render_board()
        self._render_history()

    def undo_turn(self) -> None:
        controller = self.controller
        if controller is None or self._ai_busy or self._training_busy or controller.game_over:
            return
        try:
            controller.undo_turn()
        except (RuntimeError, ValueError) as exc:
            messagebox.showinfo("Nothing to undo", str(exc), parent=self.root)
            return
        self._selected_square = None
        self._legal_destinations.clear()
        self.status_var.set(controller.status)
        self.render_board()
        self._render_history()
        self._update_controls()

    def resign(self) -> None:
        controller = self.controller
        if controller is None or self._ai_busy or self._training_busy:
            return
        if controller.game_over:
            self._finalize_game()
            return
        if not messagebox.askyesno(
            "Resign game?",
            "Record this game as a human resignation?",
            parent=self.root,
            default=messagebox.NO,
        ):
            return
        controller.resign_human()
        self.status_var.set(controller.status)
        self._finalize_game()

    def flip_board(self) -> None:
        self._flipped = not self._flipped
        self.render_board()

    def _update_controls(self) -> None:
        active_game = self.controller is not None and not self.controller.game_over
        busy = self._ai_busy or self._training_busy
        session_locked = self._session_confirmed_games > 0
        needs_save_retry = (
            self.controller is not None
            and self.controller.game_over
            and self._finalized_game_id != self.controller.game_id
        )
        fixed_state = tk.DISABLED if busy or session_locked else tk.NORMAL
        self.checkpoint_entry.configure(state=fixed_state)
        self.browse_button.configure(state=fixed_state)
        self.dataset_entry.configure(state=fixed_state)
        self.device_combo.configure(state="disabled" if busy or session_locked else "readonly")
        self.color_combo.configure(state="disabled" if busy else "readonly")
        self.training_check.configure(state=tk.DISABLED if busy else tk.NORMAL)
        self.new_button.configure(state=tk.DISABLED if busy else tk.NORMAL)
        move_state = tk.NORMAL if active_game and not busy else tk.DISABLED
        self.undo_button.configure(state=move_state)
        self.resign_button.configure(
            text="Retry Save" if needs_save_retry else "Resign",
            state=tk.NORMAL if needs_save_retry and not busy else move_state,
        )
        can_train = (
            self._session_confirmed_games > 0
            and self._training_session is not None
            and (
                self.controller is None
                or (
                    self.controller.game_over and self._finalized_game_id == self.controller.game_id
                )
            )
            and not busy
        )
        self.done_button.configure(
            text="Training AI..." if self._training_busy else "I'm Done — Train AI",
            state=tk.NORMAL if can_train else tk.DISABLED,
        )
        self.flip_button.configure(state=tk.DISABLED if self._training_busy else tk.NORMAL)

    def _render_history(self) -> None:
        self.history_text.configure(state=tk.NORMAL)
        self.history_text.delete("1.0", tk.END)
        controller = self.controller
        if controller is not None:
            replay = chess.Board()
            lines: list[str] = []
            for ply, move in enumerate(controller.move_history):
                san = replay.san(move)
                move_number = ply // 2 + 1
                prefix = f"{move_number}." if ply % 2 == 0 else f"{move_number}..."
                lines.append(f"{prefix} {san}")
                replay.push(move)
            self.history_text.insert("1.0", "\n".join(lines) or "No moves yet.")
            self.history_text.see(tk.END)
        self.history_text.configure(state=tk.DISABLED)

    def render_board(self) -> None:
        canvas = getattr(self, "canvas", None)
        if canvas is None:
            return
        canvas.delete("all")
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        self._board_size = float(max(8, min(width, height)))
        self._board_left = (width - self._board_size) / 2
        self._board_top = (height - self._board_size) / 2
        square_size = self._board_size / 8

        controller = self.controller
        board = controller.board if controller is not None else chess.Board()
        last_move = board.move_stack[-1] if board.move_stack else None
        check_square = board.king(board.turn) if board.is_check() else None

        for row in range(8):
            for column in range(8):
                square = grid_to_square(row, column, flipped=self._flipped)
                x0 = self._board_left + column * square_size
                y0 = self._board_top + row * square_size
                x1 = x0 + square_size
                y1 = y0 + square_size
                color = LIGHT_SQUARE if (row + column) % 2 == 0 else DARK_SQUARE
                if last_move is not None and square in (last_move.from_square, last_move.to_square):
                    color = LAST_MOVE_SQUARE
                if square == check_square:
                    color = CHECK_SQUARE
                if square == self._selected_square:
                    color = SELECTED_SQUARE
                canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=color)

                piece = board.piece_at(square)
                if piece is not None:
                    canvas.create_text(
                        (x0 + x1) / 2,
                        (y0 + y1) / 2,
                        text=PIECE_SYMBOLS[(piece.color, piece.piece_type)],
                        font=("Segoe UI Symbol", max(16, int(square_size * 0.62))),
                        fill="#111111",
                    )
                if square in self._legal_destinations:
                    radius = square_size * (0.16 if piece is None else 0.38)
                    canvas.create_oval(
                        (x0 + x1) / 2 - radius,
                        (y0 + y1) / 2 - radius,
                        (x0 + x1) / 2 + radius,
                        (y0 + y1) / 2 + radius,
                        outline=LEGAL_MARKER,
                        width=max(3, int(square_size * 0.06)),
                        fill=LEGAL_MARKER if piece is None else "",
                    )

                if column == 0:
                    rank_label = chess.square_name(square)[1]
                    canvas.create_text(
                        x0 + 5,
                        y0 + 4,
                        text=rank_label,
                        anchor=tk.NW,
                        font=("Segoe UI Semibold", max(8, int(square_size * 0.12))),
                    )
                if row == 7:
                    file_label = chess.square_name(square)[0]
                    canvas.create_text(
                        x1 - 5,
                        y1 - 3,
                        text=file_label,
                        anchor=tk.SE,
                        font=("Segoe UI Semibold", max(8, int(square_size * 0.12))),
                    )

    def close(self) -> None:
        if self._training_busy:
            messagebox.showwarning(
                "Training is still running",
                "Wait for automatic training to finish before closing this window.",
                parent=self.root,
            )
            return
        self._closing = True
        self._worker_token += 1
        self._training_token += 1
        self.root.destroy()


def launch_gui(
    checkpoint: str | Path | None = None,
    human_color: str = "white",
    device: str = "auto",
    opening_book: str | Path | None = None,
    search_simulations: int = 0,
    c_puct: float = 1.5,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    pgn_dir: str | Path = DEFAULT_PGN_DIR,
    training_enabled: bool = False,
) -> None:
    """Open the local workbench and block until its window closes."""

    root = tk.Tk()
    NeuralChessApp(
        root,
        checkpoint=checkpoint,
        human_color=human_color,
        device=device,
        opening_book=opening_book,
        search_simulations=search_simulations,
        c_puct=c_puct,
        dataset_path=dataset_path,
        pgn_dir=pgn_dir,
        training_enabled=training_enabled,
    )
    root.mainloop()
