const byId = (id) => document.getElementById(id);
const board = byId("board");
const status = byId("status");
const letters = "abcdefgh";
const names = { p: "pawn", n: "knight", b: "bishop", r: "rook", q: "queen", k: "king" };
let side = "white";
let flipped = false;
let state;
let selected;
let busy = false;
let retryAction;

function position(fen) {
  const squares = new Map();
  fen.split(" ")[0].split("/").forEach((rank, i) => {
    let file = 0;
    for (const char of rank) {
      if (/^[1-8]$/.test(char)) file += Number(char);
      else squares.set(letters[file++] + (8 - i), char);
    }
  });
  return squares;
}

function render() {
  if (!state) return;
  const focus = document.activeElement?.dataset.square;
  const pieces = position(state.fen);
  const lastMove = state.history.at(-1);
  const targets = selected ? state.legal.filter(move => move.startsWith(selected)) : [];
  byId("top-player").textContent = flipped ? (side === "white" ? "You · White" : "Scratch · White") : (side === "black" ? "You · Black" : "Scratch · Black");
  byId("bottom-player").textContent = flipped ? (side === "black" ? "You · Black" : "Scratch · Black") : (side === "white" ? "You · White" : "Scratch · White");
  const fragment = document.createDocumentFragment();
  for (let index = 0; index < 64; index++) {
    const row = Math.floor(index / 8), col = index % 8;
    const file = letters[flipped ? 7 - col : col], rank = flipped ? 1 + row : 8 - row;
    const square = file + rank;
    const piece = pieces.get(square);
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.square = square;
    const color = piece === piece?.toUpperCase() ? "white" : "black";
    button.className = "square " + ((row + col) % 2 ? "dark" : "light");
    button.setAttribute("aria-label", piece ? `${square}, ${color} ${names[piece.toLowerCase()]}` : square);
    button.setAttribute("aria-pressed", String(square === selected));
    if (lastMove && [lastMove.slice(0, 2), lastMove.slice(2, 4)].includes(square)) button.classList.add("last");
    if (square === state.check) button.classList.add("check");
    if (square === selected) button.classList.add("selected");
    if (targets.some(move => move.slice(2, 4) === square)) button.classList.add(piece ? "capture" : "target");
    if (piece) {
      const img = document.createElement("img");
      img.src = `/pieces/${color[0]}${piece.toUpperCase()}.png`;
      img.alt = "";
      img.draggable = false;
      button.append(img);
    }
    for (const [show, value, kind] of [[col === 0, rank, "rank"], [row === 7, file, "file"]]) {
      if (!show) continue;
      const label = document.createElement("span");
      label.className = kind;
      label.textContent = value;
      label.setAttribute("aria-hidden", "true");
      button.append(label);
    }
    button.addEventListener("click", () => selectSquare(square));
    fragment.append(button);
  }
  board.replaceChildren(fragment);
  if (focus) board.querySelector(`[data-square="${focus}"]`)?.focus({ preventScroll: true });
  const rows = [];
  for (let i = 0; i < state.san.length; i += 2) {
    const row = document.createElement("tr");
    for (const value of [i / 2 + 1, state.san[i], state.san[i + 1] ?? ""]) {
      row.insertCell().textContent = value;
    }
    rows.push(row);
  }
  byId("moves").replaceChildren(...rows);
  byId("empty").hidden = rows.length > 0;
  byId("history").scrollTop = byId("history").scrollHeight;
}

function describe() {
  if (state.result === "1/2-1/2") return `Draw by ${state.termination}.`;
  if (state.result) {
    const playerWon = state.result === (side === "white" ? "1-0" : "0-1");
    return playerWon ? "You win by checkmate." : "Scratch wins by checkmate.";
  }
  return state.turn === side ? (state.check ? "Your turn · Check" : "Your turn") : "Scratch to move";
}

async function request(payload) {
  const response = await fetch("/api/play", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(25_000),
  });
  if (!response.headers.get("Content-Type")?.includes("application/json")) throw new Error("Game service unavailable.");
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Game service unavailable.");
  state = data;
  render();
}

async function run(action, message) {
  if (busy) return;
  busy = true;
  retryAction = null;
  byId("retry").hidden = true;
  byId("new-game").disabled = true;
  byId("side").disabled = true;
  status.textContent = message;
  try {
    await action();
    status.textContent = describe();
  } catch (error) {
    status.textContent = error.message;
    retryAction = () => run(action, message);
    byId("retry").hidden = false;
  } finally {
    busy = false;
    byId("new-game").disabled = false;
    byId("side").disabled = false;
  }
}

async function reply() {
  if (!state.result && state.turn !== side) {
    status.textContent = "Scratch is thinking…";
    await request({ side, history: state.history, reply: true });
  }
}

function submit(move) {
  const history = state.history;
  let accepted = false;
  return run(async () => {
    if (!accepted) {
      selected = undefined;
      await request({ side, history, move });
      accepted = true;
    }
    await reply();
  }, "Playing move…");
}

function selectSquare(square) {
  if (busy || !state || state.result || state.turn !== side) return;
  const matches = selected ? state.legal.filter(move => move.startsWith(selected + square)) : [];
  if (matches.length === 1) return submit(matches[0]);
  if (matches.length > 1) {
    const dialog = byId("promotion");
    dialog.returnValue = "";
    dialog.addEventListener("close", () => {
      const move = matches.find(item => item[4] === dialog.returnValue);
      if (move) submit(move);
    }, { once: true });
    dialog.showModal();
    return;
  }
  selected = selected !== square && state.legal.some(move => move.startsWith(square)) ? square : undefined;
  render();
}

function start() {
  side = byId("side").value;
  flipped = side === "black";
  selected = undefined;
  state = undefined;
  board.replaceChildren();
  byId("moves").replaceChildren();
  byId("empty").hidden = false;
  run(async () => {
    if (!state) await request({ side, history: [] });
    await reply();
  }, "Setting up the board…");
}

byId("new-game").addEventListener("click", start);
byId("flip").addEventListener("click", () => { flipped = !flipped; render(); });
byId("retry").addEventListener("click", () => retryAction?.());
start();
