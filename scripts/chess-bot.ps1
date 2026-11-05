# Dot-source this file to expose chess-bot in the current PowerShell session.
function global:chess-bot {
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    & 'C:\Users\ammar\OneDrive\Desktop\chess-AI\.venv\Scripts\python.exe' -m chess_ai.online.matches @args
    if ($LASTEXITCODE -ne 0) { throw "chess-bot exited with code $LASTEXITCODE" }
}
