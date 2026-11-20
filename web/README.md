# Play the champion on Vercel

The production workbench is available at [Neural Chess](https://neural-chess-champion.vercel.app).

The public workbench uses the desktop board palette and licensed piece art. The API rebuilds
each position from its UCI move history with python-chess, including repetition history.
It runs the registered champion through ONNX Runtime on CPU with the same PUCT implementation
and a five-second soft search deadline. Actual move latency also includes network and startup
time. The browser keeps the game history for the current tab.

From the repository root, export the registered champion and verify its ONNX outputs against
PyTorch. The export stages only the inference model, search and encoding modules, and piece
art in the web directory.

```powershell
Set-Location 'C:\Users\ammar\OneDrive\Desktop\chess-AI'
& .\.venv\Scripts\python.exe -m pip install onnx==1.23.2 onnxruntime==1.32.0
& .\.venv\Scripts\python.exe scripts/export_web_champion.py
```

Deploy the staged web directory from an authenticated Vercel CLI. For the first deployment,
choose the intended Vercel team and the `neural-chess-champion` project at the prompts.

```powershell
npx --yes vercel@latest login
npx --yes vercel@latest deploy web --prod
```

The generated ONNX model and staged runtime are excluded from Git. The Vercel Python
function receives the weights as a runtime file. It serves board state and moves through
`/api/play`, and Vercel serves the browser interface from `public`.
