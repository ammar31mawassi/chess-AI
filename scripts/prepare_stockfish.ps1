# Update the separately licensed official checkout and optionally build it.
[CmdletBinding()]
param([switch]$Build, [string]$MsysRoot = 'C:\msys64')
$ErrorActionPreference = 'Stop'
$repo = Join-Path $PSScriptRoot '..\references\github\stockfish'
$repo = [IO.Path]::GetFullPath($repo)
function Invoke-GitChecked {
    param([string[]]$GitArgs)
    & git @GitArgs
    if ($LASTEXITCODE -ne 0) { throw "Git failed: $GitArgs" }
}
if (-not (Test-Path -LiteralPath $repo)) {
    Invoke-GitChecked -GitArgs @('clone', '--depth', '1', '--branch', 'master', 'https://github.com/official-stockfish/Stockfish.git', $repo)
} else {
    $dirty = & git -C $repo status --porcelain
    if ($LASTEXITCODE -ne 0 -or $dirty) { throw 'Stockfish checkout must be a clean Git repository.' }
    $branch = & git -C $repo branch --show-current
    if ($branch -ne 'master') { throw 'Stockfish checkout must already be on master; no branch is changed automatically.' }
    $remote = & git -C $repo remote get-url origin
    if ($remote -notin @('https://github.com/official-stockfish/Stockfish.git', 'https://github.com/official-stockfish/Stockfish')) {
        throw 'Stockfish origin must be the official repository.'
    }
    # Do not replace the shallow boundary with --depth 1 on an existing checkout.
    Invoke-GitChecked -GitArgs @('-C', $repo, 'fetch', '--deepen', '500', 'origin', 'master')
    & git -C $repo merge-base HEAD origin/master *> $null
    if ($LASTEXITCODE -ne 0) {
        $shallow = & git -C $repo rev-parse --is-shallow-repository
        if ($shallow -eq 'true') { Invoke-GitChecked -GitArgs @('-C', $repo, 'fetch', '--unshallow', 'origin', 'master') }
    }
    Invoke-GitChecked -GitArgs @('-C', $repo, 'merge', '--ff-only', 'origin/master')
}
$commit = & git -C $repo rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Cannot read Stockfish commit.' }
Write-Host "Stockfish source commit: $commit"
if ($Build) {
    $bash = Join-Path $MsysRoot 'usr\bin\bash.exe'
    $compiler = Join-Path $MsysRoot 'mingw64\bin\g++.exe'
    if (-not (Test-Path -LiteralPath $bash) -or -not (Test-Path -LiteralPath $compiler)) {
        throw 'Install MSYS2 make and mingw-w64-x86_64-gcc first; see docs/stockfish_teacher.md.'
    }
    # bash receives the source path as a positional argument, never shell-interpolated text.
    $env:MSYSTEM = 'MINGW64'
    & $bash -lc 'set -e; cd "$(cygpath -u "$1")"; export PATH=/mingw64/bin:/usr/bin:$PATH; make -j2 build ARCH=x86-64 COMP=mingw' stockfish-build (Join-Path $repo 'src')
    if ($LASTEXITCODE -ne 0) { throw 'Stockfish build failed.' }
    $exe = Join-Path $repo 'src\stockfish.exe'
    if (-not (Test-Path -LiteralPath $exe)) { throw 'Build did not create stockfish.exe.' }
    $buildDir = Join-Path (Split-Path $repo) "stockfish-build\$commit"
    New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
    $builtExe = Join-Path $buildDir 'stockfish.exe'
    $digest = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLowerInvariant()
    if (Test-Path -LiteralPath $builtExe) {
        if ((Get-FileHash -LiteralPath $builtExe -Algorithm SHA256).Hash.ToLowerInvariant() -ne $digest) {
            throw 'An immutable build exists with a different digest; choose a separate build location.'
        }
    } else { Copy-Item -LiteralPath $exe -Destination $builtExe }
    $provenance = @{ source_commit = $commit; executable_sha256 = $digest; license = 'GPL-3.0'; source_repository = 'https://github.com/official-stockfish/Stockfish' }
    $provenance | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $buildDir 'stockfish.provenance.json') -Encoding utf8NoBOM
    Copy-Item -LiteralPath (Join-Path $repo 'Copying.txt') -Destination (Join-Path $buildDir 'Copying.txt') -Force
    "https://github.com/official-stockfish/Stockfish/tree/$commit" | Set-Content -LiteralPath (Join-Path $buildDir 'SOURCE.txt') -Encoding utf8NoBOM
    Write-Host "Built immutable executable: $builtExe"
}
