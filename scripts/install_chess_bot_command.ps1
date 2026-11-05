$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$wrapper = Join-Path $projectRoot 'scripts\chess-bot.ps1'
$documents = [Environment]::GetFolderPath('MyDocuments')
$profilePaths = @(
    (Join-Path $documents 'PowerShell\profile.ps1'),
    (Join-Path $documents 'WindowsPowerShell\profile.ps1')
)
$line = ". '" + $wrapper.Replace("'", "''") + "'"
foreach ($profilePath in $profilePaths) {
    $existing = if (Test-Path -LiteralPath $profilePath) {
        [IO.File]::ReadAllText($profilePath)
    } else { '' }
    if (-not $existing.Contains($line)) {
        [IO.Directory]::CreateDirectory((Split-Path -Parent $profilePath)) | Out-Null
        if (Test-Path -LiteralPath $profilePath) {
            Copy-Item -LiteralPath $profilePath -Destination ($profilePath + '.chess-bot.bak')
        }
        [IO.File]::WriteAllText($profilePath, $existing + "`r`n# Chess AI terminal command`r`n" + $line + "`r`n", [Text.UTF8Encoding]::new($true))
    }
}
. $wrapper
Write-Output 'Installed chess-bot for Windows PowerShell and PowerShell. No games started.'
