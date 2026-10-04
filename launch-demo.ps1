$ErrorActionPreference = 'Stop'
$env:GAUSSIAN_STUDIO_BUILT = '1'
$studioRoot = $PSScriptRoot
$studioElectron = Join-Path $studioRoot 'node_modules/electron/dist/electron.exe'
if (-not (Test-Path -LiteralPath $studioElectron)) { throw 'Install locked dependencies and Electron first.' }
if (-not (Test-Path -LiteralPath (Join-Path $studioRoot 'dist/index.html'))) { throw 'Build the frontend first.' }
Start-Process -FilePath $studioElectron -ArgumentList @("`"$studioRoot`"") -WorkingDirectory $studioRoot
