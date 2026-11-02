$ErrorActionPreference = 'Stop'

# Official native binaries; verify the release asset before putting it on PATH.
$version = '4.14.2'
$archives = @{
    'AMD64' = @{
        Name = 'x86_64'
        SHA256 = '6993ce645e53192cec592ee23186200b574ac4c59097ecde8d9cfd681b47ae84'
    }
    'ARM64' = @{
        Name = 'aarch64'
        SHA256 = '47ea5a0ed661bea53962d8fb8ee9a1769092d979a6f7c73d3081499b3305b2ab'
    }
}
$archive = $archives[$env:PROCESSOR_ARCHITECTURE]
if (-not $archive) {
    throw "Unsupported ccache host architecture: $env:PROCESSOR_ARCHITECTURE"
}

$name = "ccache-$version-windows-$($archive.Name)"
$zip = Join-Path $env:RUNNER_TEMP "$name.zip"
$destination = Join-Path $env:RUNNER_TEMP 'ccache-tools'
Invoke-WebRequest "https://github.com/ccache/ccache/releases/download/v$version/$name.zip" -OutFile $zip
if ((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash -ne $archive.SHA256) {
    throw "SHA256 mismatch for $name.zip"
}
Expand-Archive -LiteralPath $zip -DestinationPath $destination -Force
$ccache = Get-ChildItem -LiteralPath $destination -Filter ccache.exe -Recurse | Select-Object -First 1
if (-not $ccache) { throw 'ccache.exe missing from release archive' }
$ccache.DirectoryName >> $env:GITHUB_PATH
"CCACHE_DIR=$(Join-Path $env:RUNNER_TEMP 'ccache')" >> $env:GITHUB_ENV
& $ccache.FullName --version
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
