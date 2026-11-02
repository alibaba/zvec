$ErrorActionPreference = 'Stop'
if ($env:GITHUB_ACTIONS -ne 'true') { throw 'This benchmark is only for GitHub Actions' }
if ($env:PERF_JOBS -notmatch '^[1-9][0-9]*$') { throw 'PERF_JOBS must be a positive integer' }
if ($env:VisualStudioVersion -ne $env:PERF_VS_VERSION) { throw 'Unexpected Visual Studio version' }

$harness = Join-Path $env:GITHUB_WORKSPACE 'harness'
$seed = Join-Path $env:GITHUB_WORKSPACE 'source'
$results = Join-Path $env:RUNNER_TEMP 'zvec-build-perf-results'
New-Item -ItemType Directory -Path $results -Force | Out-Null
$sourceSha = & git -C $seed rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Could not read source revision' }
$machine = [ordered]@{
    source_sha = $sourceSha
    harness_sha = $env:GITHUB_SHA
    platform = $env:PERF_PLATFORM
    order = $env:PERF_ORDER
    jobs = [int]$env:PERF_JOBS
    requested_defender_performance_mode = ($env:PERF_DEFENDER_PERFORMANCE_MODE -eq 'true')
    image_version = $env:ImageVersion
    runner_arch = $env:RUNNER_ARCH
    cpu = @(Get-CimInstance Win32_Processor | Select-Object Name, Manufacturer, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed)
    system = Get-CimInstance Win32_ComputerSystem | Select-Object Model, TotalPhysicalMemory, NumberOfLogicalProcessors
    os = Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber
    toolchain = @{ vs = $env:VisualStudioVersion; vc = $env:VCToolsVersion; sdk = $env:WindowsSDKVersion; compiler = (Get-Command cl.exe).Source; compiler_version = (Get-Item (Get-Command cl.exe).Source).VersionInfo.FileVersion }
    tools = @(& python -m pip freeze)
}
try { $machine.defender = Get-MpComputerStatus | Select-Object AntivirusEnabled, RealTimeProtectionEnabled } catch { $machine.defender = "$($_.Exception.Message)" }
try { $machine.defender_preferences = Get-MpPreference | Select-Object PerformanceModeStatus } catch { $machine.defender_preferences = "$($_.Exception.Message)" }
$machine | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath "$results/machine.json" -Encoding utf8

function Get-DefenderSnapshot {
    try {
        $status = Get-MpComputerStatus
        $preference = Get-MpPreference
        return [pscustomobject]@{
            AntivirusEnabled = $status.AntivirusEnabled
            RealTimeProtectionEnabled = $status.RealTimeProtectionEnabled
            PerformanceModeStatus = [int]$preference.PerformanceModeStatus
        }
    } catch { return [pscustomobject]@{ error = $_.Exception.Message } }
}

function Assert-DefenderExperiment($Snapshot) {
    if ($env:PERF_DEFENDER_PERFORMANCE_MODE -eq 'true' -and
        (-not $Snapshot.AntivirusEnabled -or -not $Snapshot.RealTimeProtectionEnabled -or
         $Snapshot.PerformanceModeStatus -ne 0)) {
        throw 'Defender experiment requires real-time protection ON and performance mode Enabled (0)'
    }
}

function Save-BuildSnapshot($BuildDir, $Destination) {
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    $dirs = @{
        root = $BuildDir
        arrow = Join-Path $BuildDir 'thirdparty/arrow/arrow/src/ARROW.BUILD-build'
        lz4 = Join-Path $BuildDir 'thirdparty/lz4/lz4/src/Lz4.BUILD-build'
    }
    foreach ($name in $dirs.Keys) {
        foreach ($file in @('.ninja_log', 'CMakeCache.txt', 'compile_commands.json', 'build.ninja', 'CMakeFiles/rules.ninja')) {
            $path = Join-Path $dirs[$name] $file
            if (Test-Path -LiteralPath $path) { Copy-Item -LiteralPath $path -Destination "$Destination/$name-$([IO.Path]::GetFileName($file))" }
        }
    }
}

function Invoke-MeasuredCommand($Executable, [string[]]$CommandArgs, $Log) {
    $timer = [Diagnostics.Stopwatch]::StartNew()
    # Windows PowerShell represents native stderr as ErrorRecord objects. pip
    # writes routine build progress there; use its exit code to detect failure.
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Executable @CommandArgs 2>&1 | Tee-Object -FilePath $Log | ForEach-Object { Write-Host $_ }
        $exitCode = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previousPreference }
    $timer.Stop()
    if ($exitCode -ne 0) { throw "$Executable exited with $exitCode; see $Log" }
    return $timer.Elapsed.TotalSeconds
}

$cases = if ($env:PERF_ORDER -eq 'devdrive-first') { @('devdrive', 'ntfs') } else { @('ntfs', 'devdrive') }
$failures = @()
foreach ($case in $cases) {
    $caseRoot = if ($case -eq 'ntfs') { Join-Path $env:RUNNER_TEMP 'zvec-perf-ntfs' } else { Join-Path $env:PERF_DEV_ROOT 'zvec-perf' }
    $source = Join-Path $caseRoot 'source'
    $build = Join-Path $source 'build'
    $out = Join-Path $results $case
    New-Item -ItemType Directory -Path $source, $out, "$caseRoot/tmp", "$caseRoot/wheels" -Force | Out-Null
    if (Test-Path -LiteralPath "$source/CMakeLists.txt") { throw "Refusing to reuse a source/build tree: $source" }
    # Keep each submodule's Git metadata: apply_patch_once invokes git apply
    # from nested source directories and relies on the submodule root.
    & robocopy.exe $seed $source /E /NFL /NDL /NJH /NJS /NP
    if ($LASTEXITCODE -ge 8) { throw "Source copy failed for $case" }
    $copiedSha = & git -C $source rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $copiedSha -ne $sourceSha) { throw 'Source copy revision mismatch' }
    $moduleRoots = @(& git -C $source submodule foreach --quiet --recursive 'git rev-parse --show-toplevel')
    if ($LASTEXITCODE -ne 0 -or -not $moduleRoots.Count) { throw 'Could not validate copied submodules' }
    foreach ($moduleRoot in $moduleRoots) {
        if (-not $moduleRoot.Replace('\', '/').StartsWith($source.Replace('\', '/') + '/', [StringComparison]::OrdinalIgnoreCase)) {
            throw "Submodule still points outside the copied source: $moduleRoot"
        }
    }
    $saved = @{}
    $envValues = @{
        CMAKE_GENERATOR = 'Ninja'; CMAKE_BUILD_PARALLEL_LEVEL = $env:PERF_JOBS
        CMAKE_EXPORT_COMPILE_COMMANDS = 'ON'; NINJA_STATUS = '[%f/%t %r running %es] '
        SCCACHE_DISABLE = '1'; CCACHE_DISABLE = '1'; PIP_NO_CACHE_DIR = '1'
        SETUPTOOLS_SCM_PRETEND_VERSION = '0.8.0.dev0'; TEMP = "$caseRoot/tmp"; TMP = "$caseRoot/tmp"
    }
    foreach ($key in $envValues.Keys) {
        $saved[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
        [Environment]::SetEnvironmentVariable($key, $envValues[$key], 'Process')
    }
    $phaseFile = Join-Path $out 'phase.txt'
    $caseDrive = [IO.Path]::GetPathRoot($caseRoot).TrimEnd('\')
    $sampler = Start-Job -ArgumentList $out, $phaseFile, $caseDrive -ScriptBlock {
        param($Destination, $PhaseFile, $Drive)
        $ErrorActionPreference = 'Stop'
        while ($true) {
            try {
                $cpu = Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter "Name='_Total'"
                $mem = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
                $disk = Get-CimInstance Win32_PerfFormattedData_PerfDisk_LogicalDisk -Filter "Name='$Drive'"
                $procs = @(Get-Process cl, c1xx, link, ninja, MSBuild -ErrorAction SilentlyContinue)
                [pscustomobject]@{
                    utc = [DateTime]::UtcNow.ToString('o'); phase = (Get-Content -LiteralPath $PhaseFile -ErrorAction SilentlyContinue)
                    cpu_percent = $cpu.PercentProcessorTime; available_mb = $mem.AvailableMBytes
                    page_reads_sec = $mem.PageReadsPersec; disk_bytes_sec = $disk.DiskBytesPersec
                    cl = @($procs | Where-Object ProcessName -eq 'cl').Count
                    c1xx = @($procs | Where-Object ProcessName -eq 'c1xx').Count
                    link = @($procs | Where-Object ProcessName -eq 'link').Count
                    ninja = @($procs | Where-Object ProcessName -eq 'ninja').Count
                } | Export-Csv -LiteralPath "$Destination/resources.csv" -NoTypeInformation -Append
            } catch { $_ | Out-String | Add-Content -LiteralPath "$Destination/sampler-errors.txt" }
            Start-Sleep -Seconds 5
        }
    }
    $record = [ordered]@{ case = $case; source_sha = $sourceSha; source = $source; jobs = [int]$env:PERF_JOBS; compiler_cache = 'disabled'; status = 'running' }
    try {
        $record.defender_before = Get-DefenderSnapshot
        Assert-DefenderExperiment $record.defender_before
        $record.volume = Get-Volume -DriveLetter $caseDrive.Substring(0, 1) | Select-Object DriveLetter, FileSystemType, Size, SizeRemaining
        $record.native_dev_drive = ($case -eq 'devdrive') -and ($env:PERF_NATIVE_DEV_DRIVE -eq 'True')
        if ($case -eq 'devdrive' -and $env:PERF_NATIVE_DEV_DRIVE -eq 'True') { & fsutil.exe devdrv query $env:PERF_DEV_ROOT | Out-File "$out/devdrive.txt" }
        'pip-wheel' | Set-Content -LiteralPath $phaseFile
        Push-Location -LiteralPath $source
        try {
            $pipArgs = @('-m', 'pip', 'wheel', '-v', '.', '--no-deps', '--no-build-isolation', '--wheel-dir', "$caseRoot/wheels",
                '--config-settings=cmake.define.BUILD_TOOLS=ON', '--config-settings=cmake.define.ENABLE_WERROR=ON',
                '--config-settings=cmake.define.OVERRIDE_GIT_DESCRIBE=v0.8.0',
                '--config-settings=cmake.define.CMAKE_C_COMPILER_LAUNCHER=', '--config-settings=cmake.define.CMAKE_CXX_COMPILER_LAUNCHER=',
                "--config-settings=cmake.define.CMAKE_PROJECT_INCLUDE=$harness/.github/cmake/perf-test-build.cmake")
            $record.pip_seconds = Invoke-MeasuredCommand 'python' $pipArgs "$out/pip.log"
            Save-BuildSnapshot $build "$out/main"
            'test-compile' | Set-Content -LiteralPath $phaseFile
            $record.test_compile_seconds = Invoke-MeasuredCommand 'cmake' @('--build', $build, '--target', 'zvec_ci_test_build', '--parallel', $env:PERF_JOBS) "$out/test-compile.log"
            Save-BuildSnapshot $build "$out/tests"
            & ctest --test-dir $build --show-only=json-v1 | Set-Content -LiteralPath "$out/tests.json" -Encoding utf8
            if ($LASTEXITCODE -ne 0) { throw 'Could not list the compiled tests' }
            $record.defender_after = Get-DefenderSnapshot
            Assert-DefenderExperiment $record.defender_after
            $record.status = 'success'
        } finally { Pop-Location }
    } catch {
        $record.status = 'failure'; $record.error = $_.Exception.Message
        $failures += "$case`: $($_.Exception.Message)"
        Write-Warning $record.error
    } finally {
        if (-not $record.Contains('defender_after')) { $record.defender_after = Get-DefenderSnapshot }
        Stop-Job $sampler
        Receive-Job $sampler | Out-File "$out/sampler-output.txt"
        Remove-Job $sampler
        $record | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath "$out/measurement.json" -Encoding utf8
        foreach ($key in $saved.Keys) { [Environment]::SetEnvironmentVariable($key, $saved[$key], 'Process') }
    }
    & python "$harness/.github/scripts/analyze-windows-build.py" $out
    if ($LASTEXITCODE -ne 0) { $failures += "Could not analyze $case" }
}
if ($failures.Count) { throw ($failures -join "`n") }
