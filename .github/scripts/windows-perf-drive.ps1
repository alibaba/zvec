param([ValidateSet('Create', 'Detach')][string]$Action)
$ErrorActionPreference = 'Stop'

# Only attach/format this newly created file on an ephemeral Actions runner.
# Never select or reformat an existing runner disk.
if ($env:GITHUB_ACTIONS -ne 'true') { throw 'This helper is only for GitHub Actions' }
$vhdPath = Join-Path $env:RUNNER_TEMP 'zvec-build-perf.vhdx'
$scriptPath = Join-Path $env:RUNNER_TEMP 'zvec-build-perf-diskpart.txt'
if ($Action -eq 'Detach') {
    if (Test-Path -LiteralPath $vhdPath) {
        @("select vdisk file=`"$vhdPath`"", 'detach vdisk') |
            Set-Content -LiteralPath $scriptPath -Encoding Ascii
        & diskpart.exe /s $scriptPath
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    exit 0
}
if (Test-Path -LiteralPath $vhdPath) { throw "Refusing to reuse $vhdPath" }
@("create vdisk file=`"$vhdPath`" maximum=61440 type=expandable",
  "select vdisk file=`"$vhdPath`"", 'attach vdisk') |
    Set-Content -LiteralPath $scriptPath -Encoding Ascii
& diskpart.exe /s $scriptPath
if ($LASTEXITCODE -ne 0) { throw 'diskpart could not create/attach the benchmark VHDX' }
$disk = Get-DiskImage -ImagePath $vhdPath | Get-Disk
if (-not $disk -or $disk.PartitionStyle -ne 'RAW') { throw 'Expected a new RAW VHDX' }
$partition = $disk | Initialize-Disk -PartitionStyle GPT -PassThru |
    New-Partition -AssignDriveLetter -UseMaximumSize
$native = ([Environment]::OSVersion.Version.Build -ge 22621) -and
    (Get-Command Format-Volume).Parameters.ContainsKey('DevDrive')
if ($native) {
    $partition | Format-Volume -DevDrive -NewFileSystemLabel 'ZvecPerf' -Confirm:$false -Force | Out-Null
} else {
    # Server 2022 supports ReFS, but not native Dev Drive. Label that distinction.
    $partition | Format-Volume -FileSystem ReFS -NewFileSystemLabel 'ZvecPerf' -Confirm:$false -Force | Out-Null
}
$root = "$($partition.DriveLetter):\"
if ((Get-Volume -DriveLetter $partition.DriveLetter).FileSystemType -ne 'ReFS') {
    throw 'Benchmark volume is not ReFS'
}
"PERF_DEV_ROOT=$root" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
"PERF_NATIVE_DEV_DRIVE=$native" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
# Default runs preserve Defender. The optional ARM experiment enables both
# real-time protection (for NTFS) and performance mode (for trusted Dev Drive).
# It never disables Defender or detaches antivirus filters.
if ($env:PERF_DEFENDER_PERFORMANCE_MODE -eq 'true') {
    if (-not $native) { throw 'Defender performance mode requires native Dev Drive' }
    Start-Service -Name WinDefend
    Set-MpPreference -DisableRealtimeMonitoring $false
    Set-MpPreference -PerformanceModeStatus Enabled
    # Defender applies changes asynchronously. Keep a short status history so
    # a runner policy override can be distinguished from service startup lag.
    $deadline = [DateTime]::UtcNow.AddSeconds(90)
    $states = @()
    $enabled = $false
    do {
        $defender = Get-MpComputerStatus
        $preference = Get-MpPreference
        $state = [pscustomobject]@{
            utc = [DateTime]::UtcNow.ToString('o')
            AntivirusEnabled = $defender.AntivirusEnabled
            RealTimeProtectionEnabled = $defender.RealTimeProtectionEnabled
            AMRunningMode = $defender.AMRunningMode
            IsTamperProtected = $defender.IsTamperProtected
            DisableRealtimeMonitoring = $preference.DisableRealtimeMonitoring
            PerformanceModeStatus = [int]$preference.PerformanceModeStatus
        }
        $states += $state
        Write-Host ($state | ConvertTo-Json -Compress)
        # The PowerShell enum is Enabled=0, Disabled=1 (not a Boolean).
        $enabled = $state.AntivirusEnabled -and $state.RealTimeProtectionEnabled -and
            $state.PerformanceModeStatus -eq 0
        if ($enabled) { break }
        Start-Sleep -Seconds 5
    } while ([DateTime]::UtcNow -lt $deadline)
    $setupResults = Join-Path $env:RUNNER_TEMP 'zvec-build-perf-results'
    New-Item -ItemType Directory -Path $setupResults -Force | Out-Null
    $states | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$setupResults/defender-setup.json" -Encoding utf8
    if (-not $enabled) {
        throw "Defender setup did not settle after 90 seconds: $($state | ConvertTo-Json -Compress)"
    }
    Write-Host 'Defender: real-time protection ON; performance mode Enabled (0)'
}
if ($native) {
    & fsutil.exe devdrv query $root
    if ($LASTEXITCODE -ne 0) { throw 'Native Dev Drive verification failed' }
}
Write-Host "Benchmark volume: $root (native Dev Drive: $native; backing file: $vhdPath)"
