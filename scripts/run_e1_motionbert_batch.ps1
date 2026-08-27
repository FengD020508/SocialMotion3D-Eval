param(
    [Parameter(Mandatory = $true)]
    [string]$Manifest,

    [Parameter(Mandatory = $true)]
    [string]$InputRoot,

    [Parameter(Mandatory = $true)]
    [string]$AnnotationRoot,

    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,

    [string]$TargetTrackRoot,
    [string]$MotionExtractionRoot = "D:\Project\MotionExtraction",
    [string]$CondaEnv = "motionbert",
    [string[]]$Clips = @(),
    [switch]$OverwriteCrops,
    [switch]$OverwritePoseTracking
)

$ErrorActionPreference = "Stop"
$env:SSLKEYLOGFILE = ""
$env:CONDA_NO_PLUGINS = "true"
$env:MPLCONFIGDIR = Join-Path $MotionExtractionRoot ".mplconfig"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
$MotionBertRoot = Join-Path $MotionExtractionRoot "MotionBERT"
$AlphaPoseRunner = Join-Path $MotionExtractionRoot "scripts\run_alphapose_frames_motionbert.ps1"
$Cleaner = Join-Path $PSScriptRoot "clean_alphapose_target.py"
$Smoother = Join-Path $MotionExtractionRoot "demo_stage\scripts\oneeuro_pose_smooth.py"
$Cropper = Join-Path $PSScriptRoot "prepare_motionbert_iddped.py"

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$cropArgs = @(
    "run", "--no-capture-output", "-n", $CondaEnv,
    "python", $Cropper,
    "--manifest", $Manifest,
    "--input-root", $InputRoot,
    "--annotation-root", $AnnotationRoot,
    "--output-root", $OutputRoot
)
if ($TargetTrackRoot) {
    $cropArgs += @("--target-track-root", $TargetTrackRoot)
}
if ($Clips.Count -gt 0) {
    $cropArgs += "--clips"
    $cropArgs += $Clips
}
if ($OverwriteCrops) {
    $cropArgs += "--overwrite"
}
& conda @cropArgs
if ($LASTEXITCODE -ne 0) { throw "Target crop preparation failed." }

$sceneDirs = Get-ChildItem -LiteralPath $OutputRoot -Directory | Where-Object { $_.Name -match '^\d{2}_' } | Sort-Object Name
if ($Clips.Count -gt 0) {
    $requested = @{}; foreach ($clip in $Clips) { $requested[$clip] = $true }
    $sceneDirs = $sceneDirs | Where-Object { $requested[$_.Name.Substring(0, 2)] }
}

$batch = @()
foreach ($sceneDir in $sceneDirs) {
    $scene = $sceneDir.Name
    $video = Join-Path $sceneDir.FullName "focus_crop.mp4"
    $alphaWork = Join-Path $sceneDir.FullName "alphapose_work"
    $alphaRaw = Join-Path $sceneDir.FullName "alphapose_raw"
    $alphaJson = Join-Path $alphaRaw "alphapose-results.json"
    $cleanJson = Join-Path $sceneDir.FullName "alphapose_clean.json"
    $cleanReport = Join-Path $sceneDir.FullName "alphapose_clean.report.json"
    $smoothJson = Join-Path $sceneDir.FullName "alphapose_clean_oneeuro.json"
    $smoothReport = Join-Path $sceneDir.FullName "oneeuro_report.json"
    $overlayVideo = Join-Path $sceneDir.FullName "focus_2d_pose_oneeuro.mp4"
    $liftingDir = Join-Path $sceneDir.FullName "motionbert_lifting"
    $x3d = Join-Path $liftingDir "X3D.npy"

    Write-Host "=== $scene ==="
    if ($OverwritePoseTracking) {
        $resolvedScene = [IO.Path]::GetFullPath($sceneDir.FullName)
        $resolvedRoot = $OutputRoot.TrimEnd('\') + '\'
        if (-not $resolvedScene.StartsWith($resolvedRoot, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to overwrite outputs outside E1 root: $resolvedScene"
        }
        foreach ($derived in @($cleanJson, $cleanReport, $smoothJson, $smoothReport, $overlayVideo, $liftingDir)) {
            if (Test-Path -LiteralPath $derived) { Remove-Item -LiteralPath $derived -Recurse -Force }
        }
    }
    if (-not (Test-Path -LiteralPath $x3d)) {
        if (-not (Test-Path -LiteralPath $alphaJson)) {
            & $AlphaPoseRunner -Video $video -WorkDir $alphaWork -OutDir $alphaRaw -CondaEnv $CondaEnv
            if ($LASTEXITCODE -ne 0) { throw "AlphaPose failed for $scene" }
        }
        if (-not (Test-Path -LiteralPath $cleanJson)) {
            & conda run --no-capture-output -n $CondaEnv python $Cleaner `
                --json-in $alphaJson --video $video --json-out $cleanJson --report-out $cleanReport
            if ($LASTEXITCODE -ne 0) { throw "AlphaPose cleanup failed for $scene" }
        }
        if (-not (Test-Path -LiteralPath $smoothJson)) {
            & conda run --no-capture-output -n $CondaEnv python $Smoother `
                --json-in $cleanJson --json-out $smoothJson --report-out $smoothReport `
                --video $video --out-video $overlayVideo --layout halpe26 --label alphapose26_oneeuro `
                --min-cutoff 1.0 --beta 0.04 --d-cutoff 1.0 --conf-min 0.05 --conf-threshold 0.15
            if ($LASTEXITCODE -ne 0) { throw "OneEuro smoothing failed for $scene" }
        }
        Push-Location $MotionBertRoot
        try {
            & conda run --no-capture-output -n $CondaEnv python infer_wild.py `
                --vid_path $video --json_path $smoothJson --out_path $liftingDir
            if ($LASTEXITCODE -ne 0) { throw "MotionBERT failed for $scene" }
        }
        finally {
            Pop-Location
        }
    }

    if (-not (Test-Path -LiteralPath $x3d)) { throw "Missing X3D.npy after processing $scene" }
    $framesDir = Join-Path $alphaWork "frames"
    if (Test-Path -LiteralPath $framesDir) {
        $resolvedFrames = [IO.Path]::GetFullPath($framesDir)
        $resolvedRoot = $OutputRoot.TrimEnd('\') + '\'
        if (-not $resolvedFrames.StartsWith($resolvedRoot, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove frames outside E1 output root: $resolvedFrames"
        }
        Remove-Item -LiteralPath $resolvedFrames -Recurse -Force
    }

    $array = & conda run --no-capture-output -n $CondaEnv python -c "import numpy as np; x=np.load(r'$x3d'); print(f'{x.shape[0]},{x.shape[1]},{x.shape[2]}')"
    $batch += [ordered]@{
        scene = $scene
        x3d = $x3d
        shape = ($array | Select-Object -Last 1).Trim()
        completed = (Get-Date).ToString("o")
    }
}

$report = [ordered]@{
    method = "MotionBERT"
    input = "IDD-PeD target-centered crops"
    pose_2d = "AlphaPose Halpe26 plus OneEuro"
    checkpoint = "FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin"
    scenes = $batch
}
$reportPath = Join-Path $OutputRoot "motionbert_batch_report.json"
[IO.File]::WriteAllText($reportPath, ($report | ConvertTo-Json -Depth 6) + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
Write-Host "MotionBERT batch report: $reportPath"
