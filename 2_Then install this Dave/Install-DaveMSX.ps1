#Requires -Version 5

[CmdletBinding()]
param([string]$LttsPath, [switch]$Silent, [ValidateRange(0,127)][int]$SpeedBase = 52)

# ---- self-elevate ----
$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    $a = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    if ($LttsPath) { $a += " -LttsPath `"$LttsPath`"" }
    if ($Silent)   { $a += " -Silent" }
    $a += " -SpeedBase $SpeedBase"
    Start-Process powershell -Verb RunAs -ArgumentList $a
    return
}

$ErrorActionPreference = 'Stop'
$pkg = $PSScriptRoot

$TOK_MSX   = 'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Speech\Voices\Tokens\LQDaveMSX'
$TOK_STOCK = 'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Speech\Voices\Tokens\LQDave'
$NEWCLSID  = '{671D377E-54E7-4904-8A2F-418DACF46DF0}'

function Resolve-Ltts([string]$Override) {
    if ($Override) { return $Override.TrimEnd('\') }
    foreach ($t in @($TOK_MSX, $TOK_STOCK)) {
        try { $vp = (Get-ItemProperty $t -Name VoicePath -ErrorAction Stop).VoicePath
              if ($vp) { return $vp.TrimEnd('\') } } catch { }
    }
    return 'C:\Program Files (x86)\Loquendo\LTTS'
}

try {
    $LTTS = Resolve-Ltts $LttsPath
    Write-Host "Loquendo LTTS : $LTTS"
    if (-not (Test-Path (Join-Path $LTTS 'LoqTTS6.dll'))) {
        throw "No Loquendo TTS install found under `"$LTTS`". Install a working Loquendo 6 voice first, or pass -LttsPath."
    }
    $restore = Join-Path $LTTS '_DaveMSX-restore'

    # ---- 1) BACK UP everything we are about to replace/remove (first install only,
    #         so re-running never overwrites the pristine stock backup) ----
    New-Item -ItemType Directory -Force -Path $restore | Out-Null
    $backupSet = @(
        'LoqTTS6.dll','LoqTTS6_util.dll',                       # engine we overwrite
        'Dave.vde','LoqEnglish6.2.dll','EnglishUs6.2.lde',      # stock Dave we remove
        'EnglishUs\EnglishUs6.2.atm','EnglishUs\EnglishUs6.2.lex','EnglishUs\EnglishUs6.2.phd'
    )
    foreach ($rel in $backupSet) {
        $src = Join-Path $LTTS $rel; $bak = Join-Path $restore $rel
        if ((Test-Path $src) -and -not (Test-Path $bak)) {
            New-Item -ItemType Directory -Force -Path (Split-Path $bak) | Out-Null
            Copy-Item $src $bak -Force
        }
    }
    # stock Dave voice bank dir
    $stockDave = Join-Path $LTTS 'EnglishUs\Dave'
    $stockDaveBak = Join-Path $restore 'EnglishUs\Dave'
    if ((Test-Path $stockDave) -and -not (Test-Path $stockDaveBak)) {
        Copy-Item $stockDave $stockDaveBak -Recurse -Force
    }
    # LQDave token
    $tokBak = Join-Path $restore 'LQDave.reg'
    if ((Test-Path $TOK_STOCK) -and -not (Test-Path $tokBak)) {
        & reg.exe export 'HKLM\SOFTWARE\WOW6432Node\Microsoft\Speech\Voices\Tokens\LQDave' "$tokBak" /y | Out-Null
    }
    Write-Host "  backed up      stock engine + Dave -> _DaveMSX-restore\"

    # ---- 2) INSTALL the 6.6 stack + DaveMSX (overwrite) ----
    $ship = @(
        'LoqTTS6.dll','LoqTTS6_util.dll','loqmsx.dll','LoqEnglish6.9.dll','loqmsxsapi.dll',
        'DaveMSX.vde','EnglishUs6.9.lde',
        'EnglishUs\EnglishUs6.9.atm','EnglishUs\EnglishUs6.9.lex','EnglishUs\EnglishUs6.9.phd',
        'EnglishUs\DaveEndec\Dave-19200.16000.loqmsx.bin','EnglishUs\DaveEndec\Dave.lex','EnglishUs\DaveEndec\Dave.sde'
    )
    $manifestPaths = New-Object System.Collections.Generic.List[string]
    foreach ($rel in $ship) {
        $src = Join-Path $pkg $rel; $dst = Join-Path $LTTS $rel
        if (-not (Test-Path $src)) { throw "package file missing: $rel" }
        New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
        Copy-Item $src $dst -Force
        # engine + util are RESTORED from backup on uninstall, not deleted -> not in manifest
        if ($rel -notin @('LoqTTS6.dll','LoqTTS6_util.dll')) { $manifestPaths.Add($dst) }
        Write-Host "  installed      $rel"
    }
    # apply -SpeedBase to the installed loqmsxsapi.dll (shipped at 52)
    if ($SpeedBase -ne 52) {
        $sp = Join-Path $LTTS 'loqmsxsapi.dll'
        $b = [IO.File]::ReadAllBytes($sp)
        if ($b[0x44a0] -eq 0x40) { $b[0x4802] = [byte]$SpeedBase; $b[0x6ff6] = [byte]$SpeedBase; [IO.File]::WriteAllBytes($sp,$b)
            Write-Host "  speed base     set to $SpeedBase" }
    }

    # ---- 3) REMOVE stock Dave files (already backed up) so nothing competes ----
    foreach ($rel in @('Dave.vde','LoqEnglish6.2.dll','EnglishUs6.2.lde',
                       'EnglishUs\EnglishUs6.2.atm','EnglishUs\EnglishUs6.2.lex','EnglishUs\EnglishUs6.2.phd')) {
        $p = Join-Path $LTTS $rel
        if (Test-Path $p) { Remove-Item $p -Force }
    }
    if (Test-Path $stockDave) { Remove-Item $stockDave -Recurse -Force }
    Write-Host "  removed        stock Dave voice + 6.2 front-end (backed up)"

    # ---- 4) register the private SAPI engine + DaveMSX token; remove LQDave ----
    $ck = "HKLM:\SOFTWARE\Classes\WOW6432Node\CLSID\$NEWCLSID"
    New-Item "$ck\InprocServer32" -Force | Out-Null
    Set-ItemProperty "$ck\InprocServer32" '(default)' (Join-Path $LTTS 'loqmsxsapi.dll')
    Set-ItemProperty "$ck\InprocServer32" 'ThreadingModel' 'Both'
    Set-ItemProperty $ck '(default)' 'TTSEngine Class (DaveMSX ENDEC)'

    New-Item -Path $TOK_MSX -Force | Out-Null
    Set-ItemProperty $TOK_MSX '(default)' 'DaveMSX'
    Set-ItemProperty $TOK_MSX 'CLSID'     $NEWCLSID
    Set-ItemProperty $TOK_MSX '409'       'Loquendo Dave (ENDEC/loqmsx)'
    Set-ItemProperty $TOK_MSX 'VoiceName' 'DaveMSX'
    Set-ItemProperty $TOK_MSX 'VoicePath' "$LTTS\"
    $attr = Join-Path $TOK_MSX 'Attributes'
    New-Item -Path $attr -Force | Out-Null
    Set-ItemProperty $attr 'Age' 'Adult'; Set-ItemProperty $attr 'Gender' 'Male'
    Set-ItemProperty $attr 'Language' '409'; Set-ItemProperty $attr 'Name' 'DaveMSX'
    Set-ItemProperty $attr 'Vendor' 'Loquendo'
    if (Test-Path $TOK_STOCK) { Remove-Item $TOK_STOCK -Recurse -Force }
    Write-Host "  registered     DaveMSX token; removed stock LQDave token"

    $manifest = Join-Path $LTTS 'DaveMSX.install-manifest.txt'
    $manifestPaths | Set-Content -Path $manifest -Encoding UTF8

    Write-Host ""
    Write-Host "DaveMSX installed OK -- engine replaced with genuine 6.6, stock Dave swapped out." -ForegroundColor Green
    Write-Host "Pick 'DaveMSX' in any 32-bit SAPI5 host (e.g. Balabolka, TTSApp). Undo by running uninstall.bat or Uninstall-DaveMSX.ps1."
    Write-Host "NOTE: other Loquendo-6 voices now run on the 6.6 engine (see project README!)." -ForegroundColor Yellow
} catch {
    Write-Host ""
    Write-Host "INSTALL FAILED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "If a backup exists under _DaveMSX-restore\, run uninstall.bat or Uninstall-DaveMSX.ps1 to roll back."
}

if (-not $Silent) {
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit
}
