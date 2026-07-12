#Requires -Version 5

[CmdletBinding()]
param([string]$LttsPath, [switch]$Silent)

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    $a = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    if ($LttsPath) { $a += " -LttsPath `"$LttsPath`"" }
    if ($Silent)   { $a += " -Silent" }
    Start-Process powershell -Verb RunAs -ArgumentList $a
    return
}

$ErrorActionPreference = 'Continue'   # best-effort restore

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

$LTTS = Resolve-Ltts $LttsPath
$restore = Join-Path $LTTS '_DaveMSX-restore'
Write-Host "Loquendo LTTS : $LTTS"

# 1) DaveMSX SAPI token + private CLSID
if (Test-Path $TOK_MSX) { Remove-Item $TOK_MSX -Recurse -Force; Write-Host "  removed        SAPI5 token 'DaveMSX'" }
$privClsid = "HKLM:\SOFTWARE\Classes\WOW6432Node\CLSID\$NEWCLSID"
if (Test-Path $privClsid) { Remove-Item $privClsid -Recurse -Force; Write-Host "  removed        private SAPI engine CLSID" }

# 2) DaveMSX-added files (per manifest; engine/util are NOT here - they get restored below)
$manifest = Join-Path $LTTS 'DaveMSX.install-manifest.txt'
if (Test-Path $manifest) {
    foreach ($p in (Get-Content $manifest)) {
        if ($p -and (Test-Path $p)) { Remove-Item $p -Force }
    }
    Remove-Item $manifest -Force
    Write-Host "  removed        DaveMSX files (loqmsx, LoqEnglish6.9, loqmsxsapi, bank, front-end)"
} else {
    foreach ($rel in @('loqmsx.dll','loqmsxsapi.dll','LoqEnglish6.9.dll','DaveMSX.vde','EnglishUs6.9.lde',
                       'EnglishUs\EnglishUs6.9.atm','EnglishUs\EnglishUs6.9.lex','EnglishUs\EnglishUs6.9.phd')) {
        $p = Join-Path $LTTS $rel; if (Test-Path $p) { Remove-Item $p -Force }
    }
}
$de = Join-Path $LTTS 'EnglishUs\DaveEndec'
if (Test-Path $de) { Remove-Item $de -Recurse -Force; Write-Host "  removed        EnglishUs\DaveEndec\" }

# 3) RESTORE the original engine + stock Dave from the backup
if (Test-Path $restore) {
    Get-ChildItem $restore -Recurse -File | ForEach-Object {
        $rel = $_.FullName.Substring($restore.Length).TrimStart('\')
        if ($rel -ieq 'LQDave.reg') { return }               # handled separately
        $dst = Join-Path $LTTS $rel
        New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
        Copy-Item $_.FullName $dst -Force
    }
    Write-Host "  restored       original engine (LoqTTS6.dll/util) + stock Dave voice + 6.2 front-end"
    # LQDave token
    $tokBak = Join-Path $restore 'LQDave.reg'
    if (Test-Path $tokBak) {
        & reg.exe import "$tokBak" 2>&1 | Out-Null
        Write-Host "  restored       stock 'LQDave' token"
    } elseif (-not (Test-Path $TOK_STOCK)) {
        # fallback: recreate the standard token if the backup lacked it
        New-Item -Path $TOK_STOCK -Force | Out-Null
        Set-ItemProperty $TOK_STOCK '(default)' 'Dave'
        Set-ItemProperty $TOK_STOCK 'CLSID'     '{93443cf0-90b5-4167-a60a-fe80045086a9}'
        Set-ItemProperty $TOK_STOCK '409'       'Loquendo Dave'
        Set-ItemProperty $TOK_STOCK 'VoiceName' 'Dave'
        Set-ItemProperty $TOK_STOCK 'VoicePath' "$LTTS\"
        $attr = Join-Path $TOK_STOCK 'Attributes'; New-Item -Path $attr -Force | Out-Null
        Set-ItemProperty $attr 'Age' 'Adult'; Set-ItemProperty $attr 'Gender' 'Male'
        Set-ItemProperty $attr 'Language' '409'; Set-ItemProperty $attr 'Name' 'Dave'; Set-ItemProperty $attr 'Vendor' 'Loquendo'
        Write-Host "  restored       stock 'LQDave' token (recreated - no backup found)"
    }
    Remove-Item $restore -Recurse -Force
} else {
    Write-Host "  WARNING: no _DaveMSX-restore\ backup found -- the 6.6 engine is still in place." -ForegroundColor Yellow
    Write-Host "           Reinstall a stock Loquendo 6 voice to get a clean engine back."
}

Write-Host ""
Write-Host "DaveMSX uninstalled; original engine + stock Dave restored." -ForegroundColor Green

if (-not $Silent) {
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit
}
