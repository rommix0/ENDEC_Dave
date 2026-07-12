# Loquendo Dave - "DIGITAL ENDEC" version running on Windows via SAPI

![Demo Install Video and Voice Test](install.mp4)

## What is this?

This is a Windows port of the Loquendo Dave TTS engine, specifically the "DIGITAL ENDEC" version (a custom build of the voice). It allows you to use the Loquendo Dave voice directly from a ENDEC device, but running on a Windows machine via SAPI (Speech Application Programming Interface). Previously, this was only available on the DIGITAL ENDEC hardware, but now you can use it freely on your Windows computer.

## ⚠️ IMPORTANT - read this before installing (engine replacement)

Making DaveMSX *actually accurate* to the DIGITAL ENDEC requires the Loquendo **6.6** engine - the same generation the ENDEC itself runs. So, the installer **REPLACES** the core Loquendo engine on your machine (`LoqTTS6.dll` + `LoqTTS6_util.dll`) with genuine, patched 6.6 builds, and **removes the stock "Dave" voice** in favor of DaveMSX. This is deliberate: the 6.5 and 6.6 engines will **not** cohabitate, and a second, competing "Dave" only confuses the engine (and you).

**What this means for your *other* Loquendo 6 voices:** any other Loquendo 6 voice you have (Susan, etc.) will NOT run on the 6.6 engine. The files are incompatible; this is nothing I can fix myself due to the changes made to accommodate Dave. If you wish to use OTHER Loquendo voices, you must either uninstall DaveMSX and restore your original engine + stock Dave, or run them on a separate machine/VM. Sorry.

**The whole process, however, is FULLY reversible.** The installer backs up your original engine + stock Dave to `Loquendo\LTTS\_DaveMSX-restore\`, and `Uninstall-DaveMSX.ps1` puts the machine back *exactly* as it was.

## How to install/use

A working Loquendo TTS 6 install must be present first - it supplies the shared audio modules Loquendo provides. The DaveMSX package supplies everything else (including the 6.6 engine).

1. Install the base Loquendo 6 "Dave" from the "1_Install this Dave FIRST" folder in this repository. Run it to completion. (This is the *inaccurate* SAPI Dave - it's here only to lay down the Loquendo engine that DaveMSX builds on.)
2. Run **`install.bat`** (or `Install-DaveMSX.ps1`) from the "2_Then install this Dave" folder. It elevates, backs up your engine + stock Dave, drops in the genuine 6.6 stack + the ENDEC Dave voice bank, and registers the **"DaveMSX"** voice. Stock "Dave" is removed (but backed up - see the compatibility note above).
3. Pick **DaveMSX** in any 32-bit SAPI5 app (Balabolka, TTSApp, screen readers, etc.). To undo everything and restore your original engine + stock Dave, run **`uninstall.bat`** (or `Uninstall-DaveMSX.ps1`).

## License

GNU GPL 3. See [LICENSE](LICENSE) for details.

## Credits

- Loquendo S.p.A. (now part of Nuance Communications, who itself is now part of Microsoft) - for creating the original Loquendo Dave voice.
- The Internet Archive - for hosting the original Loquendo Dave SAPI installer and patch DLL, which made this project possible in the first place.
- The Global Weather and EAS Society (and GWES EAS Relay Network participants) - for providing a community of enthusiasts and experts who helped me understand the Loquendo Dave voice, its differences between versions, and for providing a platform to share this project with others.

## GenAI Disclosure Notice: Portions of this repository have been generated using Generative AI tools (Claude, Claude Code).
