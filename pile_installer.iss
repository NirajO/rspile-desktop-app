; pile_installer.iss – Installer for Pile Analysis Tool (Student Edition)

[Setup]
; Unique ID for this app – don’t reuse for other projects
AppId={{B64A7FA2-7E4E-4E45-B7FE-1234567890AB}
AppName=Pile Analysis Tool
AppVersion=1.0.0
AppPublisher=Niraj Ojha

; Install location (Program Files\PileAnalysisTool)
DefaultDirName={pf}\PileAnalysisTool
DefaultGroupName=Pile Analysis Tool

DisableDirPage=no
DisableProgramGroupPage=no

; Where to output the installer EXE
OutputDir=installer
OutputBaseFilename=PileAnalysisSetup

Compression=lzma
SolidCompression=yes
WizardStyle=modern

; Icon for the installer EXE itself
SetupIconFile=C:\Users\Niraj Ojha\Desktop\rspile-desktop-app\icons\app_icon.ico

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
; Optional desktop shortcut
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
; MAIN EXECUTABLE (PyInstaller output)
Source: "C:\Users\Niraj Ojha\Desktop\rspile-desktop-app\dist\PileAnalysisTool.exe"; DestDir: "{app}"; Flags: ignoreversion

; Icons folder (for QIcon('icons/app_icon.png') etc.)
Source: "C:\Users\Niraj Ojha\Desktop\rspile-desktop-app\icons\*"; DestDir: "{app}\icons"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Start Menu shortcut
Name: "{group}\Pile Analysis Tool"; Filename: "{app}\PileAnalysisTool.exe"

; Desktop shortcut (only if user checks the box)
Name: "{commondesktop}\Pile Analysis Tool"; Filename: "{app}\PileAnalysisTool.exe"; Tasks: desktopicon

[Run]
; Launch app automatically after installation
Filename: "{app}\PileAnalysisTool.exe"; Description: "Launch Pile Analysis Tool"; Flags: nowait postinstall skipifsilent
