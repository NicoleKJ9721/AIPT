Option Explicit

Dim fso, shell, repoRoot, psExe, psScript, cmd

Set fso = CreateObject("Scripting.FileSystemObject")
repoRoot = fso.GetParentFolderName(WScript.ScriptFullName)
repoRoot = fso.GetParentFolderName(repoRoot)

Set shell = CreateObject("WScript.Shell")
psExe = shell.ExpandEnvironmentStrings("%SystemRoot%") & "\System32\WindowsPowerShell\v1.0\powershell.exe"
psScript = repoRoot & "\scripts\start_services_gui.ps1"

cmd = """" & psExe & """ -NoProfile -ExecutionPolicy Bypass -STA -WindowStyle Hidden -File """ & psScript & """"
shell.Run cmd, 0, False

