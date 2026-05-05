@echo off
setlocal enabledelayedexpansion

for %%f in ("%~dp0\Ligand\*.pdbqt") do (
    set "file=%%~nf"
    echo Processing ligand !file!
    "F:\Soft\Autodock\4.2.6\vina.exe" --config config.txt --ligand "%%f"
)

endlocal

