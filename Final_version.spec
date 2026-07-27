# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = (
    collect_submodules('vtkmodules')
    + collect_submodules('vtk')
    + collect_submodules('mayavi')
    + collect_submodules('tvtk')
    + collect_submodules('traitsui')
    + collect_submodules('pyface')
)

datas = (
    collect_data_files('mayavi')
    + collect_data_files('tvtk')
    + collect_data_files('traitsui')
    + collect_data_files('pyface')
)

a = Analysis(
    ['Final_version.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PySide6'],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Final_version',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name='Final_version',
)