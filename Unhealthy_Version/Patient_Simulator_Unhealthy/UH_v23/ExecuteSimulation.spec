# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['Unhealthy_Files/GUI_Files/ExecuteSimulation.py'],
    pathex=[],
    binaries=[],
    datas=[('Unhealthy_Files/GUI_Files/templates', 'templates'), ('Unhealthy_Files/GUI_Files/static', 'static'), ('Healthy_Files', 'Healthy_Files'), ('Unhealthy_Files', 'Unhealthy_Files'), ('Trained_GAN_Path_Files', 'Trained_GAN_Path_Files')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ExecuteSimulation',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
