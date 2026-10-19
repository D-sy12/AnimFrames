# -*- mode: python ; coding: utf-8 -*-
import os

# spec 已移入 packaging/ 子目录：以 spec 所在目录为基准，工程根目录是它的上一级
ROOT = os.path.dirname(SPECPATH)
ICON = os.path.join(ROOT, 'app_icon.ico')


a = Analysis(
    [os.path.join(ROOT, 'packaging', 'installer.py')],
    pathex=[os.path.join(ROOT, 'packaging')],
    binaries=[],
    datas=[(ICON, '.'),
           (os.path.join(ROOT, 'dist', 'AnimFrames'), 'payload/AnimFrames')],
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
    name='AnimFramesSetup',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[ICON],
)
