# -*- coding: utf-8 -*-
"""
动画序列帧制作工具 - 安装程序

单文件安装向导（PyInstaller onefile，程序文件内嵌于 payload/AnimFrames）：
  - 选择安装目录（默认 %LOCALAPPDATA%\Programs\AnimFrames，无需管理员）
  - 复制程序文件（进度条）
  - 创建桌面 / 开始菜单快捷方式
  - 注册 HKCU 卸载项（控制面板「应用和功能」可见），附卸载脚本
  - 静默安装：AnimFramesSetup.exe /S [目标目录]

卸载：控制面板卸载项 或 安装目录下 uninstall.bat
"""

import base64
import os
import sys
import shutil
import subprocess
import threading
import time

APP_NAME = "动画序列帧制作工具"
APP_ID = "AnimFrames"
EXE_NAME = "AnimFrames.exe"
VERSION = "0.24"


def payload_dir() -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(base, "payload", APP_ID)
    if not os.path.isdir(p):   # 开发调试：直接用打包产物目录
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "dist", APP_ID)
    return p


def default_target() -> str:
    la = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    return os.path.join(la, "Programs", APP_ID)


def _run_ps(ps_cmd: str):
    # -EncodedCommand 以 UTF-16LE Base64 传参，彻底绕开命令行代码页：
    # 打包成 windowed exe 后经 -Command 传中文会被系统 ANSI 码页破坏
    enc = base64.b64encode(ps_cmd.encode("utf-16-le")).decode("ascii")
    subprocess.run(["powershell", "-NoProfile", "-EncodedCommand", enc],
                   creationflags=subprocess.CREATE_NO_WINDOW, check=True)


def create_shortcut(lnk_path: str, target_exe: str, work_dir: str,
                    comment: str = APP_NAME):
    ps = ("$s=(New-Object -ComObject WScript.Shell).CreateShortcut("
          f"'{lnk_path}');$s.TargetPath='{target_exe}';"
          f"$s.WorkingDirectory='{work_dir}';$s.Description='{comment}';"
          f"$s.IconLocation='{target_exe},0';$s.Save()")
    # 以结果（.lnk 已生成）而非退出码判定成功：
    # 个别环境下 PowerShell 实际执行成功却返回 1，重试兜底
    last_err = None
    for _ in range(3):
        try:
            _run_ps(ps)
        except subprocess.CalledProcessError as e:
            last_err = e
        if os.path.exists(lnk_path):
            return
        time.sleep(1)
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"创建快捷方式失败: {lnk_path}")


def write_uninstall_script(target: str, desktop_lnk: str, start_lnk: str):
    """卸载脚本：bat 把 ps1 复制到临时目录后从外部删除安装目录/快捷方式/注册表。
    ps1 由 Python 写成 UTF-8 带 BOM，避免 bat echo 中文路径编码错乱。
    注意 ps1 必须先 Set-Location 切出安装目录：双击 bat 时进程当前目录
    就在安装目录内，而 Windows 不允许删除作为进程当前目录的文件夹。"""
    ps1 = os.path.join(target, "uninstall.ps1")
    with open(ps1, "w", encoding="utf-8-sig") as f:
        f.write(f"""Start-Sleep -Seconds 1
Set-Location $env:TEMP
1..10 | ForEach-Object {{
    Remove-Item -Recurse -Force '{target}' -ErrorAction SilentlyContinue
    Remove-Item -Force '{desktop_lnk}' -ErrorAction SilentlyContinue
    Remove-Item -Force '{start_lnk}' -ErrorAction SilentlyContinue
    if ((-not (Test-Path '{target}')) -and
        (-not (Test-Path '{desktop_lnk}')) -and
        (-not (Test-Path '{start_lnk}'))) {{ break }}
    Start-Sleep -Seconds 2
}}
reg delete "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{APP_ID}" /f | Out-Null
Remove-Item -Force $MyInvocation.MyCommand.Path -ErrorAction SilentlyContinue
""")
    bat = os.path.join(target, "uninstall.bat")
    with open(bat, "w", encoding="gbk") as f:
        f.write(f"""@echo off
echo 正在卸载 {APP_NAME} ...
set TMPPS=%TEMP%\\{APP_ID}_uninstall.ps1
copy /y "{ps1}" "%TMPPS%" >nul
start "" /min powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%TMPPS%"
""")
    return bat


def register_uninstall(target: str, uninstall_bat: str):
    exe = os.path.join(target, EXE_NAME)
    key = (r"HKCU\Software\Microsoft\Windows\CurrentVersion"
           r"\Uninstall\{}".format(APP_ID))
    # (值名, 类型, 数据)；列表形式调 reg.exe，不经 cmd，中文不会乱码
    entries = [
        ("DisplayName", "REG_SZ", APP_NAME),
        ("DisplayVersion", "REG_SZ", VERSION),
        ("DisplayIcon", "REG_SZ", exe),
        ("InstallLocation", "REG_SZ", target),
        ("UninstallString", "REG_SZ", f'"{uninstall_bat}"'),
        ("NoModify", "REG_DWORD", "1"),
        ("NoRepair", "REG_DWORD", "1"),
    ]
    for name, vtype, data in entries:
        subprocess.run(["reg", "add", key, "/v", name, "/t", vtype,
                        "/d", data, "/f"],
                       creationflags=subprocess.CREATE_NO_WINDOW, check=True)


def do_install(target: str, make_desktop: bool, make_startmenu: bool,
               progress_cb=None) -> str:
    """执行安装，progress_cb(done, total)；返回主程序路径"""
    src = payload_dir()
    if not os.path.isdir(src):
        raise RuntimeError(f"安装包数据缺失: {src}")
    files = []
    for root, _dirs, names in os.walk(src):
        for n in names:
            files.append(os.path.join(root, n))
    total = len(files)
    os.makedirs(target, exist_ok=True)
    for i, fp in enumerate(files, 1):
        rel = os.path.relpath(fp, src)
        dst = os.path.join(target, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(fp, dst)
        if progress_cb and (i % 20 == 0 or i == total):
            progress_cb(i, total)

    exe = os.path.join(target, EXE_NAME)
    desktop_lnk = os.path.join(os.path.expanduser("~/Desktop"),
                               f"{APP_NAME}.lnk")
    start_dir = os.path.join(os.environ.get("APPDATA", ""),
                             r"Microsoft\Windows\Start Menu\Programs")
    start_lnk = os.path.join(start_dir, f"{APP_NAME}.lnk")
    if make_desktop:
        create_shortcut(desktop_lnk, exe, target)
    if make_startmenu:
        os.makedirs(start_dir, exist_ok=True)
        create_shortcut(start_lnk, exe, target)
    bat = write_uninstall_script(
        target,
        desktop_lnk if make_desktop else "%~n0-nonexist",
        start_lnk if make_startmenu else "%~n0-nonexist")
    register_uninstall(target, bat)
    return exe


# ---------------- 图形界面 ----------------

def run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    root = tk.Tk()
    root.title(f"{APP_NAME} - 安装")
    root.geometry("520x300")
    root.resizable(False, False)
    try:
        icon = os.path.join(getattr(sys, "_MEIPASS", ""), "app_icon.ico")
        if os.path.exists(icon):
            root.iconbitmap(icon)
    except Exception:
        pass

    pad = {"padx": 16, "pady": 6}
    tk.Label(root, text=f"安装 {APP_NAME}", font=("Microsoft YaHei UI", 14, "bold")
             ).pack(anchor="w", **pad)
    tk.Label(root, text=f"版本 {VERSION}（安装到用户目录，无需管理员权限）",
             fg="#666").pack(anchor="w", padx=16)

    row = tk.Frame(root)
    row.pack(fill="x", **pad)
    tk.Label(row, text="安装位置:").pack(side="left")
    dir_var = tk.StringVar(value=default_target())
    tk.Entry(row, textvariable=dir_var, width=44).pack(side="left", padx=6)

    def browse():
        d = filedialog.askdirectory(initialdir=os.path.dirname(dir_var.get()))
        if d:
            dir_var.set(os.path.join(d, APP_ID) if not d.endswith(APP_ID) else d)
    tk.Button(row, text="浏览...", command=browse).pack(side="left")

    opt = tk.Frame(root)
    opt.pack(fill="x", **pad)
    v_desk = tk.BooleanVar(value=True)
    v_start = tk.BooleanVar(value=True)
    tk.Checkbutton(opt, text="创建桌面快捷方式", variable=v_desk).pack(side="left")
    tk.Checkbutton(opt, text="创建开始菜单快捷方式",
                   variable=v_start).pack(side="left", padx=12)

    bar = ttk.Progressbar(root, length=480, mode="determinate")
    bar.pack(padx=16, pady=(14, 2))
    status = tk.Label(root, text="", fg="#666")
    status.pack(anchor="w", padx=16)

    btn = tk.Button(root, text="安 装", width=14, height=1,
                    bg="#2d7ff9", fg="white", font=("Microsoft YaHei UI", 10, "bold"))
    btn.pack(pady=10)

    def install():
        target = dir_var.get().strip().strip('"')
        if not target:
            return
        if os.path.isdir(target) and os.listdir(target):
            if not messagebox.askyesno(
                    "目录已存在", f"{target}\n已存在文件，将覆盖安装。继续？"):
                return
        btn.config(state="disabled", text="安装中...")

        def worker():
            try:
                def cb(done, total):
                    bar["maximum"] = total
                    bar["value"] = done
                    status.config(text=f"正在复制文件 {done}/{total}")
                exe = do_install(target, v_desk.get(), v_start.get(), cb)
            except Exception as e:
                root.after(0, lambda: messagebox.showerror("安装失败", str(e)))
                root.after(0, lambda: btn.config(state="normal", text="安 装"))
                return

            def done():
                status.config(text="安装完成")
                if messagebox.askyesno("完成", "安装完成！是否立即运行？"):
                    subprocess.Popen([exe], cwd=os.path.dirname(exe))
                root.destroy()
            root.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    btn.config(command=install)
    root.mainloop()


def main():
    try:
        _main()
    except Exception:
        import traceback
        log = os.path.join(os.environ.get("TEMP", "."),
                           "AnimFramesSetup_error.log")
        with open(log, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        raise


def _main():
    args = [a for a in sys.argv[1:]]
    silent = any(a.lower() == "/s" for a in args)
    if silent:
        rest = [a for a in args if a.lower() != "/s"]
        target = rest[0] if rest else default_target()

        def _log(d, t):
            if sys.stdout is not None:   # windowed 模式无控制台
                print(f"\r{d}/{t}", end="", flush=True)
        exe = do_install(target, True, True, _log)
        if sys.stdout is not None:
            print(f"\n安装完成: {exe}")
        return
    run_gui()


if __name__ == "__main__":
    main()
