# AnimFrames —— 动画序列帧制作工具

视频转精灵表（Sprite Sheet）一站式制作工具。将游戏/动画视频逐帧导出，经抠图、精抠、色调统一、位置调整后，拼合为精灵表或 PSD 分层文件，用于本科论文的实践配套工具。

纯 Python / PySide6 桌面应用，无需管理员权限，支持打包为 Windows 安装包。

## 功能总览（8 页工作流）

| 页面 | 功能 | 关键特性 |
|---|---|---|
| 1 视频导帧 | 视频抽取序列帧 | 逐帧 / 每 N 帧 / 每 N 秒三种模式，相同帧去重（相似度阈值可调），自定义格式、前缀、起止时间 |
| 2 帧数选择 | 挑选要使用的帧 | 缩略图网格勾选、动画预览（循环/来回/一次性）、勾选实时保存，A/D/Q/E/空格快捷键 |
| 3 批量抠图 | 一键抠掉背景 | 背景色抠像（多背景色、容差/羽化/边缘收缩/边缘净化、连通域保护）与 rembg AI 模型两种模式，参数实时预览，残留背景高亮检查 |
| 4 图片精抠 | 手动修边 | 擦除 / 背景橡皮 / 补回主体三种笔刷，逐图撤销栈，滚轮缩放画布 |
| 5 色调统一 | 各帧色调一致 | 以任一帧或自传图片为基板，直方图匹配 / Reinhard 迁移两种算法，强度可调，只动主体不动透明区 |
| 6 调整位置 | 多图层排版 | 每帧独立图层：拖拽移动、精确缩放（5%~400%）、框选成组操作、参考图层垫底，全分辨率烘焙输出 |
| 7 导出图片 | 输出成品 | 精灵表 PNG（自动近方形网格、间距、缩放）+ PSD 分层文件（每帧一图层、逐层缩放），自研 PSD 写入器零额外依赖 |
| 8 动画预览 | 检查动画衔接 | 导入多张精灵表各自成动画，独立设置 FPS 与播放方式，切换对比找卡顿（独立工具，不参与存档） |

## 环境要求

- Windows 10 / 11
- Python 3.13（开发环境版本，3.10+ 理论可用）
- 依赖：PySide6 6.11、opencv-python 5.0、numpy 2.5、rembg（含 onnxruntime）

```bash
pip install PySide6 opencv-python numpy rembg onnxruntime
```

## 快速开始

```bash
git clone https://github.com/D-sy12/AnimFrames.git
cd AnimFrames
python main_app.py
```

首次使用 rembg 模式会自动下载 AI 模型（约 170MB，缓存在用户目录）。

## 目录结构

```
AnimFrames/
├── main_app.py              # 主程序入口：组装 8 页向导、恢复会话、加载图标
├── video_to_frames.py       # 抽帧核心算法（无界面）：解码、抽帧、去重
├── gui_video_to_frames.py   # 第1页 视频导帧
├── gui_frame_selector.py    # 第2页 帧数选择
├── gui_frame_matting.py     # 第3页 批量抠图
├── gui_frame_refine.py      # 第4页 图片精抠
├── gui_frame_color.py       # 第5页 色调统一
├── gui_frame_adjust.py      # 第6页 调整位置
├── gui_frame_export.py      # 第7页 导出图片
├── gui_anim_preview.py      # 第8页 动画预览
├── app_icon.ico             # 应用图标（窗口 / exe / 快捷方式共用）
├── session.json             # 会话存档（运行时生成，不入库）
├── .gitignore
├── packaging/               # 打包发布
│   ├── installer.py         # 自研安装向导（复制文件/快捷方式/卸载注册）
│   ├── AnimFrames.spec      # 主程序打包配置（one-dir）
│   ├── AnimFramesSetup.spec # 安装包打包配置（onefile，内嵌主程序）
│   └── AnimFrames_debug.spec# 控制台调试版配置（排查打包问题用）
└── tests/
    ├── make_test_video.py   # 生成测试视频脚本（红绿蓝方块，验证抽帧与去重）
    └── test_video.mp4       # 测试视频（6秒 60帧）
```

运行/打包产生的目录（均不入库）：

| 目录 | 内容 |
|---|---|
| `frames/` | 用户工作成果：抽出的帧 + matted / refined / colored / adjusted / export 各阶段输出 |
| `build/` `dist/` | PyInstaller 中间产物与主程序产物 |
| `installer_output/` | 最终安装包 AnimFramesSetup.exe |

## 打包发布

```bash
# 1. 打主程序（产出 dist/AnimFrames/）
python -m PyInstaller --noconfirm packaging/AnimFrames.spec

# 2. 打安装包（读取 dist/AnimFrames，产出 installer_output/AnimFramesSetup.exe）
python -m PyInstaller --noconfirm --distpath installer_output packaging/AnimFramesSetup.spec
```

安装包特性：图形化向导 + 静默安装 `AnimFramesSetup.exe /S [目标目录]`；默认安装到 `%LOCALAPPDATA%\Programs\AnimFrames`；创建桌面/开始菜单快捷方式；注册 HKCU 卸载项（控制面板可见），无需管理员权限。

## 快捷键

| 页面 | 按键 | 功能 |
|---|---|---|
| 帧数选择 | A / D | 上一张 / 下一张预览 |
| 帧数选择 | 空格 | 播放 / 暂停 |
| 帧数选择 | Q / E | 选中 / 取消选中当前预览帧 |
| 批量抠图、精抠、色调统一 | A / D | 上一张 / 下一张 |
| 精抠 | Ctrl+Z、[ / ] | 撤销、调笔刷大小 |
| 调整位置 | W / A / S / D | 微移当前图（步进可调） |
| 调整位置 | Q / E | 缩小 / 放大当前图（步进可调） |
| 调整位置 | Ctrl+Z | 撤销本图 |

## 数据与存档

- **会话存档**：不换视频的前提下，关闭程序后自动保存进度（视频路径、抽帧参数、各页状态），重启恢复；开发时存工程根目录 `session.json`，安装版存 `%LOCALAPPDATA%\AnimFrames\session.json`
- **帧编号**：界面显示与导出文件名均从 1 开始
- **中文路径**：全部图像读写经过 imdecode/imencode 封装，支持中文目录
