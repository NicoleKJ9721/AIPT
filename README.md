# AIPT（AI Industrial Platform Tool）

AIPT 是一个面向工业视觉场景的本地化平台，覆盖 **项目/数据集管理、智能标注、模型训练、部署推荐、（UI）多模型串联检测** 等流程。

> 提示：为避免与训练阶段重复，本平台前端“数据增强”模块已默认**屏蔽手动增广**，仅保留“保存为数据集快照（版本固化）”能力；YOLO 训练阶段会自动进行数据增广。

## 一键启动（推荐）

双击仓库根目录下的 `start_services.bat`：

- 自动打开 GUI 启动器
- 自动检查/安装依赖（后端 pip、前端 npm）
- 若检测到 NVIDIA GPU（`nvidia-smi` 可用），会自动尝试安装 CUDA 版 PyTorch（首次下载体积较大）
- 自动拉起后端（`8000`）与前端（`5173`）
- 自动打开浏览器进入平台

## 平台使用说明（快速上手）

### 1) 项目与数据集

1. 进入 **项目管理**
2. 创建项目（按你的业务类型/缺陷类型命名）
3. 导入图片并创建/选择数据集版本

### 2) 智能标注

1. 进入 **智能标注**
2. 选择图片后进行标注（矩形框/多边形等）
3. 保存标注并在数据集中查看统计

### 3) 数据快照（位于“数据增强”模块）

1. 进入 **数据增强**（当前用于“数据集快照/版本固化”）
2. 选择项目与数据集
3. 点击 **保存为快照**，生成新的数据集版本（继承图片与标注）

### 4) 模型训练

1. 进入 **模型训练**
2. 选择数据集/模型配置，提交训练任务
3. 训练完成后，可在模型管理/导出中查看产物（以页面实际功能为准）

### 5) 部署推荐与推理

1. 进入 **部署推荐**
2. 按页面提示选择模型与推理方式，启动推理服务或会话

### 6) 多模型串联检测（UI）

左侧导航 **多模型串联检测** 提供管线编排 UI（当前仅做前端展示，后端串联推理逻辑可按项目需求接入）。

## 平台部署文档（Windows）

部署与异地安装请直接阅读：`docs/平台部署文档.md`。

为方便复制，这里同步一份关键步骤（完整版见上面文档）：

### 1) 安装前置软件（建议使用 winget）

```powershell
winget install --id Git.Git -e --source winget
winget install --id OpenJS.NodeJS.LTS -e --source winget
winget install --id Python.Python.3.11 -e --source winget
```

（可选）如遇到 `torch`/DLL 相关错误：

```powershell
winget install --id Microsoft.VCRedist.2015+.x64 -e --source winget
```

### 2) 创建纯净后端虚拟环境（repo-local）

```powershell
# 推荐：明确使用 Python 3.11（若系统有 py 启动器）
py -3.11 -m venv .venv

# 备选：若没有 py，可用 python（确保 python --version >= 3.11）
# python -m venv .venv

.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -r .\backend\requirements.txt
```

### 3) 安装前端依赖

```powershell
cd .\frontend
npm install
```

### 4) 启动

双击 `start_services.bat`（GUI 启动器）或运行：

```powershell
cmd /c start_services_cli.bat
```

## 开发/测试

后端（仓库根目录）：

```powershell
.\.venv\Scripts\python -m pytest -q
```

前端（`frontend/`）：

```powershell
npm run lint
npm run build
```
