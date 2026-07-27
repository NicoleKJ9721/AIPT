# AIPT（AI Industrial Platform Tool）

AIPT 是一个面向工业视觉场景的本地化平台，覆盖 **项目/数据集管理、智能标注、YOLO 检测与分割训练、推理部署、多模型串联检测** 等流程。项目数据、训练产物和资源默认存放在源码目录之外，便于 Windows 工位机长期运行和迁移。

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
2. 选择图片后进行矩形框/多边形标注
3. 可使用模板匹配、YOLO 预标注，或基于点提示的 FastSAM/SAM 辅助分割
4. 保存标注并在数据集中查看统计

### 3) 数据快照（位于“数据增强”模块）

1. 进入 **数据增强**（当前用于“数据集快照/版本固化”）
2. 选择项目与数据集
3. 点击 **保存为快照**，生成新的数据集版本（继承图片与标注）

### 4) 模型训练

1. 进入 **模型训练**
2. 选择数据集、检测/分割任务和模型配置；默认基线为 `YOLOv8s`
3. 提交训练任务；真实训练失败会保留失败状态和日志，测试环境才会显式启用模拟训练
4. 训练完成后，可在模型管理/导出中查看产物

### 5) 部署推荐与推理

1. 进入 **部署推荐**
2. 按页面提示选择模型与推理方式，启动推理服务或会话

### 6) 多模型串联检测

左侧导航 **多模型串联检测** 可保存并执行真实的推理管线：

- 首节点可配置**固定工位 ROI**。ROI 只作用于推理输入，不会改写原图；框、分割结果仍映射回原图坐标。
- 节点之间可依据前一节点的检测框或分割区域配置**动态 ROI**，用于粗检→精检的串联流程。
- 当动态 ROI 为空时，可选择停止、回退到配置的输入区域或跳过下一步。

固定 ROI 当前以百分比参数配置；后续版本会补充拖拽绘制、忽略区、多边形掩膜和工件定位/对齐。

### 7) 第三代泛化检测规划

“第三代”指极少样本/零样本的异常与未知缺陷发现，而不是把 SAM 分割模型当作缺陷分类器。该能力尚未集成到运行时，设计、模型分层与验收规则见 [第三代泛化检测设计](docs/THIRD_GEN_GENERALIZATION_DETECTION_DESIGN.md)。

当前没有永久性背景抠图模块。对于稳定工位，应优先使用固定 ROI、忽略区和定位校正；背景/前景掩膜应作为可版本化的派生配方，而不是覆盖原始图片。

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
