# AI 工业检测模型训练平台 - 前端开发规划

## 1. 项目概述
目标是构建一个端到端的工业检测 AI 平台，前端交互界面对标 Roboflow 和 MakeSense。核心功能涵盖项目管理、数据标注、模型训练配置及监控、模型导出与部署。

## 2. 技术栈选型
为了保证高性能交互（特别是标注环节）和现代化的开发体验，推荐以下技术栈：

- **核心框架**: [React 18](https://react.dev/) + [TypeScript](https://www.typescriptlang.org/)
- **构建工具**: [Vite](https://vitejs.dev/) (极速构建)
- **UI 组件库**: [Tailwind CSS](https://tailwindcss.com/) + [Shadcn/UI](https://ui.shadcn.com/) (高度可定制，适合构建专业工具界面)
- **状态管理**: [Zustand](https://github.com/pmndrs/zustand) (轻量级，适合复杂交互状态管理)
- **路由管理**: [React Router v6](https://reactrouter.com/)
- **图形/标注引擎**: [Konva.js](https://konvajs.org/) / [React-Konva](https://github.com/konvajs/react-konva) (高性能 Canvas 库，处理大量标注框不卡顿)
- **数据交互**: [Axios](https://axios-http.com/) + [TanStack Query](https://tanstack.com/query/latest) (高效的数据获取与缓存)

## 3. 功能模块规划

### 3.1. 工作台 (Dashboard)
- **项目列表**: 展示所有检测项目，支持搜索、筛选。
- **新建项目向导**: 设置项目名称、检测类型（目标检测/语义分割/分类）、数据集类型。
- **统计概览**: 展示项目数量、图片总数、已标注数量等。

### 3.2. 数据中心 (Data Center)
- **数据上传**: 支持拖拽上传、文件夹上传、API 导入。支持图片预览。
- **数据集管理**: 数据集版本控制 (v1, v2...)，训练集/验证集/测试集划分 (Train/Val/Test Split)。
- **数据预处理/增强**: 在线配置增强策略（旋转、裁剪、噪声、亮度调整等）。

### 3.3. 标注工作室 (Annotation Studio) - **核心难点**
- **画布交互**: 缩放 (Zoom)、平移 (Pan)。
- **标注工具**:
    - 矩形框 (Bounding Box)
    - 多边形 (Polygon)
    - 关键点 (Keypoints) - 可选
- **标签管理**: 创建/编辑/删除标签类别 (Classes)，快捷键切换。
- **辅助功能**: 十字准星、智能吸附 (Snapping)、AI 辅助标注 (SAM 集成预留)。

### 3.4. 模型训练中心 (Model Center)
- **模型配置**: 选择基础模型 (YOLOv8, YOLOv10, RT-DETR 等)，配置超参数 (Epochs, Batch Size, LR)。
- **训练监控**: 实时 WebSocket 连接，展示 Loss 曲线、mAP 曲线、训练日志。
- **训练历史**: 历史训练任务记录对比。

### 3.5. 部署与推理 (Deployment & Inference)
- **模型导出**: ONNX, TensorRT, Pt 等格式导出。
- **在线测试**: 上传新图片，实时运行模型查看效果。
- **API 密钥管理**: 生成 API Key 供外部调用。

## 4. 开发阶段规划

### 第一阶段：基础设施搭建 (当前阶段)
- 初始化 Vite 项目
- 配置 Tailwind CSS & Shadcn/UI
- 搭建路由结构 (Layout, Pages)
- 封装基础网络请求模块

### 第二阶段：标注核心引擎开发 (MVP 关键)
- 实现 Canvas 画布基础 (图片加载、缩放、移动)
- 实现矩形框绘制与编辑
- 实现标签状态管理

### 第三阶段：项目与数据管理
- 实现项目创建流程
- 实现图片上传与列表展示
- 对接标注数据保存接口

### 第四阶段：训练与后端对接
- 训练参数表单
- 训练状态可视化 (Echarts/Recharts)

## 5. 目录结构建议

```
src/
├── assets/          # 静态资源
├── components/      # 公共组件 (Button, Input, etc.)
│   ├── ui/          # Shadcn UI 组件
│   └── common/      # 通用业务组件
├── features/        # 功能模块 (按业务领域划分)
│   ├── annotation/  # 标注相关 (Canvas, Tools)
│   ├── dashboard/   # 面板相关
│   ├── dataset/     # 数据集相关
│   └── training/    # 训练相关
├── hooks/           # 通用 Hooks
├── lib/             # 工具库 (utils, constants)
├── services/        # API 服务
├── store/           # 全局状态 (Zustand)
├── types/           # TS 类型定义
└── pages/           # 页面路由组件
```
