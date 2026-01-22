# AIPT 平台开发进度报告 (AIPT Platform Development Progress Report)

**版本号**: v0.1.0 (MVP)  
**日期**: 2025-12-31  
**状态**: 原型验证阶段 (Prototype / MVP)

---

## 1. 当前开发进度概览 (Current Progress Overview)

平台已完成基础架构搭建，实现了核心的"标注-训练-推理"闭环的最简可行产品（MVP）。前后端已打通，可进行本地模型加载与图像推理。

### 1.1 已完成核心功能模块

| 模块名称 | 功能描述 | 状态 | 版本 |
| :--- | :--- | :--- | :--- |
| **工作台 (Dashboard)** | 平台概览、项目统计、快捷入口 | ✅ 已完成 | v1.0 |
| **标注工作室 (Annotation Studio)** | 基于 Canvas (Konva) 的图像标注工具，支持矩形/多边形 | ✅ 已完成 | v1.0 |
| **AI 辅助标注** | 集成后端 YOLOv8 模型，实现自动识别与预标注 | ✅ 已完成 | v1.0 |
| **模型训练 (Training)** | 训练配置界面、后端异步任务调度（基础框架） | ⚠️ 部分完成 | v0.5 |
| **数据增强 (Augmentation)** | 图像增强流水线配置界面（几何/色彩/噪声/天气） | ✅ 已完成 | v1.0 |
| **模型部署 (Deployment)** | 模型导出配置界面 | ✅ 已完成 | v1.0 |
| **系统基础** | 全局异常捕获 (Error Boundary)、一键启动脚本 | ✅ 已完成 | v1.0 |

### 1.2 基础架构组件

*   **前端架构**: React 19 + TypeScript + Vite + Tailwind CSS + Shadcn UI
    *   **状态管理**: Zustand
    *   **路由**: React Router v7
    *   **绘图引擎**: React Konva
    *   **网络请求**: Axios
*   **后端架构**: Python 3.10+ + FastAPI + Ultralytics (YOLOv8)
    *   **API 文档**: Swagger UI / OpenAPI (自动生成)
    *   **任务队列**: FastAPI BackgroundTasks (暂用)
    *   **图像处理**: Pillow (PIL)

### 1.3 稳定分支

*   **当前分支**: `main` (或 `dev`)
*   **稳定版本 Tag**: `v0.1.0-alpha`

---

## 2. 待开发功能清单 (Pending Features & Backlog)

按优先级从高到低排序：

### 2.1 P0 - 核心业务闭环完善
*   **[后端] 真实训练任务执行**: 目前 `/train` 接口仅为调度存根，需对接 `ultralytics` 训练进程并捕获日志。
*   **[后端] WebSocket 实时日志**: 实现训练进度和日志向前端的实时推送。
*   **[后端] 数据持久化**: 引入 SQLite 或 PostgreSQL 存储项目、数据集、标注信息（目前为 Mock 数据）。
*   **[系统] 文件存储服务**: 实现本地文件系统或 MinIO 对接，用于存储上传的数据集图片。

### 2.2 P1 - 业务功能增强
*   **[前端] 数据集管理**: 实现真实的文件上传、分片上传、文件夹管理。
*   **[前端] 标注数据导出**: 支持导出为 YOLO/COCO/VOC 格式。
*   **[后端] 模型版本管理**: 对训练产出的 `.pt` 文件进行版本控制和归档。

### 2.3 P2 - 系统能力扩展
*   **[系统] 用户鉴权 (Auth)**: 登录/注册、JWT Token 认证、RBAC 权限控制。
*   **[前端] 多人协作**: 标注任务的分发与审核流。
*   **[后端] 边缘端导出**: 支持导出 ONNX, TensorRT, OpenVINO 格式。

### 2.4 测试缺口
*   **前端**: 单元测试覆盖率目前为 0%，需引入 Jest/Vitest 测试核心组件（如 `AnnotationStudio` 的逻辑）。
*   **后端**: 仅有基础接口测试，缺乏业务逻辑测试（如训练异常中断、并发请求处理）。

---

## 3. 前后端协作规范 (Collaboration Standards)

### 3.1 接口文档
*   **标准**: OpenAPI 3.0 (Swagger)
*   **地址**: `http://localhost:8000/docs`
*   **要求**: 所有新接口必须定义 Pydantic Schema，包含完整的 Example Value 和 Description。

### 3.2 联调与环境
*   **本地开发**: 使用 `start_services.bat` 一键启动。
*   **跨域 (CORS)**: 后端已配置 `CORSMiddleware` 允许 `["*"]`（生产环境需收敛）。前端 `vite.config.ts` 配置了 `/api` 代理转发。
*   **端口约定**:
    *   前端: `5173`
    *   后端: `8000`

### 3.3 数据交互
*   **格式**: JSON (默认), Multipart/Form-Data (文件上传)
*   **校验**:
    *   前端: 使用 Zod 或 TypeScript 类型进行预校验。
    *   后端: 依赖 Pydantic 进行严格类型校验，验证失败返回 `422 Unprocessable Entity`。

---

## 4. 后端独立测试方案 (Backend Testing Plan)

### 4.1 测试目标
*   **单元测试覆盖率**: ≥ 85%
*   **核心模块**: `model.py` (模型加载/推理), `main.py` (API 路由逻辑)。

### 4.2 压力测试指标
*   **推理接口 (`/predict`)**:
    *   目标 QPS: ≥ 10 (单卡 3060 级别)
    *   平均响应时间: < 200ms (640x640 输入)
*   **并发**: 支持至少 5 个并发训练任务排队。

### 4.3 异常处理测试用例
*   [ ] 模型文件损坏或丢失时的加载行为。
*   [ ] 上传非图片文件或损坏图片的异常捕获。
*   [ ] 训练过程中显存溢出 (OOM) 的优雅降级或报错。
*   [ ] 训练进程被系统 Kill 后的状态恢复。

### 4.4 日志规范
*   格式: `[TIME] [LEVEL] [MODULE] - Message`
*   工具: 使用 Python 标准 `logging` 库或 `loguru`。
*   分级:
    *   `INFO`: 正常操作 (API 请求, 任务状态变更)
    *   `WARNING`: 非预期输入, 资源紧张
    *   `ERROR`: 业务逻辑失败, 异常捕获

---

## 5. 后续开发路线图 (Roadmap)

### Phase 1: 数据持久化与训练实装 (2周)
*   **W1**: 引入 SQLite/SQLAlchemy，设计数据库表结构（Project, Image, Annotation）。
*   **W2**: 实现文件上传存储，完善 `/train` 接口对接真实训练，实现 WebSocket 日志推送。

### Phase 2: 鉴权与多模型支持 (2周)
*   **W3**: 实现 JWT 登录，前端添加路由守卫。
*   **W4**: 支持上传自定义预训练模型，支持更多 YOLO 版本 (v9, v10, v11)。

### Phase 3: 质量加固与发布 (1周)
*   **W5**: 补全单元测试，进行压力测试，编写用户手册，发布 v1.0 正式版。

### 资源建议
*   **人力**: 1 前端 + 1 后端 (当前配置)
*   **算力**: 开发环境需配备至少 6GB 显存的 NVIDIA GPU。

---

## 6. 附录 (Appendix)

### 6.1 第三方核心依赖列表
**Frontend (`package.json`)**:
*   `react`: ^19.2.0
*   `vite`: ^7.2.4
*   `konva`: ^10.0.12 / `react-konva`
*   `zustand`: ^5.0.9 (状态管理)
*   `tailwindcss`: ^3.4.17
*   `lucide-react`: 图标库

**Backend (`requirements.txt`)**:
*   `fastapi`: Web 框架
*   `uvicorn`: ASGI 服务器
*   `ultralytics`: YOLO 核心库
*   `pydantic`: 数据校验
*   `python-multipart`: 文件上传支持
*   `pillow`: 图像处理

### 6.2 已知问题跟踪 (Known Issues)
1.  **HMR 状态丢失**: 前端热更新时偶尔会导致 Canvas 状态重置（已通过 ErrorBoundary 缓解白屏）。
2.  **Mock 数据**: 项目列表、数据集列表目前为硬编码，刷新即丢失修改。
3.  **训练阻塞**: 当前训练任务在主进程的 BackgroundTasks 中运行，大量计算可能轻微影响 API 响应（需迁移至 Celery 或独立进程）。
