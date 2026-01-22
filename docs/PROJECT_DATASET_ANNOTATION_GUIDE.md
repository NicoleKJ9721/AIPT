# 项目 → 数据集 → 智能标注 联动实施指南

本指南用于说明平台在不改变业务流程（创建项目 → 导入数据集 → 智能标注 → 回写标注）的前提下，如何实现跨模块的强关联与稳定联动。

## 1. 流程不变的关键约束

- **项目是上下文中心**：数据集版本、导入文件、图片与标注必须能追溯到 `project_id`
- **数据集强关联**：创建数据集版本时必须绑定 `project_id`
- **标注强关联**：标注保存到 `image_id`，而 `image_id` 由数据集上传自动生成并关联回 `project_id/dataset_id`

## 2. 前端联动（页面与路由）

### 2.1 项目管理（Projects）
- 创建项目成功后设置全局上下文并跳转：`/datasets?project_id=<id>`
- 项目列表提供快捷入口：
  - 数据集管理：`/datasets?project_id=<id>`
  - 智能标注：`/annotate?project_id=<id>`

### 2.2 数据集模块（Datasets）
- 页面顶部提供**项目选择器**，并基于 `project_id` 过滤数据集列表：`GET /api/datasets?project_id=...`
- 创建数据集版本时自动携带 `project_id`（否则后端会返回 `400 project_id is required`）
- 提供“智能标注”按钮，携带 `project_id`（可选 `dataset_id`）跳转到标注页：
  - `/annotate?project_id=...&dataset_id=...`

### 2.3 智能标注（AnnotationStudio）
- 支持项目/数据集选择器（优先使用 URL query，其次使用全局上下文）
- 数据加载逻辑：
  - 项目图片：`GET /api/projects/{project_id}/images?dataset_id=...`
  - 图片标注：`GET /api/images/{image_id}/annotations`
  - 保存标注：`PUT /api/images/{image_id}/annotations`
- 进度面板按“已完成（annotations_count>0）/总数”统计

## 3. 后端联动（数据与存储）

### 3.1 强关联规则
- `Dataset.project_id`：数据集版本绑定项目
- 上传数据集文件时自动创建 `Image`：
  - `Image.project_id = Dataset.project_id`
  - `Image.dataset_id = dataset_id`
  - `Image.dataset_file_id = file_id`
  - `Image.source_url = /datasets/{dataset_id}/files/{file_id}/download`（供前端通过 `/api` 代理加载）

### 3.2 存储结构（本地）
- 根目录优先由 `AIPT_STORAGE_DIR` 控制
- 未设置 `AIPT_STORAGE_DIR` 时：使用 `system/settings` 的 `projects_root_dir`（默认在用户目录下，独立于源码）
- 每个项目创建时会固化 `Project.storage_root`（新项目默认取 `projects_root_dir`）
- 项目维度目录：
  - `<storage_root>/projects/<project_id>/datasets/<dataset_id>/...`
  - `<storage_root>/projects/<project_id>/exports/...`

## 4. 常见问题排查

- “创建失败”：优先检查
  - 前端是否选择了项目（是否携带 `project_id`）
  - 若后端设置了 `AIPT_API_KEY`：本机访问默认放行；非本机访问写操作需携带 `X-API-Key`
  - 查看 `logs/backend.log`
- “智能标注白屏/无内容”：通常是项目/数据集中暂无图片，标注页会显示“暂无图片”提示
- 后端启动失败：使用 `start_services.bat`，并检查 `logs/backend.log`

## 5. 下一步建议（不改变流程）

- 数据集“导入”增强：支持保留 zip 内目录结构、支持标签文件（YOLO/COCO/VOC）校验与转换
- 标注与训练联动：在训练模块允许选择 `project_id/dataset_id` 并复用项目图片/标注数据
- 鉴权升级：从 `X-User/X-API-Key` 过渡到 JWT + RBAC（不影响现有最小实现）
