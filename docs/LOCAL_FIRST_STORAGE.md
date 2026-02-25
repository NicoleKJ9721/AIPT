# 本地化运行：存储与资源管理（Local-First）

本平台默认以“纯本地”模式运行：平台源码与业务/项目文件在物理路径上分离，避免把数据集、训练产物等写进仓库目录。

## 1. 目录与配置

### 1.1 App Home（配置与运行文件）

后端会在用户目录下维护一份运行时目录（默认 Windows 为 `%LOCALAPPDATA%/AIPT`；非 Windows 为 `~/.aipt`）。

- 可通过环境变量覆盖：`AIPT_HOME_DIR`
- 配置文件：`<AIPT_HOME_DIR>/config.json`

### 1.2 项目文件根目录（projects_root_dir）

用于保存项目数据、数据集文件与导出结果，目录结构固定为：

- `<projects_root_dir>/projects/<project_id>/datasets/<dataset_id>/...`
- `<projects_root_dir>/projects/<project_id>/exports/...`

优先级：

1. `AIPT_STORAGE_DIR`（显式指定，最高优先级）
2. `system/settings.projects_root_dir`（默认在用户目录下，独立于源码）

### 1.3 资源根目录（resources_root_dir）

用于保存预训练模型、训练产物与缓存资源（独立于源码目录）：

- `<resources_root_dir>/models/...`

优先级：

1. `AIPT_RESOURCES_DIR`（可选）
2. `system/settings.resources_root_dir`

## 2. 系统设置 API

前端通过 Vite 代理访问 `/api/*`，实际转发到后端 `http://127.0.0.1:8000/*`。

### 2.1 获取当前设置

- `GET /api/system/settings`

返回 `projects_root_dir/resources_root_dir` 以及最近使用列表：

- `recent_projects_root_dirs`
- `recent_resources_root_dirs`

### 2.2 更新默认路径

- `PUT /api/system/settings`

请求体示例：

```json
{
  "projects_root_dir": "D:/AIPT_PROJECTS",
  "resources_root_dir": "D:/AIPT_RESOURCES"
}
```

后端会：

- 将路径规范化为绝对路径
- 自动维护最近使用路径（最多 10 条）
- 确保目录存在（会自动创建）

### 2.3 目录选择器（可视化）

- `POST /api/system/dialogs/select-directory`

用于在后端机器上弹出原生目录选择对话框（Windows 本地运行推荐）。

> 若运行环境无 GUI（例如 headless server），该接口会返回 `501`。

## 3. 轻量鉴权说明（可选）

若设置了环境变量 `AIPT_API_KEY`：

- **本机访问（localhost/127.0.0.1）默认放行**
- 非本机访问写接口需要请求头：`X-API-Key: <AIPT_API_KEY>`

前端可在浏览器本地存储中设置：

```js
localStorage.setItem("aipt_api_key", "<your-key>");
```

## 4. SQLite 数据库位置（源码外默认）

默认策略：

- 若存在 legacy 数据库：`backend/data/aipt.db`，继续使用它避免升级后“数据消失”的错觉
- 否则：使用 `<AIPT_HOME_DIR>/aipt.db`（独立于源码）

可通过环境变量完全覆盖：

- `AIPT_DATABASE_URL`（例如 `sqlite:///D:/AIPT/aipt.db`）

## 5. 初始资源（预训练模型）迁移策略

后端启动时会优先从 `<resources_root_dir>/models/yolo26n.pt` 加载 YOLO 权重（若存在）。

若该文件不存在且仓库根目录存在 `yolo26n.pt`：

- 会自动复制到资源目录后再加载

这样可在首次启动后实现“资源独立于源码目录”的本地化落盘。

兼容性说明：若 `yolo26n.pt` 不存在，会回退尝试 `yolov8n.pt`（若存在）。
