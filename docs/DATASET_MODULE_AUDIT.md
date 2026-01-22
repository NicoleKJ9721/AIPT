# 数据集模块前端功能梳理与接口映射

本文档面向 `frontend/src/pages/Datasets.tsx`，列出数据集模块所有交互元素与对应后端 API，并说明权限/鉴权的最小实现方式。

> 说明：旧版 mock 页面已移动到 `frontend/src/pages/Datasets.mock.tsx`（仅保留参考）。

## 1. 前端交互元素与后端接口映射（当前实现）

| 前端元素 | 位置 | 预期能力 | 对应接口 |
| --- | --- | --- | --- |
| 当前项目选择器 | 标题区 | 选择/切换当前项目上下文；数据集列表随项目切换 | `GET /api/projects` + `GET /api/datasets?project_id=...` |
| 新建版本按钮 | 页面右上角 | 创建数据集版本（元数据） | `POST /api/datasets` |
| 上传数据按钮 | 页面右上角 | 选择文件上传到当前选中的版本 | `POST /api/datasets/{dataset_id}/files` |
| 智能标注按钮 | 页面右上角 | 跳转到智能标注页，并携带 `project_id`（可选 `dataset_id`） | 路由：`/annotate?project_id=...&dataset_id=...` |
| 拖拽上传区域 | “上传数据”卡片 | 拖拽文件上传到当前选中的版本 | `POST /api/datasets/{dataset_id}/files` |
| 选择文件按钮 | 拖拽上传区域底部 | 打开文件选择器并上传 | `POST /api/datasets/{dataset_id}/files` |
| 刷新按钮 | 页面右上角 | 刷新数据集版本列表 | `GET /api/datasets` |
| 查询输入框 + 查询按钮 | “数据集版本”卡片右上角 | 按 `name/version` 关键字搜索 | `GET /api/datasets?q=...` |
| 数据集版本列表（点击条目） | “数据集版本”卡片 | 选择版本并加载文件列表 | `GET /api/datasets/{dataset_id}/files` |
| 版本下载按钮 | 版本条目右侧 | 打包下载该版本 zip | `GET /api/datasets/{dataset_id}/download` |
| 版本编辑按钮 | 版本条目右侧 | 更新描述/标签/划分/公开性 | `PATCH /api/datasets/{dataset_id}` |
| 版本删除按钮 | 版本条目右侧 Trash | 删除该版本（含文件） | `DELETE /api/datasets/{dataset_id}` |
| 文件列表刷新按钮 | “文件列表”卡片右上角 | 刷新文件列表 | `GET /api/datasets/{dataset_id}/files` |
| 追加上传按钮 | “文件列表”卡片右上角 | 向当前版本追加文件 | `POST /api/datasets/{dataset_id}/files` |
| 文件下载按钮 | 文件条目右侧 | 下载单文件 | `GET /api/datasets/{dataset_id}/files/{file_id}/download` |
| 文件删除按钮 | 文件条目右侧 Trash | 删除单文件 | `DELETE /api/datasets/{dataset_id}/files/{file_id}` |

## 2. 当前不可用功能标注

当前数据集页面的按钮均已接入后端 API；但**数据集版本必须绑定项目**，因此：
- 未选择项目时，页面会提示“请先选择一个项目”，并禁用“新建版本/上传数据”等依赖项目上下文的操作
- 若前端未携带 `project_id` 调用 `POST /datasets`，后端会返回 `400 project_id is required`，前端表现为“创建失败”

如仍出现“按钮无反应/白屏”等问题，优先检查：
- 后端是否启动：`GET http://127.0.0.1:8000/health`
- 前端代理是否生效：Vite `server.proxy`（`frontend/vite.config.ts`）
- 浏览器控制台网络请求是否返回 `401/403/422`

## 3. 权限与鉴权（最小实现）

后端最小策略：
- `X-User` 作为 owner 身份标识；不传默认为 `anonymous`
- 若设置环境变量 `AIPT_API_KEY`：本机访问默认放行；非本机访问写接口需要 `X-API-Key`
- 读：public 或 owner；写：owner

前端请求头来源（`frontend/src/lib/api.ts`）：
- `X-User`：从 `localStorage.aipt_user` 读取（缺省 `anonymous`）
- `X-API-Key`：从 `localStorage.aipt_api_key` 读取（可选）

可在浏览器控制台设置：
```js
localStorage.setItem("aipt_user", "alice");
localStorage.setItem("aipt_api_key", "test-key");
```
