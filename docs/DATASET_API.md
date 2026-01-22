# 数据集模块 API 文档

默认通过前端代理访问：`/api/*` → 后端 `http://127.0.0.1:8000/*`

## 统一响应格式（JSON 接口）

除二进制下载接口外，所有数据集相关 JSON 接口统一返回：

```json
{ "code": 200, "message": "OK", "data": {} }
```

- `code`：与 HTTP 状态码一致
- `message`：简要信息
- `data`：数据体（可能为 `null`）

## 鉴权与权限（最小实现）

- `X-User`：轻量用户标识（用于 owner 校验）；不传默认为 `anonymous`
- 若设置环境变量 `AIPT_API_KEY`：
  - **本机访问（localhost/127.0.0.1）默认放行**，不强制 `X-API-Key`
  - 非本机访问需要请求头 `X-API-Key: <AIPT_API_KEY>`
- 读权限：`dataset.is_public == true` 或 `dataset.owner == X-User`
- 写权限：`dataset.owner == X-User`

## Dataset 元数据接口

### GET `/api/datasets`

列出可见数据集（public 或属于当前用户）。

Query 参数：
- `q`：模糊匹配 `name/version`
- `status`：状态过滤（如 `created` / `uploaded`）
- `owner`：owner 过滤
- `project_id`：项目过滤
- `limit`：默认 50，最大 200
- `offset`：默认 0

### POST `/api/datasets`（写）

创建数据集版本。

请求体示例：
```json
{
  "project_id": "project-uuid",
  "name": "defect-dataset",
  "version": "v1",
  "description": "line2 night shift",
  "tags": ["line=2", "shift=night"],
  "is_public": false,
  "splits": { "train": 0.7, "val": 0.2, "test": 0.1 }
}
```

说明：
- `project_id` **必填**；数据集版本与项目强关联
- `version` 可不传或传 `null`，后端会自动生成 `v1/v2...`
- `splits` 之和必须为 `1.0`

### GET `/api/datasets/{dataset_id}`

返回数据集详情（包含 `file_count/total_size_bytes` 聚合信息）。

### PATCH `/api/datasets/{dataset_id}`（写）

更新元数据字段：`name/version/description/status/tags/is_public/splits`。

### DELETE `/api/datasets/{dataset_id}`（写）

删除数据集版本（同时删除已存储文件）。

## Dataset Files 文件接口

### GET `/api/datasets/{dataset_id}/files`

列出该版本的文件列表。

### POST `/api/datasets/{dataset_id}/files`（写）

批量上传文件。

请求：`multipart/form-data`，字段名为 `files`（可重复多次）：

```bash
curl -X POST "http://127.0.0.1:8000/datasets/<id>/files" \
  -H "X-User: alice" -H "X-API-Key: <key>" \
  -F "files=@a.jpg" -F "files=@b.zip"
```

### GET `/api/datasets/{dataset_id}/files/{file_id}/download`

下载单个文件（二进制响应，**不使用统一 JSON 包装**）。

### DELETE `/api/datasets/{dataset_id}/files/{file_id}`（写）

删除单个文件（同时删除磁盘文件）。

### GET `/api/datasets/{dataset_id}/download`

打包下载整个数据集版本（zip，二进制响应，**不使用统一 JSON 包装**）。

## 存储配置

- `AIPT_STORAGE_DIR`：显式指定项目文件存储根目录（优先生效）
- 未设置 `AIPT_STORAGE_DIR` 时：后端使用 `system/settings` 中的 `projects_root_dir`（默认在用户目录下，独立于源码）
- 每个项目创建时会固化 `Project.storage_root`（新项目默认取 `projects_root_dir`），数据集文件存储在：
  - `<storage_root>/projects/<project_id>/datasets/<dataset_id>/...`

## 与智能标注联动（Project → Dataset → Image → Annotation）

数据集上传会自动生成 `Image` 记录，并将 `Image.source_url` 指向对应的数据集文件下载地址：
- `source_url`: `/datasets/{dataset_id}/files/{file_id}/download`

### GET `/api/projects/{project_id}/images`

列出项目图片（由数据集上传生成或手动创建）。

Query 参数：
- `dataset_id`：可选，按数据集过滤项目图片

### GET `/api/images/{image_id}/annotations`

获取某张图片的标注列表。

### PUT `/api/images/{image_id}/annotations`

替换某张图片的标注列表（全量覆盖）。
