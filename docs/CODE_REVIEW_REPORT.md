# 全面代码审查与优化报告（Backend/Frontend）

日期：2026-01-04  
范围：`backend/`、`frontend/` 以及启动/验证脚本与文档

## 1. 审查方法与工具

静态扫描与校验：
- 后端：`ruff`（代码质量）、`bandit`（安全扫描）、`pytest + pytest-cov`（单元/接口测试与覆盖率）
- 前端：`npm run lint`（ESLint/TS）、`npm run build`（Vite 构建）

一键复跑脚本：
- `scripts/verify.bat`
- `scripts/verify.ps1`

## 2. 高优先级问题与修复摘要

### 2.1 稳定性与资源泄漏
- 修复 Toast 订阅重复注册导致的潜在资源泄漏与 UI 异常触发（`frontend/src/components/ui/use-toast.ts`）
- 修复 Hooks 依赖不完整导致的潜在闭包问题（`frontend/src/pages/AnnotationStudio.tsx`）

### 2.2 性能与内存占用
- 数据集 ZIP 导入改为“流式落盘 + 同步计算 SHA256”，避免把 ZIP 成员整块读入内存（`backend/storage.py`、`backend/main.py`）
- 图片尺寸解析使用轻量路径读取，避免无谓的全量解码/读取（`backend/main.py`）

### 2.3 安全性与可观测性
- 清理 `except: pass` / 静默吞错：保留“最佳努力”行为，但记录 debug/exception 日志，避免问题无迹可循（`backend/db.py`、`backend/main.py`）
- WMIC 调用改为显式解析可执行路径，避免 PATH 劫持风险并满足静态扫描约束（`backend/main.py`）

### 2.4 类型安全与构建稳定
- 修复 Axios headers 类型不匹配导致的 TS 构建失败风险（`frontend/src/lib/api.ts`）
- 移除仅用于开发热更新的导出/未使用变量，消除 `react-refresh` 与 ESLint 报错（`frontend/src/components/ui/button.tsx`、`frontend/src/components/ui/badge.tsx`、`frontend/src/components/ui/select.tsx`、`frontend/src/pages/Projects.tsx`）
- 为 `webkitdirectory/directory` 目录上传属性补齐 JSX 类型（`frontend/src/types/dom.d.ts`）

## 3. 测试与质量结果

### 3.1 后端测试与覆盖率
命令：

```bash
pytest -q --cov=backend --cov-report=term-missing backend/tests
```

结果（本地）：

```text
TOTAL 2024 162 92%
42 passed
```

### 3.2 数据集模块性能基准（本地 in-process）
脚本：`backend/scripts/dataset_benchmark.py`

```bash
python backend/scripts/dataset_benchmark.py --iterations 30
```

最近一次结果（摘要）：
- failures：0
- p95：均远低于 500ms（create/list/upload/download_zip/delete）

> 说明：若环境未安装 `ultralytics`，启动时会记录“模型加载失败”日志，但不影响数据集 CRUD/上传下载与基准测试。

## 4. 仍需跟踪的改进建议（不改变现有流程）
- `backend/main.py` 体积较大（路由+业务混杂），建议按领域拆分（projects/datasets/annotations/system/hardware）以降低长期维护成本。
- 前端当前以 lint/build 为主，单元测试框架尚未引入（可评估 Vitest + React Testing Library，优先覆盖跨模块状态与路由跳转）。

