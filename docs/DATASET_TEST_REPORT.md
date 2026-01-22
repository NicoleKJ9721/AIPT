# 数据集模块测试报告

## 1. 覆盖范围

后端测试文件：
- `backend/tests/test_datasets.py`：数据集 CRUD、文件上传/下载/删除、权限与校验异常、以及项目→数据集→图片→标注的端到端联动
- `backend/tests/test_system_settings.py`：本地化运行的系统设置（projects/resources 路径）读写、鉴权与目录选择器异常分支

覆盖场景：
- 正常使用：创建 → 上传 → 列表/详情 → 下载单文件/zip → 更新 → 删除
- 边界条件：版本冲突（409）、分页/查询（q）
- 异常处理：校验失败（422）、鉴权失败（401）、权限不足（403）、资源不存在（404）

## 2. 运行方式

### 2.1 单元/接口（端到端）测试

```bash
pytest -q
```

或只跑数据集相关：

```bash
pytest -q backend/tests/test_datasets.py
```

### 2.2 覆盖率

安装依赖（一次性）：
```bash
pip install -r backend/requirements.txt
```

生成覆盖率（示例）：
```bash
pytest --cov=backend --cov-report=term-missing
```

> 注：如环境未安装 `ultralytics`，与模型相关用例会跳过；数据集模块不依赖 `ultralytics`。

### 2.3 性能基准与稳定性

脚本：`backend/scripts/dataset_benchmark.py`

```bash
python backend/scripts/dataset_benchmark.py --iterations 30
```

输出包含：create/list/upload/download_zip/delete 的 avg/p50/p95/max（ms）以及失败次数（稳定性）。

## 3. 最近一次本地运行结果（2026-01-04）

本次在 `C:\Users\28754\Desktop\AIPT` 运行结果：

### 3.1 单元/接口测试

```text
42 passed
```

> `test_main.py` 的 `/predict`、`/train` 使用 DummyDetector 覆盖，不依赖 `ultralytics`，保证纯本地可测。

### 3.2 覆盖率（pytest-cov）

命令：`pytest -q --cov=backend --cov-report=term-missing backend/tests`

```text
TOTAL 2024 162 92%
```

### 3.3 性能基准与稳定性（dataset_benchmark）

命令：`python backend/scripts/dataset_benchmark.py --iterations 30`

```text
Error loading model: ultralytics is not installed; run `pip install ultralytics` (import error: No module named 'ultralytics')
Traceback (most recent call last):
  ...
RuntimeError: ultralytics is not installed; run `pip install ultralytics` (import error: No module named 'ultralytics')
Dataset benchmark (TestClient, in-process)
iterations: 30
failures: 0
- create: avg=10.36ms p50=9.64ms p95=17.53ms max=18.06ms
- list: avg=4.80ms p50=3.80ms p95=8.50ms max=25.87ms
- upload: avg=11.10ms p50=9.75ms p95=19.89ms max=20.68ms
- download_zip: avg=5.86ms p50=5.00ms p95=9.03ms max=10.59ms
- delete: avg=10.03ms p50=9.39ms p95=13.35ms max=15.15ms
```

结论（对照交付标准）：
- 覆盖率：92%（≥ 90%）
- 响应时间：p95 远低于 500ms（在本机 TestClient 基准下）
- 稳定性：30 次迭代失败数 0
