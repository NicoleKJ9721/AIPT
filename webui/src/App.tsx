import { useEffect, useMemo, useState } from "react";

const sections = [
  {
    title: "数据导入",
    description: "批量导入与清洗，支持训练/验证/测试集划分。",
  },
  {
    title: "标注工作台",
    description: "检测、分类、分割、OBB 一体化标注与审核流程。",
  },
  {
    title: "数据增广",
    description: "可视化配置增广策略，支持快照与版本回溯。",
  },
  {
    title: "模型训练",
    description: "YOLO 模型选择、超参管理与设备调度。",
  },
  {
    title: "评估中心",
    description: "训练过程监控与模型能力多维度对比分析。",
  },
  {
    title: "模型导出",
    description: "导出 ONNX/TensorRT/OpenVINO/CoreML 等工业格式。",
  },
];

const quickActions = [
  "新建数据集",
  "启动标注任务",
  "创建训练任务",
  "查看评估报告",
  "导出部署包",
];

type DatasetSummary = {
  id: string;
  name: string;
  root: string;
  items: { id: string }[];
};

type TrainingJob = {
  id: string;
  model: string;
  status: string;
};

export default function App() {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const [datasetsResponse, jobsResponse] = await Promise.all([
          fetch("/api/datasets/"),
          fetch("/api/training/jobs"),
        ]);

        if (datasetsResponse.ok) {
          const data = await datasetsResponse.json();
          setDatasets(data);
        }

        if (jobsResponse.ok) {
          const data = await jobsResponse.json();
          setJobs(data);
        }
      } catch (err) {
        setError("无法连接后端服务，请确认 API 已启动。");
      }
    };

    load();
  }, []);

  const datasetStats = useMemo(() => {
    if (datasets.length === 0) {
      return { count: 0, items: 0 };
    }
    const items = datasets.reduce((sum, dataset) => sum + dataset.items.length, 0);
    return { count: datasets.length, items };
  }, [datasets]);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="logo">AIPT</span>
          <span className="subtitle">工业视觉平台</span>
        </div>
        <nav className="nav">
          {[
            "总览",
            "数据集",
            "标注",
            "训练",
            "评估",
            "部署",
            "系统设置",
          ].map((item) => (
            <button className="nav-item" key={item} type="button">
              {item}
            </button>
          ))}
        </nav>
        <div className="status-card">
          <h4>连接状态</h4>
          <p>{error ? "未连接" : "已连接"}</p>
          <small>{error ?? "API 状态正常"}</small>
        </div>
      </aside>

      <main className="content">
        <header className="header">
          <div>
            <h1>AI 工业检测平台控制台</h1>
            <p>覆盖数据采集、标注、训练、评估与部署的闭环流程。</p>
          </div>
          <div className="header-actions">
            <button className="primary" type="button">
              新建训练任务
            </button>
            <button className="ghost" type="button">
              导出模型
            </button>
          </div>
        </header>

        <section className="metrics">
          <div className="metric">
            <h3>数据集数量</h3>
            <p>{datasetStats.count}</p>
          </div>
          <div className="metric">
            <h3>样本总量</h3>
            <p>{datasetStats.items}</p>
          </div>
          <div className="metric">
            <h3>训练任务</h3>
            <p>{jobs.length}</p>
          </div>
          <div className="metric">
            <h3>当前模型</h3>
            <p>{jobs[0]?.model ?? "未选择"}</p>
          </div>
        </section>

        <section className="grid">
          {sections.map((section) => (
            <article className="card" key={section.title}>
              <h3>{section.title}</h3>
              <p>{section.description}</p>
              <button className="link" type="button">
                进入模块 →
              </button>
            </article>
          ))}
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>快捷操作</h2>
            <span>推荐从数据集创建开始</span>
          </div>
          <div className="panel-body">
            {quickActions.map((action) => (
              <button className="chip" key={action} type="button">
                {action}
              </button>
            ))}
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>最近数据集</h2>
            <span>来自后端 /datasets/</span>
          </div>
          <div className="panel-body list">
            {datasets.length === 0 ? (
              <p className="muted">暂无数据集，请先在数据管理中导入。</p>
            ) : (
              datasets.map((dataset) => (
                <div className="list-item" key={dataset.id}>
                  <div>
                    <strong>{dataset.name}</strong>
                    <span>{dataset.root}</span>
                  </div>
                  <span className="badge">{dataset.items.length} 张</span>
                </div>
              ))
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
