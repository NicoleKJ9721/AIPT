import { ArrowRight, Box, CloudLightning, Cpu, GitBranch, HardDrive, LineChart, Play, RefreshCw, Square, Trash2 } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { isAxiosError } from "axios";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { toast } from "@/components/ui/use-toast";
import { cn } from "@/lib/utils";
import {
  aiService,
  dashboardService,
  datasetService,
  hardwareService,
  inferenceService,
  modelService,
  projectService,
  trainService,
  type DatasetImageStatsRecord,
  type DatasetRecord,
  type HardwareDeviceRecord,
  type InferenceStatusRecord,
  type ModelEvaluationPageRecord,
  type ProjectRecord,
  type TrainedModelRecord,
  type TrainDiagnosticsRecord,
  type TrainConfig,
  type TrainJobStatusRecord,
  type TrainMetricsRecord,
} from "@/lib/api";
import { useProjectContext } from "@/store/projectContext";

type RecommendResult = {
  batch: number;
  epochs: number;
  lr0: number;
};

type EvalParams = {
  split: "train" | "val" | "test";
  limit: number;
  conf: number;
  iou: number;
  imgsz: number;
  max_det: number;
  device: string;
  half: boolean;
  augment: boolean;
  end2end: boolean;
  classes: string;
};

const DEFAULT_EVAL_PARAMS: EvalParams = {
  split: "test",
  limit: 12,
  conf: 0.25,
  iou: 0.7,
  imgsz: 640,
  max_det: 50,
  device: "",
  half: false,
  augment: false,
  end2end: false,
  classes: "",
};

const YOLO_MODELS = [
  { value: "yolo26n", label: "YOLO26n (Nano)" },
  { value: "yolo26s", label: "YOLO26s (Small)" },
  { value: "yolo26m", label: "YOLO26m (Medium)" },
  { value: "yolo26l", label: "YOLO26l (Large)" },
  { value: "yolo26x", label: "YOLO26x (X-Large)" },
] as const;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function clampInt(value: number, min: number, max: number) {
  return Math.trunc(clamp(value, min, max));
}

function parseGb(memory: string | null | undefined): number | null {
  if (!memory) return null;
  const text = String(memory).trim();
  const gb = text.match(/([0-9]+(?:\.[0-9]+)?)\s*GB/i);
  if (gb) return Number(gb[1]);
  const mb = text.match(/([0-9]+(?:\.[0-9]+)?)\s*MB/i);
  if (mb) return Number(mb[1]) / 1024;
  return null;
}

function formatPixels(pixels: number) {
  if (!Number.isFinite(pixels) || pixels <= 0) return "0";
  if (pixels >= 1e9) return `${(pixels / 1e9).toFixed(2)}Gpx`;
  if (pixels >= 1e6) return `${(pixels / 1e6).toFixed(1)}Mpx`;
  if (pixels >= 1e3) return `${(pixels / 1e3).toFixed(1)}Kpx`;
  return String(Math.round(pixels));
}

function toSafeNumber(value: string, fallback: number) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function resolveImageUrl(sourceUrl: string | null | undefined): string {
  const url = (sourceUrl || "").trim();
  if (!url) return "";
  if (url.startsWith("http://") || url.startsWith("https://") || url.startsWith("data:")) return url;
  if (url.startsWith("/api/")) return url;
  if (url.startsWith("/")) return `/api${url}`;
  return url;
}

function colorForClassId(classId: number) {
  const n = Number.isFinite(classId) ? Math.max(0, Math.trunc(classId)) : 0;
  const hue = (n * 137.508) % 360;
  return `hsl(${hue} 80% 55%)`;
}

function recommendParams(args: {
  vramGb: number | null;
  imageCount: number;
  totalPixels: number;
  imgsz: number;
}): RecommendResult {
  const vram = args.vramGb ?? 0;

  let batch = 4;
  if (vram >= 24) batch = 64;
  else if (vram >= 16) batch = 32;
  else if (vram >= 12) batch = 24;
  else if (vram >= 8) batch = 16;
  else if (vram >= 6) batch = 8;
  else batch = 4;

  const imgsz = Number.isFinite(args.imgsz) && args.imgsz > 0 ? args.imgsz : 640;
  const areaScale = (imgsz / 640) ** 2;
  if (areaScale > 0) batch = Math.floor(batch / areaScale);
  batch = clampInt(batch, 4, 64);

  const n = Math.max(0, Math.trunc(args.imageCount || 0));
  let epochs = 100;
  if (n > 0 && n < 200) epochs = 300;
  else if (n < 1000) epochs = 200;
  else if (n < 5000) epochs = 150;
  else if (n < 20000) epochs = 100;
  else epochs = 50;

  const scaleHint = args.totalPixels > 0 ? args.totalPixels / 1e9 : 0;
  if (scaleHint >= 5) epochs = Math.max(50, Math.round(epochs * 0.7));
  else if (scaleHint >= 2) epochs = Math.max(50, Math.round(epochs * 0.85));

  epochs = clampInt(epochs, 50, 300);

  let lr0 = 0.01 * (batch / 16);
  lr0 = clamp(lr0, 0.0001, 0.01);
  lr0 = Math.round(lr0 * 100000) / 100000;

  return { batch, epochs, lr0 };
}

type MetricDef = {
  label: string;
  keys: string[];
  color: string;
  dot: string;
  yHint?: [number, number];
};

const TRAIN_METRIC_DEFS: MetricDef[] = [
  { label: "train/box_loss", keys: ["train/box_loss", "box_loss"], color: "stroke-amber-500", dot: "bg-amber-500", yHint: [0, 2] },
  { label: "train/cls_loss", keys: ["train/cls_loss", "cls_loss"], color: "stroke-amber-500", dot: "bg-amber-500", yHint: [0, 2] },
  { label: "train/dfl_loss", keys: ["train/dfl_loss", "dfl_loss"], color: "stroke-amber-500", dot: "bg-amber-500", yHint: [0, 2] },
  { label: "val/box_loss", keys: ["val/box_loss"], color: "stroke-orange-500", dot: "bg-orange-500", yHint: [0, 2] },
  { label: "val/cls_loss", keys: ["val/cls_loss"], color: "stroke-orange-500", dot: "bg-orange-500", yHint: [0, 2] },
  { label: "val/dfl_loss", keys: ["val/dfl_loss"], color: "stroke-orange-500", dot: "bg-orange-500", yHint: [0, 2] },
  { label: "precision(B)", keys: ["metrics/precision(B)", "precision"], color: "stroke-emerald-500", dot: "bg-emerald-500", yHint: [0, 1] },
  { label: "recall(B)", keys: ["metrics/recall(B)", "recall"], color: "stroke-emerald-500", dot: "bg-emerald-500", yHint: [0, 1] },
  { label: "mAP50(B)", keys: ["metrics/mAP50(B)", "map50"], color: "stroke-emerald-500", dot: "bg-emerald-500", yHint: [0, 1] },
  { label: "mAP50-95(B)", keys: ["metrics/mAP50-95(B)", "map"], color: "stroke-emerald-500", dot: "bg-emerald-500", yHint: [0, 1] },
  { label: "lr/pg0", keys: ["lr/pg0"], color: "stroke-sky-500", dot: "bg-sky-500", yHint: [0, 0.02] },
  { label: "lr/pg1", keys: ["lr/pg1"], color: "stroke-sky-500", dot: "bg-sky-500", yHint: [0, 0.02] },
];

function MetricAxisChart(props: {
  label: string;
  epochs: number[];
  values: Array<number | null> | undefined;
  expectedEpochs: number;
  color?: string;
  dot?: string;
  yHint?: [number, number];
}) {
  const values = props.values || [];
  const epochs = props.epochs || [];
  const n = Math.min(epochs.length, values.length);
  const points: Array<{ x: number; y: number }> = [];
  for (let i = 0; i < n; i++) {
    const y = values[i];
    const x = epochs[i];
    if (y === null || !Number.isFinite(y) || !Number.isFinite(x)) continue;
    points.push({ x, y });
  }

  const w = 360;
  const h = 180;
  const padL = 38;
  const padR = 10;
  const padT = 12;
  const padB = 26;

  const expectedEpochs = Math.max(2, Math.trunc(props.expectedEpochs || 2));
  const fallbackXMax = expectedEpochs - 1;

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);

  const minX = points.length ? Math.min(...xs) : 0;
  const maxX = points.length ? Math.max(...xs) : fallbackXMax;

  const hint = props.yHint ?? [0, 1];
  const rawMinY = points.length ? Math.min(...ys) : hint[0];
  const rawMaxY = points.length ? Math.max(...ys) : hint[1];
  const rawSpanY = Math.max(1e-6, rawMaxY - rawMinY);
  const minY = rawMinY - rawSpanY * 0.05;
  const maxY = rawMaxY + rawSpanY * 0.05;

  const xSpan = Math.max(1e-6, maxX - minX);
  const ySpan = Math.max(1e-6, maxY - minY);

  const toX = (x: number) => padL + ((x - minX) / xSpan) * (w - padL - padR);
  const toY = (y: number) => padT + (1 - (y - minY) / ySpan) * (h - padT - padB);

  const d =
    points.length >= 2
      ? points.map((p, i) => `${i === 0 ? "M" : "L"} ${toX(p.x).toFixed(2)} ${toY(p.y).toFixed(2)}`).join(" ")
      : null;

  const last = points.length ? points[points.length - 1]?.y : null;
  const color = props.color || "stroke-primary";
  const dotColor = props.dot || "bg-primary";

  const axisY = h - padB;
  const axisX = padL;
  const x0 = padL;
  const x1 = w - padR;
  const y0 = padT;
  const y1 = h - padB;

  return (
    <div className="rounded-xl border bg-gradient-to-b from-background to-muted/25 p-3 shadow-sm hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className={cn("h-2 w-2 rounded-full", dotColor)} />
          <div className="text-xs font-medium truncate" title={props.label}>
            {props.label}
          </div>
        </div>
        <div className="text-[11px] text-muted-foreground tabular-nums">{typeof last === "number" ? last.toFixed(4) : "-"}</div>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} className="mt-2 h-36 w-full text-muted-foreground">
        <line x1={axisX} y1={y0} x2={axisX} y2={y1} stroke="currentColor" strokeOpacity={0.35} />
        <line x1={x0} y1={axisY} x2={x1} y2={axisY} stroke="currentColor" strokeOpacity={0.35} />
        <line x1={axisX} y1={toY(minY)} x2={x1} y2={toY(minY)} stroke="currentColor" strokeOpacity={0.08} />
        <line x1={axisX} y1={toY(maxY)} x2={x1} y2={toY(maxY)} stroke="currentColor" strokeOpacity={0.08} />

        {d ? <path d={d} fill="none" className={cn("stroke-[2]", color)} /> : null}

        <text x={axisX} y={y1 + 18} fontSize={10} fill="currentColor">
          {minX}
        </text>
        <text x={x1} y={y1 + 18} fontSize={10} textAnchor="end" fill="currentColor">
          {maxX}
        </text>
        <text x={axisX - 6} y={toY(maxY) + 3} fontSize={10} textAnchor="end" fill="currentColor">
          {Number.isFinite(maxY) ? maxY.toFixed(2) : "-"}
        </text>
        <text x={axisX - 6} y={toY(minY) + 3} fontSize={10} textAnchor="end" fill="currentColor">
          {Number.isFinite(minY) ? minY.toFixed(2) : "-"}
        </text>
      </svg>
    </div>
  );
}

function pickSeries(series: Record<string, Array<number | null>>, keys: string[]) {
  for (const k of keys) {
    const values = series[k];
    if (!values) continue;
    if (values.some((x) => x !== null && Number.isFinite(x))) return values;
  }
  return undefined;
}

export default function Models() {
  const [searchParams, setSearchParams] = useSearchParams();

  const projectIdInContext = useProjectContext((s) => s.projectId);
  const projectNameInContext = useProjectContext((s) => s.projectName);
  const datasetIdInContext = useProjectContext((s) => s.datasetId);

  const setProjectContext = useProjectContext((s) => s.setProject);
  const setDatasetContext = useProjectContext((s) => s.setDataset);
  const clearDatasetContext = useProjectContext((s) => s.clearDataset);

  const projectIdFromQuery = (searchParams.get("project_id") || "").trim() || null;
  const datasetIdFromQuery = (searchParams.get("dataset_id") || "").trim() || null;

  const activeProjectId = projectIdFromQuery || projectIdInContext;
  const activeDatasetId = datasetIdFromQuery || datasetIdInContext;

  const [projects, setProjects] = useState<ProjectRecord[]>([]);
  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  const [trainedModels, setTrainedModels] = useState<TrainedModelRecord[]>([]);

  const [isLoadingProjects, setIsLoadingProjects] = useState(true);
  const [isLoadingDatasets, setIsLoadingDatasets] = useState(false);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [isLoadingStats, setIsLoadingStats] = useState(false);
  const [datasetStats, setDatasetStats] = useState<DatasetImageStatsRecord | null>(null);

  const [isDetecting, setIsDetecting] = useState(false);
  const [hardwareList, setHardwareList] = useState<HardwareDeviceRecord[]>([]);
  const [primaryDeviceId, setPrimaryDeviceId] = useState<string | null>(null);

  const [trainConfig, setTrainConfig] = useState<TrainConfig>({
    data: "coco128.yaml",
    epochs: 100,
    batch: 16,
    imgsz: 640,
    lr0: 0.01,
    model: "yolo26m",
    mode: "transfer",
    output_name: "",
    base_model_id: null,
  });
  const [isTraining, setIsTraining] = useState(false);
  const [isStoppingTraining, setIsStoppingTraining] = useState(false);
  const [isResumingTraining, setIsResumingTraining] = useState(false);
  const [trainJobId, setTrainJobId] = useState<string | null>(null);
  const [trainJob, setTrainJob] = useState<TrainJobStatusRecord | null>(null);
  const [trainMetrics, setTrainMetrics] = useState<TrainMetricsRecord | null>(null);
  const [trainDiagnostics, setTrainDiagnostics] = useState<TrainDiagnosticsRecord | null>(null);
  const [isLoadingTrainDiagnostics, setIsLoadingTrainDiagnostics] = useState(false);
  const [activeTrainHint, setActiveTrainHint] = useState<{ jobId: string; status: string } | null>(null);
  const [inferenceStatus, setInferenceStatus] = useState<InferenceStatusRecord | null>(null);

  const [isEvalOpen, setIsEvalOpen] = useState(false);
  const [evalModel, setEvalModel] = useState<TrainedModelRecord | null>(null);
  const [isEvalLoading, setIsEvalLoading] = useState(false);
  const [evalPage, setEvalPage] = useState(1);
  const [evalData, setEvalData] = useState<ModelEvaluationPageRecord | null>(null);
  const [evalParams, setEvalParams] = useState<EvalParams>({ ...DEFAULT_EVAL_PARAMS });
  const [evalDraft, setEvalDraft] = useState<EvalParams>({ ...DEFAULT_EVAL_PARAMS });

  const activeProject = useMemo(() => {
    if (!activeProjectId) return null;
    return projects.find((p) => p.id === activeProjectId) ?? null;
  }, [activeProjectId, projects]);

  const activeProjectName = activeProject?.name ?? projectNameInContext ?? activeProjectId ?? null;

  const activeDataset = useMemo(() => {
    if (!activeDatasetId) return null;
    return datasets.find((d) => d.id === activeDatasetId) ?? null;
  }, [activeDatasetId, datasets]);

  const primaryDevice = useMemo(() => {
    if (!primaryDeviceId) return null;
    return hardwareList.find((d) => d.id === primaryDeviceId) ?? null;
  }, [hardwareList, primaryDeviceId]);

  const recommended = useMemo(() => {
    if (!datasetStats) return null;
    const vramGb = primaryDevice?.type.toLowerCase().includes("gpu") ? parseGb(primaryDevice.memory) : null;
    return recommendParams({
      vramGb,
      imageCount: datasetStats.image_count,
      totalPixels: datasetStats.total_pixels,
      imgsz: trainConfig.imgsz,
    });
  }, [datasetStats, primaryDevice, trainConfig.imgsz]);

  const derivedTrainProgress = useMemo(() => {
    if (!trainJob) return null;
    if (typeof trainJob.progress === "number" && Number.isFinite(trainJob.progress)) {
      return clamp(trainJob.progress, 0, 1);
    }
    const totalEpochs = Number(trainJob.config?.epochs ?? 0);
    if (!trainMetrics || trainMetrics.epochs.length === 0 || !Number.isFinite(totalEpochs) || totalEpochs <= 0) return null;
    const lastEpoch = Math.max(...trainMetrics.epochs);
    if (!Number.isFinite(lastEpoch)) return null;
    return clamp((lastEpoch + 1) / totalEpochs, 0, 1);
  }, [trainJob, trainMetrics]);

  const inferenceLock = useMemo(() => {
    const active = inferenceStatus?.active_requests ?? 0;
    const sessions = inferenceStatus?.sessions ?? [];
    if (active > 0) {
      return {
        locked: true,
        kind: "inference" as const,
        reason: `当前有推理进行中（active_requests=${active}），请等待推理完成或在“部署推荐”模块释放推理后再训练。`,
      };
    }
    if (sessions.length > 0) {
      return {
        locked: true,
        kind: "inference" as const,
        reason: `当前存在推理会话（${sessions.length}），请先在“部署推荐”模块释放推理后再训练。`,
      };
    }
    return { locked: false, kind: null as ("inference" | null), reason: "" };
  }, [inferenceStatus]);

  const trainLock = useMemo(() => {
    const status = trainJob?.status ?? activeTrainHint?.status ?? null;
    const lockedByTrain = status === "running" || status === "queued" || status === "stopping";
    const jobId = trainJob?.id ?? activeTrainHint?.jobId ?? trainJobId ?? null;
    const trainReason = lockedByTrain ? `当前已有训练任务进行中（Job: ${jobId ?? "-"}）` : "";

    const locked = lockedByTrain || inferenceLock.locked;
    let kind: "train" | "inference" | "both" | null = null;
    let reason = "";
    if (lockedByTrain && inferenceLock.locked) {
      kind = "both";
      reason = `${trainReason}；${inferenceLock.reason}`;
    } else if (lockedByTrain) {
      kind = "train";
      reason = trainReason;
    } else if (inferenceLock.locked) {
      kind = "inference";
      reason = inferenceLock.reason;
    }
    return { locked, reason, jobId, status, kind };
  }, [activeTrainHint, inferenceLock.locked, inferenceLock.reason, trainJob, trainJobId]);

  const incrementalTrainBlockedReason = useMemo(() => {
    if ((trainConfig.mode ?? "transfer") !== "incremental") return "";
    if (trainedModels.length === 0) return "增量训练需要先生成历史模型版本";
    if (!String(trainConfig.base_model_id || "").trim()) return "请选择基础模型";
    return "";
  }, [trainConfig.base_model_id, trainConfig.mode, trainedModels.length]);

  const monitoringCharts = useMemo(() => {
    const series = trainMetrics?.series || {};
    return TRAIN_METRIC_DEFS.map((d) => ({ ...d, values: pickSeries(series, d.keys) }));
  }, [trainMetrics]);

  const loadProjects = useCallback(async () => {
    try {
      setIsLoadingProjects(true);
      const data = await projectService.list();
      setProjects(data);
    } catch (err: unknown) {
      console.error(err);
      toast({
        title: "加载项目失败",
        description: "请确认后端服务已启动（http://127.0.0.1:8000）",
        variant: "destructive",
      });
    } finally {
      setIsLoadingProjects(false);
    }
  }, []);

  const loadHardware = useCallback(async () => {
    try {
      setIsDetecting(true);
      const data = await hardwareService.list();
      setHardwareList(data);

      const gpus = data.filter((d) => d.type.toLowerCase().includes("gpu"));
      const bestGpu =
        gpus
          .map((d) => ({ d, gb: parseGb(d.memory) ?? -1 }))
          .sort((a, b) => b.gb - a.gb)[0]?.d ?? null;

      setPrimaryDeviceId((prev) => prev ?? bestGpu?.id ?? data[0]?.id ?? null);
    } catch (err) {
      console.error(err);
      toast({ title: "硬件检测失败", description: "请检查后端 /hardware 接口", variant: "destructive" });
    } finally {
      setIsDetecting(false);
    }
  }, []);

  const loadTrainDiagnostics = useCallback(async () => {
    try {
      setIsLoadingTrainDiagnostics(true);
      const diag = await trainService.diagnostics();
      setTrainDiagnostics(diag);
    } catch (err) {
      console.error(err);
      setTrainDiagnostics(null);
    } finally {
      setIsLoadingTrainDiagnostics(false);
    }
  }, []);

  const loadActiveTrainJob = useCallback(async () => {
    try {
      const summary = await dashboardService.summary();
      const lastId = (summary.training?.last_job_id || "").trim();
      const lastStatus = (summary.training?.last_status || "").trim();
      if (lastId && (lastStatus === "running" || lastStatus === "queued" || lastStatus === "stopping")) {
        setActiveTrainHint({ jobId: lastId, status: lastStatus });
        setTrainJobId((prev) => prev ?? lastId);
      } else {
        setActiveTrainHint(null);
      }
    } catch (err) {
      console.error(err);
    }
  }, []);

  const loadDatasets = useCallback(async () => {
    try {
      if (!activeProjectId) {
        setDatasets([]);
        setDatasetStats(null);
        clearDatasetContext();
        return;
      }
      setIsLoadingDatasets(true);
      const data = await datasetService.list({ project_id: activeProjectId, limit: 200 });
      setDatasets(data);

      const desired = activeDatasetId;
      if (desired && data.some((d) => d.id === desired)) return;

      const next = data[0] ?? null;
      if (!next) {
        clearDatasetContext();
        setSearchParams(
          (prev) => {
            const p = new URLSearchParams(prev);
            p.delete("dataset_id");
            return p;
          },
          { replace: true }
        );
        return;
      }

      setDatasetContext(next.id, `${next.name} ${next.version}`);
      setSearchParams(
        (prev) => {
          const p = new URLSearchParams(prev);
          if (activeProjectId) p.set("project_id", activeProjectId);
          p.set("dataset_id", next.id);
          return p;
        },
        { replace: true }
      );
    } catch (err) {
      console.error(err);
      toast({ title: "加载数据集失败", description: "请检查 /datasets 接口", variant: "destructive" });
    } finally {
      setIsLoadingDatasets(false);
    }
  }, [activeDatasetId, activeProjectId, clearDatasetContext, setDatasetContext, setSearchParams]);

  const loadStats = useCallback(async () => {
    if (!activeDatasetId) {
      setDatasetStats(null);
      return;
    }
    try {
      setIsLoadingStats(true);
      const stats = await datasetService.getImageStats(activeDatasetId);
      setDatasetStats(stats);
    } catch (err) {
      console.error(err);
      toast({ title: "加载数据集统计失败", description: "请检查 /datasets/{id}/stats", variant: "destructive" });
    } finally {
      setIsLoadingStats(false);
    }
  }, [activeDatasetId]);

  const loadTrainedModels = useCallback(async () => {
    if (!activeProjectId) {
      setTrainedModels([]);
      return;
    }
    try {
      setIsLoadingModels(true);
      const models = await modelService.listByProject(activeProjectId);
      setTrainedModels(models);
    } catch (err) {
      console.error(err);
      setTrainedModels([]);
      toast({ title: "加载模型历史失败", description: "请检查 /projects/{id}/models 接口", variant: "destructive" });
    } finally {
      setIsLoadingModels(false);
    }
  }, [activeProjectId]);

  useEffect(() => {
    void loadProjects();
    void loadHardware();
    void loadTrainDiagnostics();
    void loadActiveTrainJob();
  }, [loadActiveTrainJob, loadHardware, loadProjects, loadTrainDiagnostics]);

  useEffect(() => {
    if (projectIdFromQuery) {
      if (projectIdInContext !== projectIdFromQuery) {
        setProjectContext(projectIdFromQuery, projectNameInContext);
      }
      return;
    }
    if (projectIdInContext) {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          next.set("project_id", projectIdInContext);
          return next;
        },
        { replace: true }
      );
    }
  }, [projectIdFromQuery, projectIdInContext, projectNameInContext, setProjectContext, setSearchParams]);

  useEffect(() => {
    void loadDatasets();
  }, [loadDatasets]);

  useEffect(() => {
    void loadStats();
  }, [loadStats]);

  useEffect(() => {
    void loadTrainedModels();
  }, [loadTrainedModels]);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const s = await inferenceService.status();
        if (!cancelled) setInferenceStatus(s);
      } catch (err) {
        console.error(err);
        if (!cancelled) setInferenceStatus(null);
      }
    };
    void tick();
    const timer = window.setInterval(() => void tick(), 5000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (trainJob?.status === "completed") {
      void loadTrainedModels();
    }
  }, [loadTrainedModels, trainJob?.status]);

  useEffect(() => {
    if ((trainConfig.mode ?? "transfer") !== "incremental") return;
    const current = (trainConfig.base_model_id || "").trim();
    if (current) return;
    if (trainedModels.length === 0) return;
    setTrainConfig((prev) => ({ ...prev, base_model_id: trainedModels[0].id }));
  }, [trainConfig.base_model_id, trainConfig.mode, trainedModels]);

  const handleTrain = async () => {
    if (trainLock.locked) {
      const byInference = trainLock.kind === "inference" || trainLock.kind === "both";
      toast({
        title: byInference ? "推理占用中，暂不可训练" : "已有训练任务进行中",
        description: trainLock.reason || undefined,
        variant: byInference ? "destructive" : undefined,
      });
      return;
    }
    try {
      const summary = await dashboardService.summary();
      const lastId = (summary.training?.last_job_id || "").trim();
      const lastStatus = (summary.training?.last_status || "").trim();
      if (lastId && (lastStatus === "running" || lastStatus === "queued" || lastStatus === "stopping")) {
        setActiveTrainHint({ jobId: lastId, status: lastStatus });
        setTrainJobId((prev) => prev ?? lastId);
        toast({ title: "已有训练任务进行中", description: `当前已有训练任务进行中（Job: ${lastId}）` });
        return;
      }
    } catch (err) {
      console.error(err);
    }
    if (!activeProjectId) {
      toast({ title: "请先选择项目", variant: "destructive" });
      return;
    }
    if (!activeDatasetId) {
      toast({ title: "请先选择数据集版本", variant: "destructive" });
      return;
    }

    try {
      setIsTraining(true);
      const mode = (trainConfig.mode ?? "transfer") as "transfer" | "incremental";
      const baseModelId = (trainConfig.base_model_id || "").trim();
      if (mode === "incremental" && trainedModels.length === 0) {
        toast({ title: "暂无可用历史模型", description: "请先完成一次迁移训练生成模型版本", variant: "destructive" });
        return;
      }
      if (mode === "incremental" && !baseModelId) {
        toast({ title: "请选择基础模型", description: "增量训练需要从历史模型继续训练", variant: "destructive" });
        return;
      }

      const device = (() => {
        const id = (primaryDeviceId || "").trim();
        if (id.startsWith("gpu-")) return id.slice(4);
        if (id.startsWith("cpu")) return "cpu";
        return undefined;
      })();

      const payload: TrainConfig = {
        data: `dataset:${activeDatasetId}`,
        epochs: trainConfig.epochs,
        batch: trainConfig.batch,
        imgsz: trainConfig.imgsz,
        lr0: trainConfig.lr0,
        mode,
        output_name: (trainConfig.output_name || "").trim() || null,
        project_id: activeProjectId,
        dataset_id: activeDatasetId,
        device: device ?? null,
        ...(mode === "transfer" ? { model: trainConfig.model || "yolo26m" } : { base_model_id: baseModelId }),
      };
      const response = await aiService.train(payload);
      toast({ title: "训练已启动", description: response.message });
      if (response.job_id) {
        setTrainJobId(response.job_id);
        setTrainJob(null);
        setTrainMetrics(null);
      }
    } catch (error: unknown) {
      console.error("Training Error:", error);
      let detail = "请检查后端服务";
      if (isAxiosError(error)) {
        detail = error.response?.data?.message || error.response?.data?.detail || error.message;
      } else if (error instanceof Error) {
        detail = error.message;
      }
      toast({ title: "启动训练失败", description: detail, variant: "destructive" });
    } finally {
      setIsTraining(false);
    }
  };

  const handleStopTraining = useCallback(async () => {
    const jobId = trainLock.jobId;
    if (!jobId) return;
    const ok = window.confirm("确认结束当前训练任务？（会在当前 epoch 结束后停止，之后可继续训练）");
    if (!ok) return;

    try {
      setIsStoppingTraining(true);
      setTrainJobId((prev) => prev ?? jobId);
      const updated = await trainService.stopJob(jobId);
      setTrainJob(updated);
      toast({ title: "已请求停止训练", description: `Job: ${jobId}` });
    } catch (err: unknown) {
      console.error(err);
      let detail = "请检查后端服务";
      if (isAxiosError(err)) {
        detail = err.response?.data?.message || err.response?.data?.detail || err.message;
      } else if (err instanceof Error) {
        detail = err.message;
      }
      toast({ title: "停止训练失败", description: detail, variant: "destructive" });
    } finally {
      setIsStoppingTraining(false);
    }
  }, [trainLock.jobId]);

  const handleResumeTraining = useCallback(async () => {
    const jobId = trainLock.jobId ?? trainJobId;
    if (!jobId) return;
    try {
      setIsResumingTraining(true);
      setTrainJobId(jobId);
      setTrainJob(null);
      setTrainMetrics(null);
      const updated = await trainService.resumeJob(jobId);
      setTrainJob(updated);
      toast({ title: "继续训练已启动", description: `Job: ${jobId}` });
    } catch (err: unknown) {
      console.error(err);
      let detail = "请检查后端服务";
      if (isAxiosError(err)) {
        detail = err.response?.data?.message || err.response?.data?.detail || err.message;
      } else if (err instanceof Error) {
        detail = err.message;
      }
      toast({ title: "继续训练失败", description: detail, variant: "destructive" });
    } finally {
      setIsResumingTraining(false);
    }
  }, [trainJobId, trainLock.jobId]);

  const handleDeleteModel = useCallback(
    async (model: TrainedModelRecord) => {
      const ok = window.confirm(`确认删除模型“${model.name}”？该操作不可撤销`);
      if (!ok) return;
      try {
        await modelService.delete(model.id);
        toast({ title: "已删除模型", description: model.name });
        await loadTrainedModels();
      } catch (err) {
        console.error(err);
        toast({ title: "删除失败", description: "请检查 API Key / 后端日志", variant: "destructive" });
      }
    },
    [loadTrainedModels]
  );

  const openEvaluation = useCallback((model: TrainedModelRecord) => {
    setEvalModel(model);
    setEvalPage(1);
    setEvalData(null);
    setEvalDraft({ ...DEFAULT_EVAL_PARAMS });
    setEvalParams({ ...DEFAULT_EVAL_PARAMS });
    setIsEvalOpen(true);
  }, []);

  const closeEvaluation = useCallback(() => {
    setIsEvalOpen(false);
    setEvalModel(null);
    setEvalData(null);
    setIsEvalLoading(false);
    setEvalPage(1);
    setEvalDraft({ ...DEFAULT_EVAL_PARAMS });
    setEvalParams({ ...DEFAULT_EVAL_PARAMS });
  }, []);

  const applyEvalParams = useCallback(() => {
    setEvalParams({
      ...evalDraft,
      split: evalDraft.split,
      limit: clampInt(evalDraft.limit, 1, 200),
      conf: clamp(evalDraft.conf, 0, 1),
      iou: clamp(evalDraft.iou, 0, 1),
      imgsz: clampInt(evalDraft.imgsz, 32, 8192),
      max_det: clampInt(evalDraft.max_det, 1, 300),
      device: (evalDraft.device || "").trim(),
      classes: (evalDraft.classes || "").trim(),
    });
    setEvalPage(1);
    setEvalData(null);
  }, [evalDraft]);

  useEffect(() => {
    if (!isEvalOpen || !evalModel) return;

    const datasetId = (evalModel.dataset_id || "").trim();
    if (!datasetId) {
      setEvalData(null);
      return;
    }

    let cancelled = false;
    void (async () => {
      try {
        setIsEvalLoading(true);
        const data = await modelService.evaluate(evalModel.id, {
          split: evalParams.split,
          page: evalPage,
          limit: evalParams.limit,
          conf: evalParams.conf,
          iou: evalParams.iou,
          imgsz: evalParams.imgsz,
          max_det: evalParams.max_det,
          device: evalParams.device || undefined,
          half: evalParams.half,
          augment: evalParams.augment,
          end2end: evalParams.end2end,
          classes: evalParams.classes || undefined,
        });
        if (cancelled) return;
        setEvalData(data);
      } catch (err) {
        console.error(err);
        if (!cancelled) {
          toast({ title: "加载模型性能评估失败", description: "请检查后端 /models/{id}/evaluation 接口", variant: "destructive" });
        }
      } finally {
        if (!cancelled) setIsEvalLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [evalModel, evalPage, evalParams, isEvalOpen]);

  useEffect(() => {
    if (!trainJobId) return;

    let cancelled = false;
    let timer: number | null = null;

    const poll = async () => {
      try {
        const [job, metrics] = await Promise.all([trainService.getJob(trainJobId), trainService.getMetrics(trainJobId)]);
        if (cancelled) return;

        setTrainJob(job);
        setTrainMetrics(metrics);
        if (job.status === "running" || job.status === "queued" || job.status === "stopping") {
          setActiveTrainHint({ jobId: job.id, status: job.status });
        } else {
          setActiveTrainHint(null);
        }

        if (job.status === "completed" || job.status === "failed" || job.status === "stopped") {
          if (timer !== null) {
            window.clearInterval(timer);
            timer = null;
          }
        }
      } catch (err) {
        console.error(err);
      }
    };

    void poll();
    timer = window.setInterval(() => {
      void poll();
    }, 1000);

    return () => {
      cancelled = true;
      if (timer !== null) window.clearInterval(timer);
    };
  }, [trainJobId]);

  useEffect(() => {
    if (!trainJob || trainJob.status !== "running") return;
    let cancelled = false;
    const timer = window.setInterval(() => {
      void (async () => {
        try {
          const diag = await trainService.diagnostics();
          if (!cancelled) setTrainDiagnostics(diag);
        } catch (err) {
          console.error(err);
        }
      })();
    }, 3000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [trainJob]);

  return (
    <div className="space-y-6">
      {isEvalOpen && evalModel ? (
        <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm flex items-center justify-center p-4">
          <Card className="w-[95vw] h-[92vh] max-w-none shadow-2xl flex flex-col">
            <CardHeader className="border-b py-4">
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <CardTitle className="flex items-center gap-2">
                    <LineChart className="w-5 h-5 text-primary" />
                    模型性能评估
                  </CardTitle>
                  <CardDescription className="mt-1">
                    使用测试集展示推理效果 · <span className="font-medium">{evalModel.name}</span>
                  </CardDescription>
                  {evalData?.note ? <div className="text-xs text-muted-foreground mt-1">提示：{evalData.note}</div> : null}
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={isEvalLoading || (evalData?.page ?? evalPage) <= 1}
                    onClick={() => setEvalPage((p) => Math.max(1, p - 1))}
                  >
                    <ArrowRight className="w-4 h-4 rotate-180" />
                  </Button>
                  <div className="text-xs tabular-nums text-muted-foreground min-w-[84px] text-center">
                    {evalData ? `${evalData.page} / ${Math.max(1, Math.ceil(evalData.total / evalData.limit))}` : `${evalPage} / -`}
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={
                      isEvalLoading ||
                      !evalData ||
                      evalData.page >= Math.max(1, Math.ceil(evalData.total / evalData.limit))
                    }
                    onClick={() => {
                      if (!evalData) return;
                      const totalPages = Math.max(1, Math.ceil(evalData.total / evalData.limit));
                      setEvalPage((p) => Math.min(totalPages, p + 1));
                    }}
                  >
                    <ArrowRight className="w-4 h-4" />
                  </Button>
                  <Button variant="ghost" size="sm" onClick={closeEvaluation}>
                    X
                  </Button>
                </div>
              </div>

              <div className="mt-4 flex flex-wrap items-end gap-3">
                <div className="space-y-1">
                  <div className="text-[11px] text-muted-foreground">Split</div>
                  <select
                    className="h-8 rounded-md border border-input bg-background px-2 text-xs"
                    value={evalDraft.split}
                    onChange={(e) => setEvalDraft((p) => ({ ...p, split: e.target.value as EvalParams["split"] }))}
                  >
                    <option value="train">train</option>
                    <option value="val">val</option>
                    <option value="test">test</option>
                  </select>
                </div>

                <div className="space-y-1">
                  <div className="text-[11px] text-muted-foreground">每页</div>
                  <select
                    className="h-8 rounded-md border border-input bg-background px-2 text-xs"
                    value={evalDraft.limit}
                    onChange={(e) => setEvalDraft((p) => ({ ...p, limit: toSafeNumber(e.target.value, p.limit) }))}
                  >
                    {[12, 24, 36, 48].map((n) => (
                      <option key={n} value={n}>
                        {n}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="space-y-1">
                  <div className="text-[11px] text-muted-foreground">conf</div>
                  <Input
                    className="h-8 w-24 text-xs"
                    type="number"
                    step="0.01"
                    min={0}
                    max={1}
                    value={evalDraft.conf}
                    onChange={(e) => setEvalDraft((p) => ({ ...p, conf: toSafeNumber(e.target.value, p.conf) }))}
                  />
                </div>

                <div className="space-y-1">
                  <div className="text-[11px] text-muted-foreground">iou</div>
                  <Input
                    className="h-8 w-24 text-xs"
                    type="number"
                    step="0.01"
                    min={0}
                    max={1}
                    value={evalDraft.iou}
                    onChange={(e) => setEvalDraft((p) => ({ ...p, iou: toSafeNumber(e.target.value, p.iou) }))}
                  />
                </div>

                <div className="space-y-1">
                  <div className="text-[11px] text-muted-foreground">imgsz</div>
                  <Input
                    className="h-8 w-24 text-xs"
                    type="number"
                    min={32}
                    value={evalDraft.imgsz}
                    onChange={(e) => setEvalDraft((p) => ({ ...p, imgsz: toSafeNumber(e.target.value, p.imgsz) }))}
                  />
                </div>

                <div className="space-y-1">
                  <div className="text-[11px] text-muted-foreground">max_det</div>
                  <Input
                    className="h-8 w-24 text-xs"
                    type="number"
                    min={1}
                    max={300}
                    value={evalDraft.max_det}
                    onChange={(e) => setEvalDraft((p) => ({ ...p, max_det: toSafeNumber(e.target.value, p.max_det) }))}
                  />
                </div>

                <div className="space-y-1">
                  <div className="text-[11px] text-muted-foreground">device</div>
                  <Input
                    className="h-8 w-36 text-xs"
                    placeholder="cpu / 0 / cuda:0"
                    value={evalDraft.device}
                    onChange={(e) => setEvalDraft((p) => ({ ...p, device: e.target.value }))}
                  />
                </div>

                <div className="space-y-1">
                  <div className="text-[11px] text-muted-foreground">classes</div>
                  <Input
                    className="h-8 w-32 text-xs"
                    placeholder="0,1,2 (可选)"
                    value={evalDraft.classes}
                    onChange={(e) => setEvalDraft((p) => ({ ...p, classes: e.target.value }))}
                  />
                </div>

                <div className="space-y-1">
                  <div className="text-[11px] text-muted-foreground">head</div>
                  <select
                    className="h-8 rounded-md border border-input bg-background px-2 text-xs"
                    value={evalDraft.end2end ? "end2end" : "precise"}
                    onChange={(e) => setEvalDraft((p) => ({ ...p, end2end: e.target.value === "end2end" }))}
                  >
                    <option value="end2end">端到端（更快）</option>
                    <option value="precise">高精度（end2end=false）</option>
                  </select>
                </div>

                <label className="flex items-center gap-2 text-xs text-muted-foreground select-none">
                  <input
                    type="checkbox"
                    className="h-4 w-4"
                    checked={evalDraft.half}
                    onChange={(e) => setEvalDraft((p) => ({ ...p, half: e.target.checked }))}
                  />
                  half
                </label>

                <label className="flex items-center gap-2 text-xs text-muted-foreground select-none">
                  <input
                    type="checkbox"
                    className="h-4 w-4"
                    checked={evalDraft.augment}
                    onChange={(e) => setEvalDraft((p) => ({ ...p, augment: e.target.checked }))}
                  />
                  augment
                </label>

                <Button
                  size="sm"
                  className="h-8"
                  disabled={isEvalLoading}
                  onClick={applyEvalParams}
                >
                  应用参数
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  className="h-8"
                  disabled={isEvalLoading}
                  onClick={() => setEvalDraft({ ...DEFAULT_EVAL_PARAMS })}
                >
                  重置
                </Button>
              </div>
            </CardHeader>
            <CardContent className="flex-1 min-h-0 overflow-y-auto p-4">
              {(() => {
                const datasetId = (evalModel.dataset_id || "").trim();
                if (!datasetId) {
                  return <div className="h-full flex items-center justify-center text-sm text-muted-foreground">该模型未绑定数据集，无法评估。</div>;
                }
                if (isEvalLoading && !evalData) {
                  return <div className="h-full flex items-center justify-center text-sm text-muted-foreground">加载中...</div>;
                }
                if (!evalData) {
                  return <div className="h-full flex items-center justify-center text-sm text-muted-foreground">暂无数据</div>;
                }
                if (!evalData.items.length) {
                  return <div className="h-full flex items-center justify-center text-sm text-muted-foreground">测试集暂无可展示的图片</div>;
                }

                return (
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                    {evalData.items.map((it) => {
                      const img = it.image;
                      const url = resolveImageUrl(img.source_url);
                      const imgW = Math.max(1, img.width ?? 1);
                      const imgH = Math.max(1, img.height ?? 1);
                      const dets = (it.detections || []).slice().sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0));
                      return (
                        <div
                          key={img.id}
                          className="group relative rounded-lg overflow-hidden border bg-black shadow-sm hover:shadow-md transition-shadow"
                        >
                          <div className="relative aspect-square">
                            {url ? (
                              <svg viewBox={`0 0 ${imgW} ${imgH}`} className="absolute inset-0 h-full w-full">
                                <image
                                  href={url}
                                  x="0"
                                  y="0"
                                  width={imgW}
                                  height={imgH}
                                  preserveAspectRatio="xMidYMid meet"
                                  imageRendering="optimizeQuality"
                                />
                                {dets.map((d, idx) => {
                                  const [x1, y1, x2, y2] = d.bbox;
                                  const w = Math.max(0, x2 - x1);
                                  const h = Math.max(0, y2 - y1);
                                  const color = colorForClassId(d.class_id);
                                  const label = `${d.class_name} ${Number(d.confidence).toFixed(2)}`;
                                  const fontSize = Math.max(12, Math.round(Math.min(imgW, imgH) * 0.03));
                                  const pad = Math.max(2, Math.round(fontSize * 0.2));
                                  const labelH = fontSize + pad * 2;
                                  const lx = clamp(x1, 0, Math.max(0, imgW - 1));
                                  const maxLabelY = Math.max(0, imgH - labelH);
                                  const ly = clamp(y1 - labelH, 0, maxLabelY);
                                  const maxLabelW = Math.max(0, imgW - lx);
                                  const estW = label.length * fontSize * 0.55 + pad * 2;
                                  const lw = Math.min(maxLabelW, estW);
                                  return (
                                    <g key={`${img.id}-pred-${idx}`}>
                                      <rect
                                        x={x1}
                                        y={y1}
                                        width={w}
                                        height={h}
                                        fill="none"
                                        stroke={color}
                                        strokeWidth={2}
                                        strokeOpacity={0.95}
                                        vectorEffect="non-scaling-stroke"
                                      />
                                      {lw > 0 ? (
                                        <>
                                          <rect x={lx} y={ly} width={lw} height={labelH} fill={color} fillOpacity={0.85} />
                                          <text
                                            x={lx + pad}
                                            y={ly + pad}
                                            fontSize={fontSize}
                                            fill="white"
                                            dominantBaseline="hanging"
                                            style={{ userSelect: "none" }}
                                          >
                                            {label}
                                          </text>
                                        </>
                                      ) : null}
                                    </g>
                                  );
                                })}
                              </svg>
                            ) : (
                              <div className="absolute inset-0 flex items-center justify-center text-xs text-white/70 bg-black">无预览</div>
                            )}

                            <div className="absolute top-2 left-2 right-2 flex items-center justify-between gap-2">
                              <div className="truncate text-[11px] text-white/90 bg-black/60 px-2 py-1 rounded backdrop-blur-sm">
                                {img.filename}
                              </div>
                              <div className="text-[11px] text-white/80 bg-black/50 px-2 py-1 rounded tabular-nums">
                                {dets.length ? `${dets.length} det` : "0 det"}
                              </div>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                );
              })()}
            </CardContent>
          </Card>
        </div>
      ) : null}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">模型训练与部署</h1>
          <p className="text-muted-foreground mt-2">硬件检测、参数推荐与本地训练配置</p>
        </div>
        <div className="flex items-center gap-2">
          {trainLock.kind === "train" || trainLock.kind === "both" ? (
            <Badge variant="outline" className="hidden sm:inline-flex">
              训练中
            </Badge>
          ) : null}
          {trainLock.kind === "inference" || trainLock.kind === "both" ? (
            <Badge variant="outline" className="hidden sm:inline-flex">
              推理占用
            </Badge>
          ) : null}
          {trainLock.jobId && trainLock.status === "stopped" ? (
            <Button
              variant="outline"
              className="gap-2"
              onClick={() => void handleResumeTraining()}
              disabled={isResumingTraining || trainLock.kind === "inference" || trainLock.kind === "both"}
            >
              {isResumingTraining ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
              继续训练
            </Button>
          ) : null}
          {trainLock.jobId && (trainLock.status === "running" || trainLock.status === "queued" || trainLock.status === "stopping") ? (
            <Button
              variant="destructive"
              className="gap-2"
              onClick={() => void handleStopTraining()}
              disabled={isStoppingTraining || trainLock.status === "stopping"}
              title={trainLock.status === "stopping" ? "正在停止…" : undefined}
            >
              {isStoppingTraining ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Square className="w-4 h-4" />}
              结束训练
            </Button>
          ) : null}
          <Button
            className="gap-2"
            onClick={handleTrain}
            disabled={isTraining || trainLock.locked || !!incrementalTrainBlockedReason}
            title={trainLock.locked ? trainLock.reason : incrementalTrainBlockedReason || undefined}
          >
            {isTraining ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
          {isTraining ? "启动中..." : "开始训练"}
          </Button>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <HardDrive className="w-5 h-5 text-primary" />
            训练上下文
          </CardTitle>
          <CardDescription>选择项目与数据集版本，用于参数推荐与训练绑定</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-3">
          <div className="space-y-2">
            <div className="text-sm font-medium">项目</div>
            <select
              className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              value={activeProjectId ?? ""}
              disabled={isLoadingProjects || projects.length === 0}
              onChange={(e) => {
                const nextId = (e.target.value || "").trim();
                if (!nextId) return;
                const found = projects.find((p) => p.id === nextId) ?? null;
                setProjectContext(nextId, found?.name ?? null);
                clearDatasetContext();
                setSearchParams((prev) => {
                  const next = new URLSearchParams(prev);
                  next.set("project_id", nextId);
                  next.delete("dataset_id");
                  return next;
                });
              }}
            >
              <option value="" disabled>
                {isLoadingProjects ? "加载中..." : projects.length ? "请选择项目" : "暂无项目"}
              </option>
              {projects.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
            {activeProjectName ? <div className="text-xs text-muted-foreground truncate">ID: {activeProjectId}</div> : null}
          </div>

          <div className="space-y-2">
            <div className="text-sm font-medium">数据集版本</div>
            <select
              className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              value={activeDatasetId ?? ""}
              disabled={!activeProjectId || isLoadingDatasets || datasets.length === 0}
              onChange={(e) => {
                const nextId = (e.target.value || "").trim();
                if (!nextId) return;
                const found = datasets.find((d) => d.id === nextId) ?? null;
                setDatasetContext(nextId, found ? `${found.name} ${found.version}` : null);
                setSearchParams((prev) => {
                  const next = new URLSearchParams(prev);
                  if (activeProjectId) next.set("project_id", activeProjectId);
                  next.set("dataset_id", nextId);
                  return next;
                });
              }}
            >
              <option value="" disabled>
                {!activeProjectId ? "请先选择项目" : isLoadingDatasets ? "加载中..." : datasets.length ? "请选择数据集" : "暂无数据集"}
              </option>
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} {d.version}
                </option>
              ))}
            </select>
            {activeDataset ? <div className="text-xs text-muted-foreground truncate">ID: {activeDataset.id}</div> : null}
          </div>

          <div className="space-y-2">
            <div className="text-sm font-medium">数据集规模</div>
            {isLoadingStats ? (
              <div className="text-sm text-muted-foreground flex items-center gap-2">
                <RefreshCw className="w-4 h-4 animate-spin" /> 统计中...
              </div>
            ) : datasetStats ? (
              <div className="space-y-1 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">图片数</span>
                  <span className="font-medium">{datasetStats.image_count}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">平均分辨率</span>
                  <span className="font-medium">
                    {datasetStats.avg_width && datasetStats.avg_height ? `${datasetStats.avg_width}×${datasetStats.avg_height}` : "-"}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">总像素</span>
                  <span className="font-medium">{formatPixels(datasetStats.total_pixels)}</span>
                </div>
              </div>
            ) : (
              <div className="text-sm text-muted-foreground">未选择数据集</div>
            )}
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between gap-4">
              <CardTitle className="flex items-center gap-2">
                <LineChart className="w-5 h-5 text-primary" />
                训练状态
              </CardTitle>
              <Button variant="outline" size="sm" onClick={loadTrainDiagnostics} disabled={isLoadingTrainDiagnostics} className="gap-2">
                <RefreshCw className={cn("w-4 h-4", isLoadingTrainDiagnostics ? "animate-spin" : "")} />
                刷新环境
              </Button>
            </div>
            <CardDescription>绿色=可运行，红色=环境异常；训练过程曲线在下方展示</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <div className="text-sm font-medium">训练环境</div>
              <div className="flex items-center gap-2">
                <span
                  className={cn(
                    "inline-block h-2.5 w-2.5 rounded-full",
                    trainDiagnostics?.cuda_available ? "bg-emerald-500" : "bg-red-500"
                  )}
                />
                <span className="font-medium">{trainDiagnostics?.cuda_available ? "可运行" : "环境异常"}</span>
              </div>
              <div className="text-sm text-muted-foreground">
                {trainDiagnostics?.cuda_available
                  ? "已检测到可用GPU，可开始训练。"
                  : "未检测到可用GPU或依赖不完整，请点击“刷新环境”或检查驱动/环境。"}
              </div>
              <details className="text-sm text-muted-foreground">
                <summary className="cursor-pointer select-none">查看详细诊断</summary>
                {trainDiagnostics ? (
                  <div className="mt-2 space-y-1 text-sm">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-muted-foreground">Python</span>
                      <span className="font-medium truncate" title={trainDiagnostics.python_executable}>
                        {trainDiagnostics.python_version}
                      </span>
                    </div>
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-muted-foreground">环境</span>
                      <span className="font-medium truncate" title={trainDiagnostics.conda_prefix ?? ""}>
                        {trainDiagnostics.conda_env ?? "-"}
                      </span>
                    </div>
                    {trainDiagnostics.nvidia_smi && trainDiagnostics.nvidia_smi.length > 0 ? (
                      <div className="pt-2">
                        <div className="text-xs text-muted-foreground mb-1">GPU（利用率/显存）</div>
                        <div className="space-y-1 text-xs">
                          {trainDiagnostics.nvidia_smi.map((row, idx) => (
                            <div key={`${row.name}-${idx}`} className="flex items-center justify-between gap-2">
                              <span className="truncate" title={row.name}>
                                {row.name}
                              </span>
                              <span className="tabular-nums">
                                {row.utilization_gpu_pct ?? "-"}% · {row.memory_used_mb ?? "-"} / {row.memory_total_mb ?? "-"} MB ·{" "}
                                {row.temperature_c ?? "-"}°C
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div className="text-xs text-muted-foreground pt-2">未检测到GPU信息</div>
                    )}
                  </div>
                ) : (
                  <div className="mt-2 text-sm text-muted-foreground">暂无诊断信息</div>
                )}
              </details>
            </div>

            <div className="space-y-2">
              <div className="text-sm font-medium">训练任务</div>
              {trainJobId ? (
                <div className="space-y-2">
                  <div className="text-xs text-muted-foreground truncate">Job: {trainJobId}</div>
                  <div className="flex items-center gap-2 flex-wrap">
                    <Badge variant="outline">{trainJob?.status ?? "queued"}</Badge>
                    {typeof derivedTrainProgress === "number" ? (
                      <Badge variant="secondary">{Math.round(derivedTrainProgress * 100)}%</Badge>
                    ) : null}
                    {trainJob?.error ? <Badge variant="destructive">FAILED</Badge> : null}
                  </div>
                  {trainJob?.message ? <div className="text-xs text-muted-foreground">{trainJob.message}</div> : null}
                  {trainJob?.error ? <div className="text-xs text-destructive break-words">{trainJob.error}</div> : null}
                  {typeof derivedTrainProgress === "number" ? (
                    <div className="pt-2">
                      <div className="h-2 w-full rounded bg-muted overflow-hidden">
                        <div
                          className="h-2 bg-primary/80"
                          style={{ width: `${Math.round(derivedTrainProgress * 100)}%` }}
                        />
                      </div>
                    </div>
                  ) : null}
                </div>
              ) : (
                <div className="text-sm text-muted-foreground">尚未启动训练</div>
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex justify-between items-center">
              <CardTitle className="flex items-center gap-2">
                <Cpu className="w-5 h-5 text-primary" /> 硬件资源配置
              </CardTitle>
              <Button variant="outline" size="sm" onClick={loadHardware} disabled={isDetecting} className="gap-2">
                <RefreshCw className={cn("w-4 h-4", isDetecting ? "animate-spin" : "")} />
                {isDetecting ? "检测中..." : "重新检测"}
              </Button>
            </div>
            <CardDescription>仅展示 CPU 与独显（NVIDIA CUDA）</CardDescription>
          </CardHeader>
          <CardContent>
            {hardwareList.length === 0 ? (
              <div className="text-center py-8 bg-muted/20 rounded-lg border border-dashed text-muted-foreground">
                暂无硬件信息，请点击“重新检测”
              </div>
            ) : (
              <div className="space-y-2">
                {hardwareList.map((hw) => {
                  const isSelected = primaryDeviceId === hw.id;
                  const vramGb = hw.type.toLowerCase().includes("gpu") ? parseGb(hw.memory) : null;
                  return (
                    <label
                      key={hw.id}
                      className={cn(
                        "flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors",
                        isSelected ? "border-primary/40 bg-primary/5" : "hover:bg-muted/40"
                      )}
                    >
                      <input
                        type="radio"
                        name="primary_device"
                        className="mt-1"
                        checked={isSelected}
                        onChange={() => setPrimaryDeviceId(hw.id)}
                      />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="font-medium truncate">{hw.name}</span>
                          <Badge variant="outline">{hw.type}</Badge>
                          {vramGb ? <Badge variant="secondary">{vramGb}GB</Badge> : null}
                        </div>
                        <div className="text-xs text-muted-foreground mt-1 flex flex-wrap gap-x-2 gap-y-1">
                          {hw.vendor ? <span>{hw.vendor}</span> : null}
                          {hw.compute_capability ? <span>CC {hw.compute_capability}</span> : null}
                          {typeof hw.cores === "number" ? <span>{hw.cores} Cores</span> : null}
                          {hw.status ? <span>{hw.status}</span> : null}
                        </div>
                      </div>
                    </label>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>

        <Card id="train-params">
          <CardHeader>
            <div className="flex items-center justify-between gap-4">
              <div>
                <CardTitle>训练参数</CardTitle>
                <CardDescription>默认 YOLO26m；推荐参数基于硬件与数据集规模</CardDescription>
              </div>
              <Button
                variant="outline"
                disabled={!recommended}
                onClick={() => {
                  if (!recommended) return;
                  setTrainConfig((prev) => ({
                    ...prev,
                    batch: recommended.batch,
                    epochs: recommended.epochs,
                    lr0: recommended.lr0,
                  }));
                  toast({ title: "已应用推荐参数" });
                }}
              >
                应用推荐
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <form className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <label className="text-sm font-medium">训练类型</label>
                  <div className="flex flex-wrap gap-2">
                    <Button
                      type="button"
                      variant={(trainConfig.mode ?? "transfer") === "transfer" ? "default" : "outline"}
                      onClick={() =>
                        setTrainConfig((prev) => ({
                          ...prev,
                          mode: "transfer",
                          base_model_id: null,
                        }))
                      }
                    >
                      迁移训练
                    </Button>
                    <Button
                      type="button"
                      variant={(trainConfig.mode ?? "transfer") === "incremental" ? "default" : "outline"}
                      onClick={() =>
                        setTrainConfig((prev) => ({
                          ...prev,
                          mode: "incremental",
                          base_model_id: prev.base_model_id ?? (trainedModels[0]?.id ?? null),
                        }))
                      }
                    >
                      增量训练
                    </Button>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    迁移训练：从 YOLO26 预训练开始；增量训练：从历史项目模型继续训练（生成新版本）。
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium">输出模型名称（可选）</label>
                  <Input
                    value={trainConfig.output_name ?? ""}
                    onChange={(e) => setTrainConfig((prev) => ({ ...prev, output_name: e.target.value }))}
                    placeholder="留空则自动命名"
                  />
                  <div className="text-xs text-muted-foreground">留空将自动命名，不会覆盖旧版本。</div>
                </div>
              </div>

              {(trainConfig.mode ?? "transfer") === "transfer" ? (
                <div className="space-y-2">
                  <label className="text-sm font-medium">预训练模型（YOLO26）</label>
                  <select
                    className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                    value={trainConfig.model ?? "yolo26m"}
                    onChange={(e) => setTrainConfig({ ...trainConfig, model: e.target.value })}
                  >
                    {YOLO_MODELS.map((m) => (
                      <option key={m.value} value={m.value}>
                        {m.label}
                      </option>
                    ))}
                  </select>
                </div>
              ) : (
                <div className="space-y-2">
                  <label className="text-sm font-medium">基础模型（版本历史）</label>
                  <select
                    className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                    value={trainConfig.base_model_id ?? ""}
                    onChange={(e) => setTrainConfig({ ...trainConfig, base_model_id: e.target.value || null })}
                    disabled={trainedModels.length === 0}
                  >
                    <option value="">
                      {trainedModels.length === 0 ? "暂无历史模型（请先完成一次迁移训练）" : "请选择基础模型"}
                    </option>
                    {trainedModels.map((m) => {
                      const dt = new Date(m.created_at);
                      const created = Number.isFinite(dt.getTime()) ? dt.toLocaleString() : m.created_at;
                      return (
                        <option key={m.id} value={m.id}>
                          {m.name} · {m.base_model} · {created}
                        </option>
                      );
                    })}
                  </select>
                  <div className="text-xs text-muted-foreground">
                    增量训练将从所选模型继续训练，不提供 YOLO26 预训练选择。
                  </div>
                </div>
              )}

              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Epochs</label>
                  <Input
                    type="number"
                    min={1}
                    value={trainConfig.epochs}
                    onChange={(e) => setTrainConfig({ ...trainConfig, epochs: toSafeNumber(e.target.value, trainConfig.epochs) })}
                  />
                  {recommended ? <div className="text-xs text-muted-foreground">推荐：{recommended.epochs}</div> : null}
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Batch Size</label>
                  <Input
                    type="number"
                    min={1}
                    value={trainConfig.batch}
                    onChange={(e) => setTrainConfig({ ...trainConfig, batch: toSafeNumber(e.target.value, trainConfig.batch) })}
                  />
                  {recommended ? <div className="text-xs text-muted-foreground">推荐：{recommended.batch}</div> : null}
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Image Size</label>
                  <Input
                    type="number"
                    min={64}
                    value={trainConfig.imgsz}
                    onChange={(e) => setTrainConfig({ ...trainConfig, imgsz: toSafeNumber(e.target.value, trainConfig.imgsz) })}
                  />
                  <div className="text-xs text-muted-foreground">推荐逻辑按 imgsz 自动折算 batch</div>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Learning Rate</label>
                  <Input
                    type="number"
                    step="0.0001"
                    min={0.0001}
                    max={0.01}
                    value={trainConfig.lr0 ?? ""}
                    onChange={(e) => setTrainConfig({ ...trainConfig, lr0: toSafeNumber(e.target.value, trainConfig.lr0 ?? 0.01) })}
                  />
                  {recommended ? <div className="text-xs text-muted-foreground">推荐：{recommended.lr0}</div> : null}
                </div>
              </div>
            </form>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>训练过程与模型性能评估</CardTitle>
          <CardDescription>Ultralytics results.csv 指标全局曲线（无训练时也展示坐标轴）</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {monitoringCharts.map((m) => (
              <MetricAxisChart
                key={m.label}
                label={m.label}
                epochs={trainMetrics?.epochs ?? []}
                values={m.values}
                expectedEpochs={Number(trainJob?.config?.epochs ?? trainConfig.epochs)}
                color={m.color}
                dot={m.dot}
                yHint={m.yHint}
              />
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>模型版本历史</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>版本名称</TableHead>
                <TableHead>基础架构</TableHead>
                <TableHead>mAP50</TableHead>
                <TableHead>mAP50-95</TableHead>
                <TableHead>创建时间</TableHead>
                <TableHead className="text-right">操作</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isLoadingModels ? (
                <TableRow>
                  <TableCell colSpan={6} className="text-center text-muted-foreground py-6">
                    加载中...
                  </TableCell>
                </TableRow>
              ) : trainedModels.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} className="text-center text-muted-foreground py-6">
                    暂无训练模型
                  </TableCell>
                </TableRow>
              ) : (
                trainedModels.map((m) => {
                  const map50 = typeof m.metrics?.map50 === "number" ? (m.metrics.map50 as number) : null;
                  const map = typeof m.metrics?.map === "number" ? (m.metrics.map as number) : null;
                  const createdAt = new Date(m.created_at);
                  const createdText = Number.isFinite(createdAt.getTime()) ? createdAt.toLocaleString() : m.created_at;

                  return (
                    <TableRow key={m.id}>
                      <TableCell className="font-medium">
                        <div className="flex items-center gap-2">
                          <Box className="w-4 h-4 text-primary" />
                          <span className="truncate" title={m.name}>
                            {m.name}
                          </span>
                        </div>
                      </TableCell>
                      <TableCell>{m.base_model}</TableCell>
                      <TableCell className="tabular-nums">{typeof map50 === "number" ? map50.toFixed(4) : "-"}</TableCell>
                      <TableCell className="tabular-nums">{typeof map === "number" ? map.toFixed(4) : "-"}</TableCell>
                      <TableCell className="text-muted-foreground">{createdText}</TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-center justify-end gap-2">
                          <Button
                            size="sm"
                            variant="outline"
                            className="gap-2"
                            onClick={() => {
                              setTrainConfig((prev) => ({
                                ...prev,
                                mode: "incremental",
                                base_model_id: m.id,
                                output_name: `${m.name}-inc`,
                              }));
                              document.getElementById("train-params")?.scrollIntoView({ behavior: "smooth", block: "start" });
                            }}
                          >
                            <GitBranch className="w-3 h-3" /> 增量训练
                          </Button>
                          <Button size="sm" variant="outline" className="gap-2" onClick={() => openEvaluation(m)}>
                            <LineChart className="w-3 h-3" /> 性能评估
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            className="gap-2"
                            onClick={() => {
                              window.location.href = `/deploy?project_id=${m.project_id}&model_id=${m.id}`;
                            }}
                          >
                            <CloudLightning className="w-3 h-3" /> 部署/导出
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            className="gap-2 text-destructive hover:text-destructive"
                            onClick={() => void handleDeleteModel(m)}
                          >
                            <Trash2 className="w-3 h-3" /> 删除
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                })
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
