import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowRight, ChevronDown, ChevronUp, GitMerge, Image as ImageIcon, Layers, Play, Plus, Save, Settings, Trash2, Upload } from "lucide-react";
import { isAxiosError } from "axios";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/components/ui/use-toast";
import { cn } from "@/lib/utils";
import { useProjectContext } from "@/store/projectContext";
import {
  modelService,
  pipelineService,
  projectService,
  type PipelineRunResultRecord,
  type PipelineStepSpecRecord,
  type TrainedModelRecord,
} from "@/lib/api";

type StepDraft = {
  id: string;
  title: string;
  model_id: string | null;
  conf: number;
  iou: number;
  max_det: number;
  classes: string;
  input_roi_enabled: boolean;
  input_roi_x: number;
  input_roi_y: number;
  input_roi_width: number;
  input_roi_height: number;
  crop: boolean;
  crop_padding: number;
  crop_max_regions: number | null;
  connector_source: "prev_detections" | "prev_segments";
  connector_min_conf: number;
  connector_classes: string;
  connector_padding: number;
  connector_max_regions: number | null;
  connector_on_empty: "stop" | "fallback_full" | "skip";
};

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function colorForClassId(classId: number) {
  const n = Number.isFinite(classId) ? Math.max(0, Math.trunc(classId)) : 0;
  const hue = (n * 137.508) % 360;
  return `hsl(${hue} 80% 55%)`;
}

function parseClasses(text: string): Array<number | string> | null {
  const raw = String(text || "")
    .split(/[,，]/)
    .map((s) => s.trim())
    .filter(Boolean);
  if (!raw.length) return null;
  return raw.map((s) => (s.match(/^\\d+$/) ? Number(s) : s));
}

function newStepDraft(overrides?: Partial<StepDraft>): StepDraft {
  const id = overrides?.id ?? `step_${Date.now()}_${Math.random().toString(16).slice(2)}`;
  return {
    id,
    title: overrides?.title ?? "检测节点",
    model_id: overrides?.model_id ?? null,
    conf: overrides?.conf ?? 0.25,
    iou: overrides?.iou ?? 0.7,
    max_det: overrides?.max_det ?? 50,
    classes: overrides?.classes ?? "",
    input_roi_enabled: overrides?.input_roi_enabled ?? false,
    input_roi_x: overrides?.input_roi_x ?? 0,
    input_roi_y: overrides?.input_roi_y ?? 0,
    input_roi_width: overrides?.input_roi_width ?? 1,
    input_roi_height: overrides?.input_roi_height ?? 1,
    crop: overrides?.crop ?? false,
    crop_padding: overrides?.crop_padding ?? 0.0,
    crop_max_regions: overrides?.crop_max_regions ?? null,
    connector_source: overrides?.connector_source ?? "prev_detections",
    connector_min_conf: overrides?.connector_min_conf ?? 0.0,
    connector_classes: overrides?.connector_classes ?? "",
    connector_padding: overrides?.connector_padding ?? 0.0,
    connector_max_regions: overrides?.connector_max_regions ?? null,
    connector_on_empty: overrides?.connector_on_empty ?? "stop",
  };
}

function toPipelineStepPayload(step: StepDraft, index: number): PipelineStepSpecRecord {
  const x = clamp(Number(step.input_roi_x) || 0, 0, 0.999);
  const y = clamp(Number(step.input_roi_y) || 0, 0, 0.999);
  const width = clamp(Number(step.input_roi_width) || 0, 0.001, 1 - x);
  const height = clamp(Number(step.input_roi_height) || 0, 0.001, 1 - y);

  return {
    id: step.id,
    title: step.title.trim() || "检测节点",
    model_id: step.model_id || "",
    conf: clamp(step.conf, 0, 1),
    iou: clamp(step.iou, 0, 1),
    max_det: Math.trunc(clamp(step.max_det, 1, 300)),
    classes: parseClasses(step.classes),
    // A fixed inspection ROI is intentionally a pipeline-input property, not a
    // per-crop transform. Persist it on step 1 to keep existing recipe storage compatible.
    input_roi:
      index === 0 && step.input_roi_enabled
        ? { x, y, width, height }
        : null,
    connector: step.crop
      ? {
          source: step.connector_source,
          min_conf: clamp(step.connector_min_conf, 0, 1),
          classes: parseClasses(step.connector_classes),
          padding: clamp(step.connector_padding, 0, 1),
          max_regions: step.connector_max_regions ?? null,
          on_empty: step.connector_on_empty,
        }
      : null,
    crop: step.crop,
    crop_padding: clamp(step.connector_padding, 0, 1),
    crop_max_regions: step.connector_max_regions ?? null,
  };
}

export default function MultiModelPipeline() {
  const { toast } = useToast();

  const projectIdInContext = useProjectContext((s) => s.projectId);
  const setProjectContext = useProjectContext((s) => s.setProject);

  const [projects, setProjects] = useState<Array<{ id: string; name: string }>>([]);
  const [isLoadingProjects, setIsLoadingProjects] = useState(false);

  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(projectIdInContext ?? null);
  const [models, setModels] = useState<TrainedModelRecord[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);

  const [pipelines, setPipelines] = useState<Array<{ id: string; name: string; steps?: PipelineStepSpecRecord[] | null }>>([]);
  const [isLoadingPipelines, setIsLoadingPipelines] = useState(false);
  const [pipelineId, setPipelineId] = useState<string | null>(null);
  const [pipelineName, setPipelineName] = useState("多模型串联流程");

  const [steps, setSteps] = useState<StepDraft[]>([newStepDraft({ title: "缺陷检测" })]);
  const [selectedStepId, setSelectedStepId] = useState<string | null>(steps[0]?.id ?? null);

  const [inputFile, setInputFile] = useState<File | null>(null);
  const [inputUrl, setInputUrl] = useState<string>("");
  const [inputSize, setInputSize] = useState<{ w: number; h: number } | null>(null);

  const [isSaving, setIsSaving] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [runResult, setRunResult] = useState<PipelineRunResultRecord | null>(null);
  const [viewMode, setViewMode] = useState<string>("merged");

  const svgRef = useRef<SVGSVGElement | null>(null);

  const selectedProject = useMemo(
    () => projects.find((p) => p.id === selectedProjectId) ?? null,
    [projects, selectedProjectId]
  );

  const selectedStep = useMemo(() => steps.find((s) => s.id === selectedStepId) ?? null, [steps, selectedStepId]);
  const selectedStepIndex = useMemo(() => steps.findIndex((s) => s.id === selectedStepId), [steps, selectedStepId]);

  const updateInputRoi = useCallback(
    (
      patch: Partial<
        Pick<StepDraft, "input_roi_enabled" | "input_roi_x" | "input_roi_y" | "input_roi_width" | "input_roi_height">
      >
    ) => {
      setSteps((prev) => {
        if (!prev[0]) return prev;
        const raw = { ...prev[0], ...patch };
        const x = clamp(Number(raw.input_roi_x) || 0, 0, 0.999);
        const y = clamp(Number(raw.input_roi_y) || 0, 0, 0.999);
        const width = clamp(Number(raw.input_roi_width) || 0.001, 0.001, 1 - x);
        const height = clamp(Number(raw.input_roi_height) || 0.001, 0.001, 1 - y);
        return [{ ...raw, input_roi_x: x, input_roi_y: y, input_roi_width: width, input_roi_height: height }, ...prev.slice(1)];
      });
    },
    []
  );

  const loadProjects = useCallback(async () => {
    try {
      setIsLoadingProjects(true);
      const data = await projectService.list();
      setProjects(data.map((p) => ({ id: p.id, name: p.name })));
      if (!selectedProjectId && data[0]) {
        setSelectedProjectId(data[0].id);
        setProjectContext(data[0].id, data[0].name);
      }
    } catch (err) {
      console.error(err);
      toast({ title: "加载项目失败", description: "请检查后端 /projects 接口", variant: "destructive" });
    } finally {
      setIsLoadingProjects(false);
    }
  }, [selectedProjectId, setProjectContext, toast]);

  const loadModelsAndPipelines = useCallback(async () => {
    if (!selectedProjectId) return;

    setRunResult(null);
    setViewMode("merged");

    try {
      setIsLoadingModels(true);
      const ms = await modelService.listByProject(selectedProjectId);
      setModels(ms);
    } catch (err) {
      console.error(err);
      toast({ title: "加载模型失败", description: "请检查后端 /projects/{id}/models", variant: "destructive" });
    } finally {
      setIsLoadingModels(false);
    }

    try {
      setIsLoadingPipelines(true);
      const ps = await pipelineService.listByProject(selectedProjectId);
      setPipelines(ps.map((p) => ({ id: p.id, name: p.name, steps: p.steps })));
    } catch (err) {
      console.error(err);
      toast({ title: "加载流程失败", description: "请检查后端 /projects/{id}/pipelines", variant: "destructive" });
    } finally {
      setIsLoadingPipelines(false);
    }
  }, [selectedProjectId, toast]);

  useEffect(() => {
    void loadProjects();
  }, [loadProjects]);

  useEffect(() => {
    void loadModelsAndPipelines();
  }, [loadModelsAndPipelines]);

  useEffect(() => {
    if (!inputFile) {
      setInputUrl("");
      setInputSize(null);
      return;
    }
    const url = URL.createObjectURL(inputFile);
    setInputUrl(url);
    setInputSize(null);

    const img = new window.Image();
    img.onload = () => {
      setInputSize({ w: img.naturalWidth || 1, h: img.naturalHeight || 1 });
    };
    img.onerror = () => setInputSize(null);
    img.src = url;

    return () => URL.revokeObjectURL(url);
  }, [inputFile]);

  const displayDetections = useMemo(() => {
    if (!runResult) return [];
    if (viewMode === "merged") return runResult.merged_detections || [];
    if (viewMode === "final") return runResult.final_detections ?? runResult.merged_detections ?? [];
    const step = runResult.steps.find((s) => s.step_id === viewMode);
    return step?.detections || [];
  }, [runResult, viewMode]);

  const handleAddStep = () => {
    const next = newStepDraft({ title: `检测节点 ${steps.length + 1}` });
    setSteps((prev) => [...prev, next]);
    setSelectedStepId(next.id);
  };

  const handleDeleteStep = (id: string) => {
    setSteps((prev) => prev.filter((s) => s.id !== id));
    setSelectedStepId((cur) => (cur === id ? null : cur));
  };

  const moveStep = (stepId: string, direction: -1 | 1) => {
    setSteps((prev) => {
      const idx = prev.findIndex((s) => s.id === stepId);
      if (idx < 0) return prev;
      const nextIdx = idx + direction;
      if (nextIdx < 0 || nextIdx >= prev.length) return prev;
      const copy = [...prev];
      const tmp = copy[idx];
      copy[idx] = copy[nextIdx];
      copy[nextIdx] = tmp;
      return copy;
    });
  };

  const resetPipeline = useCallback(() => {
    const nextSteps = [newStepDraft({ title: "缺陷检测" })];
    setPipelineId(null);
    setPipelineName("多模型串联流程");
    setSteps(nextSteps);
    setSelectedStepId(nextSteps[0]?.id ?? null);
    setRunResult(null);
    setViewMode("merged");
  }, []);

  const deleteSelectedPipeline = async () => {
    if (!pipelineId) return;
    const ok = window.confirm(`确定删除流程“${pipelineName || "未命名流程"}”？此操作不可撤销。`);
    if (!ok) return;
    try {
      await pipelineService.delete(pipelineId);
      toast({ title: "已删除流程", description: pipelineName });
      resetPipeline();
      await loadModelsAndPipelines();
    } catch (err) {
      console.error(err);
      const msg = isAxiosError(err) ? err.response?.data?.detail || err.message : err instanceof Error ? err.message : "未知错误";
      toast({ title: "删除失败", description: String(msg), variant: "destructive" });
    }
  };

  const applyPipelineFromSaved = (pid: string) => {
    const found = pipelines.find((p) => p.id === pid) ?? null;
    if (!found) return;
    setPipelineId(pid);
    setPipelineName(found.name);
    const nextSteps = (found.steps || []).map((s, idx) =>
      newStepDraft({
        id: s.id || `step_${idx + 1}`,
        title: s.title || `检测节点 ${idx + 1}`,
        model_id: s.model_id || null,
        conf: typeof s.conf === "number" ? s.conf : 0.25,
        iou: typeof s.iou === "number" ? s.iou : 0.7,
        max_det: typeof s.max_det === "number" ? s.max_det : 50,
        classes: Array.isArray(s.classes) ? s.classes.join(",") : "",
        input_roi_enabled: Boolean(s.input_roi),
        input_roi_x: typeof s.input_roi?.x === "number" ? s.input_roi.x : 0,
        input_roi_y: typeof s.input_roi?.y === "number" ? s.input_roi.y : 0,
        input_roi_width: typeof s.input_roi?.width === "number" ? s.input_roi.width : 1,
        input_roi_height: typeof s.input_roi?.height === "number" ? s.input_roi.height : 1,
        crop: Boolean(s.connector) || Boolean(s.crop),
        crop_padding: typeof s.crop_padding === "number" ? s.crop_padding : 0.0,
        crop_max_regions: typeof s.crop_max_regions === "number" ? s.crop_max_regions : null,
        connector_source: s.connector?.source === "prev_segments" ? "prev_segments" : "prev_detections",
        connector_min_conf: typeof s.connector?.min_conf === "number" ? s.connector.min_conf : 0,
        connector_classes: Array.isArray(s.connector?.classes) ? s.connector!.classes!.join(",") : "",
        connector_padding:
          typeof s.connector?.padding === "number"
            ? s.connector.padding
            : typeof s.crop_padding === "number"
              ? s.crop_padding
              : 0,
        connector_max_regions:
          typeof s.connector?.max_regions === "number"
            ? s.connector.max_regions
            : typeof s.crop_max_regions === "number"
              ? s.crop_max_regions
              : null,
        connector_on_empty:
          s.connector?.on_empty === "fallback_full" || s.connector?.on_empty === "skip" ? s.connector.on_empty : "stop",
      })
    );
    setSteps(nextSteps.length ? nextSteps : [newStepDraft({ title: "缺陷检测" })]);
    setSelectedStepId(nextSteps[0]?.id ?? null);
    toast({ title: "已加载流程", description: found.name });
  };

  const savePipeline = async () => {
    if (!selectedProjectId) {
      toast({ title: "请先选择项目", variant: "destructive" });
      return;
    }
    const name = pipelineName.trim();
    if (!name) {
      toast({ title: "流程名称不能为空", variant: "destructive" });
      return;
    }
    if (!steps.length) {
      toast({ title: "至少需要一个节点", variant: "destructive" });
      return;
    }
    for (const s of steps) {
      if (!s.model_id) {
        toast({ title: "存在未选择模型的节点", description: `节点：${s.title}`, variant: "destructive" });
        return;
      }
    }

    const payloadSteps = steps.map(toPipelineStepPayload);

    try {
      setIsSaving(true);
      const saved = pipelineId
        ? await pipelineService.update(pipelineId, { name, steps: payloadSteps })
        : await pipelineService.create({ project_id: selectedProjectId, name, steps: payloadSteps });
      setPipelineId(saved.id);
      await loadModelsAndPipelines();
      toast({ title: "保存成功", description: saved.name });
    } catch (err) {
      console.error(err);
      const msg = isAxiosError(err) ? err.response?.data?.detail || err.message : err instanceof Error ? err.message : "未知错误";
      toast({ title: "保存失败", description: String(msg), variant: "destructive" });
    } finally {
      setIsSaving(false);
    }
  };

  const runPipeline = async () => {
    if (!selectedProjectId) {
      toast({ title: "请先选择项目", variant: "destructive" });
      return;
    }
    if (!inputFile) {
      toast({ title: "请先选择输入图片", variant: "destructive" });
      return;
    }
    if (!steps.length) {
      toast({ title: "至少需要一个节点", variant: "destructive" });
      return;
    }
    for (const s of steps) {
      if (!s.model_id) {
        toast({ title: "存在未选择模型的节点", description: `节点：${s.title}`, variant: "destructive" });
        return;
      }
    }

    const payloadSteps = steps.map(toPipelineStepPayload);

    try {
      setIsRunning(true);
      setRunResult(null);
      setViewMode("merged");
      const res = pipelineId
        ? await pipelineService.runSaved(pipelineId, inputFile)
        : await pipelineService.runAdhoc({ project_id: selectedProjectId, steps: payloadSteps }, inputFile);
      setRunResult(res);
      setViewMode("merged");
      toast({ title: "检测完成", description: `输出 ${res.merged_detections?.length ?? 0} 个结果` });
    } catch (err) {
      console.error(err);
      const msg = isAxiosError(err) ? err.response?.data?.detail || err.message : err instanceof Error ? err.message : "未知错误";
      toast({ title: "检测失败", description: String(msg), variant: "destructive" });
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] gap-6 p-6">
      <div className="flex items-start justify-between gap-4 shrink-0">
        <div>
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3">
            <GitMerge className="w-8 h-8 text-blue-500" />
            多模型串联检测
          </h1>
          <p className="text-muted-foreground mt-2">
            选择项目与模型，编排多步检测流程；支持“上一节点检测框 → 裁剪 → 下一节点二次检测”
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={() => void loadModelsAndPipelines()} disabled={isLoadingModels || isLoadingPipelines}>
            {isLoadingModels || isLoadingPipelines ? "刷新中..." : "刷新"}
          </Button>
          <Button variant="outline" className="gap-2" onClick={() => void savePipeline()} disabled={isSaving || !selectedProjectId}>
            <Save className="w-4 h-4" />
            {isSaving ? "保存中..." : "保存流程"}
          </Button>
          <Button
            className="gap-2 bg-blue-600 hover:bg-blue-700"
            onClick={() => void runPipeline()}
            disabled={isRunning || !selectedProjectId}
          >
            {isRunning ? (
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
              <Play className="w-4 h-4" />
            )}
            开始检测
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 min-h-0 flex-1">
        <div className="lg:col-span-2 flex flex-col gap-6 min-h-0">
          <Card className="border-dashed border-2 bg-slate-50/50 dark:bg-slate-950/20 overflow-hidden relative flex flex-col min-h-0">
            <div className="absolute inset-0 bg-[radial-gradient(#e5e7eb_1px,transparent_1px)] [background-size:16px_16px] [mask-image:radial-gradient(ellipse_50%_50%_at_50%_50%,#000_70%,transparent_100%)] pointer-events-none" />
            <CardHeader className="pb-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label>项目</Label>
                  <Select
                    value={selectedProjectId ?? ""}
                    onValueChange={(v) => {
                      const nextId = (v || "").trim() || null;
                      if (!nextId) return;
                      const found = projects.find((p) => p.id === nextId) ?? null;
                      setSelectedProjectId(nextId);
                      setProjectContext(nextId, found?.name ?? null);
                      setPipelineId(null);
                      setRunResult(null);
                    }}
                    disabled={isLoadingProjects || projects.length === 0}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder={isLoadingProjects ? "加载中..." : "请选择项目"} />
                    </SelectTrigger>
                    <SelectContent>
                      {projects.map((p) => (
                        <SelectItem key={p.id} value={p.id}>
                          {p.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {selectedProject ? <div className="text-xs text-muted-foreground truncate">ID: {selectedProject.id}</div> : null}
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <Label>流程</Label>
                    <div className="flex items-center gap-1">
                      <Button variant="outline" size="icon" onClick={resetPipeline} title="新建流程" disabled={!selectedProjectId}>
                        <Plus className="w-4 h-4" />
                      </Button>
                      <Button
                        variant="outline"
                        size="icon"
                        onClick={() => void deleteSelectedPipeline()}
                        title="删除已保存流程"
                        disabled={!pipelineId}
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                  <Select
                    value={pipelineId ?? ""}
                    onValueChange={(v) => {
                      const pid = (v || "").trim();
                      if (!pid) return;
                      applyPipelineFromSaved(pid);
                    }}
                    disabled={!selectedProjectId || isLoadingPipelines}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder={!selectedProjectId ? "请先选择项目" : isLoadingPipelines ? "加载中..." : "选择已保存流程（可选）"} />
                    </SelectTrigger>
                    <SelectContent>
                      {pipelines.map((p) => (
                        <SelectItem key={p.id} value={p.id}>
                          {p.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Input value={pipelineName} onChange={(e) => setPipelineName(e.target.value)} placeholder="流程名称" />
                </div>

                <div className="space-y-2">
                  <Label>输入图片</Label>
                  <div className="flex items-center gap-2">
                    <label className="flex-1">
                      <input type="file" accept="image/*" className="hidden" onChange={(e) => setInputFile(e.target.files?.[0] ?? null)} />
                      <div className="h-10 px-3 rounded-md border bg-background flex items-center justify-between gap-2 cursor-pointer hover:bg-muted/40">
                        <span className="text-sm truncate">{inputFile ? inputFile.name : "选择图片..."}</span>
                        <Upload className="w-4 h-4 text-muted-foreground" />
                      </div>
                    </label>
                    <Button variant="outline" onClick={() => setInputFile(null)} disabled={!inputFile}>
                      清除
                    </Button>
                  </div>
                  {inputSize ? (
                    <div className="text-xs text-muted-foreground tabular-nums">
                      {inputSize.w}×{inputSize.h}
                    </div>
                  ) : null}
                </div>
              </div>
            </CardHeader>

            <CardContent className="flex-1 overflow-auto">
              <div className="p-10 min-w-max flex items-center justify-center min-h-full">
                <div className="flex items-center gap-4">
                  <div className="flex flex-col items-center gap-2">
                    <div className="w-16 h-16 rounded-full bg-green-100 border-2 border-green-500 flex items-center justify-center shadow-sm z-10">
                      <ImageIcon className="w-8 h-8 text-green-600" />
                    </div>
                    <span className="text-sm font-medium text-muted-foreground">图像输入</span>
                  </div>

                  <ArrowRight className="w-6 h-6 text-slate-300" />

                  {steps.map((node) => {
                    const isSelected = selectedStepId === node.id;
                    const modelName = node.model_id ? models.find((m) => m.id === node.model_id)?.name ?? node.model_id : "未选择模型";
                    return (
                      <div key={node.id} className="flex items-center gap-4 group">
                        <div
                          className={cn(
                            "relative w-72 rounded-xl border-2 bg-card p-4 shadow-sm transition-all cursor-pointer hover:shadow-md hover:-translate-y-1",
                            isSelected ? "border-blue-500 ring-4 ring-blue-500/10" : "border-border"
                          )}
                          onClick={() => setSelectedStepId(node.id)}
                        >
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDeleteStep(node.id);
                            }}
                            className="absolute -top-2 -right-2 w-6 h-6 rounded-full bg-red-500 text-white flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                            title="删除节点"
                          >
                            <Trash2 className="w-3 h-3" />
                          </button>

                          <div className="flex items-start justify-between gap-2">
                            <div className="min-w-0 flex-1">
                              <h3 className="font-semibold text-sm truncate">{node.title}</h3>
                              <p className="text-xs text-muted-foreground mt-1 truncate">{modelName}</p>
                            </div>
                            <Badge variant="outline" className="text-[10px] shrink-0">
                              Step
                            </Badge>
                          </div>

                          <div className="mt-3 flex items-center gap-2 text-xs flex-wrap">
                            <Badge variant="secondary" className="text-[10px]">
                              conf {node.conf.toFixed(2)}
                            </Badge>
                            <Badge variant="secondary" className="text-[10px]">
                              iou {node.iou.toFixed(2)}
                            </Badge>
                            {node.crop ? (
                              <Badge variant="outline" className="text-[10px]">
                                Connector: {node.connector_source === "prev_segments" ? "Seg ROI" : "Det ROI"}
                              </Badge>
                            ) : null}
                            {node.input_roi_enabled ? (
                              <Badge variant="outline" className="text-[10px] border-amber-500/50 text-amber-700">
                                Fixed ROI
                              </Badge>
                            ) : null}
                          </div>
                        </div>

                        <ArrowRight className="w-6 h-6 text-slate-300 group-hover:text-blue-400 transition-colors" />
                      </div>
                    );
                  })}

                  <Button
                    onClick={handleAddStep}
                    variant="outline"
                    className="w-16 h-16 rounded-full border-dashed border-2 hover:border-blue-500 hover:bg-blue-50 transition-all"
                    title="新增节点"
                  >
                    <Plus className="w-6 h-6 text-muted-foreground" />
                  </Button>

                  <div className="flex flex-col items-center gap-2">
                    <div className="w-16 h-16 rounded-full bg-purple-100 border-2 border-purple-500 flex items-center justify-center shadow-sm">
                      <Layers className="w-8 h-8 text-purple-600" />
                    </div>
                    <span className="text-sm font-medium text-muted-foreground">结果输出</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="flex flex-col min-h-0">
            <CardHeader className="pb-3 border-b py-3 px-4">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <CardTitle className="text-sm font-medium">结果预览</CardTitle>
                  <CardDescription className="text-xs">支持查看合并结果或单步输出</CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <Select value={viewMode} onValueChange={(v) => setViewMode(v)} disabled={!runResult}>
                    <SelectTrigger className="h-8 w-[180px]">
                      <SelectValue placeholder="选择查看范围" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="merged">合并输出（全部步骤）</SelectItem>
                      <SelectItem value="final">最终输出（最后一步）</SelectItem>
                      {runResult?.steps?.map((s) => (
                        <SelectItem key={s.step_id} value={s.step_id}>
                          {s.title}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Badge variant="outline" className="text-[10px] tabular-nums">
                    {runResult ? `${displayDetections.length} det` : "No result"}
                  </Badge>
                </div>
              </div>
            </CardHeader>
            <CardContent className="flex-1 min-h-[360px] p-0 relative bg-black/5 overflow-hidden">
              {inputUrl && inputSize ? (
                <svg ref={svgRef} viewBox={`0 0 ${inputSize.w} ${inputSize.h}`} className="absolute inset-0 h-full w-full">
                  <image href={inputUrl} x="0" y="0" width={inputSize.w} height={inputSize.h} preserveAspectRatio="xMidYMid meet" />
                  {steps[0]?.input_roi_enabled ? (
                    <rect
                      x={steps[0].input_roi_x * inputSize.w}
                      y={steps[0].input_roi_y * inputSize.h}
                      width={steps[0].input_roi_width * inputSize.w}
                      height={steps[0].input_roi_height * inputSize.h}
                      fill="rgba(245, 158, 11, 0.10)"
                      stroke="#d97706"
                      strokeWidth={2}
                      strokeDasharray="8 5"
                      vectorEffect="non-scaling-stroke"
                    />
                  ) : null}
                  {displayDetections.map((d, idx) => {
                    const [x1, y1, x2, y2] = d.bbox;
                    const w = Math.max(0, x2 - x1);
                    const h = Math.max(0, y2 - y1);
                    const color = colorForClassId(d.class_id);
                    const label = `${d.class_name} ${Number(d.confidence).toFixed(2)}`;
                    const fontSize = Math.max(12, Math.round(Math.min(inputSize.w, inputSize.h) * 0.03));
                    const pad = Math.max(2, Math.round(fontSize * 0.2));
                    const labelH = fontSize + pad * 2;
                    const lx = clamp(x1, 0, Math.max(0, inputSize.w - 1));
                    const maxLabelY = Math.max(0, inputSize.h - labelH);
                    const ly = clamp(y1 - labelH, 0, maxLabelY);
                    const maxLabelW = Math.max(0, inputSize.w - lx);
                    const estW = label.length * fontSize * 0.55 + pad * 2;
                    const lw = Math.min(maxLabelW, estW);
                    return (
                      <g key={`pipe-det-${idx}`}>
                        <rect x={x1} y={y1} width={w} height={h} fill="none" stroke={color} strokeWidth={2} strokeOpacity={0.95} vectorEffect="non-scaling-stroke" />
                        {lw > 0 ? (
                          <>
                            <rect x={lx} y={ly} width={lw} height={labelH} fill={color} fillOpacity={0.85} />
                            <text x={lx + pad} y={ly + pad} fontSize={fontSize} fill="white" dominantBaseline="hanging" style={{ userSelect: "none" }}>
                              {label}
                            </text>
                          </>
                        ) : null}
                      </g>
                    );
                  })}
                </svg>
              ) : (
                <div className="absolute inset-0 flex items-center justify-center text-center p-6 text-muted-foreground">
                  <div>
                    <ImageIcon className="w-10 h-10 mx-auto mb-2 text-slate-300" />
                    <div className="text-sm">选择输入图片后可在此处查看检测框叠加效果</div>
                  </div>
                </div>
              )}

              {isRunning ? (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="flex flex-col items-center gap-2 text-blue-600 bg-white/70 dark:bg-black/30 px-4 py-3 rounded-md backdrop-blur-sm">
                    <div className="w-8 h-8 border-4 border-current border-t-transparent rounded-full animate-spin" />
                    <span className="text-xs font-medium">Processing...</span>
                  </div>
                </div>
              ) : null}
            </CardContent>
          </Card>
        </div>

        <div className="lg:col-span-1 min-h-0">
          <Card className="h-full flex flex-col">
            <CardHeader className="border-b">
              <CardTitle className="flex items-center gap-2 text-sm">
                <Settings className="w-4 h-4" />
                节点配置
              </CardTitle>
              <CardDescription className="text-xs">点击左侧节点进行编辑</CardDescription>
            </CardHeader>
            <CardContent className="flex-1 min-h-0 overflow-auto p-4">
              {!selectedStep ? (
                <div className="h-full flex flex-col items-center justify-center text-center text-muted-foreground p-6">
                  <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center mb-3">
                    <Settings className="w-6 h-6 text-slate-300" />
                  </div>
                  <p className="text-sm">请点击左侧节点进行配置</p>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between gap-2">
                      <Label>节点名称</Label>
                      <div className="flex items-center gap-1">
                        <Button
                          variant="outline"
                          size="icon"
                          onClick={() => moveStep(selectedStep.id, -1)}
                          disabled={steps.findIndex((s) => s.id === selectedStep.id) <= 0}
                          title="上移"
                        >
                          <ChevronUp className="w-4 h-4" />
                        </Button>
                        <Button
                          variant="outline"
                          size="icon"
                          onClick={() => moveStep(selectedStep.id, 1)}
                          disabled={steps.findIndex((s) => s.id === selectedStep.id) >= steps.length - 1}
                          title="下移"
                        >
                          <ChevronDown className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                    <Input value={selectedStep.title} onChange={(e) => setSteps((prev) => prev.map((s) => (s.id === selectedStep.id ? { ...s, title: e.target.value } : s)))} placeholder="请输入节点名称" />
                  </div>

                  <div className="space-y-2">
                    <Label>选择模型</Label>
                    <Select
                      value={selectedStep.model_id ?? ""}
                      onValueChange={(v) => {
                        const next = (v || "").trim() || null;
                        setSteps((prev) => prev.map((s) => (s.id === selectedStep.id ? { ...s, model_id: next } : s)));
                      }}
                      disabled={!selectedProjectId || isLoadingModels}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder={!selectedProjectId ? "请先选择项目" : isLoadingModels ? "加载中..." : models.length ? "选择模型..." : "暂无模型"} />
                      </SelectTrigger>
                      <SelectContent>
                        {models.map((m) => (
                          <SelectItem key={m.id} value={m.id}>
                            <div className="flex flex-col items-start">
                              <span>{m.name}</span>
                              <span className="text-xs text-muted-foreground">{m.base_model}</span>
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  {selectedStepIndex === 0 ? (
                    <>
                      <div className="h-px w-full bg-border" />
                      <div className="flex items-center justify-between rounded-lg border border-amber-500/30 bg-amber-50/40 p-3 dark:bg-amber-950/10">
                        <div className="space-y-0.5">
                          <Label className="text-sm">固定工位 ROI（不改原图）</Label>
                          <div className="text-xs text-muted-foreground">先裁出稳定的检测区域；后续框和热图仍映射回原图坐标。</div>
                        </div>
                        <input
                          type="checkbox"
                          className="h-4 w-4 rounded border-gray-300 text-amber-600 focus:ring-amber-600"
                          checked={selectedStep.input_roi_enabled}
                          onChange={(e) => updateInputRoi({ input_roi_enabled: e.target.checked })}
                        />
                      </div>
                      {selectedStep.input_roi_enabled ? (
                        <div className="grid grid-cols-2 gap-3">
                          <div className="space-y-2">
                            <Label className="text-xs text-muted-foreground">左上 X（%）</Label>
                            <Input
                              type="number"
                              step="0.1"
                              min="0"
                              max="99.9"
                              value={Number((selectedStep.input_roi_x * 100).toFixed(3))}
                              onChange={(e) => updateInputRoi({ input_roi_x: Number(e.target.value) / 100 })}
                            />
                          </div>
                          <div className="space-y-2">
                            <Label className="text-xs text-muted-foreground">左上 Y（%）</Label>
                            <Input
                              type="number"
                              step="0.1"
                              min="0"
                              max="99.9"
                              value={Number((selectedStep.input_roi_y * 100).toFixed(3))}
                              onChange={(e) => updateInputRoi({ input_roi_y: Number(e.target.value) / 100 })}
                            />
                          </div>
                          <div className="space-y-2">
                            <Label className="text-xs text-muted-foreground">宽度（%）</Label>
                            <Input
                              type="number"
                              step="0.1"
                              min="0.1"
                              max="100"
                              value={Number((selectedStep.input_roi_width * 100).toFixed(3))}
                              onChange={(e) => updateInputRoi({ input_roi_width: Number(e.target.value) / 100 })}
                            />
                          </div>
                          <div className="space-y-2">
                            <Label className="text-xs text-muted-foreground">高度（%）</Label>
                            <Input
                              type="number"
                              step="0.1"
                              min="0.1"
                              max="100"
                              value={Number((selectedStep.input_roi_height * 100).toFixed(3))}
                              onChange={(e) => updateInputRoi({ input_roi_height: Number(e.target.value) / 100 })}
                            />
                          </div>
                          <div className="col-span-2 flex justify-end">
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              onClick={() => updateInputRoi({ input_roi_x: 0, input_roi_y: 0, input_roi_width: 1, input_roi_height: 1 })}
                            >
                              重置为全图
                            </Button>
                          </div>
                        </div>
                      ) : null}
                    </>
                  ) : (
                    <div className="rounded-md border border-dashed px-3 py-2 text-xs text-muted-foreground">
                      固定工位 ROI 仅在首节点配置；本节点可使用上一节点的动态 ROI。
                    </div>
                  )}

                  <div className="h-px w-full bg-border" />

                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-2">
                      <Label className="text-xs text-muted-foreground">conf</Label>
                      <Input type="number" step="0.01" min="0" max="1" value={selectedStep.conf} onChange={(e) => setSteps((prev) => prev.map((s) => (s.id === selectedStep.id ? { ...s, conf: Number(e.target.value) || 0 } : s)))} />
                    </div>
                    <div className="space-y-2">
                      <Label className="text-xs text-muted-foreground">iou</Label>
                      <Input type="number" step="0.01" min="0" max="1" value={selectedStep.iou} onChange={(e) => setSteps((prev) => prev.map((s) => (s.id === selectedStep.id ? { ...s, iou: Number(e.target.value) || 0 } : s)))} />
                    </div>
                    <div className="space-y-2 col-span-2">
                      <Label className="text-xs text-muted-foreground">max_det</Label>
                      <Input type="number" step="1" min="1" max="300" value={selectedStep.max_det} onChange={(e) => setSteps((prev) => prev.map((s) => (s.id === selectedStep.id ? { ...s, max_det: Number(e.target.value) || 50 } : s)))} />
                    </div>
                    <div className="space-y-2 col-span-2">
                      <Label className="text-xs text-muted-foreground">classes（可选，逗号分隔：id 或 name）</Label>
                      <Input value={selectedStep.classes} onChange={(e) => setSteps((prev) => prev.map((s) => (s.id === selectedStep.id ? { ...s, classes: e.target.value } : s)))} placeholder="例如：0,1 或 scratch,dent" />
                    </div>
                  </div>

                  <div className="h-px w-full bg-border" />

                  <div className="flex items-center justify-between rounded-lg border p-3">
                    <div className="space-y-0.5">
                      <Label className="text-sm">启用衔接逻辑（Connector → Next）</Label>
                      <div className="text-xs text-muted-foreground">开启后，下一节点将按配置的 ROI 来源与策略接收输入</div>
                    </div>
                    <input type="checkbox" className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-600" checked={selectedStep.crop} onChange={(e) => setSteps((prev) => prev.map((s) => (s.id === selectedStep.id ? { ...s, crop: e.target.checked } : s)))} />
                  </div>

                  {selectedStep.crop ? (
                    <div className="grid grid-cols-2 gap-3">
                      <div className="space-y-2 col-span-2">
                        <Label className="text-xs text-muted-foreground">ROI 来源</Label>
                        <Select
                          value={selectedStep.connector_source}
                          onValueChange={(v) =>
                            setSteps((prev) =>
                              prev.map((s) =>
                                s.id === selectedStep.id
                                  ? { ...s, connector_source: (v as "prev_detections" | "prev_segments") || "prev_detections" }
                                  : s
                              )
                            )
                          }
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="选择来源" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="prev_detections">上一节点检测框</SelectItem>
                            <SelectItem value="prev_segments">上一节点分割掩膜外接框</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <Label className="text-xs text-muted-foreground">min_conf（0~1）</Label>
                        <Input
                          type="number"
                          step="0.01"
                          min="0"
                          max="1"
                          value={selectedStep.connector_min_conf}
                          onChange={(e) =>
                            setSteps((prev) =>
                              prev.map((s) =>
                                s.id === selectedStep.id ? { ...s, connector_min_conf: Number(e.target.value) || 0 } : s
                              )
                            )
                          }
                        />
                      </div>

                      <div className="space-y-2">
                        <Label className="text-xs text-muted-foreground">padding（0~1）</Label>
                        <Input
                          type="number"
                          step="0.05"
                          min="0"
                          max="1"
                          value={selectedStep.connector_padding}
                          onChange={(e) =>
                            setSteps((prev) =>
                              prev.map((s) =>
                                s.id === selectedStep.id ? { ...s, connector_padding: Number(e.target.value) || 0 } : s
                              )
                            )
                          }
                        />
                      </div>

                      <div className="space-y-2 col-span-2">
                        <Label className="text-xs text-muted-foreground">connector classes（可选，逗号分隔）</Label>
                        <Input
                          value={selectedStep.connector_classes}
                          onChange={(e) =>
                            setSteps((prev) =>
                              prev.map((s) =>
                                s.id === selectedStep.id ? { ...s, connector_classes: e.target.value } : s
                              )
                            )
                          }
                          placeholder="例如：0,1 或 scratch,dent"
                        />
                      </div>

                      <div className="space-y-2">
                        <Label className="text-xs text-muted-foreground">max_regions（可选）</Label>
                        <Input
                          type="number"
                          step="1"
                          min="1"
                          max="200"
                          value={selectedStep.connector_max_regions ?? ""}
                          onChange={(e) =>
                            setSteps((prev) =>
                              prev.map((s) =>
                                s.id === selectedStep.id
                                  ? { ...s, connector_max_regions: e.target.value ? Number(e.target.value) : null }
                                  : s
                              )
                            )
                          }
                        />
                      </div>

                      <div className="space-y-2">
                        <Label className="text-xs text-muted-foreground">on_empty 策略</Label>
                        <Select
                          value={selectedStep.connector_on_empty}
                          onValueChange={(v) =>
                            setSteps((prev) =>
                              prev.map((s) =>
                                s.id === selectedStep.id
                                  ? {
                                      ...s,
                                      connector_on_empty:
                                        (v as "stop" | "fallback_full" | "skip") || "stop",
                                    }
                                  : s
                              )
                            )
                          }
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="选择策略" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="stop">无ROI即停止</SelectItem>
                            <SelectItem value="fallback_full">无ROI回退全图</SelectItem>
                            <SelectItem value="skip">无ROI跳过衔接</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  ) : null}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
