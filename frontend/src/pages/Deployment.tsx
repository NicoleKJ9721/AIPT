import { useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { isAxiosError } from "axios";
import { Code2, Copy, Cpu, GitMerge, RefreshCw, Server, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "@/components/ui/use-toast";
import {
  inferenceService,
  modelService,
  pipelineService,
  projectService,
  type InferenceFormat,
  type InferenceKind,
  type InferenceSessionRecord,
  type InferenceStatusRecord,
  type PipelineRecord,
  type ProjectRecord,
  type TrainedModelRecord,
} from "@/lib/api";
import { useProjectContext } from "@/store/projectContext";

type DeployTargetType = "model" | "pipeline";

const FORMAT_HINT: Record<InferenceFormat, string> = {
  openvino: "OpenVINO 仅支持 Intel CPU 推理（建议用于 CPU-only 场景）",
  tensorrt: "TensorRT 仅支持 NVIDIA 显卡推理（建议用于高吞吐/低延迟场景）",
};

const GPU_ESTIMATES: Array<{ gpu: string; vram: string; perf: string; note: string }> = [
  { gpu: "RTX 2080", vram: "8GB", perf: "1.0×", note: "入门 GPU（建议小模型/低并发）" },
  { gpu: "RTX 3080", vram: "10–12GB", perf: "≈ 1.8×", note: "性价比高（中等并发）" },
  { gpu: "RTX 3090", vram: "24GB", perf: "≈ 2.0×", note: "大显存（更适合大模型/更大 batch）" },
  { gpu: "RTX 4060 Ti", vram: "8–16GB", perf: "≈ 1.6×", note: "功耗低（中低并发）" },
  { gpu: "RTX 4090", vram: "24GB", perf: "≈ 3.0×", note: "旗舰（高并发/高吞吐）" },
  { gpu: "RTX 5090", vram: "待实测", perf: "待实测", note: "新卡请以实测为准" },
];

type InferenceParams = {
  conf: string;
  iou: string;
  imgsz: string;
  max_det: string;
  classes: string;
};

const DEFAULT_PARAMS: InferenceParams = {
  conf: "0.25",
  iou: "0.7",
  imgsz: "640",
  max_det: "50",
  classes: "",
};

function clampNum(n: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, n));
}

function toNumberOrUndefined(v: string): number | undefined {
  const n = Number(v);
  if (!Number.isFinite(n)) return undefined;
  return n;
}

function safeJsonStringify(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function buildCodeSnippet(
  lang: "python" | "cpp" | "csharp",
  opts: {
    baseUrl: string;
    projectId: string;
    targetType: DeployTargetType;
    targetId: string;
    format: InferenceFormat;
    device?: string;
    end2end: boolean;
    params: InferenceParams;
  }
): string {
  const conf = clampNum(toNumberOrUndefined(opts.params.conf) ?? 0.25, 0, 1);
  const iou = clampNum(toNumberOrUndefined(opts.params.iou) ?? 0.7, 0, 1);
  const imgsz = Math.max(32, Math.round(toNumberOrUndefined(opts.params.imgsz) ?? 640));
  const maxDet = Math.max(1, Math.round(toNumberOrUndefined(opts.params.max_det) ?? 50));
  const classes = (opts.params.classes || "").trim();

  const createPayload =
    opts.targetType === "model"
      ? {
          project_id: opts.projectId,
          kind: "model" as InferenceKind,
          target_id: opts.targetId,
          format: opts.format,
          ...(opts.device ? { device: opts.device } : {}),
          end2end: opts.end2end,
        }
      : {
          project_id: opts.projectId,
          kind: "pipeline" as InferenceKind,
          target_id: opts.targetId,
          ...(opts.device ? { device: opts.device } : {}),
        };

  const predictParams = {
    conf,
    iou,
    imgsz,
    max_det: maxDet,
    ...(classes ? { classes } : {}),
  };

  if (lang === "python") {
    return `import requests
import time

BASE_URL = "${opts.baseUrl}"

create_payload = ${safeJsonStringify(createPayload)}
r0 = time.perf_counter()
r = requests.post(f"{BASE_URL}/inference/sessions", json=create_payload, timeout=1200)
r.raise_for_status()
session_id = r.json()["data"]["id"]
print("create_session:", round(time.perf_counter() - r0, 3), "s", "session_id =", session_id)

params = ${safeJsonStringify(predictParams)}
with open("test.jpg", "rb") as f:
    rr = requests.post(
        f"{BASE_URL}/inference/sessions/{session_id}/predict",
        files={"file": ("test.jpg", f, "image/jpeg")},
        params=params,
        timeout=1200,
    )
    rr.raise_for_status()
    print(rr.json()["data"])

requests.delete(f"{BASE_URL}/inference/sessions/{session_id}", timeout=30).raise_for_status()`;
  }

  if (lang === "csharp") {
    return `// C# 示例（统一会话接口）
// create payload:
${safeJsonStringify(createPayload)}
// predict params:
${safeJsonStringify(predictParams)}
// endpoint:
// POST ${opts.baseUrl}/inference/sessions
// POST ${opts.baseUrl}/inference/sessions/{session_id}/predict
// DELETE ${opts.baseUrl}/inference/sessions/{session_id}`;
  }

  return `// C++ 示例（统一会话接口）
// create payload:
${safeJsonStringify(createPayload)}
// predict params:
${safeJsonStringify(predictParams)}
// endpoint:
// POST ${opts.baseUrl}/inference/sessions
// POST ${opts.baseUrl}/inference/sessions/{session_id}/predict
// DELETE ${opts.baseUrl}/inference/sessions/{session_id}`;
}

export default function Deployment() {
  const [searchParams, setSearchParams] = useSearchParams();
  const projectIdFromQuery = (searchParams.get("project_id") || "").trim() || null;
  const modelIdFromQuery = (searchParams.get("model_id") || "").trim() || null;
  const pipelineIdFromQuery = (searchParams.get("pipeline_id") || "").trim() || null;
  const targetTypeFromQuery = (searchParams.get("target_type") || "").trim().toLowerCase();

  const projectIdInContext = useProjectContext((s) => s.projectId);
  const setProjectContext = useProjectContext((s) => s.setProject);

  const [projects, setProjects] = useState<ProjectRecord[]>([]);
  const [models, setModels] = useState<TrainedModelRecord[]>([]);
  const [pipelines, setPipelines] = useState<PipelineRecord[]>([]);
  const [isLoadingProjects, setIsLoadingProjects] = useState(true);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [isLoadingPipelines, setIsLoadingPipelines] = useState(false);

  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(projectIdFromQuery || projectIdInContext || null);
  const [targetType, setTargetType] = useState<DeployTargetType>(
    targetTypeFromQuery === "pipeline" ? "pipeline" : "model"
  );
  const [selectedModelId, setSelectedModelId] = useState<string | null>(modelIdFromQuery || null);
  const [selectedPipelineId, setSelectedPipelineId] = useState<string | null>(pipelineIdFromQuery || null);

  const [format, setFormat] = useState<InferenceFormat>("tensorrt");
  const [end2end, setEnd2end] = useState<boolean>(false);
  const [device, setDevice] = useState<string>("0");
  const [params, setParams] = useState<InferenceParams>({ ...DEFAULT_PARAMS });

  const [status, setStatus] = useState<InferenceStatusRecord | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [isClosing, setIsClosing] = useState(false);

  const [codeLang, setCodeLang] = useState<"python" | "cpp" | "csharp">("python");

  const selectedTargetId = targetType === "model" ? selectedModelId : selectedPipelineId;

  const matchingSession: InferenceSessionRecord | null = useMemo(() => {
    if (!status || !selectedTargetId) return null;
    const dev = (device || "").trim();
    return (
      status.sessions.find((s) => {
        if (s.kind !== targetType) return false;
        if (s.target_id !== selectedTargetId) return false;
        if (targetType === "pipeline") return (s.device || "").trim() === dev;
        if (s.format !== format) return false;
        if ((s.end2end ?? false) !== end2end) return false;
        if (format === "openvino") return true;
        return (s.device || "").trim() === dev;
      }) ?? null
    );
  }, [device, end2end, format, selectedTargetId, status, targetType]);

  const refreshStatus = useCallback(async () => {
    try {
      setIsRefreshing(true);
      const s = await inferenceService.status();
      setStatus(s);
    } catch (err) {
      console.error(err);
      setStatus(null);
      toast({ title: "获取推理状态失败", description: "请检查后端是否已启动", variant: "destructive" });
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  const loadProjects = useCallback(async () => {
    try {
      setIsLoadingProjects(true);
      const data = await projectService.list();
      setProjects(data);
      if (!selectedProjectId && data[0]) {
        setSelectedProjectId(data[0].id);
        setProjectContext(data[0].id, data[0].name);
      }
    } catch (err) {
      console.error(err);
      toast({ title: "加载项目失败", description: "请确认后端服务已启动", variant: "destructive" });
    } finally {
      setIsLoadingProjects(false);
    }
  }, [selectedProjectId, setProjectContext]);

  const loadTargets = useCallback(async () => {
    if (!selectedProjectId) {
      setModels([]);
      setPipelines([]);
      setSelectedModelId(null);
      setSelectedPipelineId(null);
      return;
    }

    try {
      setIsLoadingModels(true);
      const data = await modelService.listByProject(selectedProjectId);
      setModels(data);
      setSelectedModelId((prev) => prev && data.some((m) => m.id === prev) ? prev : data[0]?.id ?? null);
    } catch (err) {
      console.error(err);
      setModels([]);
      toast({ title: "加载模型失败", description: "请确认已完成训练并生成模型版本", variant: "destructive" });
    } finally {
      setIsLoadingModels(false);
    }

    try {
      setIsLoadingPipelines(true);
      const data = await pipelineService.listByProject(selectedProjectId);
      setPipelines(data);
      setSelectedPipelineId((prev) => prev && data.some((p) => p.id === prev) ? prev : data[0]?.id ?? null);
    } catch (err) {
      console.error(err);
      setPipelines([]);
      toast({ title: "加载流程失败", description: "请先在“多模型串联检测”中保存流程", variant: "destructive" });
    } finally {
      setIsLoadingPipelines(false);
    }
  }, [selectedProjectId]);

  useEffect(() => {
    void loadProjects();
  }, [loadProjects]);

  useEffect(() => {
    void loadTargets();
  }, [loadTargets]);

  useEffect(() => {
    void refreshStatus();
  }, [refreshStatus]);

  useEffect(() => {
    const next = new URLSearchParams(searchParams);
    if (selectedProjectId) next.set("project_id", selectedProjectId);
    else next.delete("project_id");
    next.set("target_type", targetType);
    if (targetType === "model") {
      if (selectedModelId) next.set("model_id", selectedModelId);
      else next.delete("model_id");
      next.delete("pipeline_id");
    } else {
      if (selectedPipelineId) next.set("pipeline_id", selectedPipelineId);
      else next.delete("pipeline_id");
      next.delete("model_id");
    }
    setSearchParams(next, { replace: true });
  }, [searchParams, selectedModelId, selectedPipelineId, selectedProjectId, setSearchParams, targetType]);

  const baseUrl = useMemo(() => "http://127.0.0.1:8000", []);

  const snippet = useMemo(() => {
    if (!selectedProjectId || !selectedTargetId) return "";
    return buildCodeSnippet(codeLang, {
      baseUrl,
      projectId: selectedProjectId,
      targetType,
      targetId: selectedTargetId,
      format,
      device: targetType === "model" && format === "tensorrt" ? device : undefined,
      end2end,
      params,
    });
  }, [baseUrl, codeLang, device, end2end, format, params, selectedProjectId, selectedTargetId, targetType]);

  const copySnippet = async () => {
    if (!snippet) return;
    try {
      await navigator.clipboard.writeText(snippet);
      toast({ title: "已复制到剪贴板" });
    } catch (err) {
      console.error(err);
      toast({ title: "复制失败", description: "浏览器权限限制，请手动复制", variant: "destructive" });
    }
  };

  const createSession = async () => {
    if (!selectedProjectId) {
      toast({ title: "请先选择项目", variant: "destructive" });
      return;
    }
    if (!selectedTargetId) {
      toast({ title: targetType === "model" ? "请先选择模型" : "请先选择已保存流程", variant: "destructive" });
      return;
    }
    try {
      setIsCreating(true);
      const payload =
        targetType === "model"
          ? {
              project_id: selectedProjectId,
              kind: "model" as InferenceKind,
              target_id: selectedTargetId,
              format,
              end2end,
              device: format === "tensorrt" ? (device || "").trim() || undefined : undefined,
            }
          : {
              project_id: selectedProjectId,
              kind: "pipeline" as InferenceKind,
              target_id: selectedTargetId,
              device: (device || "").trim() || undefined,
            };
      const created = await inferenceService.createSession(payload);
      toast({ title: "推理申请成功", description: `Session: ${created.id}` });
      await refreshStatus();
    } catch (err) {
      console.error(err);
      let msg = "未知错误";
      if (isAxiosError(err)) {
        msg = err.response?.data?.message || err.response?.data?.detail || err.message;
      } else if (err instanceof Error) {
        msg = err.message;
      }
      toast({ title: "推理申请失败", description: String(msg), variant: "destructive" });
    } finally {
      setIsCreating(false);
    }
  };

  const closeSession = async () => {
    if (!matchingSession) return;
    try {
      setIsClosing(true);
      await inferenceService.closeSession(matchingSession.id);
      toast({ title: "已释放推理会话" });
      await refreshStatus();
    } catch (err) {
      console.error(err);
      let msg = "未知错误";
      if (isAxiosError(err)) {
        msg = err.response?.data?.message || err.response?.data?.detail || err.message;
      } else if (err instanceof Error) {
        msg = err.message;
      }
      toast({ title: "释放失败", description: String(msg), variant: "destructive" });
    } finally {
      setIsClosing(false);
    }
  };

  const inferenceLockedTraining = (status?.sessions?.length ?? 0) > 0 || (status?.active_requests ?? 0) > 0;

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">部署推荐（统一会话申请）</h1>
          <p className="text-muted-foreground mt-2">
            支持模型会话与多模型串联流程会话（Pipeline）；会话存在期间将锁定训练资源
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={() => void refreshStatus()} disabled={isRefreshing} className="gap-2">
            <RefreshCw className={isRefreshing ? "w-4 h-4 animate-spin" : "w-4 h-4"} />
            刷新状态
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Server className="w-5 h-5 text-primary" /> 推理申请
              </CardTitle>
              <CardDescription>
                会话存在期间会禁止训练（避免资源争抢）
                {inferenceLockedTraining ? <Badge className="ml-2">训练已锁定</Badge> : null}
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4 md:grid-cols-2">
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
              </div>

              <div className="space-y-2">
                <Label>目标类型</Label>
                <Select value={targetType} onValueChange={(v) => setTargetType(v as DeployTargetType)}>
                  <SelectTrigger>
                    <SelectValue placeholder="请选择目标类型" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="model">模型</SelectItem>
                    <SelectItem value="pipeline">管线（多模型流程）</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {targetType === "model" ? (
                <div className="space-y-2 md:col-span-2">
                  <Label>模型</Label>
                  <Select
                    value={selectedModelId ?? ""}
                    onValueChange={(v) => setSelectedModelId((v || "").trim() || null)}
                    disabled={!selectedProjectId || isLoadingModels || models.length === 0}
                  >
                    <SelectTrigger>
                      <SelectValue
                        placeholder={
                          !selectedProjectId ? "请先选择项目" : isLoadingModels ? "加载中..." : models.length ? "请选择模型" : "暂无模型"
                        }
                      />
                    </SelectTrigger>
                    <SelectContent>
                      {models.map((m) => (
                        <SelectItem key={m.id} value={m.id}>
                          {m.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              ) : (
                <div className="space-y-2 md:col-span-2">
                  <Label>已保存流程</Label>
                  <Select
                    value={selectedPipelineId ?? ""}
                    onValueChange={(v) => setSelectedPipelineId((v || "").trim() || null)}
                    disabled={!selectedProjectId || isLoadingPipelines || pipelines.length === 0}
                  >
                    <SelectTrigger>
                      <SelectValue
                        placeholder={
                          !selectedProjectId
                            ? "请先选择项目"
                            : isLoadingPipelines
                              ? "加载中..."
                              : pipelines.length
                                ? "请选择流程"
                                : "暂无已保存流程"
                        }
                      />
                    </SelectTrigger>
                    <SelectContent>
                      {pipelines.map((p) => (
                        <SelectItem key={p.id} value={p.id}>
                          {p.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}

              {targetType === "model" ? (
                <>
                  <div className="space-y-2">
                    <Label>推理格式</Label>
                    <Select value={format} onValueChange={(v) => setFormat(v as InferenceFormat)}>
                      <SelectTrigger>
                        <SelectValue placeholder="请选择格式" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="tensorrt">TensorRT（默认）</SelectItem>
                        <SelectItem value="openvino">OpenVINO</SelectItem>
                      </SelectContent>
                    </Select>
                    <div className="text-xs text-muted-foreground flex items-center gap-2">
                      <Cpu className="w-3.5 h-3.5" />
                      <span>{FORMAT_HINT[format]}</span>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label>设备（device）</Label>
                    <Input
                      value={device}
                      onChange={(e) => setDevice(e.target.value)}
                      placeholder={format === "tensorrt" ? "0 / cuda:0" : "cpu"}
                      disabled={format === "openvino"}
                    />
                  </div>

                  <div className="md:col-span-2 space-y-2">
                    <Label>推理头（YOLO26）</Label>
                    <Select value={end2end ? "end2end" : "precise"} onValueChange={(v) => setEnd2end(v === "end2end")}>
                      <SelectTrigger>
                        <SelectValue placeholder="请选择推理头" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="end2end">端到端（更快）</SelectItem>
                        <SelectItem value="precise">高精度（end2end=false）</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </>
              ) : (
                <div className="md:col-span-2 rounded-lg border p-3 text-sm text-muted-foreground flex items-center gap-2">
                  <GitMerge className="w-4 h-4" />
                  Pipeline 会话将使用保存时定义的串联步骤与 connector 逻辑，创建后会冻结快照。
                </div>
              )}

              <div className="md:col-span-2 flex items-center justify-between gap-3 pt-2">
                <div className="text-sm text-muted-foreground">
                  {matchingSession ? (
                    <span>
                      当前会话：
                      <Badge variant="outline" className="ml-1 mr-1">
                        {matchingSession.id.slice(0, 8)}
                      </Badge>
                      {matchingSession.kind === "pipeline" ? "pipeline" : matchingSession.format}
                    </span>
                  ) : (
                    <span>当前未申请该目标的会话</span>
                  )}
                </div>
                <div className="flex gap-2">
                  {matchingSession ? (
                    <Button variant="outline" className="gap-2" onClick={() => void closeSession()} disabled={isClosing}>
                      {isClosing ? <RefreshCw className="w-4 h-4 animate-spin" /> : <X className="w-4 h-4" />}
                      释放推理
                    </Button>
                  ) : (
                    <Button className="gap-2" onClick={() => void createSession()} disabled={isCreating || !selectedTargetId}>
                      {isCreating ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Server className="w-4 h-4" />}
                      申请推理
                    </Button>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Code2 className="w-5 h-5 text-primary" /> 网络申请代码例程
              </CardTitle>
              <CardDescription>统一会话接口：`kind + target_id`</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 md:grid-cols-4">
                <div className="md:col-span-2 space-y-2">
                  <Label>推理参数（请求示例）</Label>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-1">
                      <Label className="text-xs">conf</Label>
                      <Input value={params.conf} onChange={(e) => setParams((p) => ({ ...p, conf: e.target.value }))} />
                    </div>
                    <div className="space-y-1">
                      <Label className="text-xs">iou</Label>
                      <Input value={params.iou} onChange={(e) => setParams((p) => ({ ...p, iou: e.target.value }))} />
                    </div>
                    <div className="space-y-1">
                      <Label className="text-xs">imgsz</Label>
                      <Input value={params.imgsz} onChange={(e) => setParams((p) => ({ ...p, imgsz: e.target.value }))} />
                    </div>
                    <div className="space-y-1">
                      <Label className="text-xs">max_det</Label>
                      <Input value={params.max_det} onChange={(e) => setParams((p) => ({ ...p, max_det: e.target.value }))} />
                    </div>
                    <div className="space-y-1 col-span-2">
                      <Label className="text-xs">classes（可选）</Label>
                      <Input value={params.classes} onChange={(e) => setParams((p) => ({ ...p, classes: e.target.value }))} />
                    </div>
                  </div>
                </div>

                <div className="md:col-span-2 space-y-2">
                  <Label>语言</Label>
                  <Tabs value={codeLang} onValueChange={(v) => setCodeLang(v as "python" | "cpp" | "csharp")}>
                    <TabsList className="grid w-full grid-cols-3">
                      <TabsTrigger value="python">Python</TabsTrigger>
                      <TabsTrigger value="cpp">C++</TabsTrigger>
                      <TabsTrigger value="csharp">C#</TabsTrigger>
                    </TabsList>
                    <TabsContent value="python">{null}</TabsContent>
                    <TabsContent value="cpp">{null}</TabsContent>
                    <TabsContent value="csharp">{null}</TabsContent>
                  </Tabs>
                </div>
              </div>

              <div className="rounded-lg border bg-slate-950 p-3">
                <pre className="text-xs text-slate-100 whitespace-pre-wrap break-all max-h-[420px] overflow-auto">{snippet || "// 请选择项目与目标（模型或流程）"}</pre>
              </div>

              <div className="flex justify-end">
                <Button variant="outline" className="gap-2" onClick={() => void copySnippet()} disabled={!snippet}>
                  <Copy className="w-4 h-4" />
                  复制代码
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>会话状态</CardTitle>
              <CardDescription>
                active_requests={status?.active_requests ?? 0} · sessions={status?.sessions?.length ?? 0}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!status?.sessions?.length ? (
                <div className="text-sm text-muted-foreground">当前无活跃会话</div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>会话</TableHead>
                      <TableHead>类型</TableHead>
                      <TableHead>目标</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {status.sessions.map((s) => (
                      <TableRow key={s.id}>
                        <TableCell className="font-mono text-xs">{s.id.slice(0, 8)}</TableCell>
                        <TableCell>
                          <Badge variant="outline">{s.kind}</Badge>
                        </TableCell>
                        <TableCell className="text-xs truncate max-w-[200px]" title={s.target_name}>
                          {s.target_name}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>GPU 推荐</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {GPU_ESTIMATES.map((g) => (
                  <div key={g.gpu} className="rounded-md border p-2">
                    <div className="text-sm font-medium">{g.gpu}</div>
                    <div className="text-xs text-muted-foreground">
                      {g.vram} · {g.perf}
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">{g.note}</div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
