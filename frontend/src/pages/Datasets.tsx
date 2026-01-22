import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Download, FileImage, FolderPlus, FolderUp, KeyRound, RefreshCw, ScanEye, Trash2, Upload } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { toast } from "@/components/ui/use-toast";
import { datasetService, projectService, type DatasetFileRecord, type DatasetRecord, type DatasetSplits, type ProjectRecord } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useProjectContext } from "@/store/projectContext";

function formatDate(iso: string | null | undefined) {
  if (!iso) return "-";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "-";
  return d.toISOString().split("T")[0];
}

function formatBytes(bytes: number) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const idx = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / Math.pow(1024, idx);
  return `${value.toFixed(idx === 0 ? 0 : 1)} ${units[idx]}`;
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function parseSplits(text: string): DatasetSplits | null {
  const parts = text
    .split(/[,\s]+/)
    .map((p) => p.trim())
    .filter(Boolean)
    .map((p) => Number(p));
  if (parts.length !== 3 || parts.some((n) => !Number.isFinite(n))) return null;
  const [train, val, test] = parts;
  const sum = train + val + test;
  if (Math.abs(sum - 1) > 1e-6) return null;
  if (train < 0 || val < 0 || test < 0) return null;
  return { train, val, test };
}

function normalizeTags(text: string): string[] | null {
  const tags = text
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean);
  return tags.length ? Array.from(new Set(tags)) : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function getErrorStatus(err: unknown): number | null {
  if (!isRecord(err)) return null;
  const response = err["response"];
  if (!isRecord(response)) return null;
  const status = response["status"];
  return typeof status === "number" ? status : null;
}

function getErrorMessage(err: unknown, fallback: string): string {
  if (isRecord(err)) {
    const response = err["response"];
    if (isRecord(response)) {
      const data = response["data"];
      if (isRecord(data)) {
        const message = data["message"];
        if (typeof message === "string" && message.trim()) return message;
        const detail = data["detail"];
        if (typeof detail === "string" && detail.trim()) return detail;
      }
    }
    const message = err["message"];
    if (typeof message === "string" && message.trim()) return message;
  }
  return fallback;
}

export default function Datasets() {
  const navigate = useNavigate();
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
  const preferredDatasetId = datasetIdFromQuery || datasetIdInContext;

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const folderInputRef = useRef<HTMLInputElement | null>(null);
  const preferredDatasetIdRef = useRef<string | null>(preferredDatasetId);

  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingFiles, setIsLoadingFiles] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isDownloading, setIsDownloading] = useState<string | null>(null);

  const [projects, setProjects] = useState<ProjectRecord[]>([]);
  const [isLoadingProjects, setIsLoadingProjects] = useState(true);

  const [queryInput, setQueryInput] = useState("");
  const [query, setQuery] = useState("");
  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);
  const [files, setFiles] = useState<DatasetFileRecord[]>([]);

  const [showAuth, setShowAuth] = useState(false);
  const [authUser, setAuthUser] = useState(() => {
    try {
      return localStorage.getItem("aipt_user") || "anonymous";
    } catch {
      return "anonymous";
    }
  });
  const [apiKey, setApiKey] = useState(() => {
    try {
      return localStorage.getItem("aipt_api_key") || "";
    } catch {
      return "";
    }
  });

  const selectedDataset = useMemo(
    () => datasets.find((d) => d.id === selectedDatasetId) ?? null,
    [datasets, selectedDatasetId]
  );

  const activeProject = useMemo(() => {
    if (!activeProjectId) return null;
    return projects.find((p) => p.id === activeProjectId) ?? null;
  }, [activeProjectId, projects]);

  const activeProjectName = activeProject?.name ?? projectNameInContext ?? (activeProjectId || null);

  const loadProjects = useCallback(async () => {
    try {
      setIsLoadingProjects(true);
      const data = await projectService.list();
      setProjects(data);
    } catch (err) {
      console.error(err);
      toast({
        title: "加载项目失败",
        description: getErrorMessage(err, "请确认后端服务已启动（http://127.0.0.1:8000）"),
        variant: "destructive",
      });
    } finally {
      setIsLoadingProjects(false);
    }
  }, []);

  useEffect(() => {
    preferredDatasetIdRef.current = preferredDatasetId;
  }, [preferredDatasetId]);

  const loadDatasets = useCallback(async () => {
    try {
      setIsLoading(true);
      if (!activeProjectId) {
        setDatasets([]);
        setSelectedDatasetId(null);
        setFiles([]);
        return;
      }
      const data = await datasetService.list({ q: query.trim() || undefined, project_id: activeProjectId });
      setDatasets(data);
      setSelectedDatasetId((prev) => {
        const preferred = preferredDatasetIdRef.current;
        if (preferred && data.some((d) => d.id === preferred)) return preferred;
        if (prev && data.some((d) => d.id === prev)) return prev;
        return data[0]?.id ?? null;
      });
    } catch (err) {
      console.error(err);
      const status = getErrorStatus(err);
      if (status === 404) {
        toast({
          title: "后端未包含数据集接口",
          description: "检测到 /datasets 返回 404，请重启后端服务（start_services.bat）或确认后端已更新。",
          variant: "destructive",
        });
        return;
      }
      toast({
        title: "加载数据集失败",
        description: getErrorMessage(err, "请确认后端服务已启动（http://127.0.0.1:8000）"),
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  }, [activeProjectId, query]);

  const loadFiles = useCallback(async (datasetId: string) => {
    try {
      setIsLoadingFiles(true);
      const data = await datasetService.listFiles(datasetId);
      setFiles(data);
    } catch (err) {
      console.error(err);
      toast({
        title: "加载文件失败",
        description: getErrorMessage(err, "请稍后重试"),
        variant: "destructive",
      });
    } finally {
      setIsLoadingFiles(false);
    }
  }, []);

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
    void loadProjects();
  }, [loadProjects]);

  useEffect(() => {
    if (!activeProject) return;
    if (projectIdInContext !== activeProject.id) return;
    if (projectNameInContext === activeProject.name) return;
    setProjectContext(activeProject.id, activeProject.name);
  }, [activeProject, projectIdInContext, projectNameInContext, setProjectContext]);

  useEffect(() => {
    setSelectedDatasetId(null);
    setFiles([]);
  }, [activeProjectId]);

  useEffect(() => {
    loadDatasets();
  }, [loadDatasets]);

  useEffect(() => {
    if (!selectedDatasetId) {
      setFiles([]);
      return;
    }
    loadFiles(selectedDatasetId);
  }, [selectedDatasetId, loadFiles]);

  useEffect(() => {
    if (!selectedDatasetId) {
      clearDatasetContext();
      if (datasetIdFromQuery) {
        setSearchParams(
          (prev) => {
            const next = new URLSearchParams(prev);
            next.delete("dataset_id");
            return next;
          },
          { replace: true }
        );
      }
      return;
    }

    const selected = datasets.find((d) => d.id === selectedDatasetId);
    if (!selected) return;
    setDatasetContext(selected.id, `${selected.name} ${selected.version}`);

    if (datasetIdFromQuery !== selected.id) {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          next.set("dataset_id", selected.id);
          if (activeProjectId) next.set("project_id", activeProjectId);
          return next;
        },
        { replace: true }
      );
    }
  }, [
    activeProjectId,
    clearDatasetContext,
    datasetIdFromQuery,
    datasets,
    selectedDatasetId,
    setDatasetContext,
    setSearchParams,
  ]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (!selectedDatasetId) {
      toast({ title: "请先选择或创建数据集版本", variant: "destructive" });
      return;
    }
    const dropped = Array.from(e.dataTransfer.files || []);
    await handleUpload(dropped);
  };

  const handlePickFiles = () => {
    if (!activeProjectId) {
      toast({ title: "请先选择项目", variant: "destructive" });
      return;
    }
    if (!selectedDatasetId) {
      toast({ title: "请先选择或创建数据集版本", variant: "destructive" });
      return;
    }
    fileInputRef.current?.click();
  };

  const handlePickFolder = () => {
    if (!activeProjectId) {
      toast({ title: "请先选择项目", variant: "destructive" });
      return;
    }
    if (!selectedDatasetId) {
      toast({ title: "请先选择或创建数据集版本", variant: "destructive" });
      return;
    }
    folderInputRef.current?.click();
  };

  const handleUpload = async (picked: File[]) => {
    if (!selectedDatasetId) return;
    if (!picked.length) return;
    try {
      setIsUploading(true);
      await datasetService.uploadFiles(selectedDatasetId, picked);
      toast({ title: "上传成功", description: `已上传 ${picked.length} 个文件` });
      await Promise.all([loadDatasets(), loadFiles(selectedDatasetId)]);
    } catch (err: unknown) {
      console.error(err);
      toast({
        title: "上传失败",
        description: getErrorMessage(err, "请检查 API Key / 权限设置或后端日志"),
        variant: "destructive",
      });
    } finally {
      setIsUploading(false);
    }
  };

  const handleSaveAuth = async () => {
    const user = authUser.trim() || "anonymous";
    const key = apiKey.trim();
    try {
      localStorage.setItem("aipt_user", user);
      if (key) localStorage.setItem("aipt_api_key", key);
      else localStorage.removeItem("aipt_api_key");
    } catch {
      toast({ title: "保存失败", description: "无法写入 localStorage", variant: "destructive" });
      return;
    }
    toast({ title: "已保存", description: `X-User=${user}${key ? "（已设置 Key）" : ""}` });
    setShowAuth(false);
    await loadDatasets();
    if (selectedDatasetId) await loadFiles(selectedDatasetId);
  };

  const handleCreateDataset = async () => {
    if (!activeProjectId) {
      toast({ title: "请先选择项目", description: "数据集必须绑定到项目后创建", variant: "destructive" });
      return;
    }
    const name = window.prompt("数据集名称（必填）", selectedDataset?.name || "");
    if (!name || !name.trim()) return;
    const version = window.prompt("版本（可选，留空自动生成 v1/v2...）", "");
    const description = window.prompt("描述（可选）", selectedDataset?.description || "") ?? "";
    const tagsText = window.prompt("标签（逗号分隔，可选）", (selectedDataset?.tags || []).join(", ")) ?? "";
    const isPublic = window.confirm("是否设置为公开数据集？（公开后其他用户可读取）");
    const splitsText =
      window.prompt(
        "划分 train,val,test（必须和为 1），例如：0.7,0.2,0.1",
        selectedDataset ? `${selectedDataset.splits.train},${selectedDataset.splits.val},${selectedDataset.splits.test}` : "0.7,0.2,0.1"
      ) ?? "0.7,0.2,0.1";
    const splits = parseSplits(splitsText);
    if (!splits) {
      toast({ title: "划分格式不合法", description: "请确保输入 3 个数字且和为 1.0", variant: "destructive" });
      return;
    }

    try {
      const created = await datasetService.create({
        project_id: activeProjectId,
        name: name.trim(),
        version: version?.trim() ? version.trim() : null,
        description,
        tags: normalizeTags(tagsText),
        is_public: isPublic,
        splits,
      });
      toast({ title: "创建成功", description: `${created.name} ${created.version}` });
      await loadDatasets();
      setSelectedDatasetId(created.id);
    } catch (err: unknown) {
      console.error(err);
      toast({
        title: "创建失败",
        description: getErrorMessage(err, "请检查 API Key / 权限设置或后端日志"),
        variant: "destructive",
      });
    }
  };

  const handleEditDataset = async (d: DatasetRecord) => {
    const description = window.prompt("描述", d.description) ?? d.description;
    const tagsText = window.prompt("标签（逗号分隔）", (d.tags || []).join(", ")) ?? (d.tags || []).join(", ");
    const isPublic = window.confirm("是否设置为公开数据集？（公开后其他用户可读取）");
    const splitsText = window.prompt("划分 train,val,test（必须和为 1）", `${d.splits.train},${d.splits.val},${d.splits.test}`);
    const splits = splitsText ? parseSplits(splitsText) : d.splits;
    if (splitsText && !splits) {
      toast({ title: "划分格式不合法", description: "请确保输入 3 个数字且和为 1.0", variant: "destructive" });
      return;
    }

    try {
      await datasetService.update(d.id, {
        description,
        tags: normalizeTags(tagsText),
        is_public: isPublic,
        splits: splits || d.splits,
      });
      toast({ title: "保存成功" });
      await loadDatasets();
    } catch (err: unknown) {
      console.error(err);
      toast({
        title: "保存失败",
        description: getErrorMessage(err, "请检查 API Key / 权限设置或后端日志"),
        variant: "destructive",
      });
    }
  };

  return (
    <div className="space-y-6">
      <input
        ref={fileInputRef}
        type="file"
        multiple
        className="hidden"
        onChange={async (e) => {
          const picked = Array.from(e.target.files || []);
          e.target.value = "";
          await handleUpload(picked);
        }}
      />
      <input
        ref={folderInputRef}
        type="file"
        multiple
        className="hidden"
        webkitdirectory
        directory
        onChange={async (e) => {
          const picked = Array.from(e.target.files || []);
          e.target.value = "";
          await handleUpload(picked);
        }}
      />

      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">数据集管理</h1>
          <p className="text-muted-foreground mt-2">上传、预览和管理您的训练数据</p>
          <div className="mt-3 flex flex-wrap items-center gap-2 text-sm">
            <span className="text-muted-foreground">当前项目</span>
            {isLoadingProjects ? (
              <span className="text-muted-foreground">加载中...</span>
            ) : projects.length === 0 ? (
              <span className="text-muted-foreground">暂无项目，请先在项目管理创建</span>
            ) : (
              <select
                className="h-9 rounded-md border border-input bg-background px-3 py-1 text-sm"
                value={activeProjectId ?? ""}
                onChange={(e) => {
                  const nextId = (e.target.value || "").trim();
                  if (!nextId) return;
                  const found = projects.find((p) => p.id === nextId) ?? null;
                  setProjectContext(nextId, found?.name ?? null);
                  setSearchParams((prev) => {
                    const next = new URLSearchParams(prev);
                    next.set("project_id", nextId);
                    next.delete("dataset_id");
                    return next;
                  });
                }}
              >
                <option value="" disabled>
                  请选择项目
                </option>
                {projects.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </select>
            )}
            {activeProjectName ? <Badge variant="secondary">{activeProjectName}</Badge> : null}
          </div>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" className="gap-2" onClick={() => setShowAuth(true)}>
            <KeyRound className="w-4 h-4" /> 权限设置
          </Button>
          <Button variant="outline" className="gap-2" onClick={handleCreateDataset} disabled={!activeProjectId}>
            <FolderPlus className="w-4 h-4" /> 新建版本
          </Button>
          <Button className="gap-2" onClick={handlePickFiles} disabled={isUploading || !selectedDatasetId}>
            {isUploading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />} 上传数据
          </Button>
          <Button variant="outline" className="gap-2" onClick={handlePickFolder} disabled={isUploading || !selectedDatasetId}>
            <FolderUp className="w-4 h-4" /> Upload Folder
          </Button>
          <Button
            variant="outline"
            className="gap-2"
            disabled={!activeProjectId}
            onClick={() => {
              if (!activeProjectId) return;
              const qs = new URLSearchParams({ project_id: activeProjectId });
              if (selectedDatasetId) qs.set("dataset_id", selectedDatasetId);
              navigate(`/annotate?${qs.toString()}`);
            }}
          >
            <ScanEye className="w-4 h-4" /> 智能标注
          </Button>
          <Button variant="outline" className="gap-2" onClick={loadDatasets} disabled={isLoading}>
            <RefreshCw className={cn("w-4 h-4", isLoading ? "animate-spin" : "")} /> 刷新
          </Button>
        </div>
      </div>

      {showAuth ? (
        <Card>
          <CardHeader>
            <CardTitle>权限设置</CardTitle>
            <CardDescription>
              设置 `X-User` 与可选 `X-API-Key`（当后端设置了 `AIPT_API_KEY` 时，写操作需要 Key）。
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <div className="text-sm font-medium">X-User</div>
                <Input value={authUser} onChange={(e) => setAuthUser(e.target.value)} placeholder="anonymous" />
              </div>
              <div className="space-y-2">
                <div className="text-sm font-medium">X-API-Key（可选）</div>
                <Input
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="如后端设置了 AIPT_API_KEY，请填写"
                />
              </div>
            </div>
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setShowAuth(false)}>
                取消
              </Button>
              <Button onClick={handleSaveAuth}>保存</Button>
            </div>
          </CardContent>
        </Card>
      ) : null}

      <div className="grid gap-6 md:grid-cols-3">
        {/* Upload Area */}
        <Card className="col-span-2">
          <CardHeader>
            <CardTitle>上传数据</CardTitle>
            <CardDescription>
              {selectedDataset ? `当前目标：${selectedDataset.name} ${selectedDataset.version}` : "请先选择或创建数据集版本"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div
              className={cn(
                "border-2 border-dashed rounded-lg p-12 flex flex-col items-center justify-center transition-colors",
                isDragging ? "border-primary bg-primary/5" : "border-muted-foreground/25 hover:border-primary/50"
              )}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mb-4">
                <Upload className="w-8 h-8 text-muted-foreground" />
              </div>
              <h3 className="text-lg font-semibold mb-2">拖拽文件到此处</h3>
              <p className="text-sm text-muted-foreground mb-6 text-center">
                支持 JPG/PNG 图片或 ZIP 压缩包（将按原样保存）
              </p>
              <Button onClick={handlePickFiles} disabled={isUploading}>
                选择文件
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Dataset Stats */}
        <Card>
          <CardHeader>
            <CardTitle>版本统计</CardTitle>
            <CardDescription>{selectedDataset ? `${selectedDataset.name} ${selectedDataset.version}` : "未选择"}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-between items-center pb-4 border-b">
              <span className="text-muted-foreground">文件数</span>
              <span className="font-bold text-xl">{selectedDataset?.file_count ?? 0}</span>
            </div>
            <div className="flex justify-between items-center pb-4 border-b">
              <span className="text-muted-foreground">总大小</span>
              <span className="font-bold text-xl">{formatBytes(selectedDataset?.total_size_bytes ?? 0)}</span>
            </div>
            <div className="flex justify-between items-center pb-4 border-b">
              <span className="text-muted-foreground">更新时间</span>
              <span className="font-medium">{formatDate(selectedDataset?.updated_at)}</span>
            </div>
            <div className="space-y-2">
              <div className="text-sm font-medium">数据集划分</div>
              <div className="flex h-4 rounded-full overflow-hidden bg-muted">
                <div
                  className="bg-blue-500"
                  style={{ width: `${(selectedDataset?.splits.train ?? 0.7) * 100}%` }}
                  title={`Train ${(selectedDataset?.splits.train ?? 0.7) * 100}%`}
                />
                <div
                  className="bg-green-500"
                  style={{ width: `${(selectedDataset?.splits.val ?? 0.2) * 100}%` }}
                  title={`Val ${(selectedDataset?.splits.val ?? 0.2) * 100}%`}
                />
                <div
                  className="bg-yellow-500"
                  style={{ width: `${(selectedDataset?.splits.test ?? 0.1) * 100}%` }}
                  title={`Test ${(selectedDataset?.splits.test ?? 0.1) * 100}%`}
                />
              </div>
              <div className="flex text-xs text-muted-foreground justify-between">
                <span>Train {Math.round((selectedDataset?.splits.train ?? 0.7) * 100)}%</span>
                <span>Val {Math.round((selectedDataset?.splits.val ?? 0.2) * 100)}%</span>
                <span>Test {Math.round((selectedDataset?.splits.test ?? 0.1) * 100)}%</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Dataset List */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between gap-4">
            <div>
              <CardTitle>数据集版本</CardTitle>
              <CardDescription>点击条目选择，支持搜索（name/version）</CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Input
                placeholder="搜索..."
                value={queryInput}
                onChange={(e) => setQueryInput(e.target.value)}
                className="w-64"
              />
              <Button
                variant="outline"
                onClick={() => {
                  setQuery(queryInput);
                }}
              >
                查询
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {!activeProjectId ? (
            <div className="py-10 text-center text-muted-foreground">请先选择一个项目</div>
          ) : isLoading ? (
            <div className="flex items-center justify-center py-10 text-muted-foreground">
              <RefreshCw className="w-4 h-4 animate-spin mr-2" /> 加载中...
            </div>
          ) : datasets.length === 0 ? (
            <div className="py-10 text-center text-muted-foreground">暂无数据集版本</div>
          ) : (
            <div className="space-y-3">
              {datasets.map((d) => (
                <div
                  key={d.id}
                  className={cn(
                    "flex items-center justify-between p-4 border rounded-lg transition-colors cursor-pointer",
                    selectedDatasetId === d.id ? "bg-muted/60 border-primary/30" : "hover:bg-muted/40"
                  )}
                  onClick={() => setSelectedDatasetId(d.id)}
                >
                  <div className="flex items-center gap-4 min-w-0">
                    <div className="w-10 h-10 bg-blue-100 text-blue-600 rounded flex items-center justify-center shrink-0">
                      <FileImage className="w-5 h-5" />
                    </div>
                    <div className="min-w-0">
                      <div className="font-medium flex items-center gap-2 flex-wrap">
                        <span className="truncate max-w-[320px]">{d.name}</span>
                        <Badge variant="outline">{d.version}</Badge>
                        {d.is_public ? <Badge variant="secondary">公开</Badge> : <Badge variant="outline">私有</Badge>}
                      </div>
                      <div className="text-xs text-muted-foreground flex flex-wrap gap-x-2 gap-y-1">
                        <span>{d.file_count} 文件</span>
                        <span>|</span>
                        <span>{formatBytes(d.total_size_bytes)}</span>
                        <span>|</span>
                        <span>{formatDate(d.updated_at)}</span>
                        <span>|</span>
                        <span>{d.status}</span>
                      </div>
                      {d.description ? (
                        <div className="text-xs text-muted-foreground mt-1 truncate">{d.description}</div>
                      ) : null}
                    </div>
                  </div>
                  <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
                    <Button
                      variant="outline"
                      size="sm"
                      className="gap-2"
                      onClick={async () => {
                        try {
                          setIsDownloading(d.id);
                          const blob = await datasetService.downloadDatasetZipBlob(d.id);
                          downloadBlob(blob, `${d.name}_${d.version}.zip`);
                        } catch (err: unknown) {
                          console.error(err);
                          toast({
                            title: "下载失败",
                            description: getErrorMessage(err, "请稍后重试"),
                            variant: "destructive",
                          });
                        } finally {
                          setIsDownloading(null);
                        }
                      }}
                      disabled={isDownloading === d.id}
                    >
                      {isDownloading === d.id ? (
                        <RefreshCw className="w-4 h-4 animate-spin" />
                      ) : (
                        <Download className="w-4 h-4" />
                      )}
                      下载
                    </Button>
                    <Button variant="outline" size="sm" onClick={() => handleEditDataset(d)}>
                      编辑
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="text-destructive hover:text-destructive hover:bg-destructive/10"
                      onClick={async () => {
                        if (!confirm(`确认删除数据集版本：${d.name} ${d.version} ？`)) return;
                        try {
                          await datasetService.delete(d.id);
                          toast({ title: "已删除" });
                          await loadDatasets();
                        } catch (err: unknown) {
                          console.error(err);
                          toast({
                            title: "删除失败",
                            description: getErrorMessage(err, "请检查 API Key / 权限设置或后端日志"),
                            variant: "destructive",
                          });
                        }
                      }}
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Files */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between gap-4">
            <div>
              <CardTitle>文件列表</CardTitle>
              <CardDescription>
                {selectedDataset ? `${selectedDataset.name} ${selectedDataset.version}` : "请选择一个数据集版本查看文件"}
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                onClick={() => selectedDatasetId && loadFiles(selectedDatasetId)}
                disabled={!selectedDatasetId || isLoadingFiles}
                className="gap-2"
              >
                <RefreshCw className={cn("w-4 h-4", isLoadingFiles ? "animate-spin" : "")} />
                刷新文件
              </Button>
              <Button onClick={handlePickFiles} disabled={!selectedDatasetId || isUploading} className="gap-2">
                {isUploading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
                追加上传
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {!selectedDatasetId ? (
            <div className="py-10 text-center text-muted-foreground">未选择数据集版本</div>
          ) : isLoadingFiles ? (
            <div className="flex items-center justify-center py-10 text-muted-foreground">
              <RefreshCw className="w-4 h-4 animate-spin mr-2" /> 加载中...
            </div>
          ) : files.length === 0 ? (
            <div className="py-10 text-center text-muted-foreground">暂无文件</div>
          ) : (
            <div className="space-y-2">
              {files.map((f) => (
                <div key={f.id} className="flex items-center justify-between p-3 border rounded-lg hover:bg-muted/40 transition-colors">
                  <div className="min-w-0">
                    <div className="font-medium truncate">{f.filename}</div>
                    <div className="text-xs text-muted-foreground flex flex-wrap gap-x-2 gap-y-1">
                      <span>{formatBytes(f.size_bytes)}</span>
                      <span>|</span>
                      <span>{formatDate(f.created_at)}</span>
                      <span>|</span>
                      <span className="truncate max-w-[320px]">{f.sha256}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      className="gap-2"
                      onClick={async () => {
                        if (!selectedDatasetId) return;
                        try {
                          const blob = await datasetService.downloadFileBlob(selectedDatasetId, f.id);
                          downloadBlob(blob, f.filename);
                        } catch (err: unknown) {
                          console.error(err);
                          toast({
                            title: "下载失败",
                            description: getErrorMessage(err, "请稍后重试"),
                            variant: "destructive",
                          });
                        }
                      }}
                    >
                      <Download className="w-4 h-4" />
                      下载
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="text-destructive hover:text-destructive hover:bg-destructive/10"
                      onClick={async () => {
                        if (!selectedDatasetId) return;
                        if (!confirm(`确认删除文件：${f.filename} ？`)) return;
                        try {
                          await datasetService.deleteFile(selectedDatasetId, f.id);
                          toast({ title: "已删除" });
                          await Promise.all([loadDatasets(), loadFiles(selectedDatasetId)]);
                        } catch (err: unknown) {
                          console.error(err);
                          toast({
                            title: "删除失败",
                            description: getErrorMessage(err, "请检查 API Key / 权限设置或后端日志"),
                            variant: "destructive",
                          });
                        }
                      }}
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
