import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { AlertTriangle, Crop, FolderUp, RefreshCw, RotateCw, ScanEye, Trash2, Upload } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "@/components/ui/use-toast";
import { cn } from "@/lib/utils";
import {
  datasetService,
  imageService,
  projectService,
  type DatasetRecord,
  type ImageRecord,
  type ProjectRecord,
} from "@/lib/api";
import { useProjectContext } from "@/store/projectContext";

function resolveImageUrl(sourceUrl: string | null | undefined): string {
  const url = (sourceUrl || "").trim();
  if (!url) return "";
  if (url.startsWith("http://") || url.startsWith("https://") || url.startsWith("data:")) return url;
  if (url.startsWith("/api/")) return url;
  if (url.startsWith("/")) return `/api${url}`;
  return url;
}

type QualityResult = {
  brightness: number;
  sharpness: number;
  flags: string[];
};

function clamp(n: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, n));
}

async function loadImageElement(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Failed to load image"));
    img.src = url;
  });
}

async function computeQuality(url: string): Promise<{ brightness: number; sharpness: number }> {
  const img = await loadImageElement(url);
  const w0 = img.naturalWidth || img.width || 1;
  const h0 = img.naturalHeight || img.height || 1;
  const scale = Math.min(1, 256 / Math.max(w0, h0));
  const w = Math.max(8, Math.round(w0 * scale));
  const h = Math.max(8, Math.round(h0 * scale));

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("Canvas not supported");

  ctx.drawImage(img, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h).data;

  const grays = new Float32Array(w * h);
  let sum = 0;
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const gray = 0.299 * r + 0.587 * g + 0.114 * b;
    grays[p] = gray;
    sum += gray;
  }
  const brightness = sum / (w * h) / 255;

  // Laplacian variance as a simple blur metric.
  let lapSum = 0;
  let lapSqSum = 0;
  let count = 0;
  const idx = (x: number, y: number) => y * w + x;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const c = grays[idx(x, y)];
      const l = grays[idx(x - 1, y)];
      const r = grays[idx(x + 1, y)];
      const u = grays[idx(x, y - 1)];
      const d = grays[idx(x, y + 1)];
      const lap = -4 * c + l + r + u + d;
      lapSum += lap;
      lapSqSum += lap * lap;
      count++;
    }
  }
  const mean = count ? lapSum / count : 0;
  const variance = count ? lapSqSum / count - mean * mean : 0;

  return { brightness, sharpness: variance };
}

function classifyQuality(img: ImageRecord, metrics: { brightness: number; sharpness: number }): QualityResult {
  const flags: string[] = [];

  const w = img.width ?? 0;
  const h = img.height ?? 0;
  if (w > 0 && h > 0 && (w < 512 || h < 512)) flags.push("低分辨率");

  if (metrics.brightness < 0.2) flags.push("过暗");
  if (metrics.brightness > 0.85) flags.push("过亮");

  // Heuristic threshold on Laplacian variance (after downsample).
  if (metrics.sharpness < 120) flags.push("疑似模糊");

  return { brightness: metrics.brightness, sharpness: metrics.sharpness, flags };
}

function formatPercent(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "0%";
  if (n >= 1) return "100%";
  return `${Math.round(n * 100)}%`;
}

export default function ProjectWorkspace() {
  const navigate = useNavigate();
  const params = useParams();
  const projectId = (params.projectId || "").trim();
  const [searchParams] = useSearchParams();

  const setProjectContext = useProjectContext((s) => s.setProject);
  const setDatasetContext = useProjectContext((s) => s.setDataset);
  const datasetIdInContext = useProjectContext((s) => s.datasetId);

  const [project, setProject] = useState<ProjectRecord | null>(null);
  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(datasetIdInContext);
  const [images, setImages] = useState<ImageRecord[]>([]);

  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [isChecking, setIsChecking] = useState(false);
  const [checkProgress, setCheckProgress] = useState<{ done: number; total: number } | null>(null);

  const [qualityMap, setQualityMap] = useState<Record<string, QualityResult>>({});
  const [onlyProblems, setOnlyProblems] = useState(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const folderInputRef = useRef<HTMLInputElement | null>(null);
  const didHintRef = useRef(false);

  const [cropTarget, setCropTarget] = useState<ImageRecord | null>(null);
  const cropImgRef = useRef<HTMLImageElement | null>(null);
  const [cropRect, setCropRect] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const cropDragRef = useRef<{ x: number; y: number; active: boolean }>({ x: 0, y: 0, active: false });

  const activeDataset = useMemo(
    () => datasets.find((d) => d.id === selectedDatasetId) ?? null,
    [datasets, selectedDatasetId]
  );

  const filteredImages = useMemo(() => {
    if (!onlyProblems) return images;
    return images.filter((img) => (qualityMap[img.id]?.flags?.length ?? 0) > 0);
  }, [images, onlyProblems, qualityMap]);

  const annotationProgress = useMemo(() => {
    const total = images.length;
    if (!total) return 0;
    const done = images.filter((img) => (img.annotations_count || 0) > 0).length;
    return done / total;
  }, [images]);

  const loadAll = useCallback(async () => {
    if (!projectId) return;
    setIsLoading(true);
    try {
      const p = await projectService.get(projectId);
      setProject(p);
      setProjectContext(p.id, p.name);

      const ds = await datasetService.list({ project_id: projectId, limit: 200 });
      setDatasets(ds);

      const preferred = (searchParams.get("dataset_id") || "").trim() || selectedDatasetId || ds[0]?.id || null;
      setSelectedDatasetId(preferred);
      if (preferred) {
        const d0 = ds.find((d) => d.id === preferred) ?? ds[0] ?? null;
        if (d0) setDatasetContext(d0.id, `${d0.name} ${d0.version}`);
      }
    } catch (err) {
      console.error(err);
      toast({ title: "加载失败", description: "请检查后端服务是否启动", variant: "destructive" });
    } finally {
      setIsLoading(false);
    }
  }, [projectId, searchParams, selectedDatasetId, setDatasetContext, setProjectContext]);

  const loadImages = useCallback(async () => {
    if (!projectId || !selectedDatasetId) {
      setImages([]);
      return;
    }
    try {
      const data = await imageService.list(projectId, { dataset_id: selectedDatasetId });
      setImages(data);
    } catch (err) {
      console.error(err);
      setImages([]);
    }
  }, [projectId, selectedDatasetId]);

  useEffect(() => {
    void loadAll();
  }, [loadAll]);

  useEffect(() => {
    if (didHintRef.current) return;
    if ((searchParams.get("step") || "").trim() === "import") {
      didHintRef.current = true;
      toast({ title: "下一步：导入图片", description: "请在“数据集与导入”区域上传图片，然后进行质量检查。" });
    }
  }, [searchParams]);

  useEffect(() => {
    void loadImages();
  }, [loadImages]);

  const ensureDataset = useCallback(async () => {
    if (!projectId) return null;
    const name = window.prompt("数据集名称（建议用 raw / train 等）", "raw");
    if (!name || !name.trim()) return null;
    const version = window.prompt("版本（可留空自动生成 v1/v2...）", "") || null;
    const description = window.prompt("描述（可选）", "") || "";
    try {
      const created = await datasetService.create({
        project_id: projectId,
        name: name.trim(),
        version: version?.trim() ? version.trim() : null,
        description,
        tags: null,
        is_public: false,
        splits: { train: 0.7, val: 0.2, test: 0.1 },
      });
      toast({ title: "数据集已创建", description: `${created.name} ${created.version}` });
      const ds = await datasetService.list({ project_id: projectId, limit: 200 });
      setDatasets(ds);
      setSelectedDatasetId(created.id);
      setDatasetContext(created.id, `${created.name} ${created.version}`);
      return created.id;
    } catch (err) {
      console.error(err);
      toast({ title: "创建数据集失败", description: "请检查后端或权限设置", variant: "destructive" });
      return null;
    }
  }, [projectId, setDatasetContext]);

  const uploadFiles = useCallback(
    async (files: File[]) => {
      if (!projectId) return;
      let dsId = selectedDatasetId;
      if (!dsId) {
        dsId = await ensureDataset();
      }
      if (!dsId) return;
      if (!files.length) return;

      setIsUploading(true);
      try {
        await datasetService.uploadFiles(dsId, files);
        toast({ title: "导入完成", description: `已导入 ${files.length} 个文件` });
        setQualityMap({});
        setOnlyProblems(false);
        await loadImages();
      } catch (err) {
        console.error(err);
        toast({ title: "导入失败", description: "请检查后端服务或文件格式", variant: "destructive" });
      } finally {
        setIsUploading(false);
      }
    },
    [ensureDataset, loadImages, projectId, selectedDatasetId]
  );

  const handlePickFiles = () => fileInputRef.current?.click();
  const handlePickFolder = () => folderInputRef.current?.click();

  const handleDeleteImage = useCallback(
    async (img: ImageRecord) => {
      const ok = window.confirm("确认删除此图片？该操作不可撤销");
      if (!ok) return;
      try {
        await imageService.delete(img.id);
        setImages((prev) => prev.filter((x) => x.id !== img.id));
        setQualityMap((prev) => {
          const next = { ...prev };
          delete next[img.id];
          return next;
        });
      } catch (err) {
        console.error(err);
        toast({ title: "删除失败", description: "若图片已有标注，请先清空标注再删除", variant: "destructive" });
      }
    },
    []
  );

  const handleRotate = useCallback(async (img: ImageRecord) => {
    try {
      const updated = await imageService.edit(img.id, { rotate: 90 });
      setImages((prev) => prev.map((x) => (x.id === img.id ? { ...x, ...updated } : x)));
      setQualityMap((prev) => {
        const next = { ...prev };
        delete next[img.id];
        return next;
      });
    } catch (err) {
      console.error(err);
      toast({ title: "旋转失败", description: "若图片已有标注，将禁止编辑", variant: "destructive" });
    }
  }, []);

  const openCrop = useCallback((img: ImageRecord) => {
    setCropTarget(img);
    setCropRect(null);
  }, []);

  const handleCropMouseDown = (e: React.MouseEvent) => {
    if (!cropImgRef.current) return;
    const rect = cropImgRef.current.getBoundingClientRect();
    const x = clamp(e.clientX - rect.left, 0, rect.width);
    const y = clamp(e.clientY - rect.top, 0, rect.height);
    cropDragRef.current = { x, y, active: true };
    setCropRect({ x, y, w: 1, h: 1 });
  };

  const handleCropMouseMove = (e: React.MouseEvent) => {
    if (!cropDragRef.current.active || !cropImgRef.current) return;
    const rect = cropImgRef.current.getBoundingClientRect();
    const x2 = clamp(e.clientX - rect.left, 0, rect.width);
    const y2 = clamp(e.clientY - rect.top, 0, rect.height);
    const x1 = cropDragRef.current.x;
    const y1 = cropDragRef.current.y;
    const x = Math.min(x1, x2);
    const y = Math.min(y1, y2);
    const w = Math.abs(x2 - x1);
    const h = Math.abs(y2 - y1);
    setCropRect({ x, y, w, h });
  };

  const handleCropMouseUp = () => {
    cropDragRef.current.active = false;
  };

  const applyCrop = useCallback(async () => {
    if (!cropTarget || !cropImgRef.current || !cropRect) return;
    const imgEl = cropImgRef.current;
    const rect = imgEl.getBoundingClientRect();
    const naturalW = imgEl.naturalWidth || cropTarget.width || 1;
    const naturalH = imgEl.naturalHeight || cropTarget.height || 1;

    const scaleX = naturalW / rect.width;
    const scaleY = naturalH / rect.height;
    const x = Math.round(cropRect.x * scaleX);
    const y = Math.round(cropRect.y * scaleY);
    const w = Math.round(cropRect.w * scaleX);
    const h = Math.round(cropRect.h * scaleY);

    if (w < 2 || h < 2) {
      toast({ title: "裁剪区域太小", variant: "destructive" });
      return;
    }

    try {
      const updated = await imageService.edit(cropTarget.id, { crop: { x, y, width: w, height: h } });
      setImages((prev) => prev.map((x0) => (x0.id === cropTarget.id ? { ...x0, ...updated } : x0)));
      setCropTarget(null);
      setCropRect(null);
      setQualityMap((prev) => {
        const next = { ...prev };
        delete next[cropTarget.id];
        return next;
      });
    } catch (err) {
      console.error(err);
      toast({ title: "裁剪失败", description: "若图片已有标注，将禁止编辑", variant: "destructive" });
    }
  }, [cropRect, cropTarget]);

  const runQualityCheck = useCallback(async () => {
    if (!images.length) return;
    setIsChecking(true);
    setCheckProgress({ done: 0, total: images.length });
    try {
      const next: Record<string, QualityResult> = {};

      const concurrency = 4;
      let idx = 0;
      let done = 0;
      const worker = async () => {
        while (idx < images.length) {
          const i = idx++;
          const img = images[i];
          const url = resolveImageUrl(img.source_url);
          try {
            const metrics = await computeQuality(url);
            next[img.id] = classifyQuality(img, metrics);
          } catch {
            next[img.id] = { brightness: 0, sharpness: 0, flags: ["无法检测"] };
          }
          done++;
          setCheckProgress({ done, total: images.length });
        }
      };

      await Promise.all(Array.from({ length: concurrency }, () => worker()));
      setQualityMap(next);
      toast({ title: "质量检查完成", description: "可使用“仅看问题图”快速筛选" });
    } finally {
      setIsChecking(false);
      setCheckProgress(null);
    }
  }, [images]);

  const goAnnotate = () => {
    if (!projectId || !selectedDatasetId) {
      toast({ title: "请先选择项目与数据集", variant: "destructive" });
      return;
    }
    const qs = new URLSearchParams();
    qs.set("project_id", projectId);
    qs.set("dataset_id", selectedDatasetId);
    navigate(`/annotate?${qs.toString()}`);
  };

  if (!projectId) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>项目不存在</CardTitle>
        </CardHeader>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <h1 className="text-3xl font-bold tracking-tight truncate">
            项目管理 / {project?.name || projectId}
          </h1>
          <div className="mt-2 text-sm text-muted-foreground flex flex-wrap items-center gap-2">
            <span>标注进度：{formatPercent(annotationProgress)}</span>
            {activeDataset ? (
              <Badge variant="outline">{activeDataset.name} {activeDataset.version}</Badge>
            ) : (
              <Badge variant="outline">未选择数据集</Badge>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={() => void loadAll()} disabled={isLoading} className="gap-2">
            <RefreshCw className={cn("h-4 w-4", isLoading ? "animate-spin" : "")} />
            刷新
          </Button>
          <Button onClick={goAnnotate} className="gap-2" disabled={!selectedDatasetId}>
            <ScanEye className="h-4 w-4" />
            进入标注
          </Button>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>流程</CardTitle>
          <CardDescription>创建项目 → 导入图片 → 质量检查 → 进入标注</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-3">
          <div className="p-4 rounded-lg border bg-muted/20">
            <div className="font-medium">1. 导入图片</div>
            <div className="text-sm text-muted-foreground mt-1">支持批量上传 / ZIP / 文件夹导入</div>
          </div>
          <div className="p-4 rounded-lg border bg-muted/20">
            <div className="font-medium">2. 质量检查</div>
            <div className="text-sm text-muted-foreground mt-1">一键筛选模糊/过暗/低分辨率</div>
          </div>
          <div className="p-4 rounded-lg border bg-muted/20">
            <div className="font-medium">3. 进入标注</div>
            <div className="text-sm text-muted-foreground mt-1">自动继承当前项目与数据集上下文</div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>数据集与导入</CardTitle>
          <CardDescription>在项目内完成数据集管理（不再依赖独立数据集模块）</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div className="flex flex-wrap items-center gap-2">
              <select
                className="h-9 rounded-md border border-input bg-background px-3 text-sm"
                value={selectedDatasetId ?? ""}
                onChange={(e) => {
                  const id = (e.target.value || "").trim() || null;
                  setSelectedDatasetId(id);
                  const d = datasets.find((x) => x.id === id) ?? null;
                  if (id && d) setDatasetContext(id, `${d.name} ${d.version}`);
                  setQualityMap({});
                }}
                disabled={datasets.length === 0}
              >
                {datasets.length === 0 ? <option value="">暂无数据集</option> : null}
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} {d.version}
                  </option>
                ))}
              </select>
              <Button variant="outline" onClick={() => void ensureDataset()} className="gap-2">
                <Upload className="h-4 w-4" />
                新建数据集
              </Button>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Button variant="outline" onClick={handlePickFiles} disabled={isUploading} className="gap-2">
                <Upload className="h-4 w-4" />
                上传图片
              </Button>
              <Button variant="outline" onClick={handlePickFolder} disabled={isUploading} className="gap-2">
                <FolderUp className="h-4 w-4" />
                上传文件夹
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*,.zip"
                multiple
                className="hidden"
                onChange={(e) => {
                  const files = Array.from(e.target.files || []);
                  e.target.value = "";
                  void uploadFiles(files);
                }}
              />
                <input
                  ref={folderInputRef}
                  type="file"
                  webkitdirectory="true"
                  multiple
                className="hidden"
                onChange={(e) => {
                  const files = Array.from(e.target.files || []);
                  e.target.value = "";
                  void uploadFiles(files);
                }}
              />
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Button onClick={() => void runQualityCheck()} disabled={isChecking || filteredImages.length === 0} className="gap-2">
              {isChecking ? <RefreshCw className="h-4 w-4 animate-spin" /> : <AlertTriangle className="h-4 w-4" />}
              一键质量检查
            </Button>
            <label className="flex items-center gap-2 text-sm text-muted-foreground">
              <input type="checkbox" checked={onlyProblems} onChange={(e) => setOnlyProblems(e.target.checked)} />
              仅看问题图
            </label>
            {checkProgress ? (
              <span className="text-sm text-muted-foreground">
                检测中：{checkProgress.done}/{checkProgress.total}
              </span>
            ) : null}
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {filteredImages.map((img) => {
          const q = qualityMap[img.id] || null;
          const url = resolveImageUrl(img.source_url);
          return (
            <Card key={img.id} className="overflow-hidden">
              <div className="relative aspect-square bg-muted">
                <img
                  src={url}
                  alt={img.filename}
                  className="absolute inset-0 h-full w-full object-contain bg-muted"
                  loading="lazy"
                />
                <div className="absolute right-2 top-2 flex gap-1">
                  <Button variant="secondary" size="icon" className="h-8 w-8" onClick={() => void handleRotate(img)} title="旋转">
                    <RotateCw className="h-4 w-4" />
                  </Button>
                  <Button variant="secondary" size="icon" className="h-8 w-8" onClick={() => openCrop(img)} title="裁剪">
                    <Crop className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="destructive"
                    size="icon"
                    className="h-8 w-8"
                    onClick={() => void handleDeleteImage(img)}
                    title="删除"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
              <CardContent className="p-3 space-y-2">
                <div className="text-sm font-medium truncate" title={img.filename}>
                  {img.filename}
                </div>
                <div className="text-xs text-muted-foreground">
                  {img.width && img.height ? `${img.width}×${img.height}` : "未知尺寸"}
                </div>
                {q ? (
                  <div className="flex flex-wrap gap-1">
                    {q.flags.length === 0 ? (
                      <Badge variant="outline" className="border-emerald-500/40 text-emerald-700">
                        通过
                      </Badge>
                    ) : (
                      q.flags.map((f) => (
                        <Badge key={f} variant="outline" className="border-amber-500/40 text-amber-700">
                          {f}
                        </Badge>
                      ))
                    )}
                  </div>
                ) : (
                  <div className="text-xs text-muted-foreground">未检测</div>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Crop modal */}
      {cropTarget ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
          <Card className="w-full max-w-3xl">
            <CardHeader>
              <CardTitle>裁剪图片</CardTitle>
              <CardDescription>拖拽选择裁剪区域，然后点击“应用裁剪”</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="relative w-full overflow-auto max-h-[60vh]">
                <img
                  ref={cropImgRef}
                  src={resolveImageUrl(cropTarget.source_url)}
                  alt={cropTarget.filename}
                  className="max-w-full h-auto select-none"
                  onMouseDown={handleCropMouseDown}
                  onMouseMove={handleCropMouseMove}
                  onMouseUp={handleCropMouseUp}
                  draggable={false}
                />
                {cropRect ? (
                  <div
                    className="absolute border-2 border-primary bg-primary/10 pointer-events-none"
                    style={{
                      left: cropRect.x,
                      top: cropRect.y,
                      width: cropRect.w,
                      height: cropRect.h,
                    }}
                  />
                ) : null}
              </div>
              <div className="flex justify-end gap-2">
                <Button variant="outline" onClick={() => setCropTarget(null)}>
                  取消
                </Button>
                <Button onClick={() => void applyCrop()} disabled={!cropRect} className="gap-2">
                  <Crop className="h-4 w-4" />
                  应用裁剪
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      ) : null}
    </div>
  );
}
