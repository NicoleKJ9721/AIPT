import { useState, useRef, useEffect, useCallback, Fragment } from "react";
import { Stage, Layer, Rect, Line, Group, Image as KonvaImage, Label as KonvaLabel, Tag as KonvaTag, Text } from "react-konva";
import useImage from "use-image";
import Konva from "konva";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { 
  Undo, 
  Redo, 
  ZoomIn, 
  ZoomOut, 
  ArrowRight,
  Keyboard,
  CheckCircle2,
  AlertCircle,
  Layers,
  ChevronLeft,
  ChevronRight,
  Wand2,
  X,
  Trash2
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  aiService,
  annotationService,
  datasetService,
  imageService,
  labelService,
  projectService,
  smartAnnotationService,
  type AnnotationCreatePayload,
  type AnnotationRecord,
  type DatasetRecord,
  type ImageRecord,
  type LabelClassRecord,
  type ProjectRecord,
} from "@/lib/api";
import { v4 as uuidv4 } from "uuid";
import { toast } from "@/components/ui/use-toast";
import { AnnotationToolbar } from "@/components/annotation/AnnotationToolbar";
import { LayerPanel } from "@/components/annotation/LayerPanel";
import type { Annotation, ImageItem, LabelClass } from "@/types/annotation";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useProjectContext } from "@/store/projectContext";

const DEFAULT_CLASSES: LabelClass[] = [
    // PV EL defect defaults (English short codes recommended for on-box display)
    { id: "1", name: "hd", color: "#ef4444", shortcut: "1" }, // black dot
    { id: "2", name: "crack", color: "#3b82f6", shortcut: "2" },
    { id: "3", name: "scratch", color: "#22c55e", shortcut: "3" },
    { id: "4", name: "broken", color: "#eab308", shortcut: "4" },
    { id: "5", name: "finger", color: "#a855f7", shortcut: "5" },
];

function resolveImageUrl(sourceUrl: string | null | undefined): string {
  const url = (sourceUrl || "").trim();
  if (!url) return "";
  if (url.startsWith("http://") || url.startsWith("https://") || url.startsWith("data:")) return url;
  if (url.startsWith("/api/")) return url;
  if (url.startsWith("/")) return `/api${url}`;
  return url;
}

function toImageItem(img: ImageRecord): ImageItem {
  const annotationsCount = typeof img.annotations_count === "number" ? img.annotations_count : 0;
  return {
    id: img.id,
    url: resolveImageUrl(img.source_url),
    name: img.filename,
    status: annotationsCount > 0 ? "completed" : "pending",
    annotationsCount,
    datasetId: img.dataset_id ?? null,
    datasetFileId: img.dataset_file_id ?? null,
  };
}

function toFrontendAnnotation(ann: AnnotationRecord): Annotation {
  return {
    id: ann.id,
    type: ann.type,
    label: ann.label,
    color: ann.color,
    visible: ann.visible,
    x: ann.x ?? undefined,
    y: ann.y ?? undefined,
    width: ann.width ?? undefined,
    height: ann.height ?? undefined,
    points: ann.points ?? undefined,
  };
}

function toBackendAnnotationPayload(ann: Annotation): AnnotationCreatePayload {
  return {
    id: ann.id,
    type: ann.type,
    label: ann.label,
    color: ann.color,
    visible: ann.visible ?? true,
    x: ann.x ?? null,
    y: ann.y ?? null,
    width: ann.width ?? null,
    height: ann.height ?? null,
    points: ann.points ? ann.points.map((p) => Number(p)) : null,
  };
}

const LABEL_COLOR_POOL = [
  "#ef4444",
  "#f97316",
  "#f59e0b",
  "#eab308",
  "#84cc16",
  "#22c55e",
  "#10b981",
  "#14b8a6",
  "#06b6d4",
  "#0ea5e9",
  "#3b82f6",
  "#6366f1",
  "#8b5cf6",
  "#a855f7",
  "#d946ef",
  "#ec4899",
  "#f43f5e",
  "#334155",
  "#78716c",
  "#111827",
];

const MIN_COLOR_DELTA_E = 20;

function normalizeLabel(value: string | null | undefined): string {
  return (value || "").trim();
}

function hashString(value: string): number {
  let hash = 0;
  for (let i = 0; i < value.length; i++) {
    hash = (hash * 31 + value.charCodeAt(i)) | 0;
  }
  return Math.abs(hash);
}

type LabColor = { l: number; a: number; b: number };

function normalizeHexColor(value: string | null | undefined): string | null {
  const candidate = (value || "").trim();
  if (!candidate) return null;
  if (!/^#[0-9a-fA-F]{6}$/.test(candidate)) return null;
  return candidate.toLowerCase();
}

function hexToRgb(color: string): { r: number; g: number; b: number } | null {
  const hex = normalizeHexColor(color)?.slice(1);
  if (!hex) return null;
  const r = Number.parseInt(hex.slice(0, 2), 16);
  const g = Number.parseInt(hex.slice(2, 4), 16);
  const b = Number.parseInt(hex.slice(4, 6), 16);
  if (![r, g, b].every((n) => Number.isFinite(n))) return null;
  return { r, g, b };
}

function srgbToLinear(c: number): number {
  const v = c / 255;
  return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
}

function rgbToXyz(rgb: { r: number; g: number; b: number }): { x: number; y: number; z: number } {
  const r = srgbToLinear(rgb.r);
  const g = srgbToLinear(rgb.g);
  const b = srgbToLinear(rgb.b);

  const x = r * 0.4124 + g * 0.3576 + b * 0.1805;
  const y = r * 0.2126 + g * 0.7152 + b * 0.0722;
  const z = r * 0.0193 + g * 0.1192 + b * 0.9505;
  return { x, y, z };
}

function xyzToLab(xyz: { x: number; y: number; z: number }): LabColor {
  const Xn = 0.95047;
  const Yn = 1.0;
  const Zn = 1.08883;

  const fx = xyz.x / Xn;
  const fy = xyz.y / Yn;
  const fz = xyz.z / Zn;

  const f = (t: number) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const xr = f(fx);
  const yr = f(fy);
  const zr = f(fz);

  const l = 116 * yr - 16;
  const a = 500 * (xr - yr);
  const b = 200 * (yr - zr);
  return { l, a, b };
}

function hexToLab(color: string): LabColor | null {
  const rgb = hexToRgb(color);
  if (!rgb) return null;
  return xyzToLab(rgbToXyz(rgb));
}

function deltaE76(a: LabColor, b: LabColor): number {
  const dl = a.l - b.l;
  const da = a.a - b.a;
  const db = a.b - b.b;
  return Math.sqrt(dl * dl + da * da + db * db);
}

function pickDistinctColor(existingColors: string[], preferred?: string | null): string {
  const preferredHex = normalizeHexColor(preferred);
  if (preferredHex && !existingColors.some((c) => normalizeHexColor(c) === preferredHex)) {
    return preferredHex;
  }

  const existingLabs = existingColors
    .map(normalizeHexColor)
    .filter((c): c is string => Boolean(c))
    .map((c) => hexToLab(c))
    .filter((c): c is LabColor => Boolean(c));

  if (existingLabs.length === 0) {
    return normalizeHexColor(LABEL_COLOR_POOL[0]) || "#ef4444";
  }

  let bestColor = normalizeHexColor(LABEL_COLOR_POOL[0]) || "#ef4444";
  let bestMin = -Infinity;
  for (const candidate of LABEL_COLOR_POOL) {
    const hex = normalizeHexColor(candidate);
    const lab = hex ? hexToLab(hex) : null;
    if (!hex || !lab) continue;
    const min = Math.min(...existingLabs.map((e) => deltaE76(lab, e)));
    if (min >= MIN_COLOR_DELTA_E) return hex;
    if (min > bestMin) {
      bestMin = min;
      bestColor = hex;
    }
  }
  return bestColor;
}

function nextShortcut(existing: LabelClass[]): string {
  const used = new Set(existing.map((c) => (c.shortcut || "").trim()).filter(Boolean));
  for (let i = 1; i <= 9; i++) {
    const s = String(i);
    if (!used.has(s)) return s;
  }
  return String(existing.length + 1);
}

function rectIou(a: [number, number, number, number], b: [number, number, number, number]): number {
  const ix1 = Math.max(a[0], b[0]);
  const iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]);
  const iy2 = Math.min(a[3], b[3]);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const denom = areaA + areaB - inter;
  if (denom <= 0) return 0;
  return inter / denom;
}

function ensureClassesIncludeLabels(existing: LabelClass[], labels: string[]): LabelClass[] {
  const wanted = labels.map(normalizeLabel).filter(Boolean);
  if (!wanted.length) return existing;
  const existingNames = new Set(existing.map((c) => normalizeLabel(c.name)));
  const missing = wanted.filter((l) => !existingNames.has(l));
  if (!missing.length) return existing;

  const next = [...existing];
  for (const label of missing) {
    const color = pickDistinctColor(next.map((c) => c.color));
    next.push({
      id: `cls_${hashString(label)}_${Date.now()}`,
      name: label,
      color,
      shortcut: nextShortcut(next),
    });
  }
  return next;
}

function toFrontendLabelClass(record: LabelClassRecord): LabelClass {
  return {
    id: record.id,
    name: normalizeLabel(record.name),
    color: normalizeHexColor(record.color) || "#ef4444",
    shortcut: (record.shortcut || "").trim() || "",
  };
}

export default function AnnotationStudio() {
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
  const activeDatasetId = datasetIdFromQuery || datasetIdInContext;

  const [projects, setProjects] = useState<ProjectRecord[]>([]);
  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  const [isLoadingProjects, setIsLoadingProjects] = useState(true);
  const [isLoadingDatasets, setIsLoadingDatasets] = useState(false);
  const [isLoadingImages, setIsLoadingImages] = useState(false);
  const [isLoadingAnnotations, setIsLoadingAnnotations] = useState(false);

  const [selectedTool, setSelectedTool] = useState<"select" | "rect" | "polygon" | "move">("select");
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [classes, setClasses] = useState<LabelClass[]>(DEFAULT_CLASSES);
  const [selectedClassId, setSelectedClassId] = useState<string>(DEFAULT_CLASSES[0].id);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [draggingId, setDraggingId] = useState<string | null>(null);

  type SmartMode = "detect" | "segment" | null;
  const [isSmartPanelOpen, setIsSmartPanelOpen] = useState(false);
  const [smartTab, setSmartTab] = useState<Exclude<SmartMode, null>>("detect");
  const [smartMode, setSmartMode] = useState<SmartMode>(null);
  const [smartBusy, setSmartBusy] = useState(false);

  const [smartDetectScope, setSmartDetectScope] = useState<"image" | "dataset">("dataset");
  const [smartDetectThreshold, setSmartDetectThreshold] = useState(0.6);
  const [smartDetectMaxImages, setSmartDetectMaxImages] = useState<string>("");
  const [smartDetectMaxDet, setSmartDetectMaxDet] = useState(20);
  const [smartDetectMinDistance, setSmartDetectMinDistance] = useState<string>("");
  const [smartDetectDedupIou, setSmartDetectDedupIou] = useState(0.8);
  const [smartDetectOnlyUnannotated, setSmartDetectOnlyUnannotated] = useState(true);
  const smartDetectStartRef = useRef<{ x: number; y: number } | null>(null);
  const [smartDetectDraft, setSmartDetectDraft] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null);

  const [smartSegTolerance, setSmartSegTolerance] = useState(0.08);
  const [smartSegSimplify, setSmartSegSimplify] = useState(2.0);
  const [smartSegEngine, setSmartSegEngine] = useState("auto");
  const [autoLabelConf, setAutoLabelConf] = useState(0.25);
  const [autoLabelIou, setAutoLabelIou] = useState(0.7);
  const [autoLabelMaxDet, setAutoLabelMaxDet] = useState(50);
  const [autoLabelDedupIou, setAutoLabelDedupIou] = useState(0.8);

  const cancelSmartMode = useCallback(() => {
    smartDetectStartRef.current = null;
    setSmartDetectDraft(null);
    setSmartBusy(false);
    setSmartMode(null);
  }, []);

  // Load label classes from backend (project-scoped). Seed defaults when missing.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!activeProjectId) {
        setClasses(DEFAULT_CLASSES);
        setSelectedClassId(DEFAULT_CLASSES[0].id);
        return;
      }

      try {
        let data = await labelService.list(activeProjectId);
        if (cancelled) return;

        if (!data || data.length === 0) {
          for (const cls of DEFAULT_CLASSES) {
            try {
              await labelService.create(activeProjectId, {
                name: cls.name,
                color: cls.color,
                shortcut: cls.shortcut,
              });
            } catch {
              // ignore conflicts during seeding
            }
          }
          data = await labelService.list(activeProjectId);
          if (cancelled) return;
        }

        const next = (data || []).map(toFrontendLabelClass).filter((c) => c.id && c.name);
        if (next.length > 0) {
          setClasses(next);
          setSelectedClassId((prev) => (next.some((c) => c.id === prev) ? prev : next[0].id));
          return;
        }

        setClasses(DEFAULT_CLASSES);
        setSelectedClassId(DEFAULT_CLASSES[0].id);
      } catch (err) {
        console.error(err);
        toast({
          title: "加载标签类别失败",
          description: "请检查后端服务/权限设置或后端日志",
          variant: "destructive",
        });
        setClasses(DEFAULT_CLASSES);
        setSelectedClassId(DEFAULT_CLASSES[0].id);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [activeProjectId]);
  
  // Image Navigation State
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [imageList, setImageList] = useState<ImageItem[]>([]);
  
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  
  const [history, setHistory] = useState<Annotation[][]>([]);
  const [historyStep, setHistoryStep] = useState(-1);
  
  // Polygon drawing state
  const [currentPolyPoints, setCurrentPolyPoints] = useState<number[]>([]);
  const [isDrawingPoly, setIsDrawingPoly] = useState(false);
  const [mousePos, setMousePos] = useState<{x: number, y: number} | null>(null);

  // Auto-save state
  const [saveStatus, setSaveStatus] = useState<"saved" | "saving" | "unsaved">("saved");
  const [showShortcuts, setShowShortcuts] = useState(false);

  // New UX Features
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [showCrosshair, setShowCrosshair] = useState(false);
  const [activeTab, setActiveTab] = useState("layers");
  
  // View State Management (Scale & Position per image)
  const [viewStates, setViewStates] = useState<Record<number, {scale: number, x: number, y: number}>>({});
  const [labelPicker, setLabelPicker] = useState<{
    open: boolean;
    annotationId: string | null;
    x: number;
    y: number;
    query: string;
  }>({ open: false, annotationId: null, x: 0, y: 0, query: "" });

  const stageRef = useRef<Konva.Stage>(null);
  const imgRef = useRef<Konva.Image>(null);
  const didAutoFitRef = useRef<Record<string, boolean>>({});
  
  // Load current image
  const [image] = useImage(imageList[currentImageIndex]?.url || "");
  
  const containerRef = useRef<HTMLDivElement>(null);
  const labelPickerRef = useRef<HTMLDivElement>(null);
  const [stageDimensions, setStageDimensions] = useState({ width: 1000, height: 600 });
  const currentImage = imageList[currentImageIndex] ?? null;
  const currentImageId = currentImage?.id ?? null;
  const imageCanvasWidth = image?.width || 1024;
  const imageCanvasHeight = image?.height || 768;

  useEffect(() => {
      console.log("Container Ref Changed:", containerRef.current);
      const checkSize = () => {
          if (containerRef.current) {
              const { offsetWidth, offsetHeight } = containerRef.current;
              console.log("Stage Dimensions Updated:", offsetWidth, offsetHeight);
              setStageDimensions({
                  width: offsetWidth,
                  height: offsetHeight
              });
          }
      };
      checkSize();
      window.addEventListener('resize', checkSize);
      return () => window.removeEventListener('resize', checkSize);
  }, []);

  const openLabelPicker = useCallback((annotationId: string, clientX?: number, clientY?: number) => {
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const rawX = typeof clientX === "number" ? clientX - rect.left : rect.width / 2;
    const rawY = typeof clientY === "number" ? clientY - rect.top : rect.height / 2;

    // Keep the menu inside the canvas container.
    const x = Math.max(8, Math.min(rect.width - 8, rawX));
    const y = Math.max(8, Math.min(rect.height - 8, rawY));

    setLabelPicker({ open: true, annotationId, x, y, query: "" });
  }, []);

  const closeLabelPicker = useCallback(() => {
    setLabelPicker((prev) => ({ ...prev, open: false, annotationId: null, query: "" }));
  }, []);

  useEffect(() => {
    if (!labelPicker.open) return;

    const onMouseDown = (e: MouseEvent) => {
      const panel = labelPickerRef.current;
      if (!panel) return;
      if (!panel.contains(e.target as Node)) {
        closeLabelPicker();
      }
    };

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        closeLabelPicker();
      }
    };

    window.addEventListener("mousedown", onMouseDown);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [closeLabelPicker, labelPicker.open]);

  // Close the inline label picker when switching images to avoid stale state.
  useEffect(() => {
    closeLabelPicker();
  }, [closeLabelPicker, currentImageId]);

  // --- Project/Dataset/Image data loading ---
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
    let cancelled = false;
    (async () => {
      try {
        setIsLoadingProjects(true);
        const data = await projectService.list();
        if (cancelled) return;
        setProjects(data);
      } catch (err) {
        console.error(err);
        toast({
          title: "加载项目失败",
          description: "请检查后端服务是否已启动（http://127.0.0.1:8000）",
          variant: "destructive",
        });
      } finally {
        if (!cancelled) setIsLoadingProjects(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (activeProjectId) return;
    if (!projects.length) return;
    const first = projects[0];
    setProjectContext(first.id, first.name);
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        next.set("project_id", first.id);
        next.delete("dataset_id");
        return next;
      },
      { replace: true }
    );
  }, [activeProjectId, projects, setProjectContext, setSearchParams]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!activeProjectId) {
        setDatasets([]);
        clearDatasetContext();
        setImageList([]);
        return;
      }
      try {
        setIsLoadingDatasets(true);
        const data = await datasetService.list({ project_id: activeProjectId, limit: 200 });
        if (cancelled) return;
        setDatasets(data);

        const desired = activeDatasetId;
        if (desired && data.some((d) => d.id === desired)) {
          return;
        }
        if (data[0]) {
          const nextDataset = data[0];
          setDatasetContext(nextDataset.id, `${nextDataset.name} ${nextDataset.version}`);
          setSearchParams((prev) => {
            const next = new URLSearchParams(prev);
            next.set("project_id", activeProjectId);
            next.set("dataset_id", nextDataset.id);
            return next;
          });
        } else {
          clearDatasetContext();
          setSearchParams((prev) => {
            const next = new URLSearchParams(prev);
            next.delete("dataset_id");
            return next;
          });
        }
      } catch (err) {
        console.error(err);
        toast({
          title: "加载数据集失败",
          description: "请稍后重试或查看后端日志",
          variant: "destructive",
        });
      } finally {
        if (!cancelled) setIsLoadingDatasets(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [activeProjectId, activeDatasetId, clearDatasetContext, setDatasetContext, setSearchParams]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!activeProjectId) return;
      try {
        setIsLoadingImages(true);
        const data = await imageService.list(activeProjectId, activeDatasetId ? { dataset_id: activeDatasetId } : undefined);
        if (cancelled) return;
        const items = data.map(toImageItem);
        setImageList(items);
        setCurrentImageIndex(0);
        setViewStates({});
        setSelectedId(null);
        setAnnotations([]);
        setHistory([[]]);
        setHistoryStep(0);
        setSaveStatus("saved");
      } catch (err) {
        console.error(err);
        toast({
          title: "加载图片失败",
          description: "请确认项目/数据集中已导入图片",
          variant: "destructive",
        });
        setImageList([]);
      } finally {
        if (!cancelled) setIsLoadingImages(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [activeProjectId, activeDatasetId]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!currentImageId) {
        setAnnotations([]);
        setHistory([[]]);
        setHistoryStep(0);
        setSelectedId(null);
        setSaveStatus("saved");
        return;
      }
      try {
        setIsLoadingAnnotations(true);
        const data = await annotationService.listByImage(currentImageId);
        if (cancelled) return;
        const anns = data.map(toFrontendAnnotation);
        setClasses((prev) => ensureClassesIncludeLabels(prev, anns.map((a) => a.label)));
        setAnnotations(anns);
        setHistory([anns]);
        setHistoryStep(0);
        setSelectedId(null);
        setSaveStatus("saved");
      } catch (err) {
        console.error(err);
        toast({
          title: "加载标注失败",
          description: "请稍后重试或查看后端日志",
          variant: "destructive",
        });
      } finally {
        if (!cancelled) setIsLoadingAnnotations(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [currentImageId]);

  useEffect(() => {
    if (!currentImageId) return;
    if (!image) return;
    if (viewStates[currentImageIndex]) return;
    if (didAutoFitRef.current[currentImageId]) return;

    const w = stageDimensions.width || 1;
    const h = stageDimensions.height || 1;
    const fit = Math.min(w / imageCanvasWidth, h / imageCanvasHeight);
    const nextScale = Math.max(0.1, Math.min(5, Number.isFinite(fit) && fit > 0 ? fit : 1));
    const nextPos = {
      x: (w - imageCanvasWidth * nextScale) / 2,
      y: (h - imageCanvasHeight * nextScale) / 2,
    };

    didAutoFitRef.current[currentImageId] = true;
    setScale(nextScale);
    setPosition(nextPos);
  }, [
    currentImageId,
    currentImageIndex,
    didAutoFitRef,
    image,
    imageCanvasHeight,
    imageCanvasWidth,
    stageDimensions.height,
    stageDimensions.width,
    viewStates,
  ]);

  // --- Zoom Logic ---
  const handleWheel = (e: Konva.KonvaEventObject<WheelEvent>) => {
      e.evt.preventDefault();
      const stage = stageRef.current;
      if (!stage) return;

      const oldScale = stage.scaleX();
      const pointer = stage.getPointerPosition();
      if (!pointer) return;

      const mousePointTo = {
          x: (pointer.x - stage.x()) / oldScale,
          y: (pointer.y - stage.y()) / oldScale,
      };

      // Scroll direction: down (+deltaY) -> zoom out, up (-deltaY) -> zoom in
      const direction = e.evt.deltaY > 0 ? -1 : 1;
      const scaleBy = 1.1;
      
      let newScale = direction > 0 ? oldScale * scaleBy : oldScale / scaleBy;
      
      // Limit scale (10% - 500%)
      newScale = Math.max(0.1, Math.min(5, newScale));

      const newPos = {
          x: pointer.x - mousePointTo.x * newScale,
          y: pointer.y - mousePointTo.y * newScale,
      };

      setScale(newScale);
      setPosition(newPos);
  };

  const handleResetView = () => {
      const w = stageDimensions.width || 1;
      const h = stageDimensions.height || 1;
      const fit = Math.min(w / imageCanvasWidth, h / imageCanvasHeight);
      const nextScale = Math.max(0.1, Math.min(5, Number.isFinite(fit) && fit > 0 ? fit : 1));
      const nextPos = {
          x: (w - imageCanvasWidth * nextScale) / 2,
          y: (h - imageCanvasHeight * nextScale) / 2,
      };
      setScale(nextScale);
      setPosition(nextPos);
  };

  const handleAIAutoLabel = async () => {
    try {
      const currentImage = imageList[currentImageIndex];
      if (!currentImage?.url) {
        toast({
          title: "暂无可标注图片",
          description: "请先在数据集模块导入图片后再进行标注",
          variant: "destructive",
        });
        return;
      }

      const response = await fetch(currentImage.url);
      const blob = await response.blob();
      const file = new File([blob], currentImage.name, { type: blob.type });

      const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
      const conf = clamp01(Number(autoLabelConf) || 0.25);
      const iou = clamp01(Number(autoLabelIou) || 0.7);
      const maxDet = Math.max(1, Math.floor(Number(autoLabelMaxDet) || 50));
      const result = await aiService.predict(file, { conf, iou, max_det: maxDet });

      const detectedLabels = Array.from(new Set(result.detections.map((det) => normalizeLabel(det.class)).filter(Boolean)));
      const byName = new Map(classes.map((c) => [normalizeLabel(c.name), c] as const));
      const createdClasses: LabelClass[] = [];
      if (detectedLabels.length > 0) {
        for (const name of detectedLabels) {
          if (byName.has(name)) continue;
          const color = pickDistinctColor([...classes, ...createdClasses].map((c) => c.color));
          const shortcut = nextShortcut([...classes, ...createdClasses]);

          if (activeProjectId) {
            try {
              const created = await labelService.create(activeProjectId, { name, color, shortcut });
              const nextClass = toFrontendLabelClass(created);
              createdClasses.push(nextClass);
              byName.set(normalizeLabel(nextClass.name), nextClass);
            } catch (error) {
              console.error("Create label for AI detections failed:", error);
              const nextClass: LabelClass = {
                id: `cls_${hashString(name)}_${Date.now()}`,
                name,
                color,
                shortcut,
              };
              createdClasses.push(nextClass);
              byName.set(normalizeLabel(nextClass.name), nextClass);
            }
          } else {
            const nextClass: LabelClass = {
              id: `cls_${hashString(name)}_${Date.now()}`,
              name,
              color,
              shortcut,
            };
            createdClasses.push(nextClass);
            byName.set(normalizeLabel(nextClass.name), nextClass);
          }
        }
      }
      if (createdClasses.length > 0) {
        setClasses((prev) => [...prev, ...createdClasses]);
      }

      const dedupIou = clamp01(Number(autoLabelDedupIou) || 0.8);
      const existingRects = annotations
        .filter((a) => a.type === "rect" && typeof a.x === "number" && typeof a.y === "number" && typeof a.width === "number" && typeof a.height === "number")
        .map((a) => [Number(a.x), Number(a.y), Number(a.x) + Number(a.width), Number(a.y) + Number(a.height)] as [number, number, number, number]);
      const newRects: Array<[number, number, number, number]> = [];
      const sortedDets = [...result.detections].sort((a, b) => (Number(b.confidence) || 0) - (Number(a.confidence) || 0));

      const newAnnotations: Annotation[] = sortedDets.flatMap((det) => {
        const x1 = Number(det.bbox[0]) || 0;
        const y1 = Number(det.bbox[1]) || 0;
        const x2 = Number(det.bbox[2]) || 0;
        const y2 = Number(det.bbox[3]) || 0;
        if (x2 <= x1 || y2 <= y1) return [];
        const rect: [number, number, number, number] = [x1, y1, x2, y2];
        if (existingRects.some((r) => rectIou(rect, r) >= dedupIou)) return [];
        if (newRects.some((r) => rectIou(rect, r) >= dedupIou)) return [];
        newRects.push(rect);

        const label = normalizeLabel(det.class) || "unknown";
        const labelClass = byName.get(label) || classes[0];
        return [
          {
            id: uuidv4(),
            type: "rect",
            x: x1,
            y: y1,
            width: x2 - x1,
            height: y2 - y1,
            label,
            color: labelClass.color,
            visible: true,
          },
        ];
      });

      addToHistory([...annotations, ...newAnnotations]);
      toast({
        title: "模型自动标注完成",
        description: `新增 ${newAnnotations.length} 个目标（阈值 conf=${conf.toFixed(2)}, 去重 IoU=${dedupIou.toFixed(2)}）`,
      });
    } catch (error) {
      console.error("AI Auto Label Error:", error);
      toast({
        title: "模型自动标注失败",
        description: "请检查后端服务、模型加载状态和控制台日志",
        variant: "destructive",
      });
    }
  };
  const saveCurrentAnnotations = useCallback(
    async (opts?: { silent?: boolean; annotationsOverride?: Annotation[] }): Promise<boolean> => {
      if (!currentImageId) return true;
      if (isDrawingPoly) {
        if (!opts?.silent) {
          toast({ title: "请先完成多边形绘制后再保存", variant: "destructive" });
        }
        return false;
      }

      try {
        setSaveStatus("saving");
        const annsToSave = opts?.annotationsOverride ?? annotations;
        const payload = annsToSave.map(toBackendAnnotationPayload);
        const saved = await annotationService.replaceByImage(currentImageId, payload);
        setSaveStatus("saved");
        setImageList((prev) =>
          prev.map((img) =>
            img.id === currentImageId
              ? {
                  ...img,
                  status: saved.length > 0 ? "completed" : "pending",
                  annotationsCount: saved.length,
                }
              : img
          )
        );
        return true;
      } catch (error) {
        console.error("Save Annotations Error:", error);
        setSaveStatus("unsaved");
        if (!opts?.silent) {
          toast({
            title: "保存失败",
            description: "请检查 API Key / 权限设置或后端日志",
            variant: "destructive",
          });
        }
        return false;
      }
    },
    [annotations, currentImageId, isDrawingPoly]
  );

  const enterSmartDetectMode = useCallback(() => {
    if (!activeProjectId || !activeDatasetId || !currentImageId) {
      toast({ title: "请先选择项目/数据集/图片", variant: "destructive" });
      return;
    }
    setSmartTab("detect");
    setIsSmartPanelOpen(false);
    setSmartDetectDraft(null);
    smartDetectStartRef.current = null;
    setSmartMode("detect");
    toast({ title: "智能检测已开启", description: "请在画布上拖拽框选一个目标作为参考（Esc 退出）" });
  }, [activeDatasetId, activeProjectId, currentImageId]);

  const enterSmartSegmentMode = useCallback(() => {
    if (!activeProjectId || !activeDatasetId || !currentImageId) {
      toast({ title: "请先选择项目/数据集/图片", variant: "destructive" });
      return;
    }
    setSmartTab("segment");
    setIsSmartPanelOpen(false);
    setSmartDetectDraft(null);
    smartDetectStartRef.current = null;
    setSmartMode("segment");
    toast({ title: "智能分割已开启", description: "请在目标上点击一个点（每次点击标注一个，Esc 退出）" });
  }, [activeDatasetId, activeProjectId, currentImageId]);

  const runSmartDetect = useCallback(
    async (box: [number, number, number, number]) => {
      if (!activeDatasetId || !currentImageId) return;
      if (smartBusy) return;

      const currentClass = classes.find((c) => c.id === selectedClassId) || classes[0];
      if (!currentClass?.name) {
        toast({ title: "请先选择标签类别", variant: "destructive" });
        return;
      }

      const ok = await saveCurrentAnnotations({ silent: true });
      if (!ok) return;

      const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
      const clamp = (v: number, min: number, max: number) => Math.max(min, Math.min(max, v));

      const maxW = Math.max(1, imageCanvasWidth);
      const maxH = Math.max(1, imageCanvasHeight);
      const x1 = clamp(Math.min(box[0], box[2]), 0, maxW);
      const y1 = clamp(Math.min(box[1], box[3]), 0, maxH);
      const x2 = clamp(Math.max(box[0], box[2]), 0, maxW);
      const y2 = clamp(Math.max(box[1], box[3]), 0, maxH);
      if (x2 - x1 < 3 || y2 - y1 < 3) return;

      const maxImages = smartDetectMaxImages.trim() ? Number(smartDetectMaxImages) : undefined;
      const minDistance = smartDetectMinDistance.trim() ? Number(smartDetectMinDistance) : undefined;

      setSmartBusy(true);
      try {
        const result = await smartAnnotationService.detect({
          dataset_id: activeDatasetId,
          reference_image_id: currentImageId,
          label: currentClass.name,
          color: currentClass.color,
          box: [x1, y1, x2, y2],
          scope: smartDetectScope,
          max_images: Number.isFinite(maxImages as number) ? Math.max(1, Math.floor(maxImages as number)) : undefined,
          threshold: clamp01(Number(smartDetectThreshold) || 0.6),
          max_det_per_image: Math.max(1, Math.floor(Number(smartDetectMaxDet) || 20)),
          min_distance: Number.isFinite(minDistance as number) ? Math.max(1, Math.floor(minDistance as number)) : undefined,
          dedup_iou: clamp01(Number(smartDetectDedupIou) || 0.8),
          only_unannotated: smartDetectOnlyUnannotated,
        });

        const fresh = await annotationService.listByImage(currentImageId);
        const anns = fresh.map(toFrontendAnnotation);
        setAnnotations(anns);
        setHistory([anns]);
        setHistoryStep(0);
        setSelectedId(null);
        setSaveStatus("saved");

        toast({
          title: "智能检测完成",
          description: `处理 ${result.processed_images} 张图，新增 ${result.created_annotations} 个标注${result.skipped_images ? `，跳过 ${result.skipped_images} 张` : ""}`,
        });
      } catch (error) {
        console.error("Smart detect failed:", error);
        toast({ title: "智能检测失败", description: "请检查后端服务或控制台日志", variant: "destructive" });
      } finally {
        setSmartBusy(false);
        setSmartMode(null);
        setSmartDetectDraft(null);
        smartDetectStartRef.current = null;
      }
    },
    [
      activeDatasetId,
      classes,
      currentImageId,
      imageCanvasHeight,
      imageCanvasWidth,
      saveCurrentAnnotations,
      selectedClassId,
      smartBusy,
      smartDetectDedupIou,
      smartDetectMaxDet,
      smartDetectMaxImages,
      smartDetectMinDistance,
      smartDetectOnlyUnannotated,
      smartDetectScope,
      smartDetectThreshold,
    ]
  );

  const handleCompleteToAugmentation = useCallback(async () => {
    if (!activeProjectId || !activeDatasetId) {
      toast({ title: "请先选择项目与数据集", variant: "destructive" });
      return;
    }

    const ok = window.confirm("确认完成标注并进入数据增强？");
    if (!ok) return;

    if (saveStatus === "unsaved") {
      const saved = await saveCurrentAnnotations();
      if (!saved) return;
    }

    const projectName = projects.find((p) => p.id === activeProjectId)?.name ?? projectNameInContext ?? null;
    const ds = datasets.find((d) => d.id === activeDatasetId) ?? null;

    setProjectContext(activeProjectId, projectName);
    setDatasetContext(activeDatasetId, ds ? `${ds.name} ${ds.version}` : null);

    navigate(`/augmentation?project_id=${encodeURIComponent(activeProjectId)}&dataset_id=${encodeURIComponent(activeDatasetId)}`);
  }, [
    activeDatasetId,
    activeProjectId,
    datasets,
    navigate,
    projectNameInContext,
    projects,
    saveCurrentAnnotations,
    saveStatus,
    setDatasetContext,
    setProjectContext,
  ]);

  // --- Image Navigation Handlers with State Persistence ---
  const saveCurrentViewState = useCallback(() => {
    setViewStates((prev) => ({
      ...prev,
      [currentImageIndex]: { scale, x: position.x, y: position.y },
    }));
  }, [currentImageIndex, position.x, position.y, scale]);

  const restoreViewState = useCallback(
    (index: number) => {
      const state = viewStates[index];
      if (state) {
        setScale(state.scale);
        setPosition({ x: state.x, y: state.y });
        return;
      }
      setScale(1);
      setPosition({ x: 0, y: 0 });
    },
    [viewStates]
  );

  const handlePrevImage = useCallback(async () => {
      if (currentImageIndex > 0) {
          if (saveStatus === "unsaved") {
              const ok = await saveCurrentAnnotations();
              if (!ok) return;
          }
          saveCurrentViewState();
          const newIndex = currentImageIndex - 1;
          setCurrentImageIndex(newIndex);
          restoreViewState(newIndex);

          setAnnotations([]); 
          setHistory([[]]);
          setHistoryStep(0);
          setSelectedId(null);
      }
  }, [currentImageIndex, restoreViewState, saveCurrentAnnotations, saveCurrentViewState, saveStatus]);

  const handleNextImage = useCallback(async () => {
      if (currentImageIndex < imageList.length - 1) {
          if (saveStatus === "unsaved") {
              const ok = await saveCurrentAnnotations();
              if (!ok) return;
          }
          saveCurrentViewState();
          const newIndex = currentImageIndex + 1;
          setCurrentImageIndex(newIndex);
          restoreViewState(newIndex);

          setAnnotations([]);
          setHistory([[]]);
          setHistoryStep(0);
          setSelectedId(null);
      }
  }, [currentImageIndex, imageList.length, restoreViewState, saveCurrentAnnotations, saveCurrentViewState, saveStatus]);

  const handleJumpToImage = async (index: number) => {
      if (index >= 0 && index < imageList.length) {
          if (saveStatus === "unsaved") {
              const ok = await saveCurrentAnnotations();
              if (!ok) return;
          }
          saveCurrentViewState();
          setCurrentImageIndex(index);
          restoreViewState(index);

          setAnnotations([]);
          setHistory([[]]);
          setHistoryStep(0);
          setSelectedId(null);
      }
  };

  // Apply filters to image
  useEffect(() => {
    if (imgRef.current) {
        imgRef.current.cache();
        imgRef.current.filters([Konva.Filters.Brighten, Konva.Filters.Contrast]);
        imgRef.current.brightness((brightness - 100) / 100);
        imgRef.current.contrast((contrast - 100) / 100);
        imgRef.current.getLayer()?.batchDraw();
    }
  }, [image, brightness, contrast]);

  // --- Initialization ---
  useEffect(() => {
    setHistory([[]]);
    setHistoryStep(0);
  }, []);

  // --- Auto-save (debounced) ---
  useEffect(() => {
    if (!currentImageId) return;
    if (historyStep <= 0) return; // Don't save on initial load
    const timer = setTimeout(() => {
      void saveCurrentAnnotations({ silent: true });
    }, 800);
    return () => clearTimeout(timer);
  }, [annotations, currentImageId, historyStep, saveCurrentAnnotations]);

  // --- History Management ---
  const addToHistory = useCallback((newAnnotations: Annotation[]) => {
    const newHistory = history.slice(0, historyStep + 1);
    newHistory.push(newAnnotations);
    setHistory(newHistory);
    setHistoryStep(newHistory.length - 1);
    setAnnotations(newAnnotations);
    setSaveStatus("unsaved");
  }, [history, historyStep]);

  const runSmartSegment = useCallback(
    async (point: { x: number; y: number }) => {
      if (!currentImageId) return;
      if (smartBusy) return;

      const currentClass = classes.find((c) => c.id === selectedClassId) || classes[0];
      if (!currentClass?.name) {
        toast({ title: "请先选择标签类别", variant: "destructive" });
        return;
      }

      setSmartBusy(true);
      try {
        const result = await smartAnnotationService.segment({
          image_id: currentImageId,
          point: [point.x, point.y],
          tolerance: Number(smartSegTolerance) || 0.08,
          simplify: Number(smartSegSimplify) || 2.0,
          engine: smartSegEngine,
        });

        if (!result.points || result.points.length < 6) {
          toast({ title: "未找到可分割区域", description: "请换一个点或增大容差", variant: "destructive" });
          return;
        }

        const newAnn: Annotation = {
          id: uuidv4(),
          type: "polygon",
          points: result.points,
          label: currentClass.name,
          color: currentClass.color,
          visible: true,
        };
        addToHistory([...annotations, newAnn]);
      } catch (error) {
        console.error("Smart segment failed:", error);
        toast({ title: "智能分割失败", description: "请检查后端服务或控制台日志", variant: "destructive" });
      } finally {
        setSmartBusy(false);
      }
    },
    [addToHistory, annotations, classes, currentImageId, selectedClassId, smartBusy, smartSegEngine, smartSegSimplify, smartSegTolerance]
  );

  const handleUndo = useCallback(() => {
    if (historyStep > 0) {
      const prevStep = historyStep - 1;
      setHistoryStep(prevStep);
      setAnnotations(history[prevStep]);
      setSelectedId(null);
      setSaveStatus(prevStep === 0 ? "saved" : "unsaved");
    }
  }, [history, historyStep]);

  const handleRedo = useCallback(() => {
    if (historyStep < history.length - 1) {
      const nextStep = historyStep + 1;
      setHistoryStep(nextStep);
      setAnnotations(history[nextStep]);
      setSelectedId(null);
      setSaveStatus(nextStep === 0 ? "saved" : "unsaved");
    }
  }, [history, historyStep]);

  const handleDelete = useCallback(() => {
      if (selectedId) {
          const newAnns = annotations.filter(a => a.id !== selectedId);
          addToHistory(newAnns);
          setSelectedId(null);
      }
  }, [addToHistory, annotations, selectedId]);

  const handleClearAll = () => {
      if (annotations.length > 0) {
          if (window.confirm("确定要清空当前图片的所有标注吗？")) {
            addToHistory([]);
            setSelectedId(null);
          }
      }
  };

  const handleClassSelect = useCallback((clsId: string) => {
      try {
          console.log('Class selected:', clsId);
          setSelectedClassId(clsId);
          
          if (selectedId) {
              const cls = classes.find(c => c.id === clsId);
              if (cls) {
                  const newAnns = annotations.map(a => 
                      a.id === selectedId ? { ...a, label: cls.name, color: cls.color } : a
                  );
                  addToHistory(newAnns);
              }
          }
      } catch (error) {
          console.error('Failed to select class:', error);
      }
   }, [classes, selectedId, annotations, addToHistory]);

  const handleAddClass = useCallback(() => {
    void (async () => {
      if (!activeProjectId) {
        toast({ title: "未选择项目", description: "请先选择项目后再新增标签", variant: "destructive" });
        return;
      }

      const raw = window.prompt("请输入新的英文标签短码（如 hd / crack）", "");
      const nextName = normalizeLabel(raw);
      if (!nextName) return;

      if (classes.some((c) => normalizeLabel(c.name) === nextName)) {
        toast({ title: "标签已存在", description: nextName, variant: "destructive" });
        return;
      }

      const color = pickDistinctColor(classes.map((c) => c.color));
      const shortcut = nextShortcut(classes);

      try {
        const created = await labelService.create(activeProjectId, { name: nextName, color, shortcut });
        const nextClass = toFrontendLabelClass(created);
        setClasses((prev) => [...prev, nextClass]);
        setSelectedClassId(nextClass.id);
        toast({ title: "已新增标签类别", description: nextName });
      } catch (error) {
        console.error("Create label failed:", error);
        toast({ title: "新增标签失败", description: "请检查后端服务/权限设置或后端日志", variant: "destructive" });
      }
    })();
  }, [activeProjectId, classes]);

  const handleEditClass = useCallback(
    (classId: string) => {
      void (async () => {
        const cls = classes.find((c) => c.id === classId);
        if (!cls) return;
        if (!activeProjectId) {
          toast({ title: "未选择项目", description: "请先选择项目后再编辑标签", variant: "destructive" });
          return;
        }

        const rawName = window.prompt("编辑标签名称（英文短码）", cls.name);
        if (rawName === null) return;
        const nextName = normalizeLabel(rawName);
        if (!nextName) {
          toast({ title: "标签名称不能为空", variant: "destructive" });
          return;
        }

        const rawColor = window.prompt("编辑标签颜色（#RRGGBB），留空保持不变", cls.color);
        if (rawColor === null) return;
        const colorInput = (rawColor || "").trim();
        const nextColor = colorInput ? normalizeHexColor(colorInput) : cls.color;
        if (colorInput && !nextColor) {
          toast({ title: "颜色格式不正确", description: "请输入 #RRGGBB", variant: "destructive" });
          return;
        }

        if (classes.some((c) => c.id !== cls.id && normalizeLabel(c.name) === nextName)) {
          toast({ title: "目标标签已存在", description: nextName, variant: "destructive" });
          return;
        }

        const payload: { name?: string; color?: string } = {};
        if (nextName !== cls.name) payload.name = nextName;
        if (nextColor !== cls.color) payload.color = nextColor || cls.color;
        if (Object.keys(payload).length === 0) return;

        try {
          const result = await labelService.update(activeProjectId, cls.id, payload);
          const updatedClass = toFrontendLabelClass(result.label);

          setClasses((prev) => prev.map((c) => (c.id === cls.id ? updatedClass : c)));

          const updatedAnnotations = annotations.map((a) =>
            normalizeLabel(a.label) === normalizeLabel(cls.name)
              ? { ...a, label: updatedClass.name, color: updatedClass.color }
              : a
          );
          setAnnotations(updatedAnnotations);
          setHistory([updatedAnnotations]);
          setHistoryStep(0);
          setSelectedId(null);
          void saveCurrentAnnotations({ silent: true, annotationsOverride: updatedAnnotations });

          toast({
            title: "标签已更新",
            description: `已同步更新 ${result.updated} 条标注`,
          });
        } catch (error) {
          console.error("Update label failed:", error);
          toast({ title: "编辑标签失败", description: "请检查后端服务/权限设置或后端日志", variant: "destructive" });
        }
      })();
    },
    [activeProjectId, annotations, classes, saveCurrentAnnotations]
  );

  const handleDeleteClass = useCallback(
    (classId: string) => {
      void (async () => {
        const cls = classes.find((c) => c.id === classId);
        if (!cls) return;
        if (!activeProjectId) {
          toast({ title: "未选择项目", description: "请先选择项目后再删除标签", variant: "destructive" });
          return;
        }
        if (classes.length <= 1) {
          toast({ title: "至少保留一个标签类别", variant: "destructive" });
          return;
        }

        const ok = window.confirm(`确认删除标签类别 “${cls.name}” ？\n（仅未被使用的标签可删除）`);
        if (!ok) return;

        try {
          await labelService.delete(activeProjectId, cls.id);
          const remaining = classes.filter((c) => c.id !== cls.id);
          setClasses(remaining);
          setSelectedClassId((prev) => (prev === cls.id ? remaining[0]?.id || DEFAULT_CLASSES[0].id : prev));
          toast({ title: "标签已删除", description: cls.name });
        } catch (error) {
          console.error("Delete label failed:", error);
          toast({ title: "删除失败", description: "该标签可能已被使用，请先重命名/替换相关标注", variant: "destructive" });
        }
      })();
    },
    [activeProjectId, classes]
  );

  const applyLabelClassToAnnotation = useCallback(
    (annotationId: string, classId: string) => {
      const cls = classes.find((c) => c.id === classId);
      if (!cls) return;

      const updatedAnnotations = annotations.map((a) =>
        a.id === annotationId ? { ...a, label: cls.name, color: cls.color } : a
      );

      setSelectedClassId(classId);
      addToHistory(updatedAnnotations);
      setSelectedId(annotationId);
      closeLabelPicker();
      void saveCurrentAnnotations({ silent: true, annotationsOverride: updatedAnnotations });
    },
    [addToHistory, annotations, classes, closeLabelPicker, saveCurrentAnnotations]
  );

  const handleEditAnnotationLabel = useCallback(
    (annotationId: string) => {
      openLabelPicker(annotationId);
    },
    [openLabelPicker]
  );

  const handleDeleteImageAtIndex = useCallback(
    async (index: number) => {
      const target = imageList[index];
      if (!target) return;
      const annCount = target.annotationsCount ?? 0;
      if (annCount > 0) {
        toast({
          title: "该图片已包含标注",
          description: "当前仅支持删除无缺陷（无标注）图片",
          variant: "destructive",
        });
        return;
      }

      const ok = window.confirm("确认删除此图片？该操作不可撤销");
      if (!ok) return;

      try {
        const wasCurrent = index === currentImageIndex;
        const prevLen = imageList.length;
        let nextIndex = currentImageIndex;
        if (prevLen <= 1) {
          nextIndex = 0;
        } else if (wasCurrent) {
          nextIndex = Math.min(currentImageIndex, prevLen - 2);
        } else if (index < currentImageIndex) {
          nextIndex = currentImageIndex - 1;
        }

        await imageService.delete(target.id);

        setImageList((prev) => prev.filter((img) => img.id !== target.id));
        setViewStates({});
        setCurrentImageIndex(nextIndex);

        if (wasCurrent) {
          setAnnotations([]);
          setHistory([[]]);
          setHistoryStep(0);
          setSelectedId(null);
          setSaveStatus("saved");
        }

        toast({ title: "已删除图片", description: target.name });
      } catch (error) {
        console.error("Delete image failed:", error);
        toast({ title: "删除失败", description: "请检查后端服务/权限设置或后端日志", variant: "destructive" });
      }
    },
    [currentImageIndex, imageList]
  );

  // --- Keyboard Shortcuts ---
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
        // Ignore if input is focused
        if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

        if (e.key === "Escape") {
          if (smartMode) {
            cancelSmartMode();
            return;
          }
          if (isSmartPanelOpen) {
            setIsSmartPanelOpen(false);
            return;
          }
        }

        // Tools
        if (e.key.toLowerCase() === 'v') setSelectedTool('select');
        if (e.key.toLowerCase() === 'r') setSelectedTool('rect');
        if (e.key.toLowerCase() === 'p') setSelectedTool('polygon');
        if (e.key.toLowerCase() === ' ') {
            e.preventDefault(); 
            setSelectedTool('move');
        }

        // Image Navigation
        if (e.key === 'ArrowRight' && !e.ctrlKey) handleNextImage();
        if (e.key === 'ArrowLeft' && !e.ctrlKey) handlePrevImage();

        // Actions
        if ((e.key === "Delete" || e.key === "Backspace")) {
          if (selectedId) {
            handleDelete();
          } else {
            const current = imageList[currentImageIndex];
            const annCount = current?.annotationsCount ?? 0;
            if (current && annCount === 0) {
              void handleDeleteImageAtIndex(currentImageIndex);
            }
          }
        }
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
            e.preventDefault();
            if (e.shiftKey) handleRedo();
            else handleUndo();
        }
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') { // Redo alternative
            e.preventDefault();
            handleRedo();
        }
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
             e.preventDefault();
             void saveCurrentAnnotations();
        }
        
        // Zoom
        if (e.key === '=' || e.key === '+') setScale(s => Math.min(5, s + 0.1));
        if (e.key === '-') setScale(s => Math.max(0.1, s - 0.1));
        if (e.key === '0') { setScale(1); setPosition({x:0, y:0}); }

        // Class Selection
        const num = parseInt(e.key);
        if (!isNaN(num) && num > 0 && num <= classes.length) {
            handleClassSelect(classes[num - 1].id);
        }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [
    cancelSmartMode,
    classes,
    currentImageIndex,
    handleClassSelect,
    handleDelete,
    handleDeleteImageAtIndex,
    handleNextImage,
    handlePrevImage,
    handleRedo,
    handleUndo,
    imageList,
    isSmartPanelOpen,
    saveCurrentAnnotations,
    selectedId,
    smartMode,
  ]);


  const getRelativePointerPosition = (node: Konva.Stage) => {
    const transform = node.getAbsoluteTransform().copy();
    transform.invert();
    const pos = node.getStage().getPointerPosition();
    return transform.point(pos!);
  };

  const getCurrentClass = () => classes.find(c => c.id === selectedClassId) || classes[0];

  // --- Canvas Interaction Handlers ---
  const handleMouseDown = (e: Konva.KonvaEventObject<MouseEvent>) => {
    const stage = e.target.getStage();
    if (!stage) return;
    const pos = getRelativePointerPosition(stage);

    if (smartMode === "segment") {
      void runSmartSegment(pos);
      return;
    }

    if (smartMode === "detect") {
      smartDetectStartRef.current = { x: pos.x, y: pos.y };
      setSmartDetectDraft({ x1: pos.x, y1: pos.y, x2: pos.x, y2: pos.y });
      return;
    }

    if (selectedTool === "select") {
      const clickedOnEmpty = e.target === e.target.getStage();
      if (clickedOnEmpty) {
        setSelectedId(null);
      }
      return;
    }

    if (selectedTool === "rect") {
      const newId = Math.random().toString(36).substr(2, 9);
      const currentClass = getCurrentClass();
      const newAnnotation: Annotation = {
        id: newId,
        type: "rect",
        x: pos.x,
        y: pos.y,
        width: 0,
        height: 0,
        label: currentClass.name,
        color: currentClass.color,
        visible: true
      };
      setAnnotations([...annotations, newAnnotation]);
      setSelectedId(newId);
    }

    if (selectedTool === "polygon") {
        if (!isDrawingPoly) {
            setIsDrawingPoly(true);
            setCurrentPolyPoints([pos.x, pos.y]);
        } else {
            setCurrentPolyPoints([...currentPolyPoints, pos.x, pos.y]);
        }
    }
  };

  const handleMouseMove = (e: Konva.KonvaEventObject<MouseEvent>) => {
    const stage = e.target.getStage();
    if (!stage) return;
    const pos = getRelativePointerPosition(stage);
    setMousePos(pos);

    if (smartMode === "detect" && smartDetectStartRef.current) {
      setSmartDetectDraft((prev) => (prev ? { ...prev, x2: pos.x, y2: pos.y } : prev));
      return;
    }

    if (selectedTool === "rect" && selectedId) {
      const updatedAnnotations = annotations.map((ann) => {
        if (ann.id === selectedId) {
          return {
            ...ann,
            width: pos.x - (ann.x || 0),
            height: pos.y - (ann.y || 0),
          };
        }
        return ann;
      });
      setAnnotations(updatedAnnotations);
    }
  };

  const handleMouseUp = () => {
    if (smartMode === "detect") {
      const draft = smartDetectDraft;
      if (!draft || !smartDetectStartRef.current) {
        smartDetectStartRef.current = null;
        setSmartDetectDraft(null);
        return;
      }
      smartDetectStartRef.current = null;
      setSmartDetectDraft(null);
      void runSmartDetect([draft.x1, draft.y1, draft.x2, draft.y2]);
      return;
    }

    if (selectedTool === "rect" && selectedId) {
      addToHistory(annotations);
      // Don't deselect to allow immediate resizing if needed, or deselect to draw next?
      // Roboflow style: keep selected to edit, but need to click again to draw new. 
      // For continuous drawing, we might want to clear selection. 
      // Let's clear selection to allow drawing another one immediately if tool is still rect.
      setSelectedId(null); 
    }
  };

  const finishPolygon = () => {
      if (currentPolyPoints.length < 6) return; // At least 3 points
      
      const newId = Math.random().toString(36).substr(2, 9);
      const currentClass = getCurrentClass();
      const newAnnotation: Annotation = {
          id: newId,
          type: "polygon",
          points: currentPolyPoints,
          label: currentClass.name,
          color: currentClass.color,
          visible: true
      };
      
      const newAnns = [...annotations, newAnnotation];
      setAnnotations(newAnns);
      addToHistory(newAnns);
      setIsDrawingPoly(false);
      setCurrentPolyPoints([]);
  }

  const handleStageDblClick = () => {
      if (selectedTool === 'polygon' && isDrawingPoly) {
          finishPolygon();
      }
  }

  const toggleVisibility = (id: string, e: React.MouseEvent) => {
      e.stopPropagation();
      const newAnns = annotations.map(a => a.id === id ? { ...a, visible: !a.visible } : a);
      addToHistory(newAnns);
  }

  // --- Preload Images ---
  useEffect(() => {
    const preloadImages = () => {
        const nextIdx = currentImageIndex + 1;
        const prevIdx = currentImageIndex - 1;
        
        if (nextIdx < imageList.length) {
            const img = new window.Image();
            img.src = imageList[nextIdx].url;
        }
        if (prevIdx >= 0) {
            const img = new window.Image();
            img.src = imageList[prevIdx].url;
        }
        
        // Preload one more ahead if possible for smoother sequence
        if (nextIdx + 1 < imageList.length) {
            const img = new window.Image();
            img.src = imageList[nextIdx + 1].url;
        }
    };
    
    preloadImages();
  }, [currentImageIndex, imageList]);

  // Scroll current thumbnail into view
  useEffect(() => {
      const el = document.getElementById(`thumb-${currentImageIndex}`);
      if (el) {
          el.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
      }
  }, [currentImageIndex]);

  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  return (
    <div className="flex flex-col h-[calc(100vh-6rem)]">
      {/* Top Bar Info & Navigation */}
      <Card className="px-4 py-2 flex flex-col gap-2 md:flex-row md:justify-between md:items-center bg-card rounded-none border-x-0 border-t-0 z-20 shadow-sm shrink-0">
          <div className="flex flex-col gap-2 md:flex-row md:items-center md:gap-4 min-w-0 w-full">
              <div className="flex flex-wrap items-center gap-2 shrink-0">
                  <select
                      className="h-8 rounded-md border border-input bg-background px-2 py-1 text-sm"
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
                          {isLoadingProjects ? "加载项目中..." : "请选择项目"}
                      </option>
                      {projects.map((p) => (
                          <option key={p.id} value={p.id}>
                              {p.name}
                          </option>
                      ))}
                  </select>

                  <select
                      className="h-8 rounded-md border border-input bg-background px-2 py-1 text-sm"
                      value={activeDatasetId ?? ""}
                      disabled={!activeProjectId || isLoadingDatasets || datasets.length === 0}
                      onChange={(e) => {
                          if (!activeProjectId) return;
                          const nextId = (e.target.value || "").trim();
                          if (!nextId) return;
                          const found = datasets.find((d) => d.id === nextId) ?? null;
                          setDatasetContext(nextId, found ? `${found.name} ${found.version}` : null);
                          setSearchParams((prev) => {
                              const next = new URLSearchParams(prev);
                              next.set("project_id", activeProjectId);
                              next.set("dataset_id", nextId);
                              return next;
                          });
                      }}
                  >
                      <option value="" disabled>
                          {!activeProjectId ? "请先选择项目" : isLoadingDatasets ? "加载数据集中..." : "请选择数据集"}
                      </option>
                      {datasets.map((d) => (
                          <option key={d.id} value={d.id}>
                              {d.name} {d.version}
                          </option>
                      ))}
                  </select>
              </div>
              <div className="hidden md:block h-8 w-px bg-border" />
              {imageList.length === 0 ? (
                  <div className="flex items-center gap-2 text-sm text-muted-foreground min-w-0">
                      <span className="truncate">
                          {isLoadingImages ? "加载图片中..." : "暂无图片，请先在数据集模块导入图片"}
                      </span>
                      <Button
                          variant="outline"
                          size="sm"
                          className="h-8"
                          disabled={!activeProjectId}
                          onClick={() => {
                              if (!activeProjectId) return;
                              navigate(`/projects/${encodeURIComponent(activeProjectId)}`);
                          }}
                      >
                          去数据集
                      </Button>
                  </div>
              ) : (
                  <div className="flex items-center gap-2">
                      <Button
                          variant="outline"
                          size="icon"
                          onClick={handlePrevImage}
                          disabled={currentImageIndex === 0 || isLoadingImages || isLoadingAnnotations}
                      >
                          <ChevronLeft className="w-4 h-4" />
                      </Button>
                      <div className="flex flex-col items-center min-w-[120px] gap-1">
                          <div className="flex items-center gap-2 text-sm font-semibold">
                              <Input
                                  className="w-12 h-6 px-1 text-center py-0"
                                  defaultValue={currentImageIndex + 1}
                                  key={currentImageIndex}
                                  onKeyDown={(e) => {
                                      if (e.key === "Enter") {
                                          const val = parseInt(e.currentTarget.value);
                                          if (!isNaN(val) && val >= 1 && val <= imageList.length) {
                                              void handleJumpToImage(val - 1);
                                          } else {
                                              e.currentTarget.value = (currentImageIndex + 1).toString();
                                          }
                                      }
                                  }}
                                  onBlur={(e) => {
                                      const val = parseInt(e.target.value);
                                      if (!isNaN(val) && val >= 1 && val <= imageList.length) {
                                          void handleJumpToImage(val - 1);
                                      } else {
                                          e.target.value = (currentImageIndex + 1).toString();
                                      }
                                  }}
                              />
                              <span className="text-muted-foreground">/ {imageList.length}</span>
                          </div>
                          <span
                              className="text-xs text-muted-foreground truncate max-w-[120px]"
                              title={imageList[currentImageIndex]?.name || ""}
                          >
                              {imageList[currentImageIndex]?.name || "-"}
                          </span>
                      </div>
                      <Button
                          variant="outline"
                          size="icon"
                          onClick={handleNextImage}
                          disabled={currentImageIndex === imageList.length - 1 || isLoadingImages || isLoadingAnnotations}
                      >
                          <ChevronRight className="w-4 h-4" />
                      </Button>
                  </div>
              )}
              <div className="hidden md:block h-8 w-px bg-border" />
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  {saveStatus === "saved" && <><CheckCircle2 className="w-4 h-4 text-green-500" /> 已保存</>}
                  {saveStatus === "saving" && <><AlertCircle className="w-4 h-4 text-yellow-500 animate-pulse" /> 保存中...</>}
                  {saveStatus === "unsaved" && <span className="text-yellow-500">未保存</span>}
              </div>
              <div className="hidden md:block h-8 w-px bg-border" />
              <div className="flex items-center gap-2 flex-wrap">
                  <Button variant="ghost" size="sm" onClick={() => setScale((s) => Math.max(0.1, s - 0.1))}>
                      <ZoomOut className="w-4 h-4" />
                  </Button>
                  <Button variant="ghost" size="sm" onClick={handleResetView} title="重置视图" className="w-16 px-0 text-sm">
                      {Math.round(scale * 100)}%
                  </Button>
                  <Button variant="ghost" size="sm" onClick={() => setScale((s) => Math.min(5, s + 0.1))}>
                      <ZoomIn className="w-4 h-4" />
                  </Button>
                  <div className="w-px h-4 bg-border mx-2" />
                   <Button
                       variant="ghost"
                       size="icon"
                       className="h-9 w-9"
                       title="智能标注"
                       onClick={() => setIsSmartPanelOpen(true)}
                       disabled={!currentImageId || isLoadingImages || isLoadingAnnotations}
                   >
                       <Wand2 className="w-4 h-4 text-purple-500" />
                   </Button>
                  <div className="w-px h-4 bg-border mx-2" />
                  <Button variant="ghost" size="icon" onClick={handleUndo} disabled={historyStep <= 0}>
                      <Undo className="w-4 h-4" />
                  </Button>
                  <Button variant="ghost" size="icon" onClick={handleRedo} disabled={historyStep >= history.length - 1}>
                      <Redo className="w-4 h-4" />
                  </Button>
              </div>
          </div>
          <div className="flex items-center gap-2 flex-wrap justify-between md:justify-end w-full md:w-auto">
              <Button
                size="sm"
                className="ml-2 gap-2"
                onClick={() => void handleCompleteToAugmentation()}
                disabled={!currentImageId || isLoadingAnnotations || isLoadingImages}
              >
                <ArrowRight className="w-4 h-4" /> 完成
              </Button>
              <div className="w-px h-4 bg-border mx-2" />
              <Button variant="ghost" size="icon" onClick={() => setIsSidebarOpen(!isSidebarOpen)} title={isSidebarOpen ? "收起侧边栏" : "展开侧边栏"}>
                 <Layers className={`w-5 h-5 transition-transform ${isSidebarOpen ? '' : 'rotate-180'}`} />
              </Button>
          </div>
      </Card>

      {/* Main Workspace */}
      <div className="flex-1 flex min-h-0 relative">
        {/* Shortcuts Modal Overlay */}
        {showShortcuts && (
            <div className="absolute inset-0 z-50 bg-background/80 backdrop-blur-sm flex items-center justify-center p-4">
                <Card className="w-full max-w-2xl shadow-2xl">
                    <CardHeader className="flex flex-row items-center justify-between">
                        <CardTitle className="flex items-center gap-2"><Keyboard className="w-5 h-5"/> 键盘快捷键</CardTitle>
                        <Button variant="ghost" size="icon" onClick={() => setShowShortcuts(false)}>×</Button>
                    </CardHeader>
                    <CardContent>
                        <div className="grid grid-cols-2 gap-8">
                            <div className="space-y-4">
                                <h4 className="font-medium text-primary">图片导航</h4>
                                <div className="grid grid-cols-2 gap-2 text-sm">
                                    <span className="text-muted-foreground">上一张</span> <Badge variant="outline">← (Left)</Badge>
                                    <span className="text-muted-foreground">下一张</span> <Badge variant="outline">→ (Right)</Badge>
                                </div>
                                <h4 className="font-medium text-primary mt-4">工具选择</h4>
                                <div className="grid grid-cols-2 gap-2 text-sm">
                                    <span className="text-muted-foreground">选择工具</span> <Badge variant="outline">V</Badge>
                                    <span className="text-muted-foreground">矩形工具</span> <Badge variant="outline">R</Badge>
                                    <span className="text-muted-foreground">多边形工具</span> <Badge variant="outline">P</Badge>
                                    <span className="text-muted-foreground">移动画布</span> <Badge variant="outline">Space</Badge>
                                </div>
                            </div>
                            <div className="space-y-4">
                                <h4 className="font-medium text-primary">编辑操作</h4>
                                <div className="grid grid-cols-2 gap-2 text-sm">
                                    <span className="text-muted-foreground">撤销</span> <Badge variant="outline">Ctrl + Z</Badge>
                                    <span className="text-muted-foreground">重做</span> <Badge variant="outline">Ctrl + Shift + Z</Badge>
                                    <span className="text-muted-foreground">删除选中</span> <Badge variant="outline">Del / Backspace</Badge>
                                    <span className="text-muted-foreground">保存</span> <Badge variant="outline">Ctrl + S</Badge>
                                </div>
                                <h4 className="font-medium text-primary mt-4">标签切换</h4>
                                <div className="grid grid-cols-2 gap-2 text-sm">
                                    <span className="text-muted-foreground">切换类别</span> <Badge variant="outline">1 - 9</Badge>
                                </div>
                            </div>
                        </div>
                    </CardContent>
                </Card>
            </div>
        )}

        {/* Smart Annotation Modal */}
        {isSmartPanelOpen && (
          <div className="absolute inset-0 z-50 bg-background/80 backdrop-blur-sm flex items-center justify-center p-4">
            <Card className="w-full max-w-2xl shadow-2xl">
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <Wand2 className="w-5 h-5 text-purple-500" /> 智能标注
                </CardTitle>
                <Button variant="ghost" size="icon" onClick={() => setIsSmartPanelOpen(false)}>
                  <X className="w-4 h-4" />
                </Button>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm text-muted-foreground">标签</span>
                  <Select value={selectedClassId} onValueChange={(v) => setSelectedClassId(v)}>
                    <SelectTrigger className="w-[220px]">
                      <SelectValue placeholder="选择标签" />
                    </SelectTrigger>
                    <SelectContent>
                      {classes.length === 0 ? (
                        <div className="px-2 py-1.5 text-sm text-muted-foreground">暂无可用标签</div>
                      ) : (
                        classes.map((c) => (
                          <SelectItem key={c.id} value={c.id}>
                            <span className="flex items-center gap-2">
                              <span className="inline-block w-2 h-2 rounded-full" style={{ background: c.color }} />
                              <span className="truncate">{c.name}</span>
                            </span>
                          </SelectItem>
                        ))
                      )}
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center gap-2">
                  <Button size="sm" variant={smartTab === "detect" ? "default" : "outline"} onClick={() => setSmartTab("detect")}>
                    目标检测（框选）
                  </Button>
                  <Button size="sm" variant={smartTab === "segment" ? "default" : "outline"} onClick={() => setSmartTab("segment")}>
                    实例分割（点选）
                  </Button>
                </div>

                {smartTab === "detect" ? (
                  <div className="space-y-3">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div className="space-y-1">
                        <div className="text-xs text-muted-foreground">范围</div>
                        <Select value={smartDetectScope} onValueChange={(v) => setSmartDetectScope(v as "image" | "dataset")}>
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="image">仅当前图片</SelectItem>
                            <SelectItem value="dataset">整个数据集</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-1">
                        <div className="text-xs text-muted-foreground">阈值（0-1）</div>
                        <Input type="number" step="0.01" min="0" max="1" value={smartDetectThreshold} onChange={(e) => setSmartDetectThreshold(Number(e.target.value))} />
                      </div>
                      <div className="space-y-1">
                        <div className="text-xs text-muted-foreground">最多图片数（可选）</div>
                        <Input type="number" min="1" placeholder="留空=全部" value={smartDetectMaxImages} onChange={(e) => setSmartDetectMaxImages(e.target.value)} />
                      </div>
                      <div className="space-y-1">
                        <div className="text-xs text-muted-foreground">单图最多目标</div>
                        <Input type="number" min="1" max="500" value={smartDetectMaxDet} onChange={(e) => setSmartDetectMaxDet(Number(e.target.value))} />
                      </div>
                      <div className="space-y-1">
                        <div className="text-xs text-muted-foreground">最小间隔（可选）</div>
                        <Input type="number" min="1" placeholder="留空=自动" value={smartDetectMinDistance} onChange={(e) => setSmartDetectMinDistance(e.target.value)} />
                      </div>
                      <div className="space-y-1">
                        <div className="text-xs text-muted-foreground">去重 IoU（0-1）</div>
                        <Input type="number" step="0.01" min="0" max="1" value={smartDetectDedupIou} onChange={(e) => setSmartDetectDedupIou(Number(e.target.value))} />
                      </div>
                    </div>

                    <label className="flex items-center gap-2 text-xs text-muted-foreground">
                      <input
                        type="checkbox"
                        className="h-4 w-4 rounded border-input"
                        checked={smartDetectOnlyUnannotated}
                        onChange={(e) => setSmartDetectOnlyUnannotated(e.target.checked)}
                      />
                      仅处理未标注图片（推荐）
                    </label>

                    <div className="rounded-md border p-3 space-y-2">
                      <div className="text-sm font-medium">模型自动标注（单图）</div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        <div className="space-y-1">
                          <div className="text-xs text-muted-foreground">置信度 conf（0-1）</div>
                          <Input type="number" step="0.01" min="0" max="1" value={autoLabelConf} onChange={(e) => setAutoLabelConf(Number(e.target.value))} />
                        </div>
                        <div className="space-y-1">
                          <div className="text-xs text-muted-foreground">NMS IoU（0-1）</div>
                          <Input type="number" step="0.01" min="0" max="1" value={autoLabelIou} onChange={(e) => setAutoLabelIou(Number(e.target.value))} />
                        </div>
                        <div className="space-y-1">
                          <div className="text-xs text-muted-foreground">max_det</div>
                          <Input type="number" min="1" max="1000" value={autoLabelMaxDet} onChange={(e) => setAutoLabelMaxDet(Number(e.target.value))} />
                        </div>
                        <div className="space-y-1">
                          <div className="text-xs text-muted-foreground">新增去重 IoU（0-1）</div>
                          <Input type="number" step="0.01" min="0" max="1" value={autoLabelDedupIou} onChange={(e) => setAutoLabelDedupIou(Number(e.target.value))} />
                        </div>
                      </div>
                    </div>

                    <div className="text-xs text-muted-foreground">
                      提示：进入模式后，在画布上拖拽框选一个目标作为参考，系统会在所选范围内自动批量标注相似目标。
                    </div>

                    <div className="flex flex-wrap items-center gap-2">
                      <Button size="sm" onClick={enterSmartDetectMode} disabled={smartBusy || !currentImageId || classes.length === 0}>
                        进入框选模式
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => {
                          setIsSmartPanelOpen(false);
                          void handleAIAutoLabel();
                        }}
                        disabled={smartBusy || !currentImageId || classes.length === 0}
                      >
                        使用训练模型自动标注（单图）
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div className="space-y-1 md:col-span-2">
                        <div className="text-xs text-muted-foreground">分割引擎</div>
                        <Select value={smartSegEngine} onValueChange={(v) => setSmartSegEngine(v)}>
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="auto">自动（推荐）</SelectItem>
                            <SelectItem value="fastsam_tensorrt">FastSAM TensorRT（更快）</SelectItem>
                            <SelectItem value="fastsam">FastSAM（通用）</SelectItem>
                            <SelectItem value="sam">SAM（边界更细）</SelectItem>
                            <SelectItem value="flood">传统 flood（兜底）</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-1">
                        <div className="text-xs text-muted-foreground">容差（0-1）</div>
                        <Input type="number" step="0.01" min="0" max="1" value={smartSegTolerance} onChange={(e) => setSmartSegTolerance(Number(e.target.value))} />
                      </div>
                      <div className="space-y-1">
                        <div className="text-xs text-muted-foreground">简化程度</div>
                        <Input type="number" step="0.5" min="0" value={smartSegSimplify} onChange={(e) => setSmartSegSimplify(Number(e.target.value))} />
                      </div>
                    </div>

                    <div className="text-xs text-muted-foreground">提示：进入模式后，在目标上点击一个点（每次点击标注一个实例）。</div>

                    <div className="flex flex-wrap items-center gap-2">
                      <Button size="sm" onClick={enterSmartSegmentMode} disabled={smartBusy || !currentImageId || classes.length === 0}>
                        进入点选模式
                      </Button>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        )}

        {/* Smart Annotation Status */}
        {smartMode && (
          <div className="absolute top-3 left-3 z-40 rounded-lg border bg-background/90 backdrop-blur-sm shadow-sm px-3 py-2 flex items-center gap-3">
            <Wand2 className="w-4 h-4 text-purple-500" />
            <div className="text-sm">
              {smartMode === "detect" ? "智能检测：拖拽框选参考目标" : "智能分割：点击目标上的点"}
              {smartBusy ? <span className="ml-2 text-muted-foreground">处理中...</span> : null}
            </div>
            <Button size="sm" variant="outline" onClick={cancelSmartMode} disabled={smartBusy}>
              退出
            </Button>
          </div>
        )}

        {/* Floating Toolbar */}
        <AnnotationToolbar
            selectedTool={selectedTool}
            setSelectedTool={setSelectedTool}
            setShowShortcuts={setShowShortcuts}
            onResetView={handleResetView}
            onClearAll={handleClearAll}
            hasAnnotations={annotations.length > 0}
        />

         {/* Canvas Area */}
         <div ref={containerRef} className="flex-1 bg-neutral-100/50 relative overflow-hidden flex items-center justify-center transition-all duration-300">
             <div className="relative shadow-2xl border-4 border-white ring-1 ring-black/5 bg-white">
                 <Stage
                    ref={stageRef}
                    width={stageDimensions.width}
                    height={stageDimensions.height}
                    scaleX={scale}
                    scaleY={scale}
                    x={position.x}
                    y={position.y}
                    draggable={selectedTool === "move"}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                     onMouseUp={handleMouseUp}
                     onDblClick={handleStageDblClick}
                     className="cursor-crosshair"
                     style={{
                       cursor: smartMode ? (smartBusy ? "progress" : "crosshair") : selectedTool === "move" ? "grab" : selectedTool === "select" ? "default" : "crosshair",
                     }}
                     onWheel={handleWheel}
                 >
                     <Layer imageSmoothingEnabled>
                         <KonvaImage 
                             ref={imgRef}
                             image={image} 
                             width={imageCanvasWidth} 
                             height={imageCanvasHeight} 
                         />
                     </Layer>
                    {/* Crosshair Overlay */}
                     {showCrosshair && mousePos && (
                         <Layer listening={false}>
                             <Line
                                 points={[0, mousePos.y, imageCanvasWidth, mousePos.y]}
                                 stroke="rgba(255, 0, 0, 0.5)"
                                strokeWidth={1 / scale}
                                dash={[5 / scale, 5 / scale]}
                            />
                            <Line
                                points={[mousePos.x, 0, mousePos.x, imageCanvasHeight]}
                                stroke="rgba(255, 0, 0, 0.5)"
                                strokeWidth={1 / scale}
                                dash={[5 / scale, 5 / scale]}
                             />
                         </Layer>
                     )}

                     {smartMode === "detect" && smartDetectDraft && (
                       <Layer listening={false}>
                         <Rect
                           x={Math.min(smartDetectDraft.x1, smartDetectDraft.x2)}
                           y={Math.min(smartDetectDraft.y1, smartDetectDraft.y2)}
                           width={Math.abs(smartDetectDraft.x2 - smartDetectDraft.x1)}
                           height={Math.abs(smartDetectDraft.y2 - smartDetectDraft.y1)}
                           fill="rgba(0,0,0,0.08)"
                           stroke={classes.find((c) => c.id === selectedClassId)?.color || "#a855f7"}
                           strokeWidth={2 / scale}
                           dash={[6 / scale, 4 / scale]}
                         />
                       </Layer>
                     )}
                     <Layer>
                     {annotations.map((ann) => {
                         if (ann.visible === false) return null;
                         const isSelected = selectedId === ann.id;
                        const isHovered = hoveredId === ann.id;
                        const isDragging = draggingId === ann.id;
                        const strokeWidth = (isHovered ? 3 : 2) / scale;
                        const strokeColor = isSelected ? "#FF0000" : ann.color;
                        const dash = isSelected ? [6 / scale, 4 / scale] : undefined;
                        const nodeOpacity = isDragging ? 0.3 : 1;
                        const labelText = normalizeLabel(ann.label);
                        const labelFontSize = 12 / scale;
                        const labelPadding = 4 / scale;
                        const labelStrokeWidth = 1 / scale;

                        const labelPos = (() => {
                            if (ann.type === "polygon" && ann.points && ann.points.length >= 2) {
                                let minX = Number.POSITIVE_INFINITY;
                                let minY = Number.POSITIVE_INFINITY;
                                for (let i = 0; i < ann.points.length; i += 2) {
                                    minX = Math.min(minX, ann.points[i] ?? 0);
                                    minY = Math.min(minY, ann.points[i + 1] ?? 0);
                                }
                                const ox = ann.x ?? 0;
                                const oy = ann.y ?? 0;
                                return { x: minX + ox, y: minY + oy };
                            }
                            return { x: ann.x ?? 0, y: ann.y ?? 0 };
                        })();

                        const labelYOffset = labelFontSize + labelPadding * 2 + 2 / scale;
                        const labelX = labelPos.x + 1 / scale;
                        const labelY = Math.max(0, labelPos.y - labelYOffset);
                        const labelNode = labelText ? (
                             <KonvaLabel
                                 x={labelX}
                                 y={labelY}
                                 onClick={(e) => {
                                     e.cancelBubble = true;
                                     setSelectedTool("select");
                                     setSelectedId(ann.id);
                                 }}
                                 onDblClick={(e) => {
                                     e.cancelBubble = true;
                                     openLabelPicker(ann.id, e.evt.clientX, e.evt.clientY);
                                 }}
                             >
                                <KonvaTag
                                    fill="rgba(17, 24, 39, 0.85)"
                                    stroke={ann.color}
                                    strokeWidth={labelStrokeWidth}
                                    cornerRadius={2 / scale}
                                />
                                <Text
                                    text={labelText}
                                    fontSize={labelFontSize}
                                    padding={labelPadding}
                                    fill="#ffffff"
                                    fontStyle="bold"
                                    fontFamily="monospace"
                                />
                            </KonvaLabel>
                         ) : null;
                        const bbox = (() => {
                          if (ann.type === "rect") {
                            return {
                              x: ann.x ?? 0,
                              y: ann.y ?? 0,
                              width: ann.width ?? 0,
                              height: ann.height ?? 0,
                            };
                          }
                          if (ann.type === "polygon" && ann.points && ann.points.length >= 2) {
                            let minX = Number.POSITIVE_INFINITY;
                            let minY = Number.POSITIVE_INFINITY;
                            let maxX = Number.NEGATIVE_INFINITY;
                            let maxY = Number.NEGATIVE_INFINITY;
                            for (let i = 0; i < ann.points.length; i += 2) {
                              const px = ann.points[i] ?? 0;
                              const py = ann.points[i + 1] ?? 0;
                              minX = Math.min(minX, px);
                              minY = Math.min(minY, py);
                              maxX = Math.max(maxX, px);
                              maxY = Math.max(maxY, py);
                            }
                            const ox = ann.x ?? 0;
                            const oy = ann.y ?? 0;
                            return {
                              x: minX + ox,
                              y: minY + oy,
                              width: Math.max(0, maxX - minX),
                              height: Math.max(0, maxY - minY),
                            };
                          }
                          return { x: 0, y: 0, width: 0, height: 0 };
                        })();

                        const editIconSize = 12 / scale;
                        const editPad = 1 / scale;
                        const editX = Math.max(0, bbox.x + Math.max(0, bbox.width - editIconSize - editPad));
                        const editY = Math.max(0, bbox.y + editPad);
                        const editNode = (
                          <Group
                            x={editX}
                            y={editY}
                            onClick={(e) => {
                              e.cancelBubble = true;
                              setSelectedTool("select");
                              setSelectedId(ann.id);
                              openLabelPicker(ann.id, e.evt.clientX, e.evt.clientY);
                            }}
                          >
                            <Rect
                              width={editIconSize}
                              height={editIconSize}
                              fill="rgba(17, 24, 39, 0.85)"
                              stroke="rgba(255,255,255,0.6)"
                              strokeWidth={labelStrokeWidth}
                              cornerRadius={2 / scale}
                            />
                            <Text
                              text="✎"
                              width={editIconSize}
                              height={editIconSize}
                              fontSize={9 / scale}
                              fill="#ffffff"
                              fontStyle="bold"
                              align="center"
                              verticalAlign="middle"
                              fontFamily="monospace"
                              listening={false}
                            />
                          </Group>
                        );

                        if (ann.type === "rect") {
                            return (
                                <Fragment key={ann.id}>
                                    <Rect
                                        id={ann.id}
                                        x={ann.x}
                                        y={ann.y}
                                        width={ann.width}
                                        height={ann.height}
                                        stroke={strokeColor}
                                        strokeWidth={strokeWidth}
                                        dash={dash}
                                        fillEnabled={false}
                                        opacity={nodeOpacity}
                                        draggable={selectedTool === "select"}
                                        onMouseEnter={() => setHoveredId(ann.id)}
                                        onMouseLeave={() => setHoveredId((prev) => (prev === ann.id ? null : prev))}
                                        onClick={(e) => {
                                            e.cancelBubble = true;
                                            setSelectedTool("select");
                                            setSelectedId(ann.id);
                                        }}
                                         onDblClick={(e) => {
                                             e.cancelBubble = true;
                                             openLabelPicker(ann.id, e.evt.clientX, e.evt.clientY);
                                         }}
                                        onDragStart={() => {
                                          setSelectedId(ann.id);
                                          setDraggingId(ann.id);
                                        }}
                                        onDragMove={(e) => {
                                            const nextX = e.target.x();
                                            const nextY = e.target.y();
                                            setAnnotations((prev) =>
                                                prev.map((a) => (a.id === ann.id ? { ...a, x: nextX, y: nextY } : a))
                                            );
                                        }}
                                        onDragEnd={(e) => {
                                            setDraggingId((prev) => (prev === ann.id ? null : prev));
                                            const newAnns = annotations.map(a => 
                                                a.id === ann.id ? { ...a, x: e.target.x(), y: e.target.y() } : a
                                            );
                                            addToHistory(newAnns);
                                        }}
                                        onTransformEnd={(e) => {
                                            const node = e.target;
                                            const newAnns = annotations.map(a => 
                                                a.id === ann.id ? {
                                                    ...a,
                                                    x: node.x(),
                                                    y: node.y(),
                                                    width: Math.max(5, node.width() * node.scaleX()),
                                                    height: Math.max(5, node.height() * node.scaleY()),
                                                } : a
                                            );
                                            node.scaleX(1);
                                            node.scaleY(1);
                                            addToHistory(newAnns);
                                        }}
                                    />
                                     {editNode}
                                     {labelNode}
                                 </Fragment>
                             );
                         } else if (ann.type === "polygon" && ann.points) {
                             return (
                                 <Fragment key={ann.id}>
                                    <Line
                                        id={ann.id}
                                        x={ann.x ?? 0}
                                        y={ann.y ?? 0}
                                        points={ann.points}
                                        stroke={strokeColor}
                                        strokeWidth={strokeWidth}
                                        dash={dash}
                                        fillEnabled={false}
                                        opacity={nodeOpacity}
                                        closed={true}
                                        draggable={selectedTool === "select"}
                                        onMouseEnter={() => setHoveredId(ann.id)}
                                        onMouseLeave={() => setHoveredId((prev) => (prev === ann.id ? null : prev))}
                                        onClick={(e) => {
                                            e.cancelBubble = true;
                                            setSelectedTool("select");
                                            setSelectedId(ann.id);
                                        }}
                                        onDblClick={(e) => {
                                            e.cancelBubble = true;
                                            openLabelPicker(ann.id, e.evt.clientX, e.evt.clientY);
                                         }}
                                        onDragStart={() => {
                                          setSelectedId(ann.id);
                                          setDraggingId(ann.id);
                                        }}
                                        onDragMove={(e) => {
                                            const nextX = e.target.x();
                                            const nextY = e.target.y();
                                            setAnnotations((prev) =>
                                                prev.map((a) => (a.id === ann.id ? { ...a, x: nextX, y: nextY } : a))
                                            );
                                        }}
                                        onDragEnd={(e) => {
                                            setDraggingId((prev) => (prev === ann.id ? null : prev));
                                            const newAnns = annotations.map(a => 
                                                a.id === ann.id ? { ...a, x: e.target.x(), y: e.target.y() } : a
                                            );
                                            addToHistory(newAnns);
                                        }}
                                    />
                                     {editNode}
                                     {labelNode}
                                 </Fragment>
                             );
                         }
                        return null;
                    })}
                    
                    {/* Drawing Polygon Preview */}
                    {isDrawingPoly && (
                        <>
                            <Line
                                points={mousePos ? [...currentPolyPoints, mousePos.x, mousePos.y] : currentPolyPoints}
                                stroke={getCurrentClass().color}
                                strokeWidth={2 / scale}
                                dash={[5, 5]}
                            />
                            {currentPolyPoints.map((_, i) => {
                                    if (i % 2 !== 0) return null;
                                    return (
                                        <Rect
                                        key={i}
                                        x={currentPolyPoints[i] - 3 / scale}
                                        y={currentPolyPoints[i+1] - 3 / scale}
                                        width={6 / scale}
                                        height={6 / scale}
                                        fill={getCurrentClass().color}
                                        />
                                    )
                            })}
                        </>
                    )}
                     </Layer>
                 </Stage>
             </div>

             {mousePos && (smartMode || selectedTool === "rect" || selectedTool === "polygon" || isDrawingPoly) ? (
               <div className="absolute bottom-2 right-2 z-40 pointer-events-none rounded bg-background/70 px-2 py-1 text-xs text-muted-foreground shadow-sm backdrop-blur-sm">
                 x: {Math.round(Math.max(0, Math.min(imageCanvasWidth - 1, mousePos.x)))} y:{" "}
                 {Math.round(Math.max(0, Math.min(imageCanvasHeight - 1, mousePos.y)))}
               </div>
             ) : null}

             {labelPicker.open && labelPicker.annotationId && (
               <div
                 ref={labelPickerRef}
                 className="absolute z-50 w-64 rounded-md border bg-background shadow-lg p-2"
                 style={{ left: labelPicker.x, top: labelPicker.y }}
                 onMouseDown={(e) => e.stopPropagation()}
               >
                 <div className="text-xs text-muted-foreground mb-2">选择标签类别（可搜索）</div>
                 <Input
                   value={labelPicker.query}
                   onChange={(e) => setLabelPicker((prev) => ({ ...prev, query: e.target.value }))}
                   placeholder="搜索标签..."
                   className="h-8"
                   autoFocus
                 />
                 <div className="mt-2 max-h-56 overflow-auto space-y-1">
                   {(() => {
                     const q = normalizeLabel(labelPicker.query);
                     const target = annotations.find((a) => a.id === labelPicker.annotationId) || null;
                     const currentLabel = target ? normalizeLabel(target.label) : "";
                     const filtered = classes.filter((c) => {
                       const name = normalizeLabel(c.name);
                       return !q || name.includes(q);
                     });

                     if (filtered.length === 0) {
                       return <div className="text-xs text-muted-foreground px-2 py-2">无匹配标签</div>;
                     }

                     return filtered.map((cls) => {
                       const active = normalizeLabel(cls.name) === currentLabel;
                       return (
                         <button
                           key={cls.id}
                           type="button"
                           className={`w-full flex items-center gap-2 px-2 py-1.5 rounded text-left text-sm hover:bg-muted ${
                             active ? "bg-primary/10" : ""
                           }`}
                           onClick={() => applyLabelClassToAnnotation(labelPicker.annotationId as string, cls.id)}
                         >
                           <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: cls.color }} />
                           <span className="flex-1 truncate font-medium">{cls.name}</span>
                           {active && <span className="text-[10px] text-primary">当前</span>}
                         </button>
                       );
                     });
                   })()}
                 </div>
               </div>
             )}
         </div>

        {/* Right Sidebar - Info Panel */}
        {isSidebarOpen && (
        <LayerPanel
            isSidebarOpen={isSidebarOpen}
            activeTab={activeTab}
            setActiveTab={setActiveTab}
            brightness={brightness}
            setBrightness={setBrightness}
            contrast={contrast}
            setContrast={setContrast}
            showCrosshair={showCrosshair}
            setShowCrosshair={setShowCrosshair}
            annotations={annotations}
            selectedId={selectedId}
            setSelectedId={setSelectedId}
            onDelete={handleDelete}
            onToggleVisibility={toggleVisibility}
            classes={classes}
            selectedClassId={selectedClassId}
            onClassSelect={handleClassSelect}
            onClassAdd={handleAddClass}
            onClassEdit={handleEditClass}
            onClassDelete={handleDeleteClass}
            onEditAnnotationLabel={handleEditAnnotationLabel}
            currentImageIndex={currentImageIndex}
            totalImages={imageList.length}
            images={imageList}
        />
        )}
      </div>

      {/* Bottom Thumbnail Bar */}
      <div className="h-32 bg-background border-t p-2 overflow-x-auto whitespace-nowrap">
          <div className="flex gap-2 h-full items-center">
               {imageList.map((img, idx) => (
                   <div
                     key={img.id}
                     id={`thumb-${idx}`}
                     className={`relative inline-block h-24 w-32 rounded-lg overflow-hidden border-2 cursor-pointer transition-all shrink-0 ${
                         currentImageIndex === idx ? "border-primary ring-2 ring-primary/20" : 
                         img.status === 'completed' ? "border-green-500/50" : "border-transparent hover:border-primary/50"
                     }`}
                     onClick={() => handleJumpToImage(idx)}
                   >
                       <img src={img.url} alt={img.name} className="w-full h-full object-cover" />
                       {img.annotationsCount === 0 && (
                         <button
                           type="button"
                           className="absolute top-1 left-1 rounded bg-black/30 p-0.5 hover:bg-black/40"
                           title="删除无缺陷图片"
                           onClick={(e) => {
                             e.stopPropagation();
                             void handleDeleteImageAtIndex(idx);
                           }}
                         >
                           <Trash2 className="h-4 w-4 text-red-500 opacity-50" />
                         </button>
                       )}
                       <div className="absolute bottom-0 left-0 right-0 bg-black/60 text-white text-[10px] px-1 py-0.5 truncate">
                           {idx + 1}. {img.name}
                       </div>
                       {img.status === 'completed' && (
                           <div className="absolute top-1 right-1 bg-green-500 rounded-full p-0.5">
                               <CheckCircle2 className="w-3 h-3 text-white" />
                           </div>
                       )}
                   </div>
               ))}
          </div>
      </div>
    </div>
  );
}
