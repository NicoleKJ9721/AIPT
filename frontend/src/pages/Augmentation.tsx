import { useCallback, useEffect, useMemo, useState } from "react";
import { 
    RotateCw, Sun, Zap, Layers, CloudRain, 
    Save, RotateCcw, ArrowRight,
    CheckCircle2, Image as ImageIcon,
    Download, EyeOff,
    Pencil,
    Trash2
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  annotationService,
  datasetService,
  imageService,
  projectService,
  type AnnotationCreatePayload,
  type AnnotationRecord,
  type DatasetImageStatsRecord,
  type DatasetRecord,
  type ImageRecord,
  type ProjectRecord,
} from "@/lib/api";
import { toast } from "@/components/ui/use-toast";
import { useProjectContext } from "@/store/projectContext";

function resolveImageUrl(sourceUrl: string | null | undefined): string {
  const url = (sourceUrl || "").trim();
  if (!url) return "";
  if (url.startsWith("http://") || url.startsWith("https://") || url.startsWith("data:")) return url;
  if (url.startsWith("/api/")) return url;
  if (url.startsWith("/")) return `/api${url}`;
  return url;
}

function isLegacyAugmentedFilename(filename: string): boolean {
  const name = (filename || "").trim();
  if (!name) return false;
  return /_aug_\d{10,}_/i.test(name);
}

type AugmentConfig = {
  flipH: boolean;
  flipV: boolean;
  rotate: boolean;
  rotateDeg: number;
  shear: number;
  translate: number;
  hue: number;
  saturation: number;
  brightness: number;
  contrast: number;
  blur: number;
  noise: boolean;
  noiseAmount: number;
  weather: "none" | "rain" | "snow" | "fog";
  mosaic: boolean;
  mixup: boolean;
  cutout: boolean;
};

const DEFAULT_AUGMENT_CONFIG: AugmentConfig = {
  flipH: false,
  flipV: false,
  rotate: true,
  rotateDeg: 30,
  shear: 0,
  translate: 0,
  hue: 0,
  saturation: 100,
  brightness: 100,
  contrast: 100,
  blur: 0,
  noise: false,
  noiseAmount: 20,
  weather: "none",
  mosaic: false,
  mixup: false,
  cutout: false,
};

const WEATHER_OPTIONS = ["none", "rain", "snow", "fog"] as const;
const MANUAL_AUGMENTATION_ENABLED = false;

function baseAugmentConfig(): AugmentConfig {
  return {
    flipH: false,
    flipV: false,
    rotate: false,
    rotateDeg: 0,
    shear: 0,
    translate: 0,
    hue: 0,
    saturation: 100,
    brightness: 100,
    contrast: 100,
    blur: 0,
    noise: false,
    noiseAmount: 20,
    weather: "none",
    mosaic: false,
    mixup: false,
    cutout: false,
  };
}

type AugmentVariant = { key: string; label: string; cfg: AugmentConfig };

function buildAugmentVariants(cfg: AugmentConfig): AugmentVariant[] {
  const variants: AugmentVariant[] = [];

  if (cfg.flipH) variants.push({ key: "flipH", label: "Flip H", cfg: { ...baseAugmentConfig(), flipH: true } });
  if (cfg.flipV) variants.push({ key: "flipV", label: "Flip V", cfg: { ...baseAugmentConfig(), flipV: true } });

  if (cfg.rotateDeg) {
    variants.push({
      key: "rotate",
      label: "Rotate",
      cfg: { ...baseAugmentConfig(), rotate: true, rotateDeg: cfg.rotateDeg },
    });
  }

  if (cfg.shear) variants.push({ key: "shear", label: "Shear", cfg: { ...baseAugmentConfig(), shear: cfg.shear } });

  if (cfg.hue) variants.push({ key: "hue", label: "Hue", cfg: { ...baseAugmentConfig(), hue: cfg.hue } });
  if (cfg.saturation !== 100) variants.push({ key: "saturation", label: "Saturation", cfg: { ...baseAugmentConfig(), saturation: cfg.saturation } });
  if (cfg.brightness !== 100) variants.push({ key: "brightness", label: "Brightness", cfg: { ...baseAugmentConfig(), brightness: cfg.brightness } });
  if (cfg.contrast !== 100) variants.push({ key: "contrast", label: "Contrast", cfg: { ...baseAugmentConfig(), contrast: cfg.contrast } });

  if (cfg.blur) variants.push({ key: "blur", label: "Blur", cfg: { ...baseAugmentConfig(), blur: cfg.blur } });
  if (cfg.noise) variants.push({ key: "noise", label: "Noise", cfg: { ...baseAugmentConfig(), noise: true, noiseAmount: cfg.noiseAmount } });
  if (cfg.weather !== "none")
    variants.push({
      key: `weather_${cfg.weather}`,
      label: `Weather ${cfg.weather}`,
      cfg: { ...baseAugmentConfig(), weather: cfg.weather },
    });

  if (cfg.mosaic) variants.push({ key: "mosaic", label: "Mosaic", cfg: { ...baseAugmentConfig(), mosaic: true } });
  if (cfg.mixup) variants.push({ key: "mixup", label: "Mixup", cfg: { ...baseAugmentConfig(), mixup: true } });
  if (cfg.cutout) variants.push({ key: "cutout", label: "Cutout", cfg: { ...baseAugmentConfig(), cutout: true } });

  return variants;
}

function mulberry32(seed: number): () => number {
  return () => {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp(n: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, n));
}

function basenameNoExt(filename: string): string {
  const base = (filename || "").trim().replaceAll("\\", "/").split("/").pop() || "image";
  const dot = base.lastIndexOf(".");
  return dot > 0 ? base.slice(0, dot) : base;
}

function makePreviewVariant(base: AugmentConfig, tab: string, seed: number): AugmentConfig {
  const rand = mulberry32(seed);
  const next: AugmentConfig = { ...base };
  const jitter = (val: number, amount: number) => val + (rand() - 0.5) * 2 * amount;

  if (tab === "geometric") {
    if (next.rotate) next.rotateDeg = clamp(jitter(next.rotateDeg, Math.max(3, next.rotateDeg * 0.25)), 0, 180);
    next.translate = clamp(jitter(next.translate, 0.05), 0, 0.5);
    next.shear = clamp(jitter(next.shear, 5), -45, 45);
  } else if (tab === "color") {
    next.hue = clamp(jitter(next.hue, 15), -180, 180);
    next.saturation = clamp(jitter(next.saturation, 20), 0, 200);
    next.brightness = clamp(jitter(next.brightness, 20), 0, 200);
    next.contrast = clamp(jitter(next.contrast, 20), 0, 200);
  } else if (tab === "noise") {
    next.blur = clamp(jitter(next.blur, 1), 0, 10);
    if (next.noise) next.noiseAmount = clamp(jitter(next.noiseAmount, 10), 0, 100);
  } else if (tab === "advanced") {
    // Cutout position/size is randomized at render time.
    next.mixup = base.mixup;
  }

  return next;
}

type XY = { x: number; y: number };

function transformPoint(pt: XY, cfg: AugmentConfig, w: number, h: number): XY {
  const imgW = w || 1;
  const imgH = h || 1;

  let x = pt.x - imgW / 2;
  let y = pt.y - imgH / 2;

  const sx = cfg.flipH ? -1 : 1;
  const sy = cfg.flipV ? -1 : 1;
  x *= sx;
  y *= sy;

  if (cfg.shear) {
    const sh = Math.tan((cfg.shear * Math.PI) / 180);
    x = x + sh * y;
  }

  if (cfg.rotate) {
    const t = (cfg.rotateDeg * Math.PI) / 180;
    const cos = Math.cos(t);
    const sin = Math.sin(t);
    const nx = x * cos - y * sin;
    const ny = x * sin + y * cos;
    x = nx;
    y = ny;
  }

  const translatePx = cfg.translate * Math.min(imgW, imgH);
  x += translatePx;
  y += translatePx;

  x += imgW / 2;
  y += imgH / 2;

  return { x, y };
}

function transformAnnotation(
  ann: AnnotationRecord,
  cfg: AugmentConfig,
  imgW: number,
  imgH: number
): AnnotationCreatePayload | null {
  if (ann.type === "rect") {
    const x0 = ann.x ?? 0;
    const y0 = ann.y ?? 0;
    const w0 = ann.width ?? 0;
    const h0 = ann.height ?? 0;
    if (w0 <= 0 || h0 <= 0) return null;

    const corners = [
      transformPoint({ x: x0, y: y0 }, cfg, imgW, imgH),
      transformPoint({ x: x0 + w0, y: y0 }, cfg, imgW, imgH),
      transformPoint({ x: x0, y: y0 + h0 }, cfg, imgW, imgH),
      transformPoint({ x: x0 + w0, y: y0 + h0 }, cfg, imgW, imgH),
    ];
    const xs = corners.map((p) => p.x);
    const ys = corners.map((p) => p.y);
    const x1 = clamp(Math.min(...xs), 0, imgW);
    const y1 = clamp(Math.min(...ys), 0, imgH);
    const x2 = clamp(Math.max(...xs), 0, imgW);
    const y2 = clamp(Math.max(...ys), 0, imgH);
    const bw = Math.max(0, x2 - x1);
    const bh = Math.max(0, y2 - y1);
    if (bw < 1 || bh < 1) return null;
    return {
      type: "rect",
      label: ann.label,
      color: ann.color,
      visible: ann.visible,
      x: x1,
      y: y1,
      width: bw,
      height: bh,
    };
  }

  if (ann.type === "polygon" && ann.points) {
    const ox = ann.x ?? 0;
    const oy = ann.y ?? 0;
    const pts = ann.points;
    if (pts.length < 6 || pts.length % 2 !== 0) return null;

    const out: number[] = [];
    for (let i = 0; i < pts.length; i += 2) {
      const p = transformPoint({ x: ox + pts[i], y: oy + pts[i + 1] }, cfg, imgW, imgH);
      out.push(clamp(p.x, 0, imgW), clamp(p.y, 0, imgH));
    }
    return {
      type: "polygon",
      label: ann.label,
      color: ann.color,
      visible: ann.visible,
      x: 0,
      y: 0,
      points: out,
    };
  }

  return null;
}

function _segmentIntersectVertical(a: XY, b: XY, x: number): XY | null {
  const dx = b.x - a.x;
  if (dx === 0) return null;
  const t = (x - a.x) / dx;
  if (!Number.isFinite(t)) return null;
  return { x, y: a.y + t * (b.y - a.y) };
}

function _segmentIntersectHorizontal(a: XY, b: XY, y: number): XY | null {
  const dy = b.y - a.y;
  if (dy === 0) return null;
  const t = (y - a.y) / dy;
  if (!Number.isFinite(t)) return null;
  return { x: a.x + t * (b.x - a.x), y };
}

function _clipPolygon(poly: XY[], inside: (p: XY) => boolean, intersect: (a: XY, b: XY) => XY | null): XY[] {
  if (poly.length === 0) return [];
  const out: XY[] = [];
  let prev = poly[poly.length - 1]!;
  let prevIn = inside(prev);
  for (const curr of poly) {
    const currIn = inside(curr);
    if (currIn) {
      if (!prevIn) {
        const pt = intersect(prev, curr);
        if (pt) out.push(pt);
      }
      out.push(curr);
    } else if (prevIn) {
      const pt = intersect(prev, curr);
      if (pt) out.push(pt);
    }
    prev = curr;
    prevIn = currIn;
  }
  return out;
}

function _clipPolygonToRect(poly: XY[], rect: Rect2): XY[] {
  let out = poly;
  out = _clipPolygon(out, (p) => p.x >= rect.x1, (a, b) => _segmentIntersectVertical(a, b, rect.x1));
  out = _clipPolygon(out, (p) => p.x <= rect.x2, (a, b) => _segmentIntersectVertical(a, b, rect.x2));
  out = _clipPolygon(out, (p) => p.y >= rect.y1, (a, b) => _segmentIntersectHorizontal(a, b, rect.y1));
  out = _clipPolygon(out, (p) => p.y <= rect.y2, (a, b) => _segmentIntersectHorizontal(a, b, rect.y2));
  return out;
}

function _transformPointMosaic(pt: XY, placement: MosaicPlacement): XY {
  return { x: placement.dst.x1 + (pt.x - placement.src.x1), y: placement.dst.y1 + (pt.y - placement.src.y1) };
}

function transformAnnotationMosaic(
  ann: AnnotationRecord,
  placement: MosaicPlacement,
  outW: number,
  outH: number
): AnnotationCreatePayload | null {
  if (!placement) return null;
  const src = placement.src;
  const dst = placement.dst;
  if (_rectW(src) <= 0 || _rectH(src) <= 0 || _rectW(dst) <= 0 || _rectH(dst) <= 0) return null;

  if (ann.type === "rect") {
    const x0 = ann.x ?? 0;
    const y0 = ann.y ?? 0;
    const w0 = ann.width ?? 0;
    const h0 = ann.height ?? 0;
    if (w0 <= 0 || h0 <= 0) return null;

    const x1 = Math.max(x0, src.x1);
    const y1 = Math.max(y0, src.y1);
    const x2 = Math.min(x0 + w0, src.x2);
    const y2 = Math.min(y0 + h0, src.y2);
    const bw = x2 - x1;
    const bh = y2 - y1;
    if (bw < 1 || bh < 1) return null;

    const p1 = _transformPointMosaic({ x: x1, y: y1 }, placement);
    const p2 = _transformPointMosaic({ x: x2, y: y2 }, placement);
    const ox1 = clamp(Math.min(p1.x, p2.x), 0, outW);
    const oy1 = clamp(Math.min(p1.y, p2.y), 0, outH);
    const ox2 = clamp(Math.max(p1.x, p2.x), 0, outW);
    const oy2 = clamp(Math.max(p1.y, p2.y), 0, outH);
    const obw = Math.max(0, ox2 - ox1);
    const obh = Math.max(0, oy2 - oy1);
    if (obw < 1 || obh < 1) return null;

    return {
      type: "rect",
      label: ann.label,
      color: ann.color,
      visible: ann.visible,
      x: ox1,
      y: oy1,
      width: obw,
      height: obh,
    };
  }

  if (ann.type === "polygon" && ann.points) {
    const ox = ann.x ?? 0;
    const oy = ann.y ?? 0;
    const pts = ann.points;
    if (pts.length < 6 || pts.length % 2 !== 0) return null;

    const poly: XY[] = [];
    for (let i = 0; i < pts.length; i += 2) {
      poly.push({ x: ox + pts[i], y: oy + pts[i + 1] });
    }

    const clipped = _clipPolygonToRect(poly, src);
    if (clipped.length < 3) return null;

    const out: number[] = [];
    for (const p of clipped) {
      const m = _transformPointMosaic(p, placement);
      out.push(clamp(m.x, 0, outW), clamp(m.y, 0, outH));
    }
    if (out.length < 6) return null;

    return {
      type: "polygon",
      label: ann.label,
      color: ann.color,
      visible: ann.visible,
      x: 0,
      y: 0,
      points: out,
    };
  }

  return null;
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

type Rect2 = { x1: number; y1: number; x2: number; y2: number };
type MosaicPlacement = { src: Rect2; dst: Rect2 };
type MosaicMeta = { outW: number; outH: number; placements: MosaicPlacement[] };

function _rectW(r: Rect2) {
  return Math.max(0, r.x2 - r.x1);
}

function _rectH(r: Rect2) {
  return Math.max(0, r.y2 - r.y1);
}

function _buildMosaicMeta(
  sizes: Array<{ w: number; h: number }>,
  outW: number,
  outH: number,
  seed: number
): MosaicMeta {
  const rand = mulberry32(seed);
  const xc = Math.floor((0.25 + rand() * 0.5) * outW);
  const yc = Math.floor((0.25 + rand() * 0.5) * outH);

  const order = [0, 1, 2, 3];
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = order[i];
    order[i] = order[j];
    order[j] = tmp;
  }

  const placements: MosaicPlacement[] = new Array(4) as MosaicPlacement[];
  for (let quad = 0; quad < 4; quad++) {
    const idx = order[quad];
    const s = sizes[idx] ?? { w: outW, h: outH };
    const iw = Math.max(1, Math.floor(s.w));
    const ih = Math.max(1, Math.floor(s.h));

    let dst: Rect2;
    let src: Rect2;
    if (quad === 0) {
      // top-left
      dst = { x1: Math.max(xc - iw, 0), y1: Math.max(yc - ih, 0), x2: xc, y2: yc };
      src = { x1: iw - _rectW(dst), y1: ih - _rectH(dst), x2: iw, y2: ih };
    } else if (quad === 1) {
      // top-right
      dst = { x1: xc, y1: Math.max(yc - ih, 0), x2: Math.min(xc + iw, outW), y2: yc };
      src = { x1: 0, y1: ih - _rectH(dst), x2: _rectW(dst), y2: ih };
    } else if (quad === 2) {
      // bottom-left
      dst = { x1: Math.max(xc - iw, 0), y1: yc, x2: xc, y2: Math.min(yc + ih, outH) };
      src = { x1: iw - _rectW(dst), y1: 0, x2: iw, y2: _rectH(dst) };
    } else {
      // bottom-right
      dst = { x1: xc, y1: yc, x2: Math.min(xc + iw, outW), y2: Math.min(yc + ih, outH) };
      src = { x1: 0, y1: 0, x2: _rectW(dst), y2: _rectH(dst) };
    }

    placements[idx] = { src, dst };
  }

  return { outW, outH, placements };
}

async function renderMosaicPngBlob(
  urls: string[],
  opts: { seed: number }
): Promise<{ blob: Blob; meta: MosaicMeta }> {
  if (urls.length < 4) throw new Error("Mosaic requires at least 4 images");
  const imgs = await Promise.all(urls.slice(0, 4).map((u) => loadImageElement(u)));
  const outW = imgs[0].naturalWidth || imgs[0].width || 1;
  const outH = imgs[0].naturalHeight || imgs[0].height || 1;

  const canvas = document.createElement("canvas");
  canvas.width = outW;
  canvas.height = outH;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas not supported");

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, outW, outH);

  const sizes = imgs.map((im) => ({ w: im.naturalWidth || im.width || 1, h: im.naturalHeight || im.height || 1 }));
  const meta = _buildMosaicMeta(sizes, outW, outH, opts.seed);
  for (let i = 0; i < 4; i++) {
    const img = imgs[i];
    const placement = meta.placements[i];
    if (!placement) continue;
    const sw = _rectW(placement.src);
    const sh = _rectH(placement.src);
    const dw = _rectW(placement.dst);
    const dh = _rectH(placement.dst);
    if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) continue;
    ctx.drawImage(
      img,
      placement.src.x1,
      placement.src.y1,
      sw,
      sh,
      placement.dst.x1,
      placement.dst.y1,
      dw,
      dh
    );
  }

  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((b) => {
      if (!b) reject(new Error("Failed to encode PNG"));
      else resolve(b);
    }, "image/png");
  });
  return { blob, meta };
}

async function renderAugmentedPngBlob(
  url: string,
  cfg: AugmentConfig,
  opts: { mixupUrl?: string; seed: number }
): Promise<Blob> {
  const img = await loadImageElement(url);
  const w = img.naturalWidth || img.width || 1;
  const h = img.naturalHeight || img.height || 1;

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas not supported");

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.clearRect(0, 0, w, h);
  ctx.filter = `hue-rotate(${cfg.hue}deg) saturate(${cfg.saturation}%) brightness(${cfg.brightness}%) contrast(${cfg.contrast}%) blur(${cfg.blur}px)`;

  ctx.save();
  ctx.translate(w / 2, h / 2);

  const translatePx = cfg.translate * Math.min(w, h);
  ctx.translate(translatePx, translatePx);

  if (cfg.rotate) {
    ctx.rotate((cfg.rotateDeg * Math.PI) / 180);
  }
  if (cfg.shear) {
    const shear = Math.tan((cfg.shear * Math.PI) / 180);
    ctx.transform(1, 0, shear, 1, 0, 0);
  }
  ctx.scale(cfg.flipH ? -1 : 1, cfg.flipV ? -1 : 1);

  ctx.drawImage(img, -w / 2, -h / 2, w, h);
  ctx.restore();

  // Mixup: blend a second image on top.
  if (cfg.mixup && opts.mixupUrl) {
    try {
      const img2 = await loadImageElement(opts.mixupUrl);
      ctx.globalAlpha = 0.5;
      ctx.drawImage(img2, 0, 0, w, h);
      ctx.globalAlpha = 1;
    } catch {
      // best-effort only
    }
  }

  // Noise overlay (lightweight).
  if (cfg.noise) {
    const rand = mulberry32(opts.seed);
    const dots = Math.min(20000, Math.floor((w * h) / 250));
    ctx.save();
    ctx.globalAlpha = clamp(cfg.noiseAmount / 100, 0, 1);
    for (let i = 0; i < dots; i++) {
      const x = Math.floor(rand() * w);
      const y = Math.floor(rand() * h);
      const v = Math.floor(rand() * 255);
      ctx.fillStyle = `rgb(${v},${v},${v})`;
      ctx.fillRect(x, y, 1, 1);
    }
    ctx.restore();
  }

  // Weather overlays (very lightweight).
  if (cfg.weather === "fog") {
    ctx.save();
    ctx.fillStyle = "rgba(255,255,255,0.25)";
    ctx.fillRect(0, 0, w, h);
    ctx.restore();
  } else if (cfg.weather === "rain") {
    const rand = mulberry32(opts.seed);
    ctx.save();
    ctx.strokeStyle = "rgba(200,220,255,0.25)";
    ctx.lineWidth = 1;
    const drops = Math.min(3000, Math.floor((w * h) / 1500));
    for (let i = 0; i < drops; i++) {
      const x = rand() * w;
      const y = rand() * h;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + 4, y + 12);
      ctx.stroke();
    }
    ctx.restore();
  } else if (cfg.weather === "snow") {
    const rand = mulberry32(opts.seed);
    ctx.save();
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    const flakes = Math.min(2000, Math.floor((w * h) / 2500));
    for (let i = 0; i < flakes; i++) {
      const x = rand() * w;
      const y = rand() * h;
      const r = 1 + rand() * 2;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  // Cutout: random black rectangle.
  if (cfg.cutout) {
    const rand = mulberry32(opts.seed);
    const cw = Math.floor(w * (0.15 + rand() * 0.2));
    const ch = Math.floor(h * (0.15 + rand() * 0.2));
    const cx = Math.floor(rand() * (w - cw));
    const cy = Math.floor(rand() * (h - ch));
    ctx.save();
    ctx.fillStyle = "rgba(0,0,0,1)";
    ctx.fillRect(cx, cy, cw, ch);
    ctx.restore();
  }

  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) reject(new Error("Failed to encode PNG"));
      else resolve(blob);
    }, "image/png");
  });
}

export default function AugmentationConfig() {
  const [activeTab, setActiveTab] = useState("geometric");
  const [showOriginal, setShowOriginal] = useState(false);
  const [isDatasetManagerOpen, setIsDatasetManagerOpen] = useState(false);
  const [editingDatasetId, setEditingDatasetId] = useState<string | null>(null);
  const [editDatasetName, setEditDatasetName] = useState("");
  const [editDatasetVersion, setEditDatasetVersion] = useState("");
  const [galleryPage, setGalleryPage] = useState(1);

  const projectId = useProjectContext((s) => s.projectId);
  const datasetId = useProjectContext((s) => s.datasetId);
  const setProject = useProjectContext((s) => s.setProject);
  const setDataset = useProjectContext((s) => s.setDataset);
  const clearDataset = useProjectContext((s) => s.clearDataset);

  const [projects, setProjects] = useState<ProjectRecord[]>([]);
  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  const [images, setImages] = useState<ImageRecord[]>([]);
  const [isLoadingProjects, setIsLoadingProjects] = useState(true);
  const [isLoadingDatasets, setIsLoadingDatasets] = useState(false);
  const [isLoadingImages, setIsLoadingImages] = useState(false);
  const [datasetStats, setDatasetStats] = useState<DatasetImageStatsRecord | null>(null);
  const [previewSamples, setPreviewSamples] = useState<string[]>([]);
  const [isApplying, setIsApplying] = useState(false);
  const [isSnapshotOpen, setIsSnapshotOpen] = useState(false);
  const [snapshotName, setSnapshotName] = useState("");
  const [snapshotVersion, setSnapshotVersion] = useState("");
  const [snapshotDescription, setSnapshotDescription] = useState("");
  const [annotationSummaries, setAnnotationSummaries] = useState<
    Record<
      string,
      {
        labels: Array<{ label: string; color: string }>;
        boxes: Array<{ x: number; y: number; width: number; height: number; label: string; color: string }>;
      }
    >
  >({});

  const activeDataset = useMemo(
    () => datasets.find((d) => d.id === datasetId) ?? null,
    [datasetId, datasets]
  );

  const openDatasetManager = useCallback(() => {
    if (!projectId) {
      toast({ title: "请先选择项目", variant: "destructive" });
      return;
    }
    setEditingDatasetId(null);
    setEditDatasetName("");
    setEditDatasetVersion("");
    setIsDatasetManagerOpen(true);
  }, [projectId]);

  const refreshDatasets = useCallback(async () => {
    if (!projectId) return;
    const data = await datasetService.list({ project_id: projectId, limit: 200 });
    setDatasets(data);
  }, [projectId]);

  const beginEditDataset = useCallback((d: DatasetRecord) => {
    setEditingDatasetId(d.id);
    setEditDatasetName(d.name);
    setEditDatasetVersion(d.version || "");
  }, []);

  const cancelEditDataset = useCallback(() => {
    setEditingDatasetId(null);
    setEditDatasetName("");
    setEditDatasetVersion("");
  }, []);

  const saveEditDataset = useCallback(async () => {
    if (!editingDatasetId) return;
    const nextName = editDatasetName.trim();
    const nextVersion = editDatasetVersion.trim();
    if (!nextName) {
      toast({ title: "名称不能为空", variant: "destructive" });
      return;
    }
    try {
      const updated = await datasetService.update(editingDatasetId, {
        name: nextName,
        version: nextVersion || undefined,
      });
      await refreshDatasets();
      if (datasetId === editingDatasetId) {
        setDataset(editingDatasetId, `${updated.name} ${updated.version}`);
      }
      toast({ title: "数据集已更新" });
      cancelEditDataset();
    } catch (error) {
      console.error(error);
      toast({ title: "更新失败", description: "请检查 API Key / 权限设置或后端日志", variant: "destructive" });
    }
  }, [
    cancelEditDataset,
    datasetId,
    editDatasetName,
    editDatasetVersion,
    editingDatasetId,
    refreshDatasets,
    setDataset,
  ]);

  const deleteDataset = useCallback(async (d: DatasetRecord) => {
    if (!projectId) return;
    const ok = window.confirm(`确认删除数据集版本：${d.name} ${d.version} ？该操作不可撤销`);
    if (!ok) return;
    try {
      await datasetService.delete(d.id);
      await refreshDatasets();
      if (datasetId === d.id) {
        clearDataset();
        setImages([]);
      }
      toast({ title: "已删除数据集" });
    } catch (error) {
      console.error(error);
      toast({ title: "删除失败", description: "请检查 API Key / 权限设置或后端日志", variant: "destructive" });
    }
  }, [clearDataset, datasetId, projectId, refreshDatasets]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        setIsLoadingProjects(true);
        const data = await projectService.list();
        if (cancelled) return;
        setProjects(data);
        if (!projectId && data[0]) {
          setProject(data[0].id, data[0].name);
        }
      } catch (error) {
        console.error(error);
        toast({ title: "加载项目失败", description: "请检查后端服务是否启动", variant: "destructive" });
      } finally {
        if (!cancelled) setIsLoadingProjects(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [projectId, setProject]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!projectId) {
        setDatasets([]);
        clearDataset();
        setImages([]);
        return;
      }
      try {
        setIsLoadingDatasets(true);
        const data = await datasetService.list({ project_id: projectId, limit: 200 });
        if (cancelled) return;
        setDatasets(data);
        if (data.length === 0) {
          clearDataset();
          setImages([]);
          return;
        }
        if (!datasetId || !data.some((d) => d.id === datasetId)) {
          setDataset(data[0].id, `${data[0].name} ${data[0].version}`);
        }
      } catch (error) {
        console.error(error);
        toast({ title: "加载数据集失败", description: "请检查后端服务或数据集状态", variant: "destructive" });
      } finally {
        if (!cancelled) setIsLoadingDatasets(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [clearDataset, datasetId, projectId, setDataset]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!projectId) return;
      try {
        setIsLoadingImages(true);
        const data = await imageService.list(projectId, datasetId ? { dataset_id: datasetId } : undefined);
        if (cancelled) return;
        setImages(data);
      } catch (error) {
        console.error(error);
        toast({ title: "加载图片失败", description: "请先在项目管理中导入图片", variant: "destructive" });
        setImages([]);
      } finally {
        if (!cancelled) setIsLoadingImages(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [datasetId, projectId]);

  const previewImage = images[0] || null;
  const previewUrl = previewImage ? resolveImageUrl(previewImage.source_url) : "";
  const mixupImageUrl = images[1] ? resolveImageUrl(images[1].source_url) : previewUrl;

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!datasetId) {
        setDatasetStats(null);
        return;
      }
      try {
        const stats = await datasetService.getImageStats(datasetId);
        if (!cancelled) setDatasetStats(stats);
      } catch (error) {
        console.error(error);
        if (!cancelled) setDatasetStats(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  // Configuration State
  const initialConfig = MANUAL_AUGMENTATION_ENABLED ? DEFAULT_AUGMENT_CONFIG : baseAugmentConfig();
  const [config, setConfig] = useState<AugmentConfig>(() => ({ ...initialConfig }));

  const handleReset = () => {
    setConfig({ ...initialConfig });
  };

  const [showGuide, setShowGuide] = useState(false);
  const configForPreview: AugmentConfig = useMemo(() => ({ ...config }), [config]);
  const augmentVariants = useMemo(
    () => (MANUAL_AUGMENTATION_ENABLED ? buildAugmentVariants(configForPreview) : []),
    [configForPreview]
  );
  const baseImageCount = datasetStats?.image_count ?? images.length;
  const predictedImageCount = baseImageCount * (1 + augmentVariants.length);

  const GALLERY_PAGE_SIZE = 36;
  const galleryTotalPages = Math.max(1, Math.ceil(Math.max(0, images.length) / GALLERY_PAGE_SIZE));
  const galleryPageSafe = Math.min(Math.max(1, galleryPage), galleryTotalPages);
  const galleryItems = useMemo(() => {
    const start = (galleryPageSafe - 1) * GALLERY_PAGE_SIZE;
    return images.slice(start, start + GALLERY_PAGE_SIZE);
  }, [galleryPageSafe, images]);

  const legacyAugmentedCount = useMemo(() => images.filter((img) => isLegacyAugmentedFilename(img.filename)).length, [images]);

  useEffect(() => {
    setGalleryPage(1);
  }, [datasetId]);

  useEffect(() => {
    let cancelled = false;

    const clampBox = (x: number, y: number, w: number, h: number, imgW: number, imgH: number) => {
      const x1 = Math.max(0, Math.min(imgW, x));
      const y1 = Math.max(0, Math.min(imgH, y));
      const x2 = Math.max(0, Math.min(imgW, x + w));
      const y2 = Math.max(0, Math.min(imgH, y + h));
      const bw = Math.max(0, x2 - x1);
      const bh = Math.max(0, y2 - y1);
      if (bw < 1 || bh < 1) return null;
      return { x: x1, y: y1, width: bw, height: bh };
    };

    const toBox = (ann: AnnotationRecord, imgW: number, imgH: number) => {
      if (ann.type === "rect") {
        const x = Number(ann.x ?? 0);
        const y = Number(ann.y ?? 0);
        const w = Number(ann.width ?? 0);
        const h = Number(ann.height ?? 0);
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(w) || !Number.isFinite(h)) return null;
        return clampBox(x, y, w, h, imgW, imgH);
      }
      if (ann.type === "polygon" && ann.points && ann.points.length >= 6) {
        const ox = Number(ann.x ?? 0);
        const oy = Number(ann.y ?? 0);
        const pts = ann.points;
        let minX = Number.POSITIVE_INFINITY;
        let minY = Number.POSITIVE_INFINITY;
        let maxX = Number.NEGATIVE_INFINITY;
        let maxY = Number.NEGATIVE_INFINITY;
        for (let i = 0; i < pts.length; i += 2) {
          const px = ox + Number(pts[i] ?? 0);
          const py = oy + Number(pts[i + 1] ?? 0);
          if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
          minX = Math.min(minX, px);
          minY = Math.min(minY, py);
          maxX = Math.max(maxX, px);
          maxY = Math.max(maxY, py);
        }
        if (!Number.isFinite(minX) || !Number.isFinite(minY) || !Number.isFinite(maxX) || !Number.isFinite(maxY)) return null;
        return clampBox(minX, minY, maxX - minX, maxY - minY, imgW, imgH);
      }
      return null;
    };

    const run = async () => {
      const missing = galleryItems.filter((img) => img.id && !annotationSummaries[img.id]);
      if (missing.length === 0) return;

      const next: Record<string, { labels: Array<{ label: string; color: string }>; boxes: Array<{ x: number; y: number; width: number; height: number; label: string; color: string }> }> = {};

      const concurrency = 6;
      let idx = 0;
      const worker = async () => {
        while (idx < missing.length) {
          const img = missing[idx++];
          try {
            const anns = await annotationService.listByImage(img.id);
            const imgW = Math.max(1, img.width ?? 0);
            const imgH = Math.max(1, img.height ?? 0);

            const boxes: Array<{ x: number; y: number; width: number; height: number; label: string; color: string }> = [];
            const labelMap = new Map<string, string>();
            for (const ann of anns) {
              const label = String(ann.label || "").trim();
              const color = String(ann.color || "#ef4444").trim() || "#ef4444";
              if (label) labelMap.set(label, color);
              const b = imgW > 1 && imgH > 1 ? toBox(ann, imgW, imgH) : null;
              if (b && label) {
                boxes.push({ ...b, label, color });
              }
            }

            const labels = Array.from(labelMap.entries()).map(([label, color]) => ({ label, color }));
            next[img.id] = { labels, boxes };
          } catch (error) {
            console.error(error);
            next[img.id] = { labels: [], boxes: [] };
          }
        }
      };

      await Promise.all(Array.from({ length: Math.min(concurrency, missing.length) }, () => worker()));
      if (cancelled) return;
      setAnnotationSummaries((prev) => ({ ...prev, ...next }));
    };

    void run();
    return () => {
      cancelled = true;
    };
  }, [annotationSummaries, galleryItems]);

  // Generate 5 sample previews (for current method tab) for quick visual validation.
  const generateSamplePreviews = useCallback(async () => {
    if (!MANUAL_AUGMENTATION_ENABLED) {
      setPreviewSamples([]);
      return;
    }
    if (!previewUrl) {
      setPreviewSamples([]);
      return;
    }

    const urls: string[] = [];
    try {
      for (let i = 0; i < 5; i++) {
        const seed = Date.now() + i * 1337;
        const cfg = makePreviewVariant(configForPreview, activeTab, seed);
        console.log("[Augmentation] preview-sample", i + 1, { tab: activeTab, cfg });
        let blob: Blob;
        if (cfg.mosaic) {
          const mosaicUrls = images
            .slice(0, 4)
            .map((img) => resolveImageUrl(img.source_url))
            .filter(Boolean);
          if (mosaicUrls.length >= 4) {
            blob = (await renderMosaicPngBlob(mosaicUrls, { seed })).blob;
          } else {
            blob = await renderAugmentedPngBlob(previewUrl, { ...cfg, mosaic: false }, { mixupUrl: mixupImageUrl, seed });
          }
        } else {
          blob = await renderAugmentedPngBlob(previewUrl, cfg, { mixupUrl: mixupImageUrl, seed });
        }
        urls.push(URL.createObjectURL(blob));
      }
      setPreviewSamples(urls);
    } catch (error) {
      console.error("Generate preview samples failed:", error);
      urls.forEach((u) => URL.revokeObjectURL(u));
      setPreviewSamples([]);
    }
  }, [activeTab, configForPreview, images, mixupImageUrl, previewUrl]);

  // Debounce preview generation so sliders don't freeze the UI.
  useEffect(() => {
    const t = window.setTimeout(() => {
      void generateSamplePreviews();
    }, 400);
    return () => window.clearTimeout(t);
  }, [generateSamplePreviews]);

  // Revoke old object URLs to avoid memory leaks.
  useEffect(() => {
    return () => {
      previewSamples.forEach((u) => URL.revokeObjectURL(u));
    };
  }, [previewSamples]);

  // Generate CSS transform/filter based on config
  const getImageStyle = () => {
    if (!MANUAL_AUGMENTATION_ENABLED || showOriginal) return {};

    const transforms = [
        config.rotate ? `rotate(${config.rotateDeg}deg)` : '',
        config.flipH ? 'scaleX(-1)' : '',
        config.flipV ? 'scaleY(-1)' : '',
        config.shear ? `skew(${config.shear}deg)` : '',
        config.translate ? `translate(${config.translate * 100}px, ${config.translate * 100}px)` : ''
    ].filter(Boolean).join(' ');

    const filters = [
        `hue-rotate(${config.hue}deg)`,
        `saturate(${config.saturation}%)`,
        `brightness(${config.brightness}%)`,
        `contrast(${config.contrast}%)`,
        `blur(${config.blur}px)`
    ].join(' ');

    return {
        transform: transforms,
        filter: filters,
        transition: 'all 0.3s ease'
    };
  };

  const openSnapshotDialog = useCallback(() => {
    if (!projectId || !datasetId) {
      toast({ title: "请先选择项目与数据集", variant: "destructive" });
      return;
    }
    if (!activeDataset) {
      toast({ title: "数据集信息未就绪", description: "请稍后重试", variant: "destructive" });
      return;
    }
    if (images.length === 0) {
      toast({ title: "暂无图片可用于快照", description: "请先导入图片", variant: "destructive" });
      return;
    }
    setSnapshotName(activeDataset.name);
    setSnapshotVersion("");
    setSnapshotDescription(activeDataset.description || "");
    setIsSnapshotOpen(true);
  }, [activeDataset, datasetId, images.length, projectId]);

  const applyAugmentationPipeline = useCallback(async () => {
    if (!projectId || !datasetId) {
      toast({ title: "请先选择项目与数据集", variant: "destructive" });
      return;
    }
    if (!activeDataset) {
      toast({ title: "数据集信息未就绪", description: "请稍后重试", variant: "destructive" });
      return;
    }

    const name = (snapshotName || "").trim() || activeDataset.name;
    const version = (snapshotVersion || "").trim() || null;
    const description = snapshotDescription || "";
    if (!name) {
      toast({ title: "快照名称不能为空", variant: "destructive" });
      return;
    }

    const batchSize = 10;
    const variants = augmentVariants;
    const now = Date.now();

    setIsApplying(true);
    setIsSnapshotOpen(false);
    try {
      const sources = await imageService.list(projectId, { dataset_id: datasetId });
      if (sources.length === 0) {
        toast({ title: "暂无图片可用于快照", variant: "destructive" });
        return;
      }

      let runnableVariants = variants;
      if (sources.length < 2) runnableVariants = runnableVariants.filter((v) => v.key !== "mixup");
      if (sources.length < 4) runnableVariants = runnableVariants.filter((v) => v.key !== "mosaic");

      const created = await datasetService.clone(datasetId, {
        name,
        version,
        description,
        tags: activeDataset.tags || null,
        is_public: false,
        splits: activeDataset.splits,
      });

      toast({
        title: "已创建数据集快照",
        description: MANUAL_AUGMENTATION_ENABLED
          ? `${created.name} ${created.version}（正在生成增强样本…）`
          : `${created.name} ${created.version}`,
      });

      const nextDatasets = await datasetService.list({ project_id: projectId, limit: 200 });
      setDatasets(nextDatasets);
      setDataset(created.id, `${created.name} ${created.version}`);

      if (!MANUAL_AUGMENTATION_ENABLED) {
        const [finalImages, stats] = await Promise.all([
          imageService.list(projectId, { dataset_id: created.id }),
          datasetService.getImageStats(created.id),
        ]);
        setImages(finalImages);
        setDatasetStats(stats);
        toast({ title: "快照生成完成", description: `已生成快照（共 ${finalImages.length} 张图片）` });
        return;
      }

      type Task = { filename: string; sources: ImageRecord[]; cfg: AugmentConfig; mosaic?: MosaicMeta; fileId?: string };
      const tasks = new Map<string, Task>();
      const pending: File[] = [];

      const flush = async () => {
        if (pending.length === 0) return;
        const uploading = pending.splice(0, pending.length);
        const uploaded = await datasetService.uploadFiles(created.id, uploading);
        for (const rec of uploaded) {
          const t = tasks.get(rec.filename);
          if (t) t.fileId = rec.id;
        }
      };

      for (let i = 0; i < sources.length; i++) {
        const src = sources[i];
        const srcUrl = resolveImageUrl(src.source_url);
        if (!srcUrl) continue;

        for (let j = 0; j < runnableVariants.length; j++) {
          const variant = runnableVariants[j];
          const seed = now + i * 10007 + j * 97;
          const cfg = variant.cfg;

          console.log("[Augmentation] snapshot-apply", {
            projectId,
            source: src.filename,
            variant: variant.key,
            cfg,
          });

          let blob: Blob;
          let taskSources: ImageRecord[] = [src];
          let mosaic: MosaicMeta | undefined;

          if (cfg.mosaic) {
            const mosaicSources: ImageRecord[] = [];
            const mosaicUrls: string[] = [];
            for (let k = 0; k < sources.length && mosaicSources.length < 4; k++) {
              const cand = sources[(i + k) % sources.length]!;
              if (mosaicSources.some((s) => s.id === cand.id)) continue;
              const u = resolveImageUrl(cand.source_url);
              if (!u) continue;
              mosaicSources.push(cand);
              mosaicUrls.push(u);
            }
            if (mosaicUrls.length < 4) continue;
            const rendered = await renderMosaicPngBlob(mosaicUrls, { seed });
            blob = rendered.blob;
            mosaic = rendered.meta;
            taskSources = mosaicSources;
          } else {
            const mixupPartner = cfg.mixup && sources.length > 1 ? sources[(i + 1) % sources.length] : null;
            const mixupUrl = cfg.mixup ? resolveImageUrl(mixupPartner?.source_url) : undefined;
            blob = await renderAugmentedPngBlob(srcUrl, cfg, { mixupUrl, seed });
            taskSources = mixupPartner ? [src, mixupPartner] : [src];
          }
          const filename = `${basenameNoExt(src.filename)}_aug_${variant.key}_${now}_${i}_${j}.png`;
          pending.push(new File([blob], filename, { type: "image/png" }));
          tasks.set(filename, { filename, sources: taskSources, cfg, mosaic });

          if (pending.length >= batchSize) {
            await flush();
          }
        }
      }
      await flush();

      const snapshotImages = await imageService.list(projectId, { dataset_id: created.id });
      const snapshotById = new Map(snapshotImages.map((img) => [img.id, img]));
      const targetIdByFileId = new Map<string, string>();
      for (const img of snapshotImages) {
        if (img.dataset_file_id) targetIdByFileId.set(img.dataset_file_id, img.id);
      }

      const annCache = new Map<string, AnnotationRecord[]>();
      const getSourceAnnotations = async (imageId: string) => {
        const cached = annCache.get(imageId);
        if (cached) return cached;
        const anns = await annotationService.listByImage(imageId);
        annCache.set(imageId, anns);
        return anns;
      };

      const copyJobs: Array<{ targetImageId: string; sources: ImageRecord[]; cfg: AugmentConfig; mosaic?: MosaicMeta }> = [];
      for (const t of tasks.values()) {
        if (!t.fileId) continue;
        const targetImageId = targetIdByFileId.get(t.fileId);
        if (!targetImageId) continue;
        copyJobs.push({ targetImageId, sources: t.sources, cfg: t.cfg, mosaic: t.mosaic });
      }

      let idx = 0;
      const concurrency = 4;
      const worker = async () => {
        while (idx < copyJobs.length) {
          const job = copyJobs[idx++];
          const target = snapshotById.get(job.targetImageId) ?? null;
          const fallback = job.sources[0] ?? null;
          const imgW = Math.max(1, target?.width ?? fallback?.width ?? 1);
          const imgH = Math.max(1, target?.height ?? fallback?.height ?? 1);

          let payload: AnnotationCreatePayload[] = [];
          if (job.cfg.mosaic && job.mosaic && job.sources.length >= 4 && job.mosaic.placements.length >= 4) {
            for (let sIdx = 0; sIdx < 4; sIdx++) {
              const src = job.sources[sIdx];
              const placement = job.mosaic.placements[sIdx];
              if (!src || !placement) continue;
              const anns = await getSourceAnnotations(src.id);
              for (const ann of anns) {
                const out = transformAnnotationMosaic(ann, placement, imgW, imgH);
                if (out) payload.push(out);
              }
            }
          } else {
            const merged: AnnotationRecord[] = [];
            for (const src of job.sources) {
              merged.push(...(await getSourceAnnotations(src.id)));
            }
            payload = merged
              .map((a) => transformAnnotation(a, job.cfg, imgW, imgH))
              .filter(Boolean) as AnnotationCreatePayload[];
          }
          await annotationService.replaceByImage(job.targetImageId, payload);
        }
      };
      await Promise.all(Array.from({ length: concurrency }, () => worker()));

      const [finalImages, stats] = await Promise.all([
        imageService.list(projectId, { dataset_id: created.id }),
        datasetService.getImageStats(created.id),
      ]);
      setImages(finalImages);
      setDatasetStats(stats);

      toast({
        title: "快照生成完成",
        description: `已生成 ${sources.length * runnableVariants.length} 张增强样本（总计 ${sources.length * (1 + runnableVariants.length)} 张）`,
      });
    } catch (error) {
      console.error("Apply snapshot failed:", error);
      toast({ title: "保存快照失败", description: "请检查后端服务或控制台日志", variant: "destructive" });
    } finally {
      setIsApplying(false);
    }
  }, [
    activeDataset,
    augmentVariants,
    datasetId,
    projectId,
    setDataset,
    snapshotDescription,
    snapshotName,
    snapshotVersion,
  ]);

  return (
    <div className="space-y-6 h-[calc(100vh-6rem)] flex flex-col relative">
      {isSnapshotOpen && (
        <div className="absolute inset-0 z-50 bg-background/80 backdrop-blur-sm flex items-center justify-center p-6">
          <Card className="w-full max-w-lg shadow-2xl">
            <CardHeader className="border-b">
              <CardTitle>保存为数据集快照</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 pt-6">
              <div className="space-y-2">
                <div className="text-sm font-medium">名称</div>
                <Input value={snapshotName} onChange={(e) => setSnapshotName(e.target.value)} placeholder="例如：train / aug" />
              </div>
              <div className="space-y-2">
                <div className="text-sm font-medium">版本（可留空自动生成 v1/v2...）</div>
                <Input value={snapshotVersion} onChange={(e) => setSnapshotVersion(e.target.value)} placeholder="例如：v2" />
              </div>
              <div className="space-y-2">
                <div className="text-sm font-medium">描述（可选）</div>
                <textarea
                  className="w-full min-h-[96px] rounded-md border border-input bg-background px-3 py-2 text-sm"
                  value={snapshotDescription}
                  onChange={(e) => setSnapshotDescription(e.target.value)}
                  placeholder="记录本次增强目的与参数说明"
                />
              </div>
              <div className="text-xs text-muted-foreground">
                保存后将生成一个新的数据集版本，并自动继承原始标注信息。
              </div>
            </CardContent>
            <CardFooter className="justify-end gap-2 border-t py-4">
              <Button variant="outline" onClick={() => setIsSnapshotOpen(false)} disabled={isApplying}>
                取消
              </Button>
              <Button onClick={() => void applyAugmentationPipeline()} disabled={isApplying}>
                {isApplying ? "生成中..." : "生成快照"}
              </Button>
            </CardFooter>
          </Card>
        </div>
      )}
      {/* Guide Overlay */}
      {showGuide && (
        <div className="absolute inset-0 z-50 bg-background/80 backdrop-blur-sm flex items-center justify-center p-12">
            <Card className="w-full max-w-4xl shadow-2xl h-[80vh] flex flex-col">
                <CardHeader className="border-b">
                    <div className="flex justify-between items-center">
                        <CardTitle>数据集快照使用说明</CardTitle>
                        <Button variant="ghost" onClick={() => setShowGuide(false)}>×</Button>
                    </div>
                </CardHeader>
                <CardContent className="flex-1 overflow-auto p-8">
                    <div className="space-y-12">
                        <div className="flex items-start gap-6">
                            <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center shrink-0 text-primary font-bold text-xl">1</div>
                            <div className="space-y-4 flex-1">
                                <h3 className="text-xl font-bold">选择数据集 (Select Dataset)</h3>
                                <p className="text-muted-foreground">选择需要保存为快照的数据集，确保已导入图片并完成必要的标注。</p>
                                <div className="bg-muted p-4 rounded-lg border border-dashed flex items-center justify-center h-32">
                                    <div className="text-center text-muted-foreground">
                                        <ImageIcon className="w-8 h-8 mx-auto mb-2 opacity-50"/>
                                        数据集选择示意图
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div className="flex items-center justify-center">
                            <ArrowRight className="w-8 h-8 text-muted-foreground rotate-90 md:rotate-0" />
                        </div>

                        <div className="flex items-start gap-6">
                            <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center shrink-0 text-primary font-bold text-xl">2</div>
                            <div className="space-y-4 flex-1">
                                <h3 className="text-xl font-bold">保存快照 (Snapshot)</h3>
                                <p className="text-muted-foreground">点击“保存为快照”生成新的数据集版本。本平台训练阶段会自动进行数据增广，因此此处仅保留快照/版本固化能力。</p>
                                <div className="bg-muted p-4 rounded-lg border border-dashed flex items-center justify-center h-32">
                                    <div className="text-center text-muted-foreground">
                                        <Save className="w-8 h-8 mx-auto mb-2 opacity-50"/>
                                        快照生成示意图
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="flex items-center justify-center">
                            <ArrowRight className="w-8 h-8 text-muted-foreground rotate-90 md:rotate-0" />
                        </div>

                        <div className="flex items-start gap-6">
                            <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center shrink-0 text-primary font-bold text-xl">3</div>
                            <div className="space-y-4 flex-1">
                                <h3 className="text-xl font-bold">训练与部署 (Train/Deploy)</h3>
                                <p className="text-muted-foreground">使用生成的快照数据集进行模型训练与部署推荐，必要时再导出训练格式。</p>
                                <div className="bg-muted p-4 rounded-lg border border-dashed flex items-center justify-center h-32">
                                    <div className="text-center text-muted-foreground">
                                        <Download className="w-8 h-8 mx-auto mb-2 opacity-50"/>
                                        训练/部署示意图
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </CardContent>
            </Card>
        </div>
      )}

      {isDatasetManagerOpen && (
        <div className="absolute inset-0 z-50 bg-background/80 backdrop-blur-sm flex items-center justify-center p-6">
          <Card className="w-full max-w-3xl shadow-2xl">
            <CardHeader className="border-b">
              <div className="flex justify-between items-center">
                <CardTitle>数据集管理</CardTitle>
                <Button
                  variant="ghost"
                  onClick={() => {
                    setIsDatasetManagerOpen(false);
                    cancelEditDataset();
                  }}
                >
                  ×
                </Button>
              </div>
            </CardHeader>
            <CardContent className="p-6 max-h-[70vh] overflow-auto space-y-4">
              <div className="text-xs text-muted-foreground">
                支持数据集版本的重命名与删除（需要 API Key）。删除后会同步清理该版本下的图片与文件。
              </div>

              {datasets.length === 0 ? (
                <div className="text-sm text-muted-foreground">暂无数据集版本</div>
              ) : (
                <div className="space-y-2">
                  {datasets.map((d) => {
                    const isActive = d.id === datasetId;
                    const isEditing = d.id === editingDatasetId;
                    return (
                      <div key={d.id} className="flex items-center gap-2 p-3 rounded-lg border">
                        <div className="min-w-0 flex-1">
                          {isEditing ? (
                            <div className="flex flex-wrap items-center gap-2">
                              <Input
                                className="h-8 min-w-[200px]"
                                value={editDatasetName}
                                onChange={(e) => setEditDatasetName(e.target.value)}
                                placeholder="数据集名称"
                              />
                              <Input
                                className="h-8 w-28"
                                value={editDatasetVersion}
                                onChange={(e) => setEditDatasetVersion(e.target.value)}
                                placeholder="版本"
                              />
                              <Button size="sm" onClick={() => void saveEditDataset()}>
                                保存
                              </Button>
                              <Button variant="outline" size="sm" onClick={cancelEditDataset}>
                                取消
                              </Button>
                            </div>
                          ) : (
                            <div className="flex items-center gap-2 flex-wrap">
                              <span className="font-medium truncate" title={`${d.name} ${d.version}`}>
                                {d.name} {d.version}
                              </span>
                              {isActive ? <span className="text-xs text-primary">当前</span> : null}
                              <span className="text-xs text-muted-foreground">{d.status}</span>
                            </div>
                          )}
                        </div>

                        {!isEditing ? (
                          <div className="flex items-center gap-1">
                            <Button variant="ghost" size="icon" title="编辑" onClick={() => beginEditDataset(d)}>
                              <Pencil className="w-4 h-4" />
                            </Button>
                            <Button variant="ghost" size="icon" title="删除" onClick={() => void deleteDataset(d)}>
                              <Trash2 className="w-4 h-4 text-destructive" />
                            </Button>
                          </div>
                        ) : null}
                      </div>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between shrink-0">
        <div>
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3">
              数据集快照
              <Badge variant="secondary" className="text-xs font-medium">
                手动增强已禁用
              </Badge>
              <Button variant="ghost" size="sm" className="text-muted-foreground" onClick={() => setShowGuide(true)}>
                   <CheckCircle2 className="w-4 h-4 mr-1"/> 查看说明
               </Button>
           </h1>
            <div className="flex items-center gap-2 mt-2 text-sm text-muted-foreground">
              <span className="flex items-center gap-1"><ImageIcon className="w-3 h-3"/> 选择数据集</span>
              <ArrowRight className="w-3 h-3" />
              <span className="flex items-center gap-1 font-medium text-primary"><Save className="w-3 h-3"/> 生成快照</span>
              <ArrowRight className="w-3 h-3" />
              <span className="flex items-center gap-1"><Download className="w-3 h-3"/> 用于训练/部署</span>
            </div>
            <div className="flex flex-wrap items-center gap-2 mt-3">
              <select
               className="h-8 rounded-md border border-input bg-background px-2 py-1 text-sm"
               value={projectId ?? ""}
               disabled={isLoadingProjects || projects.length === 0}
               onChange={(e) => {
                 const nextId = (e.target.value || "").trim();
                 if (!nextId) return;
                 const found = projects.find((p) => p.id === nextId) ?? null;
                 setProject(nextId, found?.name ?? null);
                 clearDataset();
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
                value={datasetId ?? ""}
                disabled={!projectId || isLoadingDatasets || datasets.length === 0}
                onChange={(e) => {
                 if (!projectId) return;
                 const nextId = (e.target.value || "").trim();
                 if (!nextId) return;
                 const found = datasets.find((d) => d.id === nextId) ?? null;
                 setDataset(nextId, found ? `${found.name} ${found.version}` : null);
               }}
             >
               <option value="" disabled>
                 {!projectId ? "请先选择项目" : isLoadingDatasets ? "加载数据集中..." : "请选择数据集"}
               </option>
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} {d.version}
                  </option>
                ))}
              </select>
              <Button variant="outline" size="sm" onClick={openDatasetManager} disabled={!projectId}>
                <Layers className="w-4 h-4 mr-2" /> 数据集管理
              </Button>
              <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground min-w-0">
                <span className="truncate max-w-[260px]">
                  {isLoadingImages ? "加载预览图..." : previewImage ? `预览：${previewImage.filename}` : "暂无预览图（请先导入图片）"}
                </span>
                {datasetId ? (
                  <span className="whitespace-nowrap">
                    图片：{baseImageCount}
                    {MANUAL_AUGMENTATION_ENABLED ? ` · 增强：${augmentVariants.length} · 预计：${predictedImageCount}` : ""}
                  </span>
                ) : null}
              </div>
            </div>
          </div>
        <div className="flex gap-2">
             {MANUAL_AUGMENTATION_ENABLED ? (
               <Button variant="outline" onClick={handleReset}>
                   <RotateCcw className="w-4 h-4 mr-2" /> 重置
               </Button>
             ) : null}
               <Button onClick={openSnapshotDialog} disabled={isApplying || !datasetId || !projectId || images.length === 0}>
                   <Save className="w-4 h-4 mr-2" /> {isApplying ? "生成中..." : "保存为快照"}
               </Button>
           </div>
         </div>

      <div className="grid grid-cols-12 gap-6 flex-1 min-h-0">
        <div className="col-span-12 xl:col-span-3 flex flex-col gap-4 min-h-0">
          {/* Navigation */}
          {MANUAL_AUGMENTATION_ENABLED ? (
            <Card>
              <CardContent className="p-2 space-y-1">
                {[
                  { id: "geometric", label: "几何变换", icon: RotateCw },
                  { id: "color", label: "色彩调整", icon: Sun },
                  { id: "noise", label: "噪声注入", icon: Zap },
                  { id: "weather", label: "环境模拟", icon: CloudRain },
                  { id: "advanced", label: "高级增强", icon: Layers },
                ].map((item) => (
                  <Button
                    key={item.id}
                    variant={activeTab === item.id ? "secondary" : "ghost"}
                    className={`w-full justify-start gap-3 ${activeTab === item.id ? "bg-secondary font-medium" : ""}`}
                    onClick={() => setActiveTab(item.id)}
                  >
                    <item.icon className={`w-4 h-4 ${activeTab === item.id ? "text-primary" : "text-muted-foreground"}`} />
                    {item.label}
                  </Button>
                ))}
              </CardContent>
            </Card>
          ) : (
            <Card className="border-dashed">
              <CardHeader className="pb-3">
                <CardTitle className="text-base">本模块仅保留快照功能</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm text-muted-foreground">
                <div>手动数据增强已关闭：YOLO 训练阶段会自动进行数据增广。</div>
                <div>你仍可将当前数据集保存为新版本快照，用于训练/部署前的版本固化。</div>
              </CardContent>
            </Card>
          )}

          {/* Configuration Form */}
          <Card className={`flex flex-col min-h-0 flex-1 overflow-hidden ${MANUAL_AUGMENTATION_ENABLED ? "" : "hidden"}`}>
          <CardHeader className="border-b bg-muted/20 py-3">
            <CardTitle className="text-lg flex items-center gap-2">
                {activeTab === "geometric" && <RotateCw className="w-5 h-5" />}
                {activeTab === "color" && <Sun className="w-5 h-5" />}
                {activeTab === "noise" && <Zap className="w-5 h-5" />}
                {activeTab === "weather" && <CloudRain className="w-5 h-5" />}
                {activeTab === "advanced" && <Layers className="w-5 h-5" />}
                {activeTab === "geometric" && "几何变换参数"}
                {activeTab === "color" && "色彩调整参数"}
                {activeTab === "noise" && "噪声与模糊参数"}
                {activeTab === "weather" && "环境模拟参数"}
                {activeTab === "advanced" && "高级增强参数"}
            </CardTitle>
          </CardHeader>
          <CardContent className="p-4 overflow-y-auto flex-1 space-y-6">
             {activeTab === "geometric" && (
                 <div className="space-y-6">
                     <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <label className="font-medium">水平翻转 (Flip H)</label>
                            <input type="checkbox" className="w-4 h-4" checked={config.flipH} onChange={e => setConfig({...config, flipH: e.target.checked})} />
                        </div>
                        <div className="flex items-center justify-between">
                            <label className="font-medium">垂直翻转 (Flip V)</label>
                            <input type="checkbox" className="w-4 h-4" checked={config.flipV} onChange={e => setConfig({...config, flipV: e.target.checked})} />
                        </div>
                     </div>
                     <div className="space-y-3">
                        <div className="flex justify-between">
                            <label className="font-medium">旋转角度 (Rotate)</label>
                            <span className="text-sm text-muted-foreground">{config.rotateDeg}°</span>
                        </div>
                        <Input type="range" min="0" max="180" value={config.rotateDeg} onChange={e => setConfig({...config, rotateDeg: Number(e.target.value)})} />
                     </div>
                     <div className="space-y-3">
                        <div className="flex justify-between">
                            <label className="font-medium">错切变换 (Shear)</label>
                            <span className="text-sm text-muted-foreground">{config.shear}°</span>
                        </div>
                        <Input type="range" min="-45" max="45" value={config.shear} onChange={e => setConfig({...config, shear: Number(e.target.value)})} />
                     </div>
                 </div>
             )}

             {activeTab === "color" && (
                 <div className="space-y-6">
                     <div className="space-y-3">
                        <div className="flex justify-between">
                            <label className="font-medium">色相偏移 (Hue)</label>
                            <span className="text-sm text-muted-foreground">{config.hue}</span>
                        </div>
                        <Input type="range" min="-180" max="180" value={config.hue} onChange={e => setConfig({...config, hue: Number(e.target.value)})} />
                     </div>
                     <div className="space-y-3">
                        <div className="flex justify-between">
                            <label className="font-medium">饱和度 (Saturation)</label>
                            <span className="text-sm text-muted-foreground">{config.saturation}%</span>
                        </div>
                        <Input type="range" min="0" max="200" value={config.saturation} onChange={e => setConfig({...config, saturation: Number(e.target.value)})} />
                     </div>
                     <div className="space-y-3">
                        <div className="flex justify-between">
                            <label className="font-medium">亮度 (Brightness)</label>
                            <span className="text-sm text-muted-foreground">{config.brightness}%</span>
                        </div>
                        <Input type="range" min="0" max="200" value={config.brightness} onChange={e => setConfig({...config, brightness: Number(e.target.value)})} />
                     </div>
                     <div className="space-y-3">
                        <div className="flex justify-between">
                            <label className="font-medium">对比度 (Contrast)</label>
                            <span className="text-sm text-muted-foreground">{config.contrast}%</span>
                        </div>
                        <Input type="range" min="0" max="200" value={config.contrast} onChange={e => setConfig({...config, contrast: Number(e.target.value)})} />
                     </div>
                 </div>
             )}

             {activeTab === "noise" && (
                 <div className="space-y-6">
                     <div className="space-y-3">
                        <div className="flex justify-between">
                            <label className="font-medium">高斯模糊 (Blur)</label>
                            <span className="text-sm text-muted-foreground">{config.blur}px</span>
                        </div>
                        <Input type="range" min="0" max="10" step="0.5" value={config.blur} onChange={e => setConfig({...config, blur: Number(e.target.value)})} />
                     </div>
                     <div className="space-y-4 pt-4 border-t">
                        <div className="flex items-center justify-between">
                            <label className="font-medium">噪点注入 (Noise)</label>
                            <input type="checkbox" className="w-4 h-4" checked={config.noise} onChange={e => setConfig({...config, noise: e.target.checked})} />
                        </div>
                        {config.noise && (
                             <div className="space-y-3">
                                <div className="flex justify-between">
                                    <label className="text-sm text-muted-foreground">强度</label>
                                    <span className="text-sm text-muted-foreground">{config.noiseAmount}%</span>
                                </div>
                                <Input type="range" min="0" max="100" value={config.noiseAmount} onChange={e => setConfig({...config, noiseAmount: Number(e.target.value)})} />
                             </div>
                        )}
                     </div>
                 </div>
             )}

            {activeTab === "weather" && (
                  <div className="grid grid-cols-2 gap-4">
                      {WEATHER_OPTIONS.map((w) => (
                          <div 
                             key={w}
                             className={`p-4 border rounded-lg cursor-pointer flex flex-col items-center gap-2 transition-all ${config.weather === w ? 'border-primary bg-primary/5' : 'hover:bg-muted'}`}
                             onClick={() => setConfig({...config, weather: w})}
                          >
                             {w === 'none' && <Sun className="w-8 h-8 text-orange-500" />}
                             {w === 'rain' && <CloudRain className="w-8 h-8 text-blue-500" />}
                             {w === 'snow' && <CloudRain className="w-8 h-8 text-slate-300" />}
                             {w === 'fog' && <CloudRain className="w-8 h-8 text-gray-400" />}
                             <span className="capitalize font-medium">{w === 'none' ? 'None' : w}</span>
                         </div>
                     ))}
                 </div>
             )}

             {activeTab === "advanced" && (
                <div className="space-y-6">
                    <div className="flex items-start gap-4 p-4 bg-muted/30 rounded-lg">
                        <div className="p-2 bg-background rounded border shadow-sm">
                            <Layers className="w-6 h-6 text-purple-600" />
                        </div>
                        <div className="flex-1 space-y-2">
                            <div className="flex items-center justify-between">
                                <h3 className="font-medium">Mosaic 马赛克增强</h3>
                                <input type="checkbox" className="w-4 h-4" checked={config.mosaic} onChange={e => setConfig({...config, mosaic: e.target.checked})} />
                            </div>
                            <p className="text-sm text-muted-foreground">将 4 张训练图片随机拼接在一起。</p>
                        </div>
                    </div>

                    <div className="flex items-start gap-4 p-4 bg-muted/30 rounded-lg">
                        <div className="p-2 bg-background rounded border shadow-sm">
                            <Layers className="w-6 h-6 text-indigo-600" />
                        </div>
                        <div className="flex-1 space-y-2">
                            <div className="flex items-center justify-between">
                                <h3 className="font-medium">Mixup 数据混合</h3>
                                <input type="checkbox" className="w-4 h-4" checked={config.mixup} onChange={e => setConfig({...config, mixup: e.target.checked})} />
                            </div>
                            <p className="text-sm text-muted-foreground">将两张图片按比例叠加。</p>
                        </div>
                    </div>

                    <div className="flex items-start gap-4 p-4 bg-muted/30 rounded-lg">
                        <div className="p-2 bg-background rounded border shadow-sm">
                            <EyeOff className="w-6 h-6 text-slate-600" />
                        </div>
                        <div className="flex-1 space-y-2">
                            <div className="flex items-center justify-between">
                                <h3 className="font-medium">Cutout 随机遮挡</h3>
                                <input type="checkbox" className="w-4 h-4" checked={config.cutout} onChange={e => setConfig({...config, cutout: e.target.checked})} />
                            </div>
                            <p className="text-sm text-muted-foreground">随机生成黑色遮挡块。</p>
                        </div>
                    </div>
                </div>
             )}
          </CardContent>
        </Card>

        {/* Preview Panel */}
        <Card className="flex flex-col bg-muted/10 border-2 border-muted">
            <CardHeader className="border-b py-3 bg-background">
                <div className="flex items-center justify-between">
                    <CardTitle className="text-sm font-medium">实时预览 (Preview)</CardTitle>
                    <div className="flex items-center gap-3">
                        {MANUAL_AUGMENTATION_ENABLED ? (
                          <label className="flex items-center gap-1 text-xs text-muted-foreground select-none">
                              <input
                                  type="checkbox"
                                  className="w-4 h-4"
                                  defaultChecked
                                  disabled
                              />
                              应用到全部
                          </label>
                        ) : null}
                        <Button 
                            variant="outline" 
                            size="sm" 
                            className={`${showOriginal ? 'bg-primary text-primary-foreground' : ''}`}
                            onMouseDown={() => setShowOriginal(true)}
                            onMouseUp={() => setShowOriginal(false)}
                            onMouseLeave={() => setShowOriginal(false)}
                        >
                            <EyeOff className="w-3 h-3 mr-1" /> 原图对比
                        </Button>
                    </div>
                </div>
            </CardHeader>
             <CardContent className="flex-1 p-4 flex flex-col items-center justify-center gap-3 relative overflow-hidden bg-grid-pattern">
                 <div className="relative w-full max-w-[420px] aspect-square shadow-lg rounded-lg overflow-hidden bg-black">
                     {/* Main Image with CSS filters/transforms (default: first imported image) */}
                     {previewUrl ? (
                         <img
                             src={previewUrl}
                            alt="Preview"
                            className="w-full h-full object-contain origin-center"
                            style={getImageStyle()}
                        />
                    ) : (
                        <div className="absolute inset-0 flex items-center justify-center text-white/70 text-sm px-6 text-center">
                            暂无预览图，请先在项目管理中导入图片
                        </div>
                    )}
                    
                    {/* Noise Overlay */}
                    {!showOriginal && config.noise && (
                        <div className="absolute inset-0 pointer-events-none mix-blend-overlay" style={{
                            backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' opacity='${config.noiseAmount / 100}'/%3E%3C/svg%3E")`,
                        }} />
                    )}

                    {/* Weather Overlay - Simple Gradient Simulators */}
                    {!showOriginal && config.weather === 'rain' && (
                        <div className="absolute inset-0 pointer-events-none bg-blue-900/20" style={{
                            backgroundImage: 'linear-gradient(to bottom, transparent 95%, rgba(255,255,255,0.2) 100%)',
                            backgroundSize: '20px 20px',
                            transform: 'skewX(-10deg)'
                        }} />
                    )}
                    {!showOriginal && config.weather === 'fog' && (
                         <div className="absolute inset-0 pointer-events-none bg-white/40" />
                    )}

                    {/* Mixup Overlay */}
                    {!showOriginal && config.mixup && mixupImageUrl && (
                        <div className="absolute inset-0 pointer-events-none opacity-50 mix-blend-normal">
                              <img 
                                src={mixupImageUrl}
                                className="w-full h-full object-contain"
                                alt="Mixup"
                              />
                        </div>
                    )}

                    {/* Cutout Overlay */}
                     {!showOriginal && config.cutout && (
                         <div className="absolute w-1/4 h-1/4 bg-black top-1/4 left-1/4 pointer-events-none" />
                     )}
                 </div>

                 {/* Original Label Indicator */}
                 {showOriginal && (
                     <div className="absolute top-4 left-4 bg-black/70 text-white px-2 py-1 text-xs rounded">
                         Original
                     </div>
                 )}

                 {/* Sample previews (5 per method) */}
                 {previewSamples.length > 0 && (
                   <div className="w-full max-w-[420px] grid grid-cols-5 gap-2">
                     {previewSamples.map((u, i) => (
                       <div key={u} className="aspect-square rounded bg-black/90 overflow-hidden border border-white/10">
                         <img src={u} alt={`sample-${i + 1}`} className="w-full h-full object-contain" />
                       </div>
                     ))}
                   </div>
                 )}
             </CardContent>
             <CardFooter className="bg-background border-t p-3 text-xs text-muted-foreground flex justify-between">
                 <span>
                     {previewImage?.width && previewImage?.height
                         ? `Resolution: ${previewImage.width}x${previewImage.height}`
                         : "Resolution: -"}{" "}
                     · 预览区域 1:1
                     {datasetStats && (
                       <>
                         {" "}
                         · Images: {datasetStats.image_count} · Avg: {datasetStats.avg_width ?? "-"}×{datasetStats.avg_height ?? "-"}
                       </>
                     )}
                 </span>
                 <span>应用范围: 全部图片（快照）</span>
             </CardFooter>
         </Card>
        </div>

        {/* Gallery Panel */}
        <Card className="col-span-12 xl:col-span-9 flex flex-col min-h-0">
          <CardHeader className="border-b py-3">
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <CardTitle className="text-sm font-medium">数据集图像与标注预览</CardTitle>
                <div className="text-xs text-muted-foreground mt-1 flex flex-wrap items-center gap-x-2 gap-y-1">
                  <span className="truncate max-w-[420px]">
                    {activeDataset ? `${activeDataset.name} ${activeDataset.version}` : datasetId ? "数据集加载中..." : "请先选择数据集"}
                  </span>
                  <span className="whitespace-nowrap">共 {images.length} 张</span>
                  {datasetStats ? (
                    <span className="whitespace-nowrap">
                      · 标注统计：Images {datasetStats.image_count}
                    </span>
                  ) : null}
                  {legacyAugmentedCount > 0 ? (
                    <Badge
                      variant="outline"
                      className="border-amber-500/40 text-amber-700"
                      title="检测到旧版本生成的增强样本（可能出现明显下采样/锯齿）。如需应用最新增强算法，请重新“保存为快照”生成新样本。"
                    >
                      旧版增强样本 {legacyAugmentedCount}
                    </Badge>
                  ) : null}
                </div>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <Button
                  variant="outline"
                  size="sm"
                  disabled={galleryPageSafe <= 1}
                  onClick={() => setGalleryPage((p) => Math.max(1, p - 1))}
                >
                  <ArrowRight className="w-4 h-4 rotate-180" />
                </Button>
                <div className="text-xs tabular-nums text-muted-foreground min-w-[72px] text-center">
                  {galleryPageSafe} / {galleryTotalPages}
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={galleryPageSafe >= galleryTotalPages}
                  onClick={() => setGalleryPage((p) => Math.min(galleryTotalPages, p + 1))}
                >
                  <ArrowRight className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="flex-1 min-h-0 overflow-y-auto p-4">
            {!datasetId || images.length === 0 ? (
              <div className="h-full flex items-center justify-center text-sm text-muted-foreground">
                暂无可展示的图片，请先在项目管理中导入图片并保存快照版本。
              </div>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6 gap-3">
                {galleryItems.map((img) => {
                  const url = resolveImageUrl(img.source_url);
                  const imgW = Math.max(1, img.width ?? 1);
                  const imgH = Math.max(1, img.height ?? 1);
                  const summary = img.id ? annotationSummaries[img.id] : undefined;
                  const labels = summary?.labels ?? [];
                  const boxes = summary?.boxes ?? [];

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
                            {boxes.slice(0, 50).map((b, idx) => (
                              <rect
                                key={`${img.id}-b-${idx}`}
                                x={b.x}
                                y={b.y}
                                width={b.width}
                                height={b.height}
                                fill="none"
                                stroke={b.color || "#ef4444"}
                                strokeWidth={2}
                                strokeOpacity={0.9}
                                vectorEffect="non-scaling-stroke"
                              />
                            ))}
                          </svg>
                        ) : (
                          <div className="absolute inset-0 flex items-center justify-center text-xs text-white/70 bg-black">
                            无预览
                          </div>
                        )}

                        <div className="absolute top-2 left-2 right-2 flex items-center justify-between gap-2">
                          <div className="truncate text-[11px] text-white/90 bg-black/60 px-2 py-1 rounded backdrop-blur-sm">
                            {img.filename}
                          </div>
                          <div className="text-[11px] text-white/80 bg-black/50 px-2 py-1 rounded tabular-nums">
                            {img.id ? "标注" : "-"}
                          </div>
                        </div>

                        <div className="absolute inset-x-0 bottom-0 p-2 bg-gradient-to-t from-black/70 via-black/10 to-transparent">
                          <div className="flex flex-wrap gap-1 items-center">
                            {labels.length === 0 ? (
                              <span className="text-[11px] text-white/70">无标注</span>
                            ) : (
                              <>
                                {labels.slice(0, 3).map((l) => (
                                  <Badge
                                    key={`${img.id}-lbl-${l.label}`}
                                    variant="outline"
                                    className="bg-black/35 backdrop-blur-sm"
                                    style={{
                                      borderColor: l.color || "#ef4444",
                                      color: l.color || "#ef4444",
                                    }}
                                  >
                                    {l.label}
                                  </Badge>
                                ))}
                                {labels.length > 3 ? (
                                  <Badge variant="outline" className="bg-black/35 backdrop-blur-sm text-white/80 border-white/30">
                                    +{labels.length - 3}
                                  </Badge>
                                ) : null}
                              </>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>
       </div>
     </div>
  );
}
