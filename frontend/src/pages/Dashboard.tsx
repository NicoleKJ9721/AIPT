import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, CheckCircle2, FolderOpen, RefreshCw, Save } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { toast } from "@/components/ui/use-toast";
import { aiService, dashboardService, systemService, type DashboardSummaryRecord, type SystemSettings } from "@/lib/api";
import { cn } from "@/lib/utils";

const CONSUMER_GPU_BUILDS = [
  {
    tier: "入门配置",
    gpu: "RTX 4060 Ti (8-16GB)",
    cpu: "i5 / Ryzen 5",
    ram: "32GB",
    storage: "1TB NVMe",
    color: "bg-blue-500",
    note: "适合学习与小模型微调",
  },
  {
    tier: "均衡配置",
    gpu: "RTX 3090 / 4080",
    cpu: "i7 / Ryzen 7",
    ram: "64GB",
    storage: "2TB NVMe",
    color: "bg-purple-500",
    note: "生产环境标准配置",
  },
  {
    tier: "旗舰配置",
    gpu: "RTX 4090 (24GB)",
    cpu: "i9 / Ryzen 9",
    ram: "128GB",
    storage: "4TB NVMe RAID",
    color: "bg-orange-500",
    note: "大模型训练与高并发推理",
  },
];

type HealthRecord = {
  status: string;
  model_loaded?: boolean;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
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

function normalizePath(value: string) {
  const trimmed = (value || "").trim();
  return trimmed.replace(/[\\/]+$/, "");
}

function formatPercent(n: number) {
  if (!Number.isFinite(n) || n <= 0) return "0%";
  if (n >= 1) return "100%";
  return `${Math.round(n * 100)}%`;
}

function statusColor(status: "ok" | "warn" | "error") {
  if (status === "ok") return "bg-emerald-500";
  if (status === "warn") return "bg-amber-500";
  return "bg-red-500";
}

function trainStatusText(status: string | null | undefined) {
  switch (status) {
    case "queued":
      return "排队中";
    case "running":
      return "训练中";
    case "completed":
      return "已完成";
    case "failed":
      return "失败";
    default:
      return "未开始";
  }
}

export default function Dashboard() {
  const [settings, setSettings] = useState<SystemSettings | null>(null);
  const [summary, setSummary] = useState<DashboardSummaryRecord | null>(null);
  const [health, setHealth] = useState<HealthRecord | null>(null);

  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingMetrics, setIsLoadingMetrics] = useState(true);
  const [isSaving, setIsSaving] = useState(false);

  const [projectsRoot, setProjectsRoot] = useState("");
  const [resourcesRoot, setResourcesRoot] = useState("");

  const recentProjectsRoots = settings?.recent_projects_root_dirs ?? [];
  const recentResourcesRoots = settings?.recent_resources_root_dirs ?? [];

  const hasChanges = useMemo(() => {
    if (!settings) return false;
    return (
      normalizePath(projectsRoot) !== normalizePath(settings.projects_root_dir) ||
      normalizePath(resourcesRoot) !== normalizePath(settings.resources_root_dir)
    );
  }, [projectsRoot, resourcesRoot, settings]);

  const annotationProgress = useMemo(() => {
    if (!summary) return 0;
    if (!summary.images_total) return 0;
    return summary.images_annotated_total / summary.images_total;
  }, [summary]);

  const warnings = useMemo(() => {
    const out: string[] = [];
    if (health && health.status !== "ok") {
      out.push("后端服务异常：请先运行 `start_services.bat` 启动后端。");
    }
    if (summary?.training?.last_status === "failed") {
      out.push(`最近一次训练失败：${summary.training.last_error || "请前往模型训练模块查看日志"}`);
    }
    if (settings) {
      if (!normalizePath(settings.projects_root_dir)) out.push("项目存储路径未配置。");
      if (!normalizePath(settings.resources_root_dir)) out.push("资源存储路径未配置。");
    }
    return out;
  }, [health, settings, summary]);

  const loadSettings = async () => {
    const s = await systemService.getSettings();
    setSettings(s);
    setProjectsRoot(s.projects_root_dir);
    setResourcesRoot(s.resources_root_dir);
  };

  const loadMetrics = async () => {
    try {
      setIsLoadingMetrics(true);
      const [sum, h] = await Promise.all([
        dashboardService.summary(),
        aiService.checkHealth().catch(() => ({ status: "error" } as HealthRecord)),
      ]);
      setSummary(sum);
      setHealth(h as HealthRecord);
    } catch (err) {
      console.error(err);
      setSummary(null);
      setHealth({ status: "error" });
    } finally {
      setIsLoadingMetrics(false);
    }
  };

  const loadAll = async () => {
    try {
      setIsLoading(true);
      await Promise.all([loadSettings(), loadMetrics()]);
    } catch (err) {
      console.error(err);
      toast({
        title: "加载失败",
        description: getErrorMessage(err, "无法获取系统信息，请检查后端是否已启动（http://127.0.0.1:8000）。"),
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    void loadAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const pickDir = async (kind: "projects" | "resources") => {
    const current = kind === "projects" ? projectsRoot : resourcesRoot;
    try {
      const selected = await systemService.selectDirectory({
        title: kind === "projects" ? "选择项目文件保存目录" : "选择本地资源目录",
        initial_dir: current || undefined,
      });
      if (!selected) return;
      if (kind === "projects") setProjectsRoot(selected);
      else setResourcesRoot(selected);
    } catch (err) {
      console.error(err);
      toast({
        title: "打开选择器失败",
        description: getErrorMessage(err, "目录选择器不可用（可能缺少 GUI 环境）。"),
        variant: "destructive",
      });
    }
  };

  const save = async () => {
    if (!settings) return;
    const nextProjects = normalizePath(projectsRoot);
    const nextResources = normalizePath(resourcesRoot);
    if (!nextProjects || !nextResources) {
      toast({ title: "路径不能为空", variant: "destructive" });
      return;
    }
    try {
      setIsSaving(true);
      const updated = await systemService.updateSettings({
        projects_root_dir: nextProjects,
        resources_root_dir: nextResources,
      });
      setSettings(updated);
      setProjectsRoot(updated.projects_root_dir);
      setResourcesRoot(updated.resources_root_dir);
      toast({ title: "已保存", description: "新的默认路径将用于后续新建项目与资源存储。" });
    } catch (err) {
      console.error(err);
      toast({
        title: "保存失败",
        description: getErrorMessage(err, "请检查 API Key / 权限设置或后端日志。"),
        variant: "destructive",
      });
    } finally {
      setIsSaving(false);
    }
  };

  const backendStatus: "ok" | "warn" | "error" =
    health?.status === "ok" ? "ok" : health ? "error" : "warn";

  const trainLastStatus = summary?.training?.last_status ?? null;
  const trainIsRunning = (summary?.training?.running_jobs ?? 0) > 0 || trainLastStatus === "running";
  const trainStatus: "ok" | "warn" | "error" =
    trainLastStatus === "failed" ? "error" : trainIsRunning ? "warn" : "ok";

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">工作台</h1>
          <p className="text-muted-foreground mt-2">
            平台状态监控与本地存储配置（源码与业务文件物理分离）
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={loadAll} disabled={isLoading} className="gap-2">
            <RefreshCw className={cn("w-4 h-4", isLoading ? "animate-spin" : "")} />
            刷新
          </Button>
          <Button onClick={save} disabled={!hasChanges || isSaving || isLoading} className="gap-2">
            {isSaving ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
            保存默认路径
          </Button>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">项目总数</CardTitle>
          </CardHeader>
          <CardContent className="text-3xl font-bold">
            {isLoadingMetrics ? "…" : String(summary?.projects_total ?? 0)}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">标注进度</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="text-3xl font-bold">{isLoadingMetrics ? "…" : formatPercent(annotationProgress)}</div>
            <div className="text-sm text-muted-foreground">
              {isLoadingMetrics
                ? "加载中…"
                : `${summary?.images_annotated_total ?? 0}/${summary?.images_total ?? 0} 张已标注`}
            </div>
            <div className="h-2 w-full rounded bg-muted overflow-hidden">
              <div className="h-full bg-primary" style={{ width: `${Math.round(annotationProgress * 100)}%` }} />
            </div>
          </CardContent>
        </Card>

        <Card className={cn(trainStatus === "error" ? "border-red-500/40" : "")}>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">训练状态</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="flex items-center gap-2">
              <span className={cn("inline-block h-2.5 w-2.5 rounded-full", statusColor(trainStatus))} />
              <div className="text-xl font-semibold">{isLoadingMetrics ? "…" : trainStatusText(trainLastStatus)}</div>
            </div>
            <div className="text-sm text-muted-foreground">
              {isLoadingMetrics
                ? "加载中…"
                : summary?.training?.last_job_id
                  ? `最近任务：${summary.training.last_job_id.slice(0, 8)}`
                  : "暂无训练任务"}
            </div>
          </CardContent>
        </Card>

        <Card className={cn(backendStatus === "error" ? "border-red-500/40" : "")}>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">平台服务</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="flex items-center gap-2">
              <span className={cn("inline-block h-2.5 w-2.5 rounded-full", statusColor(backendStatus))} />
              <div className="text-xl font-semibold">{backendStatus === "ok" ? "正常" : "异常"}</div>
            </div>
            <div className="text-sm text-muted-foreground">后端：{health?.status ?? "未知"}</div>
          </CardContent>
        </Card>
      </div>

      {warnings.length > 0 ? (
        <Card className="border-red-500/40">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-red-600">
              <AlertTriangle className="h-5 w-5" />
              异常预警
            </CardTitle>
            <CardDescription>请先处理以下问题，再进行标注/训练等流程</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            {warnings.map((w) => (
              <div key={w} className="flex items-start gap-2">
                <span className="mt-1 inline-block h-2 w-2 rounded-full bg-red-500" />
                <span>{w}</span>
              </div>
            ))}
          </CardContent>
        </Card>
      ) : (
        <Card className="border-emerald-500/40">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-emerald-600">
              <CheckCircle2 className="h-5 w-5" />
              运行正常
            </CardTitle>
            <CardDescription>可以开始创建项目 → 导入图片 → 标注 → 增强 → 训练</CardDescription>
          </CardHeader>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>硬件配置建议（消费级显卡）</CardTitle>
          <CardDescription>部署推荐模块的硬件建议已迁移到工作台展示（按成本/性能分 3 档）</CardDescription>
        </CardHeader>
        <CardContent className="overflow-auto">
          <div className="min-w-[720px]">
            <table className="w-full text-sm">
              <thead className="text-muted-foreground">
                <tr className="border-b">
                  <th className="py-2 text-left font-medium">档位</th>
                  <th className="py-2 text-left font-medium">GPU</th>
                  <th className="py-2 text-left font-medium">CPU</th>
                  <th className="py-2 text-left font-medium">内存</th>
                  <th className="py-2 text-left font-medium">存储</th>
                  <th className="py-2 text-left font-medium">适用</th>
                </tr>
              </thead>
              <tbody>
                {CONSUMER_GPU_BUILDS.map((b) => (
                  <tr key={b.tier} className="border-b last:border-0 align-top">
                    <td className="py-3 pr-3 font-medium whitespace-nowrap">{b.tier}</td>
                    <td className="py-3 pr-3">{b.gpu}</td>
                    <td className="py-3 pr-3">{b.cpu}</td>
                    <td className="py-3 pr-3 whitespace-nowrap">{b.ram}</td>
                    <td className="py-3 pr-3 whitespace-nowrap">{b.storage}</td>
                    <td className="py-3 text-muted-foreground">{b.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>项目文件存储</CardTitle>
            <CardDescription>用于保存项目数据、数据集文件与导出结果（建议放在非源码目录）</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center gap-2">
              <Input value={projectsRoot} onChange={(e) => setProjectsRoot(e.target.value)} placeholder="选择目录..." />
              <Button variant="outline" onClick={() => pickDir("projects")} className="gap-2">
                <FolderOpen className="w-4 h-4" />
                选择
              </Button>
            </div>
            {recentProjectsRoots.length > 0 ? (
              <div className="text-sm text-muted-foreground">
                最近使用：
                <div className="mt-2 space-y-1">
                  {recentProjectsRoots.slice(0, 6).map((p) => (
                    <button
                      key={p}
                      className="block w-full text-left truncate hover:text-foreground"
                      onClick={() => setProjectsRoot(p)}
                      title={p}
                      type="button"
                    >
                      {p}
                    </button>
                  ))}
                </div>
              </div>
            ) : null}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>本地资源存储</CardTitle>
            <CardDescription>用于保存预训练模型、训练产物与缓存资源（独立于源码目录）</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center gap-2">
              <Input value={resourcesRoot} onChange={(e) => setResourcesRoot(e.target.value)} placeholder="选择目录..." />
              <Button variant="outline" onClick={() => pickDir("resources")} className="gap-2">
                <FolderOpen className="w-4 h-4" />
                选择
              </Button>
            </div>
            {recentResourcesRoots.length > 0 ? (
              <div className="text-sm text-muted-foreground">
                最近使用：
                <div className="mt-2 space-y-1">
                  {recentResourcesRoots.slice(0, 6).map((p) => (
                    <button
                      key={p}
                      className="block w-full text-left truncate hover:text-foreground"
                      onClick={() => setResourcesRoot(p)}
                      title={p}
                      type="button"
                    >
                      {p}
                    </button>
                  ))}
                </div>
              </div>
            ) : null}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
