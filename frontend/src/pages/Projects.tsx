import { useState, useEffect } from "react";
import { 
  Plus, Search, Filter, Folder, GitBranch, Clock, Github, 
  Trash2, Copy, Download, AlertTriangle, X, CheckCircle, 
  FileJson, RotateCcw, Database, ScanEye
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { projectService, type ProjectRecord } from "@/lib/api";
import { useNavigate } from "react-router-dom";
import { useProjectContext } from "@/store/projectContext";

// --- Types ---
interface Project {
  id: string;
  name: string;
  type: string;
  images: number;
  status: string;
  lastUpdated: string;
  commit: string;
}

interface AuditLog {
  id: string;
  action: "DELETE" | "COPY" | "EXPORT" | "UNDO";
  targetId: string;
  targetName: string;
  timestamp: string;
  details?: string;
}

const formatDate = (isoString: string) => {
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) return "-";
  return date.toISOString().split("T")[0];
};

const toUiProject = (p: ProjectRecord): Project => ({
  id: p.id,
  name: p.name,
  type: p.type,
  images: p.images_count ?? 0,
  status: p.status,
  lastUpdated: formatDate(p.updated_at),
  commit: p.latest_commit || "-",
});

// --- Simple Toast Component ---
const Toast = ({ message, type, onClose }: { message: string, type: "success" | "error" | "info", onClose: () => void }) => {
  useEffect(() => {
    const timer = setTimeout(onClose, 3000);
    return () => clearTimeout(timer);
  }, [onClose]);

  return (
    <div className={cn(
      "fixed top-4 right-4 z-50 flex items-center gap-2 px-4 py-2 rounded-md shadow-lg border animate-in slide-in-from-top-2 fade-in",
      type === "success" ? "bg-green-50 border-green-200 text-green-700" :
      type === "error" ? "bg-red-50 border-red-200 text-red-700" :
      "bg-white border-gray-200 text-gray-700"
    )}>
      {type === "success" && <CheckCircle className="w-4 h-4" />}
      {type === "error" && <AlertTriangle className="w-4 h-4" />}
      {type === "info" && <FileJson className="w-4 h-4" />}
      <span className="text-sm font-medium">{message}</span>
      <button onClick={onClose} className="ml-2 hover:opacity-70"><X className="w-3 h-3" /></button>
    </div>
  );
};

// --- Confirmation Dialog Component ---
const ConfirmDialog = ({ 
  isOpen, 
  title, 
  description, 
  onConfirm, 
  onCancel 
}: { 
  isOpen: boolean, 
  title: string, 
  description: string, 
  onConfirm: () => void, 
  onCancel: () => void 
}) => {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm animate-in fade-in">
      <Card className="w-full max-w-md shadow-xl border-destructive/20">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-destructive">
            <AlertTriangle className="w-5 h-5" />
            {title}
          </CardTitle>
          <CardDescription>{description}</CardDescription>
        </CardHeader>
        <CardContent className="flex justify-end gap-3 pt-4">
          <Button variant="outline" onClick={onCancel}>取消</Button>
          <Button variant="destructive" onClick={onConfirm}>确认删除</Button>
        </CardContent>
      </Card>
    </div>
  );
};

export default function Projects() {
  const navigate = useNavigate();
  const setProjectContext = useProjectContext((s) => s.setProject);

  const [projects, setProjects] = useState<Project[]>([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [isLoadingProjects, setIsLoadingProjects] = useState(true);
  
  // Undo Stack: Stores snapshot of projects before mutation
  const [undoStack, setUndoStack] = useState<Project[][]>([]);
  
  // Audit Logs
  const [logs, setLogs] = useState<AuditLog[]>([]);

  // UI States
  const [toast, setToast] = useState<{ message: string, type: "success" | "error" | "info" } | null>(null);
  const [deleteTargetId, setDeleteTargetId] = useState<string | null>(null);
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [newProjectType, setNewProjectType] = useState("目标检测");

  // --- Helpers ---
  const showToast = (message: string, type: "success" | "error" | "info" = "success") => {
    setToast({ message, type });
  };

  const addLog = (action: AuditLog["action"], target: Project, details?: string) => {
    const newLog: AuditLog = {
      id: Math.random().toString(36).substr(2, 9),
      action,
      targetId: target.id,
      targetName: target.name,
      timestamp: new Date().toISOString(),
      details
    };
    setLogs(prev => [newLog, ...prev]);
    console.log("[AUDIT LOG]", newLog);
  };

  const saveToUndo = () => {
    setUndoStack(prev => [...prev, [...projects]]);
  };

  const loadProjects = async () => {
    try {
      setIsLoadingProjects(true);
      const data = await projectService.list();
      setProjects(data.map(toUiProject));
    } catch (error) {
      console.error("Load Projects Error:", error);
      showToast("加载项目失败，请检查后端服务", "error");
    } finally {
      setIsLoadingProjects(false);
    }
  };

  useEffect(() => {
    void loadProjects();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- Actions ---

  // 1. Delete
  const initiateDelete = (id: string) => {
    setDeleteTargetId(id);
  };

  const confirmDelete = async () => {
    if (!deleteTargetId) return;
    const target = projects.find(p => p.id === deleteTargetId);
    if (!target) return;

    saveToUndo();
    try {
      await projectService.delete(deleteTargetId);
      setProjects(prev => prev.filter(p => p.id !== deleteTargetId));
      addLog("DELETE", target);
      showToast(`项目 "${target.name}" 已删除`);
    } catch (error) {
      console.error("Delete Project Error:", error);
      showToast("删除失败，请检查后端服务", "error");
    } finally {
      setDeleteTargetId(null);
    }
  };

  // 2. Copy
  const handleCopy = async (project: Project) => {
    saveToUndo();
    try {
      const created = await projectService.create({
        name: `${project.name}_副本`,
        type: project.type,
        status: "进行中",
        latest_commit: "Initial copy",
      });
      setProjects(prev => [toUiProject(created), ...prev]);
      addLog("COPY", project, `Created copy: ${created.id}`);
      showToast("项目已复制", "success");
    } catch (error) {
      console.error("Copy Project Error:", error);
      showToast("复制失败，请检查后端服务", "error");
    }
  };

  // 3. Export
  const handleExport = (project: Project) => {
    try {
      const exportData = {
        meta: {
          version: "1.0",
          exportDate: new Date().toISOString(),
          exportedBy: "User_Current"
        },
        project: project,
        configurations: {
            // Mock config data
            model: "YOLOv8",
            batchSize: 16,
            learningRate: 0.001
        }
      };

      const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${project.name}_${Date.now()}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      addLog("EXPORT", project);
      showToast("项目导出成功，开始下载", "info");
    } catch (error) {
      console.error("Export Error:", error);
      showToast("导出失败", "error");
    }
  };

  // 4. Undo
  const handleUndo = async () => {
    if (undoStack.length === 0) return;
    const previousState = undoStack[undoStack.length - 1];
    setUndoStack(prev => prev.slice(0, -1));
    
    const prevMap = new Map(previousState.map(p => [p.id, p]));
    const curMap = new Map(projects.map(p => [p.id, p]));
    const idsToDelete = [...curMap.keys()].filter(id => !prevMap.has(id));
    const projectsToCreate = previousState.filter(p => !curMap.has(p.id));
    const projectsToUpdate = previousState.filter(p => {
      const cur = curMap.get(p.id);
      if (!cur) return false;
      return (
        cur.name !== p.name ||
        cur.type !== p.type ||
        cur.status !== p.status ||
        cur.commit !== p.commit
      );
    });

    try {
      for (const id of idsToDelete) {
        await projectService.delete(id);
      }
      for (const p of projectsToCreate) {
        await projectService.create({
          id: p.id,
          name: p.name,
          type: p.type,
          status: p.status,
          latest_commit: p.commit === "-" ? "" : p.commit,
        });
      }
      for (const p of projectsToUpdate) {
        await projectService.update(p.id, {
          name: p.name,
          type: p.type,
          status: p.status,
          latest_commit: p.commit === "-" ? "" : p.commit,
        });
      }

      await loadProjects();
      showToast("操作已撤销", "info");
      const lastLog = logs[0];
      if (lastLog) {
        addLog("UNDO", { id: lastLog.targetId, name: lastLog.targetName } as Project, `Reverted ${lastLog.action}`);
      }
    } catch (error) {
      console.error("Undo Error:", error);
      showToast("撤销失败，请检查后端服务", "error");
      await loadProjects();
    }
  };

  const handleCreateProject = async () => {
    const name = newProjectName.trim();
    if (!name) {
      showToast("请输入项目名称", "error");
      return;
    }

    saveToUndo();
    try {
      const created = await projectService.create({ name, type: newProjectType });
      setProjects(prev => [toUiProject(created), ...prev]);
      showToast("项目已创建", "success");
      setIsCreateOpen(false);
      setNewProjectName("");
      setNewProjectType("目标检测");

      // Set global project context and jump to project workspace (dataset is managed inside).
      setProjectContext(created.id, created.name);
      navigate(`/projects/${encodeURIComponent(created.id)}?step=import`);
    } catch (error) {
      console.error("Create Project Error:", error);
      showToast("创建失败，请检查后端服务", "error");
    }
  };

  const filteredProjects = projects.filter(p => 
    p.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="space-y-6 relative">
      {/* Toast Notification */}
      {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}

      {/* Delete Confirmation Dialog */}
      <ConfirmDialog 
        isOpen={!!deleteTargetId}
        title="确认删除该项目？"
        description="此操作将永久删除该项目及其所有配置数据。此操作不可恢复（除非使用撤销）。"
        onConfirm={confirmDelete}
        onCancel={() => setDeleteTargetId(null)}
      />
      
      {/* Create Project Dialog */}
      {isCreateOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm animate-in fade-in">
          <Card className="w-full max-w-md shadow-xl">
            <CardHeader>
              <CardTitle>新建项目</CardTitle>
              <CardDescription>创建一个新的工业视觉检测项目</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">项目名称</label>
                <Input
                  value={newProjectName}
                  onChange={(e) => setNewProjectName(e.target.value)}
                  placeholder="例如：PCB 缺陷检测"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">任务类型</label>
                <select
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background"
                  value={newProjectType}
                  onChange={(e) => setNewProjectType(e.target.value)}
                >
                  <option value="目标检测">目标检测</option>
                  <option value="语义分割">语义分割</option>
                  <option value="分类">分类</option>
                </select>
              </div>
              <div className="flex justify-end gap-3 pt-2">
                <Button variant="outline" onClick={() => setIsCreateOpen(false)}>取消</Button>
                <Button onClick={handleCreateProject}>创建</Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">项目管理</h1>
          <p className="text-muted-foreground mt-2">
            全流程管理您的 AI 项目，支持本地数据处理与 Git 版本控制
          </p>
        </div>
        <div className="flex gap-2">
          {undoStack.length > 0 && (
             <Button variant="ghost" onClick={handleUndo} className="gap-2 text-muted-foreground hover:text-foreground">
                <RotateCcw className="w-4 h-4" /> 撤销操作
             </Button>
          )}
          <Button variant="outline" className="gap-2">
            <Github className="w-4 h-4" /> 关联 GitHub
          </Button>
          <Button className="gap-2" onClick={() => setIsCreateOpen(true)}>
            <Plus className="w-4 h-4" /> 新建项目
          </Button>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="搜索项目名称..."
            className="pl-9"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        <Button variant="outline" className="gap-2">
          <Filter className="w-4 h-4" /> 筛选状态
        </Button>
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">总项目数</CardTitle>
            <Folder className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{projects.length}</div>
            <p className="text-xs text-muted-foreground">
              当前活跃项目
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Git 提交总数</CardTitle>
            <GitBranch className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">156</div>
            <p className="text-xs text-muted-foreground">
              本周新增 24 次提交
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">平均训练时长</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">2.4h</div>
            <p className="text-xs text-muted-foreground">
              本地 GPU 加速
            </p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>项目列表</CardTitle>
          <CardDescription>包含最近活跃的项目及其版本信息</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>项目名称</TableHead>
                <TableHead>任务类型</TableHead>
                <TableHead>图片数量</TableHead>
                <TableHead>最新提交 (Git)</TableHead>
                <TableHead>状态</TableHead>
                <TableHead>最后更新</TableHead>
                <TableHead className="text-right">操作</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isLoadingProjects ? (
                <TableRow>
                  <TableCell colSpan={7} className="text-center py-8 text-muted-foreground">
                    正在加载项目...
                  </TableCell>
                </TableRow>
              ) : filteredProjects.length === 0 ? (
                 <TableRow>
                    <TableCell colSpan={7} className="text-center py-8 text-muted-foreground">
                        暂无项目数据
                    </TableCell>
                 </TableRow>
              ) : (
                filteredProjects.map((project) => (
                    <TableRow key={project.id}>
                    <TableCell className="font-medium">
                        <div className="flex flex-col">
                            <span>{project.name}</span>
                            <span className="text-xs text-muted-foreground">ID: {project.id}</span>
                        </div>
                    </TableCell>
                    <TableCell>
                        <Badge variant="secondary">{project.type}</Badge>
                    </TableCell>
                    <TableCell>{project.images}</TableCell>
                    <TableCell>
                        <div className="flex items-center gap-1 text-sm text-muted-foreground">
                            <GitBranch className="w-3 h-3" />
                            <span className="font-mono text-xs">{project.commit}</span>
                        </div>
                    </TableCell>
                    <TableCell>
                        <Badge
                        variant={
                            project.status === "已完成"
                            ? "default"
                            : project.status === "训练中"
                            ? "secondary"
                            : "outline"
                        }
                        className={
                            project.status === "已完成"
                            ? "bg-green-500 hover:bg-green-600"
                            : project.status === "训练中"
                            ? "bg-blue-500 text-white hover:bg-blue-600"
                            : ""
                        }
                        >
                        {project.status}
                        </Badge>
                    </TableCell>
                    <TableCell>{project.lastUpdated}</TableCell>
                    <TableCell className="text-right">
                        <div className="flex justify-end gap-1">
                            <Button
                                variant="ghost"
                                size="icon"
                                className="h-8 w-8 text-emerald-600 hover:text-emerald-700 hover:bg-emerald-50"
                                onClick={() => {
                                  setProjectContext(project.id, project.name);
                                  navigate(`/projects/${encodeURIComponent(project.id)}`);
                                }}
                                title="进入项目数据（导入/质检）"
                            >
                                <Database className="w-4 h-4" />
                            </Button>
                            <Button
                                variant="ghost"
                                size="icon"
                                className="h-8 w-8 text-purple-600 hover:text-purple-700 hover:bg-purple-50"
                                onClick={() => {
                                  setProjectContext(project.id, project.name);
                                  navigate(`/annotate?project_id=${encodeURIComponent(project.id)}`);
                                }}
                                title="进入智能标注"
                            >
                                <ScanEye className="w-4 h-4" />
                            </Button>
                            <Button 
                                variant="ghost" 
                                size="icon" 
                                className="h-8 w-8 text-blue-600 hover:text-blue-700 hover:bg-blue-50"
                                onClick={() => handleCopy(project)}
                                title="复制项目 (Copy)"
                            >
                                <Copy className="w-4 h-4" />
                            </Button>
                            <Button 
                                variant="ghost" 
                                size="icon" 
                                className="h-8 w-8 text-orange-600 hover:text-orange-700 hover:bg-orange-50"
                                onClick={() => handleExport(project)}
                                title="导出数据 (Export)"
                            >
                                <Download className="w-4 h-4" />
                            </Button>
                            <Button 
                                variant="ghost" 
                                size="icon" 
                                className="h-8 w-8 text-red-600 hover:text-red-700 hover:bg-red-50"
                                onClick={() => initiateDelete(project.id)}
                                title="删除项目 (Delete)"
                            >
                                <Trash2 className="w-4 h-4" />
                            </Button>
                        </div>
                    </TableCell>
                    </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
