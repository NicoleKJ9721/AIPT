import { useState } from "react";
import { 
  GitMerge, 
  Plus, 
  Play, 
  Save, 
  Trash2, 
  Settings, 
  Image as ImageIcon, 
  ArrowRight,
  CheckCircle2,
  Layers
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

// Mock data
const AVAILABLE_MODELS = [
  { id: "m1", name: "YOLOv8n-PCB-Defect", version: "v1.0", type: "Detection" },
  { id: "m2", name: "ResNet50-Classification", version: "v2.3", type: "Classification" },
  { id: "m3", name: "UNet-Segmentation", version: "v1.2", type: "Segmentation" },
  { id: "m4", name: "OCR-Serial-Number", version: "v1.0", type: "OCR" },
];

type PipelineNode = {
  id: string;
  type: "model" | "filter" | "output";
  title: string;
  modelId?: string;
  config: {
    threshold?: number;
    classes?: string[];
    crop?: boolean;
  };
};

const INITIAL_NODES: PipelineNode[] = [
  { 
    id: "n1", 
    type: "model", 
    title: "初级缺陷检测", 
    modelId: "m1", 
    config: { threshold: 0.5, classes: ["scratch", "dent"] } 
  },
  { 
    id: "n2", 
    type: "model", 
    title: "OCR识别", 
    modelId: "m4", 
    config: { threshold: 0.7, crop: true } 
  }
];

export default function MultiModelPipeline() {
  const [nodes, setNodes] = useState<PipelineNode[]>(INITIAL_NODES);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  const selectedNode = nodes.find(n => n.id === selectedNodeId);

  const handleAddNode = () => {
    const newNode: PipelineNode = {
      id: `n${Date.now()}`,
      type: "model",
      title: "新模型节点",
      config: { threshold: 0.5 }
    };
    setNodes([...nodes, newNode]);
    setSelectedNodeId(newNode.id);
  };

  const handleDeleteNode = (id: string) => {
    setNodes(nodes.filter(n => n.id !== id));
    if (selectedNodeId === id) setSelectedNodeId(null);
  };

  const handleRun = () => {
    setIsRunning(true);
    setTimeout(() => setIsRunning(false), 2000);
  };

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] gap-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between shrink-0">
        <div>
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3">
            <GitMerge className="w-8 h-8 text-blue-500" />
            多模型串联检测
          </h1>
          <p className="text-muted-foreground mt-2">
            构建多模型串行处理流，支持检测框裁剪、二次识别与逻辑判定
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline" className="gap-2">
            <Save className="w-4 h-4" />
            保存流程
          </Button>
          <Button className="gap-2 bg-blue-600 hover:bg-blue-700" onClick={handleRun} disabled={isRunning}>
            {isRunning ? <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : <Play className="w-4 h-4" />}
            开始检测
          </Button>
        </div>
      </div>

      <div className="flex flex-1 gap-6 min-h-0">
        {/* Left: Pipeline Canvas */}
        <div className="flex-1 flex flex-col min-w-0">
          <Card className="flex-1 border-dashed border-2 bg-slate-50/50 dark:bg-slate-950/20 overflow-hidden relative flex flex-col">
            <div className="absolute inset-0 bg-[radial-gradient(#e5e7eb_1px,transparent_1px)] [background-size:16px_16px] [mask-image:radial-gradient(ellipse_50%_50%_at_50%_50%,#000_70%,transparent_100%)] pointer-events-none" />
            
            <div className="flex-1 overflow-auto">
              <div className="p-12 min-w-max flex items-center justify-center min-h-full">
                <div className="flex items-center gap-4">
                  {/* Start Node */}
                  <div className="flex flex-col items-center gap-2">
                    <div className="w-16 h-16 rounded-full bg-green-100 border-2 border-green-500 flex items-center justify-center shadow-sm z-10">
                      <ImageIcon className="w-8 h-8 text-green-600" />
                    </div>
                    <span className="text-sm font-medium text-muted-foreground">图像输入</span>
                  </div>

                  {/* Arrow */}
                  <ArrowRight className="w-6 h-6 text-slate-300" />

                  {/* Nodes */}
                  {nodes.map((node) => (
                    <div key={node.id} className="flex items-center gap-4 group">
                      <div 
                        className={cn(
                          "relative w-64 rounded-xl border-2 bg-card p-4 shadow-sm transition-all cursor-pointer hover:shadow-md hover:-translate-y-1",
                          selectedNodeId === node.id ? "border-blue-500 ring-4 ring-blue-500/10" : "border-border"
                        )}
                        onClick={() => setSelectedNodeId(node.id)}
                      >
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex items-center gap-2">
                            <div className={cn(
                              "w-8 h-8 rounded-lg flex items-center justify-center text-white",
                              node.type === "model" ? "bg-blue-500" : "bg-orange-500"
                            )}>
                              <Layers className="w-4 h-4" />
                            </div>
                            <div>
                              <div className="font-semibold text-sm">{node.title}</div>
                              <div className="text-[10px] text-muted-foreground">ID: {node.id}</div>
                            </div>
                          </div>
                          <Button 
                            variant="ghost" 
                            size="icon" 
                            className="h-6 w-6 text-muted-foreground hover:text-destructive -mr-2 -mt-2 opacity-0 group-hover:opacity-100 transition-opacity"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDeleteNode(node.id);
                            }}
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </Button>
                        </div>
                        
                        <div className="space-y-2">
                          <div className="text-xs bg-secondary/50 p-2 rounded border flex items-center justify-between">
                            <span className="text-muted-foreground">模型</span>
                            <span className="font-medium truncate max-w-[120px]">
                              {AVAILABLE_MODELS.find(m => m.id === node.modelId)?.name || "未选择"}
                            </span>
                          </div>
                          <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
                            <Badge variant="outline" className="h-5 text-[10px] px-1.5">
                              Conf: {node.config.threshold}
                            </Badge>
                            {node.config.crop && (
                              <Badge variant="secondary" className="h-5 text-[10px] px-1.5">
                                Crop
                              </Badge>
                            )}
                          </div>
                        </div>

                        {/* Connection Handle */}
                        <div className="absolute -right-3 top-1/2 -translate-y-1/2 w-3 h-3 bg-slate-400 rounded-full border-2 border-white ring-2 ring-slate-100" />
                        <div className="absolute -left-3 top-1/2 -translate-y-1/2 w-3 h-3 bg-slate-400 rounded-full border-2 border-white ring-2 ring-slate-100" />
                      </div>

                      <ArrowRight className="w-6 h-6 text-slate-300" />
                    </div>
                  ))}

                  {/* Add Node Button */}
                  <Button 
                    variant="outline" 
                    className="h-16 w-16 rounded-2xl border-dashed border-2 gap-1 flex-col hover:border-blue-500 hover:text-blue-500 hover:bg-blue-50"
                    onClick={handleAddNode}
                  >
                    <Plus className="w-6 h-6" />
                  </Button>

                  <ArrowRight className="w-6 h-6 text-slate-300" />

                  {/* End Node */}
                  <div className="flex flex-col items-center gap-2">
                    <div className="w-16 h-16 rounded-full bg-slate-100 border-2 border-slate-300 flex items-center justify-center shadow-sm">
                      <CheckCircle2 className="w-8 h-8 text-slate-400" />
                    </div>
                    <span className="text-sm font-medium text-muted-foreground">结果输出</span>
                  </div>
                </div>
              </div>
            </div>
          </Card>
        </div>

        {/* Right: Configuration & Preview */}
        <div className="w-96 flex flex-col gap-6 shrink-0">
          {/* Node Configuration */}
          <Card className="flex-1 flex flex-col min-h-0">
            <CardHeader className="pb-3 border-b">
              <CardTitle className="text-base flex items-center gap-2">
                <Settings className="w-4 h-4" />
                节点配置
              </CardTitle>
            </CardHeader>
            <div className="flex-1 overflow-auto">
              <div className="p-4 space-y-6">
                {selectedNode ? (
                  <>
                    <div className="space-y-2">
                      <Label>节点名称</Label>
                      <Input 
                        value={selectedNode.title} 
                        onChange={(e) => {
                          const newTitle = e.target.value;
                          setNodes(nodes.map(n => n.id === selectedNode.id ? { ...n, title: newTitle } : n));
                        }}
                      />
                    </div>

                    <div className="space-y-2">
                      <Label>选择模型</Label>
                      <Select 
                        value={selectedNode.modelId} 
                        onValueChange={(val) => {
                          setNodes(nodes.map(n => n.id === selectedNode.id ? { ...n, modelId: val } : n));
                        }}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="选择模型..." />
                        </SelectTrigger>
                        <SelectContent>
                          {AVAILABLE_MODELS.map(m => (
                            <SelectItem key={m.id} value={m.id}>
                              <div className="flex flex-col items-start">
                                <span>{m.name}</span>
                                <span className="text-xs text-muted-foreground">{m.type} · {m.version}</span>
                              </div>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="h-px w-full bg-slate-200 dark:bg-slate-800" />

                    <div className="space-y-4">
                      <Label>参数设置</Label>
                      
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label className="text-xs text-muted-foreground">置信度阈值 (Conf)</Label>
                          <Input 
                            type="number" 
                            step="0.1" 
                            min="0" 
                            max="1"
                            value={selectedNode.config.threshold}
                            onChange={(e) => {
                              const val = parseFloat(e.target.value);
                              setNodes(nodes.map(n => n.id === selectedNode.id ? { ...n, config: { ...n.config, threshold: val } } : n));
                            }}
                          />
                        </div>
                        <div className="space-y-2">
                          <Label className="text-xs text-muted-foreground">IOU 阈值</Label>
                          <Input type="number" defaultValue="0.45" step="0.1" />
                        </div>
                      </div>

                      <div className="flex items-center justify-between rounded-lg border p-3 shadow-sm">
                        <div className="space-y-0.5">
                          <Label className="text-sm">启用裁剪 (Crop)</Label>
                          <div className="text-xs text-muted-foreground">将检测框作为下一级输入</div>
                        </div>
                        <input 
                          type="checkbox" 
                          className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-600"
                          checked={selectedNode.config.crop || false}
                          onChange={(e) => {
                            setNodes(nodes.map(n => n.id === selectedNode.id ? { ...n, config: { ...n.config, crop: e.target.checked } } : n));
                          }}
                        />
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="h-full flex flex-col items-center justify-center text-center text-muted-foreground p-8">
                    <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center mb-3">
                      <Settings className="w-6 h-6 text-slate-300" />
                    </div>
                    <p className="text-sm">请点击左侧节点进行配置</p>
                  </div>
                )}
              </div>
            </div>
          </Card>

          {/* Preview Area */}
          <Card className="h-64 flex flex-col shrink-0">
            <CardHeader className="pb-3 border-b py-3 min-h-[48px] px-4">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-medium">结果预览</CardTitle>
                <Badge variant="outline" className="text-[10px]">Real-time</Badge>
              </div>
            </CardHeader>
            <CardContent className="flex-1 p-0 relative bg-black/5 flex items-center justify-center overflow-hidden">
               {isRunning ? (
                 <div className="flex flex-col items-center gap-2 text-blue-600 animate-pulse">
                   <div className="w-8 h-8 border-4 border-current border-t-transparent rounded-full animate-spin" />
                   <span className="text-xs font-medium">Processing...</span>
                 </div>
               ) : (
                 <div className="text-center p-4">
                   <ImageIcon className="w-12 h-12 text-slate-300 mx-auto mb-2" />
                   <p className="text-xs text-muted-foreground">点击“开始检测”查看效果</p>
                 </div>
               )}
               {/* Mock result overlay could go here */}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
