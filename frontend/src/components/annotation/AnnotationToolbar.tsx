import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { 
    MousePointer2, 
    Square, 
    Pentagon, 
    Move, 
    Keyboard,
    RotateCcw,
    Trash2
} from "lucide-react";

interface AnnotationToolbarProps {
    selectedTool: "select" | "rect" | "polygon" | "move";
    setSelectedTool: (tool: "select" | "rect" | "polygon" | "move") => void;
    setShowShortcuts: (show: boolean) => void;
    onResetView: () => void;
    onClearAll: () => void;
    hasAnnotations: boolean;
}

export function AnnotationToolbar({
    selectedTool,
    setSelectedTool,
    setShowShortcuts,
    onResetView,
    onClearAll,
    hasAnnotations
}: AnnotationToolbarProps) {
    return (
        <div className="absolute left-4 top-4 z-10 flex flex-col gap-2">
            <Card className="flex flex-col items-center py-2 gap-2 shadow-md">
                <Button
                    variant={selectedTool === "move" ? "default" : "ghost"}
                    size="icon"
                    className="h-9 w-9"
                    title="移动画布 (Space)"
                    onClick={() => setSelectedTool("move")}
                >
                    <Move className="w-4 h-4" />
                </Button>
                <Button
                    variant={selectedTool === "select" ? "default" : "ghost"}
                    size="icon"
                    className="h-9 w-9"
                    title="选择工具 (V)"
                    onClick={() => setSelectedTool("select")}
                >
                    <MousePointer2 className="w-4 h-4" />
                </Button>
                <div className="w-6 h-px bg-border" />
                <Button
                    variant={selectedTool === "rect" ? "default" : "ghost"}
                    size="icon"
                    className="h-9 w-9"
                    title="矩形标注 (R)"
                    onClick={() => setSelectedTool("rect")}
                >
                    <Square className="w-4 h-4" />
                </Button>
                <Button
                    variant={selectedTool === "polygon" ? "default" : "ghost"}
                    size="icon"
                    className="h-9 w-9"
                    title="多边形标注 (P) - 双击结束"
                    onClick={() => setSelectedTool("polygon")}
                >
                    <Pentagon className="w-4 h-4" />
                </Button>
                <div className="flex-1" />
                <Button variant="ghost" size="icon" className="h-9 w-9" title="快捷键帮助" onClick={() => setShowShortcuts(true)}>
                    <Keyboard className="w-4 h-4" />
                </Button>
                <div className="w-6 h-px bg-border" />
                <Button variant="ghost" size="icon" className="h-9 w-9" title="恢复原图大小" onClick={onResetView}>
                    <RotateCcw className="w-4 h-4" />
                </Button>
                <Button 
                    variant="ghost" 
                    size="icon" 
                    className="h-9 w-9 text-destructive hover:text-destructive" 
                    title="清空标注信息" 
                    disabled={!hasAnnotations} 
                    onClick={onClearAll}
                >
                    <Trash2 className="w-4 h-4" />
                </Button>
            </Card>
        </div>
    );
}
