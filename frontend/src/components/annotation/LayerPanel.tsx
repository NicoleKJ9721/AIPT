import { 
    Settings, 
    Sun, 
    Contrast, 
    Crosshair, 
    Layers, 
    Pentagon, 
    Square, 
    Eye, 
    EyeOff, 
    Trash2, 
    Pencil,
    Tag, 
    Plus, 
    CheckCircle2 
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { Annotation, LabelClass, ImageItem } from "@/types/annotation";

interface LayerPanelProps {
    isSidebarOpen: boolean;
    activeTab: string;
    setActiveTab: (tab: string) => void;
    
    // Image Settings
    brightness: number;
    setBrightness: (val: number) => void;
    contrast: number;
    setContrast: (val: number) => void;
    showCrosshair: boolean;
    setShowCrosshair: (show: boolean) => void;

    // Annotations
    annotations: Annotation[];
    selectedId: string | null;
    setSelectedId: (id: string | null) => void;
    onDelete: () => void;
    onToggleVisibility: (id: string, e: React.MouseEvent) => void;

    // Classes
    classes: LabelClass[];
    selectedClassId: string;
    onClassSelect: (id: string) => void;
    onClassAdd: () => void;
    onClassEdit: (id: string) => void;
    onClassDelete: (id: string) => void;
    onEditAnnotationLabel: (annotationId: string) => void;

    // Progress
    currentImageIndex: number;
    totalImages: number;
    images?: ImageItem[];
}

export function LayerPanel({
    isSidebarOpen,
    activeTab,
    setActiveTab,
    brightness,
    setBrightness,
    contrast,
    setContrast,
    showCrosshair,
    setShowCrosshair,
    annotations,
    selectedId,
    setSelectedId,
    onDelete,
    onToggleVisibility,
    classes,
    selectedClassId,
    onClassSelect,
    onClassAdd,
    onClassEdit,
    onClassDelete,
    onEditAnnotationLabel,
    currentImageIndex,
    totalImages,
    images, // Destructure images from props
}: LayerPanelProps) {
    const completedCount = images ? images.filter((img) => img.status === "completed").length : 0;
    const progressPct = totalImages > 0 ? (completedCount / totalImages) * 100 : 0;

    return (
        <div className={`
            flex flex-col gap-4 border-l bg-background transition-all duration-300 ease-in-out z-20 absolute right-0 h-full shadow-xl
            ${isSidebarOpen ? 'w-80 translate-x-0 p-4' : 'w-0 translate-x-full overflow-hidden p-0'}
            md:relative md:translate-x-0 md:shadow-none
            ${isSidebarOpen ? 'md:w-[25%] md:min-w-[280px] md:max-w-[320px]' : 'md:w-0 md:p-0 md:overflow-hidden'}
        `}>
            <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col min-h-0">
                <div className="flex items-center justify-between px-1">
                     <TabsList className="grid w-full grid-cols-2">
                        <TabsTrigger value="layers" className="text-xs">图层 & 对象</TabsTrigger>
                        <TabsTrigger value="classes" className="text-xs">标签类别</TabsTrigger>
                    </TabsList>
                </div>
                
                {/* Image Enhancement Controls */}
                <Card className="shrink-0 mt-2 border-dashed shadow-sm">
                    <CardHeader className="py-2 px-3">
                        <CardTitle className="text-xs flex items-center gap-2 text-muted-foreground font-medium">
                            <Settings className="w-3 h-3"/> 图像增强
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="py-2 px-3 space-y-3">
                        <div className="space-y-1">
                            <div className="flex justify-between text-[10px] text-muted-foreground">
                                <span className="flex items-center gap-1"><Sun className="w-3 h-3"/> 亮度</span>
                                <span>{brightness}%</span>
                            </div>
                            <Slider 
                                defaultValue={[100]} 
                                max={200} 
                                min={0} 
                                step={1} 
                                value={[brightness]}
                                onValueChange={(v) => setBrightness(v[0])}
                                className="h-1.5"
                            />
                        </div>
                        <div className="space-y-1">
                            <div className="flex justify-between text-[10px] text-muted-foreground">
                                <span className="flex items-center gap-1"><Contrast className="w-3 h-3"/> 对比度</span>
                                <span>{contrast}%</span>
                            </div>
                            <Slider 
                                defaultValue={[100]} 
                                max={200} 
                                min={0} 
                                step={1} 
                                value={[contrast]}
                                onValueChange={(v) => setContrast(v[0])}
                                className="h-1.5"
                            />
                        </div>
                        <div className="flex items-center justify-between pt-1">
                            <span className="text-[10px] text-muted-foreground font-medium flex items-center gap-1">
                                <Crosshair className="w-3 h-3"/> 辅助准星
                            </span>
                            <input 
                                type="checkbox" 
                                checked={showCrosshair} 
                                onChange={(e) => setShowCrosshair(e.target.checked)}
                                className="w-3 h-3 accent-primary cursor-pointer"
                            />
                        </div>
                    </CardContent>
                </Card>

                <TabsContent value="layers" className="flex-1 flex flex-col min-h-0 mt-2 space-y-2">
                    {/* Annotation List */}
                    <Card className="flex-1 flex flex-col min-h-0 border-0 shadow-none bg-transparent">
                        <div className="px-2 py-1 font-medium flex justify-between items-center text-xs text-muted-foreground mb-1">
                            <div className="flex items-center gap-2">
                                <Layers className="w-3 h-3" /> 标注对象
                            </div>
                            <Badge variant="outline" className="text-[10px] h-4 px-1">{annotations.length}</Badge>
                        </div>
                        <div className="flex-1 overflow-auto p-1 space-y-1 pr-2 scrollbar-thin scrollbar-thumb-muted scrollbar-track-transparent">
                        {annotations.map((ann, idx) => (
                            <div
                            key={ann.id}
                            className={`group flex items-center gap-2 p-1.5 rounded text-xs cursor-pointer border transition-all ${
                                selectedId === ann.id ? "bg-accent border-primary shadow-sm" : "hover:bg-muted border-transparent"
                            } ${!ann.visible ? "opacity-50 grayscale" : ""}`}
                            onClick={() => setSelectedId(ann.id)}
                            >
                            <div className="text-muted-foreground text-[10px] font-mono w-4 text-center">{idx + 1}</div>
                            {ann.type === "polygon" ? <Pentagon className="w-3 h-3 text-muted-foreground" /> : <Square className="w-3 h-3 text-muted-foreground" />}
                            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: ann.color }} />
                            <span
                                className="flex-1 truncate font-medium"
                                title="双击编辑标签"
                                onDoubleClick={(e) => {
                                    e.stopPropagation();
                                    onEditAnnotationLabel(ann.id);
                                }}
                            >
                                {ann.label}
                            </span>
                            
                            <div className="flex items-center opacity-0 group-hover:opacity-100 transition-opacity">
                                <Button variant="ghost" size="icon" className="h-5 w-5" onClick={(e) => onToggleVisibility(ann.id, e)}>
                                    {ann.visible !== false ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
                                </Button>
                                <Button variant="ghost" size="icon" className="h-5 w-5 hover:text-destructive" onClick={(e) => { e.stopPropagation(); onDelete(); }}>
                                    <Trash2 className="w-3 h-3" />
                                </Button>
                            </div>
                            </div>
                        ))}
                        {annotations.length === 0 && (
                            <div className="flex flex-col items-center justify-center h-full text-muted-foreground space-y-2 p-8 opacity-50">
                            <Layers className="w-8 h-8 opacity-20" />
                            <span className="text-[10px]">暂无标注数据</span>
                            </div>
                        )}
                        </div>
                    </Card>
                </TabsContent>

                <TabsContent value="classes" className="flex-1 flex flex-col min-h-0 mt-2">
                    {/* Class Manager */}
                    <Card className="flex-1 flex flex-col min-h-0 border-0 shadow-none bg-transparent">
                        <div className="px-2 py-1 font-medium flex justify-between items-center text-xs text-muted-foreground mb-1">
                            <div className="flex items-center gap-2">
                                <Tag className="w-3 h-3" /> 标签类别
                            </div>
                            <Button variant="ghost" size="icon" className="h-5 w-5" title="新增标签类别" onClick={onClassAdd}>
                                <Plus className="w-3 h-3" />
                            </Button>
                        </div>
                        <div className="px-2 pb-1 text-[10px] text-muted-foreground">
                            双击类别名称可编辑（名称/颜色将同步更新相关标注）
                        </div>
                        <div className="p-1 space-y-1 flex-1 overflow-auto pr-2 scrollbar-thin scrollbar-thumb-muted scrollbar-track-transparent">
                            {classes.map((cls) => (
                            <div
                                key={cls.id}
                                    className={`group flex items-center gap-2 p-1.5 rounded text-xs cursor-pointer border transition-all duration-200 ${
                                        selectedClassId === cls.id ? "bg-primary/10 border-primary shadow-sm" : "hover:bg-muted border-transparent"
                                    }`}
                                    onClick={() => onClassSelect(cls.id)}
                                    onDoubleClick={(e) => {
                                        e.stopPropagation();
                                        onClassEdit(cls.id);
                                    }}
                                    title="双击编辑"
                                >
                                    <Badge variant="outline" className="w-4 h-4 flex items-center justify-center p-0 border-muted-foreground/30 text-[10px] font-mono text-muted-foreground">
                                        {cls.shortcut}
                                    </Badge>
                                    <div className="w-2.5 h-2.5 rounded-full shadow-sm" style={{ backgroundColor: cls.color }} />
                                    <span className="flex-1 font-medium">{cls.name}</span>
                                    <div
                                        className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity"
                                        onClick={(e) => e.stopPropagation()}
                                    >
                                        <Button
                                            variant="ghost"
                                            size="icon"
                                            className="h-6 w-6"
                                            title="编辑标签"
                                            onClick={() => onClassEdit(cls.id)}
                                        >
                                            <Pencil className="w-3 h-3" />
                                        </Button>
                                        <Button
                                            variant="ghost"
                                            size="icon"
                                            className="h-6 w-6 hover:text-destructive"
                                            title="删除标签"
                                            onClick={() => onClassDelete(cls.id)}
                                        >
                                            <Trash2 className="w-3 h-3" />
                                        </Button>
                                    </div>
                                    {selectedClassId === cls.id && <CheckCircle2 className="w-3 h-3 text-primary" />}
                                </div>
                            ))}
                        </div>
                    </Card>
                </TabsContent>
            </Tabs>
            
            {/* Stats / Progress */}
            <Card className="shrink-0 bg-muted/30 border-0">
                <CardContent className="p-3 space-y-3">
                    <div className="space-y-1.5">
                        <div className="flex justify-between text-[10px] text-muted-foreground">
                            <span>标注进度</span>
                            <span className="font-medium">
                                {completedCount}/{totalImages}（{Math.round(progressPct)}%）
                            </span>
                        </div>
                        <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                            <div className="h-full bg-primary transition-all duration-300" style={{ width: `${progressPct}%` }} />
                        </div>
                    </div>
                    {images && totalImages > 0 && (
                         <div className="text-[10px] text-muted-foreground truncate">
                            当前: {images[currentImageIndex]?.name}
                        </div>
                    )}
                </CardContent>
            </Card>
        </div>
    );
}
