// Legacy mock page kept for reference.
import { useState } from "react";
import { Upload, FileImage, Trash2, FolderPlus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

// Mock Data
const datasets = [
  { id: 1, name: "train_v1.zip", size: "124 MB", date: "2023-12-28", count: 450 },
  { id: 2, name: "val_v1.zip", size: "32 MB", date: "2023-12-28", count: 120 },
];

export default function Datasets() {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    // Handle file drop logic here
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">数据集管理</h1>
          <p className="text-muted-foreground mt-2">
            上传、预览和管理您的训练数据
          </p>
        </div>
        <div className="flex gap-2">
            <Button variant="outline" className="gap-2">
                <FolderPlus className="w-4 h-4" /> 新建版本
            </Button>
            <Button className="gap-2">
                <Upload className="w-4 h-4" /> 上传数据
            </Button>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        {/* Upload Area */}
        <Card className="col-span-2">
          <CardHeader>
            <CardTitle>上传数据</CardTitle>
          </CardHeader>
          <CardContent>
            <div
              className={`border-2 border-dashed rounded-lg p-12 flex flex-col items-center justify-center transition-colors ${
                isDragging
                  ? "border-primary bg-primary/5"
                  : "border-muted-foreground/25 hover:border-primary/50"
              }`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mb-4">
                <Upload className="w-8 h-8 text-muted-foreground" />
              </div>
              <h3 className="text-lg font-semibold mb-2">拖拽文件到此处</h3>
              <p className="text-sm text-muted-foreground mb-6 text-center">
                支持 JPG, PNG 格式图片 <br /> 或包含标注信息的 ZIP 压缩包
              </p>
              <Button>选择文件</Button>
            </div>
          </CardContent>
        </Card>

        {/* Dataset Stats */}
        <Card>
            <CardHeader>
                <CardTitle>当前版本统计 (v1.0)</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
                <div className="flex justify-between items-center pb-4 border-b">
                    <span className="text-muted-foreground">总图片数</span>
                    <span className="font-bold text-xl">570</span>
                </div>
                <div className="flex justify-between items-center pb-4 border-b">
                    <span className="text-muted-foreground">已标注</span>
                    <span className="font-bold text-xl text-green-600">570</span>
                </div>
                <div className="flex justify-between items-center pb-4 border-b">
                    <span className="text-muted-foreground">类别数</span>
                    <span className="font-bold text-xl">4</span>
                </div>
                <div className="space-y-2">
                    <div className="text-sm font-medium">数据集划分</div>
                    <div className="flex h-4 rounded-full overflow-hidden bg-muted">
                        <div className="bg-blue-500 w-[70%]" title="训练集 70%" />
                        <div className="bg-green-500 w-[20%]" title="验证集 20%" />
                        <div className="bg-yellow-500 w-[10%]" title="测试集 10%" />
                    </div>
                    <div className="flex text-xs text-muted-foreground justify-between">
                        <span>Train 70%</span>
                        <span>Val 20%</span>
                        <span>Test 10%</span>
                    </div>
                </div>
            </CardContent>
        </Card>
      </div>

      {/* Dataset Preview List */}
      <Card>
        <CardHeader>
          <CardTitle>数据预览</CardTitle>
        </CardHeader>
        <CardContent>
            <div className="space-y-4">
                {datasets.map((file) => (
                    <div key={file.id} className="flex items-center justify-between p-4 border rounded-lg hover:bg-muted/50 transition-colors">
                        <div className="flex items-center gap-4">
                            <div className="w-10 h-10 bg-blue-100 text-blue-600 rounded flex items-center justify-center">
                                <FileImage className="w-5 h-5" />
                            </div>
                            <div>
                                <div className="font-medium">{file.name}</div>
                                <div className="text-xs text-muted-foreground flex gap-2">
                                    <span>{file.size}</span>
                                    <span>•</span>
                                    <span>{file.count} 张图片</span>
                                    <span>•</span>
                                    <span>{file.date}</span>
                                </div>
                            </div>
                        </div>
                        <div className="flex items-center gap-2">
                             <Badge variant="outline">已处理</Badge>
                             <Button variant="ghost" size="icon" className="text-destructive hover:text-destructive hover:bg-destructive/10">
                                <Trash2 className="w-4 h-4" />
                             </Button>
                        </div>
                    </div>
                ))}
            </div>
        </CardContent>
      </Card>
    </div>
  );
}
