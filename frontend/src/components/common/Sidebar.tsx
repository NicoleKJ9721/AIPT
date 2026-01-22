import { LayoutDashboard, Box, PlaySquare, Settings, LogOut, Wand2, Rocket, ScanEye, Cpu, GitMerge } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";

const sidebarItems = [
  { icon: LayoutDashboard, label: "工作台", href: "/" },
  { icon: Box, label: "项目管理", href: "/projects" },
  { icon: ScanEye, label: "智能标注", href: "/annotate" },
  { icon: Wand2, label: "数据增强", href: "/augmentation" },
  { icon: PlaySquare, label: "模型训练", href: "/models" },
  { icon: GitMerge, label: "多模型串联检测", href: "/pipeline" },
  { icon: Rocket, label: "部署推荐", href: "/deploy" },
];

export function Sidebar() {
  const location = useLocation();

  return (
    <div className="flex flex-col h-full w-72 bg-slate-900 text-slate-50 shadow-xl transition-all duration-300">
      <div className="p-6 border-b border-slate-800">
        <div className="flex items-center gap-3 font-bold text-xl tracking-tight">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center text-white shadow-lg shadow-blue-500/20">
            <Cpu className="w-6 h-6" />
          </div>
          <div className="flex flex-col leading-none">
            <span className="text-white">AIPT</span>
            <span className="text-xs text-slate-400 font-normal mt-1">工业视觉平台</span>
          </div>
        </div>
      </div>

      <nav className="flex-1 p-4 space-y-1.5 overflow-y-auto">
        {sidebarItems.map((item) => {
          const isActive = location.pathname === item.href;
          return (
            <Link
              key={item.href}
              to={item.href}
              className={cn(
                "group flex items-center gap-3 px-4 py-3.5 rounded-xl text-sm font-medium transition-all duration-200",
                isActive
                  ? "bg-blue-600 text-white shadow-md shadow-blue-900/20"
                  : "text-slate-400 hover:bg-slate-800 hover:text-white hover:translate-x-1"
              )}
            >
              <item.icon className={cn("w-5 h-5 transition-colors", isActive ? "text-white" : "text-slate-400 group-hover:text-white")} />
              {item.label}
              {isActive && <div className="ml-auto w-1.5 h-1.5 bg-white rounded-full animate-pulse" />}
            </Link>
          );
        })}
      </nav>

      <div className="p-4 border-t border-slate-800 space-y-2 bg-slate-900/50">
        <Link
          to="/settings"
          className="flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium text-slate-400 hover:bg-slate-800 hover:text-white transition-all"
        >
          <Settings className="w-5 h-5" />
          系统设置
        </Link>
        <button
          type="button"
          className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium text-red-400 hover:bg-red-950/30 hover:text-red-300 transition-all"
          onClick={() => {
            try {
              localStorage.removeItem("aipt_user");
              localStorage.removeItem("aipt_api_key");
            } catch {
              // ignore
            }
            window.location.href = "/";
          }}
        >
          <LogOut className="w-5 h-5" />
          退出登录
        </button>
      </div>
    </div>
  );
}
