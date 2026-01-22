import { Outlet } from "react-router-dom";
import { Suspense } from "react";
import { Sidebar } from "@/components/common/Sidebar";
import { Header } from "@/components/common/Header";

export function MainLayout() {
  return (
    <div className="flex h-screen w-full bg-slate-50/50 dark:bg-slate-950 overflow-hidden font-sans">
      <Sidebar />
      <div className="flex-1 flex flex-col h-full overflow-hidden relative">
        <Header />
        <main className="flex-1 overflow-auto p-6 scroll-smooth">
          <div className="max-w-7xl mx-auto h-full flex flex-col">
            <Suspense fallback={
              <div className="flex items-center justify-center h-full">
                <div className="flex flex-col items-center gap-4">
                  <div className="animate-spin rounded-full h-10 w-10 border-4 border-primary border-t-transparent"></div>
                  <span className="text-sm text-muted-foreground animate-pulse">Loading modules...</span>
                </div>
              </div>
            }>
              <Outlet />
            </Suspense>
          </div>
        </main>
      </div>
    </div>
  );
}
