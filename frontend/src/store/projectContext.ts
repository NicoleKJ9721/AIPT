import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

type ProjectContextState = {
  projectId: string | null;
  projectName: string | null;
  datasetId: string | null;
  datasetName: string | null;
  setProject: (projectId: string, projectName?: string | null) => void;
  setDataset: (datasetId: string, datasetName?: string | null) => void;
  clearProject: () => void;
  clearDataset: () => void;
};

export const useProjectContext = create<ProjectContextState>()(
  persist(
    (set) => ({
      projectId: null,
      projectName: null,
      datasetId: null,
      datasetName: null,
      setProject: (projectId, projectName) =>
        set((state) => {
          if (state.projectId === projectId) {
            return { projectId, projectName: projectName ?? state.projectName ?? null };
          }
          return {
            projectId,
            projectName: projectName ?? null,
            datasetId: null,
            datasetName: null,
          };
        }),
      setDataset: (datasetId, datasetName) => set({ datasetId, datasetName: datasetName ?? null }),
      clearProject: () => set({ projectId: null, projectName: null, datasetId: null, datasetName: null }),
      clearDataset: () => set({ datasetId: null, datasetName: null }),
    }),
    {
      name: "aipt_project_context",
      storage: createJSONStorage(() => localStorage),
      version: 2,
      migrate: (persistedState, version) => {
        if (version === 1 && persistedState && typeof persistedState === "object") {
          const state = persistedState as Partial<ProjectContextState>;
          return {
            projectId: state.projectId ?? null,
            projectName: state.projectName ?? null,
            datasetId: null,
            datasetName: null,
          } as ProjectContextState;
        }
        return persistedState as ProjectContextState;
      },
    }
  )
);
