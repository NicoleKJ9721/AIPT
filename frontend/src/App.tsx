import { lazy } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { MainLayout } from './layouts/MainLayout';
import { Toaster } from "@/components/ui/toaster";
import { ErrorBoundary } from "@/components/ErrorBoundary";

const Dashboard = lazy(() => import('./pages/Dashboard'));
const Projects = lazy(() => import('./pages/Projects'));
const ProjectWorkspace = lazy(() => import('./pages/ProjectWorkspace'));
const Datasets = lazy(() => import('./pages/Datasets'));
const Models = lazy(() => import('./pages/Models'));
const AnnotationStudio = lazy(() => import('./pages/AnnotationStudio'));
const AugmentationConfig = lazy(() => import('./pages/Augmentation'));
const Deployment = lazy(() => import('./pages/Deployment'));
const MultiModelPipeline = lazy(() => import('./pages/MultiModelPipeline'));

function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<MainLayout />}>
            <Route index element={<Dashboard />} />
            <Route path="projects" element={<Projects />} />
            <Route path="projects/:projectId" element={<ProjectWorkspace />} />
            <Route path="datasets" element={<Datasets />} />
            <Route path="models" element={<Models />} />
            <Route path="annotate" element={<AnnotationStudio />} />
            <Route path="augmentation" element={<AugmentationConfig />} />
            <Route path="pipeline" element={<MultiModelPipeline />} />
            <Route path="deploy" element={<Deployment />} />
            <Route path="settings" element={<Dashboard />} />
          </Route>
      </Routes>
      <Toaster />
    </BrowserRouter>
    </ErrorBoundary>
  );
}

export default App;
