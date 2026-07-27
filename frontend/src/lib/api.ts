import axios, { AxiosHeaders, type InternalAxiosRequestConfig } from 'axios';

// Create axios instance with base URL pointing to the proxy
export const api = axios.create({
  baseURL: '/api',
  timeout: 30000, // 30 seconds timeout for AI processing
});

// Attach lightweight auth headers (optional)
api.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  try {
    const user = localStorage.getItem("aipt_user") || "anonymous";
    const headers = AxiosHeaders.from(config.headers);
    headers.set("X-User", user);

    const apiKey = localStorage.getItem("aipt_api_key");
    if (apiKey) {
      headers.set("X-API-Key", apiKey);
    }
    config.headers = headers;
  } catch {
    // ignore (e.g. during SSR/build)
  }
  return config;
});

export interface DetectionResult {
  filename: string;
  detections: {
    bbox: [number, number, number, number]; // [x1, y1, x2, y2]
    confidence: number;
    class: string;
    class_id: number;
  }[];
}

export interface PredictOptions {
  conf?: number;
  iou?: number;
  max_det?: number;
  imgsz?: number;
  classes?: number[];
}

export interface TrainConfig {
  data: string;
  epochs: number;
  imgsz: number;
  batch: number;
  lr0?: number;
  model?: string;
  mode?: "transfer" | "incremental";
  task?: "detect" | "segment";
  output_name?: string | null;
  base_model_id?: string | null;
  device?: string | null;
  amp?: boolean | null;
  workers?: number | null;
  cache?: boolean | "ram" | null;
  project_id?: string;
  dataset_id?: string;
}

export interface TrainResponse {
  status: string;
  message: string;
  config: TrainConfig;
  job_id?: string;
}

export const aiService = {
  // Health check
  checkHealth: async () => {
    const response = await api.get('/health');
    return response.data;
  },

  // Inference
  predict: async (file: File, options?: PredictOptions): Promise<DetectionResult> => {
    const formData = new FormData();
    formData.append('file', file);
    const params: Record<string, string | number> = {};
    if (typeof options?.conf === "number" && Number.isFinite(options.conf)) params.conf = options.conf;
    if (typeof options?.iou === "number" && Number.isFinite(options.iou)) params.iou = options.iou;
    if (typeof options?.max_det === "number" && Number.isFinite(options.max_det)) params.max_det = Math.max(1, Math.floor(options.max_det));
    if (typeof options?.imgsz === "number" && Number.isFinite(options.imgsz)) params.imgsz = Math.max(32, Math.floor(options.imgsz));
    if (Array.isArray(options?.classes) && options.classes.length > 0) {
      const clean = options.classes.map((n) => Number(n)).filter((n) => Number.isFinite(n)).map((n) => Math.floor(n));
      if (clean.length > 0) params.classes = clean.join(",");
    }

    const response = await api.post<DetectionResult>('/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      params,
    });
    return response.data;
  },

  // Training
  train: async (config: TrainConfig): Promise<TrainResponse> => {
    const response = await api.post<TrainResponse>('/train', config);
    return response.data;
  }
};

export interface TrainJobStatusRecord {
  id: string;
  status: "queued" | "running" | "stopping" | "stopped" | "completed" | "failed";
  progress?: number | null;
  message?: string | null;
  error?: string | null;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  log_path: string;
  config: TrainConfig;
}

export interface TrainLogChunkRecord {
  offset: number;
  text: string;
  eof: boolean;
}

export interface TrainMetricsRecord {
  epochs: number[];
  series: Record<string, Array<number | null>>;
}

export interface TrainDiagnosticsRecord {
  python_executable: string;
  python_version: string;
  conda_env?: string | null;
  conda_prefix?: string | null;
  ultralytics?: string | null;
  torch?: string | null;
  cuda_available?: boolean | null;
  cuda_version?: string | null;
  cudnn_version?: string | null;
  nvidia_smi?: Record<string, string>[] | null;
}

export const trainService = {
  diagnostics: async (): Promise<TrainDiagnosticsRecord> => {
    const response = await api.get<ApiResponse<TrainDiagnosticsRecord>>("/train/diagnostics");
    if (!response.data.data) {
      throw new Error("Empty diagnostics response");
    }
    return response.data.data;
  },
  getJob: async (jobId: string): Promise<TrainJobStatusRecord> => {
    const response = await api.get<ApiResponse<TrainJobStatusRecord>>(`/train/jobs/${jobId}`);
    if (!response.data.data) {
      throw new Error("Empty train job response");
    }
    return response.data.data;
  },
  stopJob: async (jobId: string): Promise<TrainJobStatusRecord> => {
    const response = await api.post<ApiResponse<TrainJobStatusRecord>>(`/train/jobs/${jobId}/stop`);
    if (!response.data.data) {
      throw new Error("Empty stop job response");
    }
    return response.data.data;
  },
  resumeJob: async (jobId: string): Promise<TrainJobStatusRecord> => {
    const response = await api.post<ApiResponse<TrainJobStatusRecord>>(`/train/jobs/${jobId}/resume`);
    if (!response.data.data) {
      throw new Error("Empty resume job response");
    }
    return response.data.data;
  },
  getLogs: async (jobId: string, offset: number): Promise<TrainLogChunkRecord> => {
    const response = await api.get<ApiResponse<TrainLogChunkRecord>>(`/train/jobs/${jobId}/logs`, {
      params: { offset },
    });
    if (!response.data.data) {
      return { offset, text: "", eof: true };
    }
    return response.data.data;
  },
  getMetrics: async (jobId: string): Promise<TrainMetricsRecord> => {
    const response = await api.get<ApiResponse<TrainMetricsRecord>>(`/train/jobs/${jobId}/metrics`);
    return response.data.data || { epochs: [], series: {} };
  },
};

export interface TrainedModelRecord {
  id: string;
  project_id: string;
  dataset_id?: string | null;
  parent_model_id?: string | null;
  name: string;
  base_model: string;
  train_mode?: string;
  train_config?: Record<string, unknown> | null;
  metrics?: Record<string, unknown> | null;
  created_at: string;
}

export interface ModelEvaluationDetectionRecord {
  bbox: [number, number, number, number];
  confidence: number;
  class_name: string;
  class_id: number;
}

export interface ModelEvaluationItemRecord {
  image: ImageRecord;
  detections: ModelEvaluationDetectionRecord[];
}

export interface ModelEvaluationPageRecord {
  model_id: string;
  dataset_id: string;
  split: "train" | "val" | "test";
  page: number;
  limit: number;
  total: number;
  note?: string | null;
  items: ModelEvaluationItemRecord[];
}

export const modelService = {
  listByProject: async (projectId: string): Promise<TrainedModelRecord[]> => {
    const response = await api.get<ApiResponse<TrainedModelRecord[]>>(`/projects/${projectId}/models`);
    return response.data.data || [];
  },
  evaluate: async (
    modelId: string,
    params?: {
      split?: "train" | "val" | "test";
      page?: number;
      limit?: number;
      conf?: number;
      iou?: number;
      imgsz?: number;
      max_det?: number;
      device?: string;
      half?: boolean;
      augment?: boolean;
      end2end?: boolean;
      classes?: string;
    }
  ): Promise<ModelEvaluationPageRecord> => {
    const response = await api.get<ApiResponse<ModelEvaluationPageRecord>>(`/models/${modelId}/evaluation`, { params });
    return response.data.data;
  },
  delete: async (modelId: string): Promise<void> => {
    await api.delete<ApiResponse<null>>(`/models/${modelId}`);
  },
};

export interface ApiResponse<T> {
  code: number;
  message: string;
  data: T;
}

export type InferenceFormat = "tensorrt" | "openvino";
export type InferenceKind = "model" | "pipeline";

export interface InferenceSessionRecord {
  id: string;
  project_id: string;
  kind: InferenceKind;
  target_id: string;
  target_name: string;
  model_id?: string | null;
  format?: InferenceFormat | null;
  device?: string | null;
  end2end?: boolean;
  created_at: string;
  last_used_at?: string | null;
  cached_artifact?: string | null;
}

export interface InferenceStatusRecord {
  active_requests: number;
  sessions: InferenceSessionRecord[];
}

export interface InferenceSessionCreatePayload {
  project_id: string;
  kind?: InferenceKind;
  target_id?: string;
  // Backward-compat for older callers.
  model_id?: string;
  format?: InferenceFormat | null;
  device?: string | null;
  end2end?: boolean | null;
  half?: boolean | null;
  int8?: boolean | null;
  workspace?: number | null;
  batch?: number | null;
  imgsz?: number | null;
}

export interface InferencePredictionRecord {
  session_id: string;
  detections: ModelEvaluationDetectionRecord[];
  merged_detections?: ModelEvaluationDetectionRecord[];
  steps?: PipelineRunStepRecord[] | null;
  note?: string | null;
}

export const inferenceService = {
  status: async (): Promise<InferenceStatusRecord> => {
    const response = await api.get<ApiResponse<InferenceStatusRecord>>("/inference/status");
    return response.data.data;
  },
  createSession: async (payload: InferenceSessionCreatePayload): Promise<InferenceSessionRecord> => {
    const response = await api.post<ApiResponse<InferenceSessionRecord>>("/inference/sessions", payload);
    return response.data.data;
  },
  closeSession: async (sessionId: string): Promise<void> => {
    await api.delete<ApiResponse<null>>(`/inference/sessions/${sessionId}`);
  },
  predict: async (
    sessionId: string,
    file: File,
    params?: { conf?: number; iou?: number; imgsz?: number; max_det?: number; classes?: string; verbose?: boolean }
  ): Promise<InferencePredictionRecord> => {
    const formData = new FormData();
    formData.append("file", file);
    const response = await api.post<ApiResponse<InferencePredictionRecord>>(`/inference/sessions/${sessionId}/predict`, formData, {
      params,
      headers: { "Content-Type": "multipart/form-data" },
    });
    return response.data.data;
  },
};

export interface PipelineConnectorSpecRecord {
  source?: "prev_detections" | "prev_segments";
  min_conf?: number;
  classes?: Array<number | string> | null;
  padding?: number;
  max_regions?: number | null;
  on_empty?: "stop" | "fallback_full" | "skip";
}

export interface PipelineInputRoiSpecRecord {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface PipelineStepSpecRecord {
  id: string;
  title: string;
  model_id: string;
  conf?: number;
  iou?: number;
  max_det?: number;
  classes?: Array<number | string> | null;
  input_roi?: PipelineInputRoiSpecRecord | null;
  connector?: PipelineConnectorSpecRecord | null;
  crop?: boolean;
  crop_padding?: number;
  crop_max_regions?: number | null;
}

export interface PipelineRecord {
  id: string;
  project_id: string;
  name: string;
  description: string;
  steps?: PipelineStepSpecRecord[] | null;
  created_at: string;
  updated_at: string;
}

export interface PipelineCreatePayload {
  project_id: string;
  name: string;
  description?: string;
  steps: PipelineStepSpecRecord[];
}

export interface PipelineUpdatePayload {
  name?: string;
  description?: string;
  steps?: PipelineStepSpecRecord[];
}

export interface PipelineRunRequestPayload {
  project_id: string;
  steps: PipelineStepSpecRecord[];
}

export interface PipelineRunStepRecord {
  step_id: string;
  title: string;
  model_id: string;
  detections: ModelEvaluationDetectionRecord[];
  duration_ms?: number | null;
  note?: string | null;
}

export interface PipelineRunResultRecord {
  project_id: string;
  pipeline_id?: string | null;
  steps: PipelineRunStepRecord[];
  final_detections?: ModelEvaluationDetectionRecord[];
  merged_detections: ModelEvaluationDetectionRecord[];
  note?: string | null;
}

export const pipelineService = {
  listByProject: async (projectId: string): Promise<PipelineRecord[]> => {
    const response = await api.get<ApiResponse<PipelineRecord[]>>(`/projects/${projectId}/pipelines`);
    return response.data.data || [];
  },
  create: async (payload: PipelineCreatePayload): Promise<PipelineRecord> => {
    const response = await api.post<ApiResponse<PipelineRecord>>("/pipelines", payload);
    return response.data.data;
  },
  update: async (pipelineId: string, payload: PipelineUpdatePayload): Promise<PipelineRecord> => {
    const response = await api.put<ApiResponse<PipelineRecord>>(`/pipelines/${pipelineId}`, payload);
    return response.data.data;
  },
  delete: async (pipelineId: string): Promise<void> => {
    await api.delete<ApiResponse<null>>(`/pipelines/${pipelineId}`);
  },
  runAdhoc: async (payload: PipelineRunRequestPayload, file: File): Promise<PipelineRunResultRecord> => {
    const formData = new FormData();
    formData.append("payload", JSON.stringify(payload));
    formData.append("file", file);
    const response = await api.post<ApiResponse<PipelineRunResultRecord>>("/pipelines/run", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return response.data.data;
  },
  runSaved: async (pipelineId: string, file: File): Promise<PipelineRunResultRecord> => {
    const formData = new FormData();
    formData.append("file", file);
    const response = await api.post<ApiResponse<PipelineRunResultRecord>>(`/pipelines/${pipelineId}/run`, formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return response.data.data;
  },
};

export interface SmartDetectPayload {
  dataset_id: string;
  reference_image_id: string;
  label: string;
  color?: string;
  box: [number, number, number, number];
  scope?: "image" | "dataset";
  max_images?: number | null;
  threshold?: number;
  max_det_per_image?: number;
  min_distance?: number | null;
  dedup_iou?: number;
  only_unannotated?: boolean;
}

export interface SmartDetectImageRecord {
  image_id: string;
  created: number;
  skipped?: boolean;
  reason?: string | null;
}

export interface SmartDetectResultRecord {
  processed_images: number;
  created_annotations: number;
  skipped_images?: number;
  template_size: [number, number];
  images: SmartDetectImageRecord[];
}

export interface SmartSegmentPayload {
  image_id: string;
  point: [number, number];
  tolerance?: number;
  simplify?: number;
  engine?: string;
}

export interface SmartSegmentResultRecord {
  points: number[];
  area: number;
}

export const smartAnnotationService = {
  detect: async (payload: SmartDetectPayload): Promise<SmartDetectResultRecord> => {
    const response = await api.post<ApiResponse<SmartDetectResultRecord>>("/smart-annotation/detect", payload);
    return response.data.data;
  },
  segment: async (payload: SmartSegmentPayload): Promise<SmartSegmentResultRecord> => {
    const response = await api.post<ApiResponse<SmartSegmentResultRecord>>("/smart-annotation/segment", payload);
    return response.data.data;
  },
};

export interface DashboardTrainingSummaryRecord {
  running_jobs: number;
  last_job_id: string | null;
  last_status: string | null;
  last_progress: number | null;
  last_message: string | null;
  last_error: string | null;
}

export interface DashboardSummaryRecord {
  projects_total: number;
  datasets_total: number;
  images_total: number;
  images_annotated_total: number;
  images_pending_total: number;
  training: DashboardTrainingSummaryRecord;
}

export const dashboardService = {
  summary: async (): Promise<DashboardSummaryRecord> => {
    const response = await api.get<ApiResponse<DashboardSummaryRecord>>("/dashboard/summary");
    return response.data.data;
  },
};

export interface HardwareDeviceRecord {
  id: string;
  name: string;
  type: string;
  vendor?: string | null;
  memory?: string | null;
  cores?: number | null;
  compute_capability?: string | null;
  status?: string;
}

export const hardwareService = {
  list: async (): Promise<HardwareDeviceRecord[]> => {
    const response = await api.get<ApiResponse<HardwareDeviceRecord[]>>("/hardware");
    return response.data.data || [];
  },
};

export interface ProjectRecord {
  id: string;
  name: string;
  type: string;
  status: string;
  latest_commit: string;
  storage_root?: string | null;
  created_at: string;
  updated_at: string;
  images_count: number;
}

export interface ProjectCreatePayload {
  id?: string;
  name: string;
  type?: string;
  status?: string;
  latest_commit?: string;
}

export const projectService = {
  list: async (): Promise<ProjectRecord[]> => {
    const response = await api.get<ProjectRecord[]>("/projects");
    return response.data;
  },
  get: async (projectId: string): Promise<ProjectRecord> => {
    const response = await api.get<ProjectRecord>(`/projects/${projectId}`);
    return response.data;
  },
  create: async (payload: ProjectCreatePayload): Promise<ProjectRecord> => {
    const response = await api.post<ProjectRecord>("/projects", payload);
    return response.data;
  },
  update: async (
    projectId: string,
    payload: Partial<ProjectCreatePayload>
  ): Promise<ProjectRecord> => {
    const response = await api.patch<ProjectRecord>(`/projects/${projectId}`, payload);
    return response.data;
  },
  delete: async (projectId: string): Promise<void> => {
    await api.delete(`/projects/${projectId}`);
  },
};

export interface ImageRecord {
  id: string;
  project_id: string;
  filename: string;
  source_url: string | null;
  dataset_id?: string | null;
  dataset_file_id?: string | null;
  width: number | null;
  height: number | null;
  created_at: string;
  annotations_count: number;
}

export interface AnnotationRecord {
  id: string;
  image_id: string;
  type: "rect" | "polygon";
  label: string;
  color: string;
  visible: boolean;
  x: number | null;
  y: number | null;
  width: number | null;
  height: number | null;
  points: number[] | null;
  created_at: string;
  updated_at: string;
}

export interface AnnotationCreatePayload {
  id?: string | null;
  type: "rect" | "polygon";
  label: string;
  color?: string;
  visible?: boolean;
  x?: number | null;
  y?: number | null;
  width?: number | null;
  height?: number | null;
  points?: number[] | null;
}

export const imageService = {
  list: async (projectId: string, params?: { dataset_id?: string }): Promise<ImageRecord[]> => {
    const response = await api.get<ImageRecord[]>(`/projects/${projectId}/images`, { params });
    return response.data;
  },
  delete: async (imageId: string): Promise<void> => {
    await api.delete<ApiResponse<null>>(`/images/${imageId}`);
  },
  edit: async (
    imageId: string,
    payload: { rotate?: 90 | 180 | 270; crop?: { x: number; y: number; width: number; height: number } }
  ): Promise<ImageRecord> => {
    const response = await api.post<ApiResponse<ImageRecord>>(`/images/${imageId}/edit`, payload);
    return response.data.data;
  },
};

export const annotationService = {
  listByImage: async (imageId: string): Promise<AnnotationRecord[]> => {
    const response = await api.get<AnnotationRecord[]>(`/images/${imageId}/annotations`);
    return response.data;
  },
  replaceByImage: async (imageId: string, payload: AnnotationCreatePayload[]): Promise<AnnotationRecord[]> => {
    const response = await api.put<AnnotationRecord[]>(`/images/${imageId}/annotations`, payload);
    return response.data;
  },
};

export interface LabelRenamePayload {
  from_label: string;
  to_label: string;
  dataset_id?: string | null;
}

export interface LabelRenameResult {
  updated: number;
}

export interface LabelClassRecord {
  id: string;
  project_id: string;
  name: string;
  color: string;
  shortcut: string;
  created_at: string;
  updated_at: string;
}

export interface LabelClassCreatePayload {
  id?: string;
  name: string;
  color?: string;
  shortcut?: string | null;
}

export interface LabelClassUpdatePayload {
  name?: string | null;
  color?: string | null;
  shortcut?: string | null;
}

export interface LabelClassUpdateResult {
  label: LabelClassRecord;
  updated: number;
}

export const labelService = {
  list: async (projectId: string): Promise<LabelClassRecord[]> => {
    const response = await api.get<ApiResponse<LabelClassRecord[]>>(`/projects/${projectId}/labels`);
    return response.data.data || [];
  },
  create: async (projectId: string, payload: LabelClassCreatePayload): Promise<LabelClassRecord> => {
    const response = await api.post<ApiResponse<LabelClassRecord>>(`/projects/${projectId}/labels`, payload);
    if (!response.data.data) throw new Error(response.data.message || "Failed to create label");
    return response.data.data;
  },
  update: async (
    projectId: string,
    labelId: string,
    payload: LabelClassUpdatePayload
  ): Promise<LabelClassUpdateResult> => {
    const response = await api.put<ApiResponse<LabelClassUpdateResult>>(`/projects/${projectId}/labels/${labelId}`, payload);
    if (!response.data.data) throw new Error(response.data.message || "Failed to update label");
    return response.data.data;
  },
  delete: async (projectId: string, labelId: string): Promise<void> => {
    await api.delete(`/projects/${projectId}/labels/${labelId}`);
  },
  rename: async (projectId: string, payload: LabelRenamePayload): Promise<LabelRenameResult> => {
    const response = await api.post<ApiResponse<LabelRenameResult>>(`/projects/${projectId}/labels/rename`, payload);
    return response.data.data || { updated: 0 };
  },
};

export interface DatasetSplits {
  train: number;
  val: number;
  test: number;
}

export interface DatasetRecord {
  id: string;
  project_id: string | null;
  name: string;
  version: string;
  description: string;
  status: string;
  owner: string;
  is_public: boolean;
  tags: string[] | null;
  splits: DatasetSplits;
  created_at: string;
  updated_at: string;
  file_count: number;
  total_size_bytes: number;
}

export interface DatasetFileRecord {
  id: string;
  dataset_id: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  sha256: string;
  created_at: string;
}

export interface DatasetImageStatsRecord {
  dataset_id: string;
  image_count: number;
  sized_image_count: number;
  avg_width: number | null;
  avg_height: number | null;
  total_pixels: number;
}

export interface DatasetCreatePayload {
  id?: string;
  project_id?: string | null;
  name: string;
  version?: string | null;
  description?: string;
  tags?: string[] | null;
  is_public?: boolean;
  splits?: DatasetSplits;
}

export interface DatasetUpdatePayload {
  name?: string;
  version?: string;
  description?: string;
  status?: string;
  tags?: string[] | null;
  is_public?: boolean;
  splits?: DatasetSplits;
}

export const datasetService = {
  list: async (params?: {
    q?: string;
    status?: string;
    owner?: string;
    project_id?: string;
    limit?: number;
    offset?: number;
  }): Promise<DatasetRecord[]> => {
    const response = await api.get<ApiResponse<DatasetRecord[]>>("/datasets", { params });
    return response.data.data || [];
  },
  get: async (datasetId: string): Promise<DatasetRecord> => {
    const response = await api.get<ApiResponse<DatasetRecord>>(`/datasets/${datasetId}`);
    return response.data.data;
  },
  create: async (payload: DatasetCreatePayload): Promise<DatasetRecord> => {
    const response = await api.post<ApiResponse<DatasetRecord>>("/datasets", payload);
    return response.data.data;
  },
  clone: async (
    datasetId: string,
    payload: Partial<Omit<DatasetCreatePayload, "project_id">> & { project_id?: never }
  ): Promise<DatasetRecord> => {
    const response = await api.post<ApiResponse<DatasetRecord>>(`/datasets/${datasetId}/clone`, payload);
    return response.data.data;
  },
  update: async (datasetId: string, payload: DatasetUpdatePayload): Promise<DatasetRecord> => {
    const response = await api.patch<ApiResponse<DatasetRecord>>(`/datasets/${datasetId}`, payload);
    return response.data.data;
  },
  delete: async (datasetId: string): Promise<void> => {
    await api.delete<ApiResponse<null>>(`/datasets/${datasetId}`);
  },
  listFiles: async (datasetId: string): Promise<DatasetFileRecord[]> => {
    const response = await api.get<ApiResponse<DatasetFileRecord[]>>(`/datasets/${datasetId}/files`);
    return response.data.data || [];
  },
  uploadFiles: async (datasetId: string, files: File[]): Promise<DatasetFileRecord[]> => {
    const formData = new FormData();
    files.forEach((f) => formData.append("files", f));
    const response = await api.post<ApiResponse<DatasetFileRecord[]>>(
      `/datasets/${datasetId}/files`,
      formData,
      { headers: { "Content-Type": "multipart/form-data" } }
    );
    return response.data.data || [];
  },
  deleteFile: async (datasetId: string, fileId: string): Promise<void> => {
    await api.delete<ApiResponse<null>>(`/datasets/${datasetId}/files/${fileId}`);
  },
  downloadFileBlob: async (datasetId: string, fileId: string): Promise<Blob> => {
    const response = await api.get(`/datasets/${datasetId}/files/${fileId}/download`, {
      responseType: "blob",
    });
    return response.data;
  },
  downloadDatasetZipBlob: async (datasetId: string): Promise<Blob> => {
    const response = await api.get(`/datasets/${datasetId}/download`, { responseType: "blob" });
    return response.data;
  },
  getImageStats: async (datasetId: string): Promise<DatasetImageStatsRecord> => {
    const response = await api.get<ApiResponse<DatasetImageStatsRecord>>(`/datasets/${datasetId}/stats`);
    if (!response.data.data) {
      return {
        dataset_id: datasetId,
        image_count: 0,
        sized_image_count: 0,
        avg_width: null,
        avg_height: null,
        total_pixels: 0,
      };
    }
    return response.data.data;
  },
};

export interface SystemSettings {
  projects_root_dir: string;
  resources_root_dir: string;
  recent_projects_root_dirs: string[];
  recent_resources_root_dirs: string[];
  default_model_resource_id: string | null;
}

export interface SystemSettingsUpdatePayload {
  projects_root_dir?: string | null;
  resources_root_dir?: string | null;
  default_model_resource_id?: string | null;
}

export interface DirectoryPickerRequest {
  title?: string | null;
  initial_dir?: string | null;
}

export const systemService = {
  getSettings: async (): Promise<SystemSettings> => {
    const response = await api.get<ApiResponse<SystemSettings>>("/system/settings");
    if (!response.data.data) {
      throw new Error("Empty system settings response");
    }
    return response.data.data;
  },
  updateSettings: async (payload: SystemSettingsUpdatePayload): Promise<SystemSettings> => {
    const response = await api.put<ApiResponse<SystemSettings>>("/system/settings", payload);
    if (!response.data.data) {
      throw new Error("Empty system settings response");
    }
    return response.data.data;
  },
  selectDirectory: async (payload: DirectoryPickerRequest): Promise<string | null> => {
    const response = await api.post<ApiResponse<string>>("/system/dialogs/select-directory", payload);
    return response.data.data ?? null;
  },
};
