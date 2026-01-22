import { useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { isAxiosError } from "axios";
import { Code2, Copy, Cpu, RefreshCw, Server, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "@/components/ui/use-toast";
import {
  inferenceService,
  modelService,
  projectService,
  type InferenceFormat,
  type InferenceSessionRecord,
  type InferenceStatusRecord,
  type ProjectRecord,
  type TrainedModelRecord,
} from "@/lib/api";
import { useProjectContext } from "@/store/projectContext";

const FORMAT_HINT: Record<InferenceFormat, string> = {
  openvino: "OpenVINO 仅支持 Intel CPU 推理（建议用于 CPU-only 场景）",
  tensorrt: "TensorRT 仅支持 NVIDIA 显卡推理（建议用于高吞吐/低延迟场景）",
};

const GPU_ESTIMATES: Array<{ gpu: string; vram: string; perf: string; note: string }> = [
  { gpu: "RTX 2080", vram: "8GB", perf: "1.0×", note: "入门 GPU（建议小模型/低并发）" },
  { gpu: "RTX 3080", vram: "10–12GB", perf: "≈ 1.8×", note: "性价比高（中等并发）" },
  { gpu: "RTX 3090", vram: "24GB", perf: "≈ 2.0×", note: "大显存（更适合大模型/更大 batch）" },
  { gpu: "RTX 4060 Ti", vram: "8–16GB", perf: "≈ 1.6×", note: "功耗低（中低并发）" },
  { gpu: "RTX 4090", vram: "24GB", perf: "≈ 3.0×", note: "旗舰（高并发/高吞吐）" },
  { gpu: "RTX 5090", vram: "待实测", perf: "待实测", note: "新卡请以实测为准" },
];

type InferenceParams = {
  conf: string;
  iou: string;
  imgsz: string;
  max_det: string;
  classes: string;
};

const DEFAULT_PARAMS: InferenceParams = {
  conf: "0.25",
  iou: "0.7",
  imgsz: "640",
  max_det: "50",
  classes: "",
};

function clampNum(n: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, n));
}

function toNumberOrUndefined(v: string): number | undefined {
  const n = Number(v);
  if (!Number.isFinite(n)) return undefined;
  return n;
}

function safeJsonStringify(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function buildCodeSnippet(
  lang: "python" | "cpp" | "csharp",
  opts: {
    baseUrl: string;
    projectId: string;
    modelId: string;
    format: InferenceFormat;
    device?: string;
    params: InferenceParams;
  }
): string {
  const conf = clampNum(toNumberOrUndefined(opts.params.conf) ?? 0.25, 0, 1);
  const iou = clampNum(toNumberOrUndefined(opts.params.iou) ?? 0.7, 0, 1);
  const imgsz = Math.max(32, Math.round(toNumberOrUndefined(opts.params.imgsz) ?? 640));
  const maxDet = Math.max(1, Math.round(toNumberOrUndefined(opts.params.max_det) ?? 50));
  const classes = (opts.params.classes || "").trim();

  const createPayload = {
    project_id: opts.projectId,
    model_id: opts.modelId,
    format: opts.format,
    ...(opts.device ? { device: opts.device } : {}),
  };

  const predictParams = {
    conf,
    iou,
    imgsz,
    max_det: maxDet,
    ...(classes ? { classes } : {}),
  };

  if (lang === "python") {
    return `import time
import requests

BASE_URL = "${opts.baseUrl}"

# 1) 申请推理会话（同一模型/参数组合只会首次加载一次，后续复用）
create_payload = ${safeJsonStringify(createPayload)}
r0 = time.perf_counter()
r = requests.post(f"{BASE_URL}/inference/sessions", json=create_payload, timeout=1200)
r.raise_for_status()
session_id = r.json()["data"]["id"]
print("create_session:", round(time.perf_counter() - r0, 3), "s", "session_id =", session_id)

# （可选）再次申请同一会话：会直接命中缓存，不会重复加载模型
r1 = time.perf_counter()
r2 = requests.post(f"{BASE_URL}/inference/sessions", json=create_payload, timeout=1200)
r2.raise_for_status()
session_id2 = r2.json()["data"]["id"]
print("create_session_again:", round(time.perf_counter() - r1, 3), "s", "session_id =", session_id2)
assert session_id2 == session_id

# 2) 发送图片进行推理
params = ${safeJsonStringify(predictParams)}
for img_path in ["test.jpg"]:
    with open(img_path, "rb") as f:
        files = {"file": (img_path, f, "image/jpeg")}
        rr = requests.post(f"{BASE_URL}/inference/sessions/{session_id}/predict", files=files, params=params, timeout=1200)
        rr.raise_for_status()
        print("predict:", img_path)
        print(rr.json()["data"])

# 3) （建议）用完后关闭会话：否则平台会禁止训练（避免抢占 GPU/CPU 资源）
requests.delete(f"{BASE_URL}/inference/sessions/{session_id}", timeout=30).raise_for_status()`;
  }

  if (lang === "csharp") {
    const deviceLine = opts.device ? `, device = "${opts.device}"` : "";
    return `using System;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

class Program {
  static async Task Main() {
    var baseUrl = "${opts.baseUrl}";
    using var http = new HttpClient();

    // 1) 申请推理会话（同一模型/参数组合只会首次加载一次，后续复用）
    var createPayload = new { project_id = "${opts.projectId}", model_id = "${opts.modelId}", format = "${opts.format}"${deviceLine} };
    var createJson = JsonSerializer.Serialize(createPayload);
    var sw = Stopwatch.StartNew();
    var createRes = await http.PostAsync($"{baseUrl}/inference/sessions", new StringContent(createJson, Encoding.UTF8, "application/json"));
    createRes.EnsureSuccessStatusCode();
    var createBody = await createRes.Content.ReadAsStringAsync();
    var sessionId = JsonDocument.Parse(createBody).RootElement.GetProperty("data").GetProperty("id").GetString();
    Console.WriteLine($"create_session: {sw.ElapsedMilliseconds} ms, session_id = {sessionId}");

    // （可选）再次申请同一会话：应直接命中缓存（不重复加载模型）
    sw.Restart();
    var createRes2 = await http.PostAsync($"{baseUrl}/inference/sessions", new StringContent(createJson, Encoding.UTF8, "application/json"));
    createRes2.EnsureSuccessStatusCode();
    var createBody2 = await createRes2.Content.ReadAsStringAsync();
    var sessionId2 = JsonDocument.Parse(createBody2).RootElement.GetProperty("data").GetProperty("id").GetString();
    Console.WriteLine($"create_session_again: {sw.ElapsedMilliseconds} ms, session_id = {sessionId2}");

    // 2) 推理
    var imgPath = "test.jpg";
    using var form = new MultipartFormDataContent();
    var bytes = await File.ReadAllBytesAsync(imgPath);
    var fileContent = new ByteArrayContent(bytes);
    fileContent.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");
    form.Add(fileContent, "file", Path.GetFileName(imgPath));

    var qs = "?conf=${conf}&iou=${iou}&imgsz=${imgsz}&max_det=${maxDet}${classes ? `&classes=${encodeURIComponent(classes)}` : ""}";
    var predRes = await http.PostAsync($"{baseUrl}/inference/sessions/{sessionId}/predict{qs}", form);
    predRes.EnsureSuccessStatusCode();
    Console.WriteLine(await predRes.Content.ReadAsStringAsync());

    // 3) （建议）关闭会话：否则平台会禁止训练
    var closeRes = await http.DeleteAsync($"{baseUrl}/inference/sessions/{sessionId}");
    closeRes.EnsureSuccessStatusCode();
  }
}`;
  }

  // cpp
  return `// C++ 示例（libcurl + nlohmann/json）
// - 申请会话：POST ${opts.baseUrl}/inference/sessions
// - 推理：POST ${opts.baseUrl}/inference/sessions/{session_id}/predict (multipart/form-data)
// - 关闭会话：DELETE ${opts.baseUrl}/inference/sessions/{session_id}
//
// 依赖：
//   - libcurl
//   - nlohmann/json (single-header)
//
// 提示：
//   - 同一模型/参数组合在服务端会话缓存后，再次申请会话不会重复加载模型（响应更快）。
//   - 推理期间平台会禁止训练（避免抢占 GPU/CPU 资源），用完请关闭会话。

#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

static size_t write_cb(char* ptr, size_t size, size_t nmemb, void* userdata) {
  auto* out = static_cast<std::string*>(userdata);
  out->append(ptr, size * nmemb);
  return size * nmemb;
}

static nlohmann::json parse_json_or_throw(const std::string& body) {
  try {
    return nlohmann::json::parse(body);
  } catch (...) {
    throw std::runtime_error("failed to parse json: " + body);
  }
}

static nlohmann::json http_post_json(const std::string& url, const nlohmann::json& payload) {
  CURL* curl = curl_easy_init();
  if (!curl) throw std::runtime_error("curl_easy_init failed");

  std::string resp;
  auto body = payload.dump();

  struct curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");
  // 可选：headers = curl_slist_append(headers, "X-User: alice");
  // 可选：headers = curl_slist_append(headers, "X-API-Key: YOUR_API_KEY");

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)body.size());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 1200L);

  auto rc = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  if (rc != CURLE_OK) throw std::runtime_error(std::string("curl error: ") + curl_easy_strerror(rc));
  if (http_code < 200 || http_code >= 300) throw std::runtime_error("HTTP " + std::to_string(http_code) + ": " + resp);
  return parse_json_or_throw(resp);
}

static nlohmann::json http_delete(const std::string& url) {
  CURL* curl = curl_easy_init();
  if (!curl) throw std::runtime_error("curl_easy_init failed");

  std::string resp;
  struct curl_slist* headers = nullptr;
  // 可选：headers = curl_slist_append(headers, "X-User: alice");
  // 可选：headers = curl_slist_append(headers, "X-API-Key: YOUR_API_KEY");

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 1200L);

  auto rc = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  if (rc != CURLE_OK) throw std::runtime_error(std::string("curl error: ") + curl_easy_strerror(rc));
  if (http_code < 200 || http_code >= 300) throw std::runtime_error("HTTP " + std::to_string(http_code) + ": " + resp);
  return resp.empty() ? nlohmann::json::object() : parse_json_or_throw(resp);
}

static nlohmann::json http_post_multipart_file(const std::string& url, const std::string& file_path) {
  CURL* curl = curl_easy_init();
  if (!curl) throw std::runtime_error("curl_easy_init failed");

  std::string resp;
  curl_mime* mime = curl_mime_init(curl);
  curl_mimepart* part = curl_mime_addpart(mime);
  curl_mime_name(part, "file");
  curl_mime_filedata(part, file_path.c_str());

  struct curl_slist* headers = nullptr;
  // 可选：headers = curl_slist_append(headers, "X-User: alice");
  // 可选：headers = curl_slist_append(headers, "X-API-Key: YOUR_API_KEY");

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 1200L);

  auto rc = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_slist_free_all(headers);
  curl_mime_free(mime);
  curl_easy_cleanup(curl);

  if (rc != CURLE_OK) throw std::runtime_error(std::string("curl error: ") + curl_easy_strerror(rc));
  if (http_code < 200 || http_code >= 300) throw std::runtime_error("HTTP " + std::to_string(http_code) + ": " + resp);
  return parse_json_or_throw(resp);
}

int main() {
  const std::string base = "${opts.baseUrl}";

  curl_global_init(CURL_GLOBAL_DEFAULT);
  const auto create_payload = nlohmann::json::parse(R"JSON(${safeJsonStringify(createPayload)})JSON");

  auto t0 = std::chrono::steady_clock::now();
  auto create1 = http_post_json(base + "/inference/sessions", create_payload);
  auto t1 = std::chrono::steady_clock::now();
  const std::string session_id = create1["data"]["id"].get<std::string>();
  std::cout << "create_session: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
            << " ms, session_id=" << session_id << std::endl;

  // （可选）再次申请同一会话：应直接命中缓存
  auto t2 = std::chrono::steady_clock::now();
  auto create2 = http_post_json(base + "/inference/sessions", create_payload);
  auto t3 = std::chrono::steady_clock::now();
  std::cout << "create_session_again: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
            << " ms, session_id=" << create2["data"]["id"].get<std::string>() << std::endl;

  // 推理参数（query string）
  // conf=${conf}, iou=${iou}, imgsz=${imgsz}, max_det=${maxDet}${classes ? `, classes=${classes}` : ""}
  const std::string qs = "?conf=${conf}&iou=${iou}&imgsz=${imgsz}&max_det=${maxDet}${classes ? `&classes=${encodeURIComponent(classes)}` : ""}";
  auto pred = http_post_multipart_file(base + "/inference/sessions/" + session_id + "/predict" + qs, "test.jpg");
  std::cout << pred.dump(2) << std::endl;

  // 关闭会话
  http_delete(base + "/inference/sessions/" + session_id);
  curl_global_cleanup();
  return 0;
}`;
}

export default function Deployment() {
  const [searchParams, setSearchParams] = useSearchParams();
  const projectIdFromQuery = (searchParams.get("project_id") || "").trim() || null;
  const modelIdFromQuery = (searchParams.get("model_id") || "").trim() || null;

  const projectIdInContext = useProjectContext((s) => s.projectId);
  const setProjectContext = useProjectContext((s) => s.setProject);

  const [projects, setProjects] = useState<ProjectRecord[]>([]);
  const [models, setModels] = useState<TrainedModelRecord[]>([]);
  const [isLoadingProjects, setIsLoadingProjects] = useState(true);
  const [isLoadingModels, setIsLoadingModels] = useState(false);

  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(projectIdFromQuery || projectIdInContext || null);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(modelIdFromQuery || null);

  const [format, setFormat] = useState<InferenceFormat>("tensorrt");
  const [device, setDevice] = useState<string>("0");
  const [params, setParams] = useState<InferenceParams>({ ...DEFAULT_PARAMS });

  const [status, setStatus] = useState<InferenceStatusRecord | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [isClosing, setIsClosing] = useState(false);

  const [codeLang, setCodeLang] = useState<"python" | "cpp" | "csharp">("python");

  const selectedProject = useMemo(
    () => projects.find((p) => p.id === selectedProjectId) ?? null,
    [projects, selectedProjectId]
  );

  const selectedModel = useMemo(() => models.find((m) => m.id === selectedModelId) ?? null, [models, selectedModelId]);

  const matchingSession: InferenceSessionRecord | null = useMemo(() => {
    if (!status || !selectedModelId) return null;
    const dev = (device || "").trim();
    return (
      status.sessions.find((s) => {
        if (s.model_id !== selectedModelId) return false;
        if (s.format !== format) return false;
        if (format === "openvino") return true;
        return (s.device || "").trim() === dev;
      }) ?? null
    );
  }, [device, format, selectedModelId, status]);

  const refreshStatus = useCallback(async () => {
    try {
      setIsRefreshing(true);
      const s = await inferenceService.status();
      setStatus(s);
    } catch (err) {
      console.error(err);
      setStatus(null);
      toast({ title: "获取推理状态失败", description: "请检查后端是否已启动", variant: "destructive" });
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  const loadProjects = useCallback(async () => {
    try {
      setIsLoadingProjects(true);
      const data = await projectService.list();
      setProjects(data);
      if (!selectedProjectId && data[0]) {
        setSelectedProjectId(data[0].id);
        setProjectContext(data[0].id, data[0].name);
      }
    } catch (err) {
      console.error(err);
      toast({ title: "加载项目失败", description: "请确认后端服务已启动", variant: "destructive" });
    } finally {
      setIsLoadingProjects(false);
    }
  }, [selectedProjectId, setProjectContext]);

  const loadModels = useCallback(async () => {
    if (!selectedProjectId) {
      setModels([]);
      setSelectedModelId(null);
      return;
    }
    try {
      setIsLoadingModels(true);
      const data = await modelService.listByProject(selectedProjectId);
      setModels(data);
      if (!selectedModelId && data[0]) setSelectedModelId(data[0].id);
    } catch (err) {
      console.error(err);
      setModels([]);
      toast({ title: "加载模型失败", description: "请确认已完成训练并生成模型版本", variant: "destructive" });
    } finally {
      setIsLoadingModels(false);
    }
  }, [selectedModelId, selectedProjectId]);

  useEffect(() => {
    void loadProjects();
  }, [loadProjects]);

  useEffect(() => {
    void loadModels();
  }, [loadModels]);

  useEffect(() => {
    void refreshStatus();
  }, [refreshStatus]);

  useEffect(() => {
    const next = new URLSearchParams(searchParams);
    if (selectedProjectId) next.set("project_id", selectedProjectId);
    else next.delete("project_id");
    if (selectedModelId) next.set("model_id", selectedModelId);
    else next.delete("model_id");
    setSearchParams(next, { replace: true });
  }, [searchParams, selectedModelId, selectedProjectId, setSearchParams]);

  const baseUrl = useMemo(() => "http://127.0.0.1:8000", []);

  const snippet = useMemo(() => {
    if (!selectedProjectId || !selectedModelId) return "";
    return buildCodeSnippet(codeLang, {
      baseUrl,
      projectId: selectedProjectId,
      modelId: selectedModelId,
      format,
      device: format === "tensorrt" ? device : undefined,
      params,
    });
  }, [baseUrl, codeLang, device, format, params, selectedModelId, selectedProjectId]);

  const copySnippet = async () => {
    if (!snippet) return;
    try {
      await navigator.clipboard.writeText(snippet);
      toast({ title: "已复制到剪贴板" });
    } catch (err) {
      console.error(err);
      toast({ title: "复制失败", description: "浏览器权限限制，请手动复制", variant: "destructive" });
    }
  };

  const createSession = async () => {
    if (!selectedProjectId) {
      toast({ title: "请先选择项目", variant: "destructive" });
      return;
    }
    if (!selectedModelId) {
      toast({ title: "请先选择模型", variant: "destructive" });
      return;
    }
    try {
      setIsCreating(true);
      const created = await inferenceService.createSession({
        project_id: selectedProjectId,
        model_id: selectedModelId,
        format,
        device: format === "tensorrt" ? (device || "").trim() || undefined : undefined,
      });
      toast({ title: "推理申请成功", description: `Session: ${created.id}` });
      await refreshStatus();
    } catch (err) {
      console.error(err);
      let msg = "未知错误";
      if (isAxiosError(err)) {
        msg = err.response?.data?.message || err.response?.data?.detail || err.message;
      } else if (err instanceof Error) {
        msg = err.message;
      }
      toast({ title: "推理申请失败", description: String(msg), variant: "destructive" });
    } finally {
      setIsCreating(false);
    }
  };

  const closeSession = async () => {
    if (!matchingSession) return;
    try {
      setIsClosing(true);
      await inferenceService.closeSession(matchingSession.id);
      toast({ title: "已释放推理会话" });
      await refreshStatus();
    } catch (err) {
      console.error(err);
      let msg = "未知错误";
      if (isAxiosError(err)) {
        msg = err.response?.data?.message || err.response?.data?.detail || err.message;
      } else if (err instanceof Error) {
        msg = err.message;
      }
      toast({ title: "释放失败", description: String(msg), variant: "destructive" });
    } finally {
      setIsClosing(false);
    }
  };

  const inferenceLockedTraining = (status?.sessions?.length ?? 0) > 0 || (status?.active_requests ?? 0) > 0;

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">部署推荐（网络推理申请）</h1>
          <p className="text-muted-foreground mt-2">
            选择项目 → 选择模型 → 选择推理格式与参数 → 申请推理会话 → 使用网络请求调用推理
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={() => void refreshStatus()} disabled={isRefreshing} className="gap-2">
            <RefreshCw className={isRefreshing ? "w-4 h-4 animate-spin" : "w-4 h-4"} />
            刷新状态
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Server className="w-5 h-5 text-primary" /> 推理申请
              </CardTitle>
              <CardDescription>
                推理会话存在期间会禁止训练（避免重复加载模型与资源冲突）
                {inferenceLockedTraining ? <Badge className="ml-2">训练已锁定</Badge> : null}
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label>项目</Label>
                <Select
                  value={selectedProjectId ?? ""}
                  onValueChange={(v) => {
                    const nextId = (v || "").trim() || null;
                    if (!nextId) return;
                    const found = projects.find((p) => p.id === nextId) ?? null;
                    setSelectedProjectId(nextId);
                    setSelectedModelId(null);
                    setProjectContext(nextId, found?.name ?? null);
                  }}
                  disabled={isLoadingProjects || projects.length === 0}
                >
                  <SelectTrigger>
                    <SelectValue placeholder={isLoadingProjects ? "加载中..." : "请选择项目"} />
                  </SelectTrigger>
                  <SelectContent>
                    {projects.map((p) => (
                      <SelectItem key={p.id} value={p.id}>
                        {p.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {selectedProject ? <div className="text-xs text-muted-foreground truncate">ID: {selectedProject.id}</div> : null}
              </div>

              <div className="space-y-2">
                <Label>模型</Label>
                <Select
                  value={selectedModelId ?? ""}
                  onValueChange={(v) => setSelectedModelId((v || "").trim() || null)}
                  disabled={!selectedProjectId || isLoadingModels || models.length === 0}
                >
                  <SelectTrigger>
                    <SelectValue
                      placeholder={
                        !selectedProjectId ? "请先选择项目" : isLoadingModels ? "加载中..." : models.length ? "请选择模型" : "暂无模型"
                      }
                    />
                  </SelectTrigger>
                  <SelectContent>
                    {models.map((m) => (
                      <SelectItem key={m.id} value={m.id}>
                        {m.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {selectedModel ? (
                  <div className="text-xs text-muted-foreground truncate">
                    Base: {selectedModel.base_model} · Created: {new Date(selectedModel.created_at).toLocaleString()}
                  </div>
                ) : null}
              </div>

              <div className="space-y-2">
                <Label>推理格式</Label>
                <Select value={format} onValueChange={(v) => setFormat(v as InferenceFormat)}>
                  <SelectTrigger>
                    <SelectValue placeholder="请选择格式" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="tensorrt">TensorRT（默认）</SelectItem>
                    <SelectItem value="openvino">OpenVINO</SelectItem>
                  </SelectContent>
                </Select>
                <div className="text-xs text-muted-foreground flex items-center gap-2">
                  <Cpu className="w-3.5 h-3.5" />
                  <span>{FORMAT_HINT[format]}</span>
                </div>
              </div>

              <div className="space-y-2">
                <Label>设备（device）</Label>
                <Input
                  value={device}
                  onChange={(e) => setDevice(e.target.value)}
                  placeholder={format === "tensorrt" ? "0 / cuda:0" : "cpu"}
                  disabled={format === "openvino"}
                />
                <div className="text-xs text-muted-foreground">
                  {format === "openvino" ? "OpenVINO 默认使用 CPU，无需设置 device" : "TensorRT 常用 device=0（第一张显卡）"}
                </div>
              </div>

              <div className="md:col-span-2 flex items-center justify-between gap-3 pt-2">
                <div className="text-sm text-muted-foreground">
                  {matchingSession ? (
                    <span>
                      当前会话：<Badge variant="outline">{matchingSession.id.slice(0, 8)}</Badge> · {matchingSession.format}
                      {matchingSession.cached_artifact ? ` · ${matchingSession.cached_artifact}` : ""}
                    </span>
                  ) : (
                    <span>当前未申请该模型/格式的推理会话</span>
                  )}
                </div>
                <div className="flex gap-2">
                  {matchingSession ? (
                    <Button variant="outline" className="gap-2" onClick={() => void closeSession()} disabled={isClosing}>
                      {isClosing ? <RefreshCw className="w-4 h-4 animate-spin" /> : <X className="w-4 h-4" />}
                      释放推理
                    </Button>
                  ) : (
                    <Button className="gap-2" onClick={() => void createSession()} disabled={isCreating || !selectedModelId}>
                      {isCreating ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Server className="w-4 h-4" />}
                      申请推理
                    </Button>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Code2 className="w-5 h-5 text-primary" /> 网络申请代码例程
              </CardTitle>
              <CardDescription>选择语言后即可直接复制（推理仅在平台后端实现，不导出模型文件）</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 md:grid-cols-4">
                <div className="md:col-span-2 space-y-2">
                  <Label>推理参数（用于请求示例）</Label>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-1">
                      <Label className="text-xs">conf</Label>
                      <Input value={params.conf} onChange={(e) => setParams((p) => ({ ...p, conf: e.target.value }))} />
                    </div>
                    <div className="space-y-1">
                      <Label className="text-xs">iou</Label>
                      <Input value={params.iou} onChange={(e) => setParams((p) => ({ ...p, iou: e.target.value }))} />
                    </div>
                    <div className="space-y-1">
                      <Label className="text-xs">imgsz</Label>
                      <Input value={params.imgsz} onChange={(e) => setParams((p) => ({ ...p, imgsz: e.target.value }))} />
                    </div>
                    <div className="space-y-1">
                      <Label className="text-xs">max_det</Label>
                      <Input value={params.max_det} onChange={(e) => setParams((p) => ({ ...p, max_det: e.target.value }))} />
                    </div>
                    <div className="col-span-2 space-y-1">
                      <Label className="text-xs">classes（可选，逗号分隔 class_id）</Label>
                      <Input value={params.classes} onChange={(e) => setParams((p) => ({ ...p, classes: e.target.value }))} placeholder="例如：0,1,2" />
                    </div>
                  </div>
                </div>
                <div className="md:col-span-2 space-y-2">
                  <Label>推理服务地址</Label>
                  <Input value={baseUrl} readOnly />
                  <div className="text-xs text-muted-foreground">
                    默认后端地址：<span className="font-mono">{baseUrl}</span>（如部署到其它机器/端口请修改）
                  </div>
                </div>
              </div>

              <Tabs value={codeLang} onValueChange={(v) => setCodeLang(v as "python" | "cpp" | "csharp")} className="w-full">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="python">Python</TabsTrigger>
                  <TabsTrigger value="cpp">C++</TabsTrigger>
                  <TabsTrigger value="csharp">C#</TabsTrigger>
                </TabsList>
                <TabsContent value="python" className="mt-4">
                  <div />
                </TabsContent>
                <TabsContent value="cpp" className="mt-4">
                  <div />
                </TabsContent>
                <TabsContent value="csharp" className="mt-4">
                  <div />
                </TabsContent>
              </Tabs>

              <div className="flex items-center justify-between gap-3">
                <div className="text-xs text-muted-foreground truncate">
                  {selectedProjectId && selectedModelId
                    ? `project_id=${selectedProjectId} · model_id=${selectedModelId} · format=${format}`
                    : "请先选择项目与模型"}
                </div>
                <Button variant="outline" size="sm" className="gap-2" onClick={() => void copySnippet()} disabled={!snippet}>
                  <Copy className="w-4 h-4" />
                  复制代码
                </Button>
              </div>

              <div className="rounded-md border bg-muted/40 p-3 font-mono text-xs whitespace-pre-wrap leading-relaxed overflow-auto max-h-[420px]">
                {snippet || "请先选择项目与模型后生成代码示例。"}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Server className="w-5 h-5 text-green-600" /> 性能预估（消费级显卡）
              </CardTitle>
              <CardDescription>仅作为参考，实际性能与模型大小、分辨率、batch、驱动与 TensorRT 版本有关</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>GPU</TableHead>
                    <TableHead>显存</TableHead>
                    <TableHead>相对性能</TableHead>
                    <TableHead>备注</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {GPU_ESTIMATES.map((r) => (
                    <TableRow key={r.gpu}>
                      <TableCell className="font-medium">{r.gpu}</TableCell>
                      <TableCell>{r.vram}</TableCell>
                      <TableCell>
                        <Badge variant="outline">{r.perf}</Badge>
                      </TableCell>
                      <TableCell className="text-muted-foreground">{r.note}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">当前推理状态</CardTitle>
              <CardDescription>用于确认是否存在推理申请/推理中任务</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">active_requests</span>
                <Badge variant="outline">{status?.active_requests ?? 0}</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">sessions</span>
                <Badge variant="outline">{status?.sessions?.length ?? 0}</Badge>
              </div>
              {status?.sessions?.length ? (
                <div className="space-y-2 pt-2 border-t">
                  {status.sessions.slice(0, 6).map((s) => (
                    <div key={s.id} className="rounded-md border bg-background p-2">
                      <div className="flex items-center justify-between gap-2">
                        <div className="font-mono text-xs truncate">{s.id}</div>
                        <Badge variant="secondary">{s.format}</Badge>
                      </div>
                      <div className="text-xs text-muted-foreground mt-1 truncate">
                        model_id={s.model_id} {s.device ? `· device=${s.device}` : ""} {s.cached_artifact ? `· ${s.cached_artifact}` : ""}
                      </div>
                      <div className="mt-2 flex justify-end">
                        <Button
                          variant="outline"
                          size="sm"
                          className="gap-2"
                          onClick={() => void inferenceService.closeSession(s.id).then(refreshStatus)}
                          disabled={isClosing}
                        >
                          <X className="w-4 h-4" />
                          释放
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-xs text-muted-foreground">暂无推理会话</div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">训练互斥说明</CardTitle>
              <CardDescription>平台策略</CardDescription>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground space-y-2">
              <div>- 有推理申请（session）或推理中（active_requests&gt;0）时，后端会拒绝开始/继续训练（HTTP 409）。</div>
              <div>- 同一模型的推理会话会复用已加载模型，避免重复加载带来的总耗时增加。</div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
