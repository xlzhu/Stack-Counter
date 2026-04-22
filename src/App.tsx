/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Plus, Minus, RotateCcw, Layers, Camera, Upload, Loader2, X, Check, Settings, Globe, Cpu, Save, Languages, Zap, Image as ImageIcon, SwitchCamera } from 'lucide-react';
import { GoogleGenAI } from "@google/genai";
import { analyzeImageLocal } from './utils/cv';

interface StackItem {
  id: string;
  color: string;
}

interface ModelConfig {
  type: 'gemini' | 'custom';
  endpoint?: string;
  apiKey?: string;
  modelName?: string;
  temperature?: number;
}

interface LocalResult {
  peaks: number[];
  intersectionPoints: { x: number; y: number }[];
  y1: number;
  y2: number;
  boundaryY1?: number;
  boundaryY2?: number;
  height: number;
  width: number;
  centerX: number;
  deskewAngle?: number;
  qualityScore?: number;
  boundarySource?: 'red_lines' | 'texture' | 'ratio';
  quarterTurnAngle?: number;
  selectedBandRatio?: number;
  candidateBands?: Array<{ ratio: number; centerX: number; count: number; qualityScore: number }>;
}

type Language = 'en' | 'zh';

const TRANSLATIONS = {
  en: {
    title: 'Stack-Counter',
    modelSettings: 'Model Settings',
    uploadPhoto: 'Upload Photo',
    scanCamera: 'Scan with Camera',
    stackEmpty: 'Stack is empty',
    aiAnalyzing: 'AI Analyzing Layers...',
    total: 'Total',
    adjust: 'ADJUST',
    resetStack: 'RESET STACK',
    scanStack: 'Scan Stack',
    zoom: 'Zoom',
    saveConfig: 'SAVE CONFIGURATION',
    exportTestImage: 'EXPORT TEST IMAGE',
    endpointUrl: 'Endpoint URL',
    apiKey: 'API Key',
    apiKeyOptional: 'Gemini API Key (Optional)',
    modelName: 'Model Name',
    temperature: 'Temperature',
    tempDesc: 'Lower values are more deterministic, higher values more creative.',
    gemini: 'Gemini',
    customApi: 'Custom API',
    leaveEmpty: 'Leave empty to use system key',
    placeholderEndpoint: 'https://api.openai.com/v1',
    placeholderModel: 'gemini-3.1-pro-preview',
    placeholderCustomModel: 'gpt-4o',
    errorDetermining: 'Could not determine count. Please try again.',
    errorCamera: 'Could not access camera. Please check permissions.',
    analysisFailed: 'Analysis failed. Please check your connection or image quality.',
    footer: 'AI-Powered Layer Detection',
    promptText: "Methodically count the number of thin stacked layers in this image (e.g., stacked trays). These layers are very dense. First, identify the boundaries of the stack. Then, count each individual layer by looking at the distinct horizontal lines on the vertical edges. Be extremely precise and do not skip any layers. Return ONLY the final integer count.",
    systemInstruction: "You are a precision counting expert. ",
    manualCountNote: "Note: For a similar stack, a manual count of {count} was previously recorded. Use this as a reference but verify the current image exactly. ",
    fastMode: 'Fast Local Mode',
    aiMode: 'AI Mode',
    viewStack: 'Stack View',
    viewImage: 'Image View',
    roiStart: 'ROI Start',
    roiEnd: 'ROI End',
    peakDistance: 'Peak Distance',
    peakProminence: 'Peak Prominence',
    localAnalyzing: 'Analyzing Locally...',
    exportFailed: 'Failed to export the ROI test image.',
    resetParams: 'Reset'
  },
  zh: {
    title: '层级计数器',
    modelSettings: '模型设置',
    uploadPhoto: '上传照片',
    scanCamera: '相机扫描',
    stackEmpty: '堆栈为空',
    aiAnalyzing: 'AI 正在分析层级...',
    total: '总计',
    adjust: '调整',
    resetStack: '重置堆栈',
    scanStack: '扫描堆栈',
    zoom: '缩放',
    saveConfig: '保存配置',
    exportTestImage: '导出测试图',
    endpointUrl: '接口地址 (Endpoint)',
    apiKey: 'API 密钥',
    apiKeyOptional: 'Gemini API 密钥 (可选)',
    modelName: '模型名称',
    temperature: '温度 (Temperature)',
    tempDesc: '较低的值更具确定性，较高的值更具创造性。',
    gemini: 'Gemini',
    customApi: '自定义 API',
    leaveEmpty: '留空则使用系统默认密钥',
    placeholderEndpoint: 'https://api.openai.com/v1',
    placeholderModel: 'gemini-3.1-pro-preview',
    placeholderCustomModel: 'gpt-4o',
    errorDetermining: '无法确定数量，请重试。',
    errorCamera: '无法访问相机，请检查权限。',
    analysisFailed: '分析失败，请检查网络连接或图像质量。',
    footer: 'AI 驱动的层级检测',
    promptText: "有条理地计算这张图片中薄堆叠层的数量（例如，堆叠的托盘）。这些层非常密集。首先，识别堆栈的边界。然后，通过观察垂直边缘上明显的水平线来计算每个单独的层。请务必极其精确，不要跳过任何层。仅返回最终的整数计数。",
    systemInstruction: "你是一个精准计数专家。",
    manualCountNote: "注意：对于类似的堆栈，之前记录的手动计数为 {count}。请将其作为参考，但要准确核实当前图像。",
    fastMode: '极速本地模式',
    aiMode: 'AI 模式',
    viewStack: '堆栈视图',
    viewImage: '图像视图',
    roiStart: '检测区域起点',
    roiEnd: '检测区域终点',
    peakDistance: '峰值间距',
    peakProminence: '峰值突起度',
    localAnalyzing: '本地分析中...',
    exportFailed: '导出 ROI 测试图失败。',
    resetParams: '恢复默认'
  }
};

const COLORS = [
  'bg-emerald-400',
  'bg-sky-400',
  'bg-indigo-400',
  'bg-violet-400',
  'bg-fuchsia-400',
  'bg-rose-400',
  'bg-orange-400',
  'bg-amber-400',
];

const ROI_HANDLE_GAP = 0.05;
const LOCAL_ANALYSIS_MAX_DIMENSION = 1920; // Increased from 1024px to 1920px for Phase 1.1

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

const compressImage = (
  dataUrl: string,
  maxWidth = 1024,
  maxHeight = 1024,
  initialQuality = 0.7,
  mimeType: 'image/jpeg' | 'image/png' = 'image/jpeg'
): Promise<string> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      let width = img.width;
      let height = img.height;

      if (width > height) {
        if (width > maxWidth) {
          height = Math.round((height * maxWidth) / width);
          width = maxWidth;
        }
      } else {
        if (height > maxHeight) {
          width = Math.round((width * maxHeight) / height);
          height = maxHeight;
        }
      }

      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(img, 0, 0, width, height);

        if (mimeType === 'image/png') {
          resolve(canvas.toDataURL('image/png'));
          return;
        }

        let quality = initialQuality;
        let compressed = canvas.toDataURL('image/jpeg', quality);

        // Ensure the base64 string is under ~1.5MB (approx 1MB actual size)
        while (compressed.length > 1500000 && quality > 0.1) {
          quality -= 0.1;
          compressed = canvas.toDataURL('image/jpeg', quality);
        }

        resolve(compressed);
      } else {
        resolve(dataUrl);
      }
    };
    img.onerror = () => reject(new Error("Failed to load image. The format might not be supported by your browser."));
    img.src = dataUrl;
  });
};

export default function App() {
  const [language, setLanguage] = useState<Language>(() => {
    const saved = localStorage.getItem('stack_counter_lang');
    if (saved === 'en' || saved === 'zh') return saved;
    return navigator.language.startsWith('zh') ? 'zh' : 'en';
  });

  useEffect(() => {
    localStorage.setItem('stack_counter_lang', language);
  }, [language]);

  const t = useMemo(() => TRANSLATIONS[language], [language]);

  const toggleLanguage = () => {
    setLanguage(prev => prev === 'en' ? 'zh' : 'en');
  };

  const [stack, setStack] = useState<StackItem[]>([]);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastManualCount, setLastManualCount] = useState<number | null>(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [analysisMode, setAnalysisMode] = useState<'ai' | 'local'>('ai');
  const [viewMode, setViewMode] = useState<'stack' | 'image'>('stack');
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [correctedImageUrl, setCorrectedImageUrl] = useState<string | null>(null);
  const [localResult, setLocalResult] = useState<LocalResult | null>(null);
  const [localParams, setLocalParams] = useState({
    roiStart: 0.12,
    roiEnd: 0.94,
    distance: 8,
    prominence: 10
  });
  const [draggingHandle, setDraggingHandle] = useState<'start' | 'end' | null>(null);
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [currentCameraId, setCurrentCameraId] = useState<string | null>(null);
  
  const [modelConfig, setModelConfig] = useState<ModelConfig>(() => {
    const saved = localStorage.getItem('stack_counter_config');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        return {
          type: parsed.type || 'gemini',
          endpoint: parsed.endpoint,
          apiKey: parsed.apiKey,
          modelName: parsed.modelName || 'gemini-3.1-pro-preview',
          temperature: parsed.temperature ?? 1
        };
      } catch (e) {
        console.error("Failed to parse saved config", e);
      }
    }
    return {
      type: 'gemini',
      modelName: 'gemini-3.1-pro-preview',
      temperature: 1
    };
  });

  useEffect(() => {
    localStorage.setItem('stack_counter_config', JSON.stringify(modelConfig));
  }, [modelConfig]);
  
  const scrollRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const overlayImageRef = useRef<HTMLImageElement>(null);

  const videoContainerRef = useRef<HTMLDivElement>(null);
  const overlayBoxRef = useRef<HTMLDivElement>(null);

  const increment = useCallback(() => {
    const newId = Math.random().toString(36).substring(2, 9);
    const color = COLORS[stack.length % COLORS.length];
    setStack((prev) => [...prev, { id: newId, color }]);
  }, [stack.length]);

  const decrement = useCallback(() => {
    setStack((prev) => prev.slice(0, -1));
  }, []);

  const reset = useCallback(() => {
    setStack([]);
    setError(null);
    setUploadedImage(null);
    setCorrectedImageUrl(null);
    setLocalResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }, []);

  const [isAdjusting, setIsAdjusting] = useState(false);
  const [adjustValue, setAdjustValue] = useState("");

  const handleAdjustSubmit = () => {
    const count = parseInt(adjustValue, 10);
    if (!isNaN(count) && count >= 0) {
      updateStackCount(count);
      setLastManualCount(count);
    }
    setIsAdjusting(false);
  };

  const updateStackCount = (count: number) => {
    const newStack: StackItem[] = [];
    for (let i = 0; i < count; i++) {
      newStack.push({
        id: Math.random().toString(36).substring(2, 9),
        color: COLORS[i % COLORS.length]
      });
    }
    setStack(newStack);
  };

  const exportRoiTestImage = useCallback(async () => {
    if (!uploadedImage || !localResult) return;

    try {
      const img = new Image();
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = uploadedImage;
      });

      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Could not create export canvas');

      ctx.drawImage(img, 0, 0);

      ctx.strokeStyle = '#ff2d2d';
      ctx.lineWidth = Math.max(3, Math.round(img.height / 220));
      ctx.setLineDash([]);

      const drawBoundary = (y: number) => {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(img.width, y);
        ctx.stroke();
      };

      drawBoundary(localResult.boundaryY1 ?? localResult.y1);
      drawBoundary(localResult.boundaryY2 ?? localResult.y2);

      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const link = document.createElement('a');
      link.href = canvas.toDataURL('image/png');
      link.download = `stack-counter-roi-test-${timestamp}.png`;
      link.click();
    } catch (err) {
      console.error('Export ROI test image failed:', err);
      setError(t.exportFailed);
    }
  }, [localResult, t.exportFailed, uploadedImage]);

  const runLocalAnalysis = useCallback(async (base64Data: string) => {
    const img = new Image();
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = base64Data;
    });

    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error("Could not get canvas context");

    ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, img.width, img.height);

    const result = analyzeImageLocal(
      imageData,
      [localParams.roiStart, localParams.roiEnd],
      localParams.distance,
      localParams.prominence
    );

    // Convert corrected ImageData (deskewed + center line) to a displayable data URL
    if (result.correctedImageData) {
      const corrCanvas = document.createElement('canvas');
      corrCanvas.width = result.correctedImageData.width;
      corrCanvas.height = result.correctedImageData.height;
      const corrCtx = corrCanvas.getContext('2d');
      if (corrCtx) {
        corrCtx.putImageData(result.correctedImageData, 0, 0);
        setCorrectedImageUrl(corrCanvas.toDataURL('image/jpeg', 0.92));
      }
    }

    setLocalResult({
      peaks: result.peaks,
      intersectionPoints: result.intersectionPoints,
      y1: result.y1,
      y2: result.y2,
      boundaryY1: result.boundaryY1,
      boundaryY2: result.boundaryY2,
      height: result.height,
      width: result.width,
      centerX: result.centerX,
      deskewAngle: result.deskewAngle,
      qualityScore: result.qualityScore,
      boundarySource: result.boundarySource,
      quarterTurnAngle: result.quarterTurnAngle,
      selectedBandRatio: result.selectedBandRatio,
      candidateBands: result.candidateBands
    });
    updateStackCount(result.count);
    setViewMode('image');
    return result;
  }, [localParams]);

  useEffect(() => {
    if (analysisMode !== 'local' || !uploadedImage || localResult === null) {
      return;
    }

    const timer = window.setTimeout(() => {
      runLocalAnalysis(uploadedImage).catch((err: any) => {
        console.error("Local Analysis Error:", err);
        setError(err?.message || t.analysisFailed);
      });
    }, 80);

    return () => window.clearTimeout(timer);
  }, [analysisMode, uploadedImage, localParams, runLocalAnalysis, t.analysisFailed]);

  useEffect(() => {
    (window as Window & {
      __STACK_COUNTER_TEST_STATE?: {
        count: number;
        analysisMode: 'ai' | 'local';
        uploadedImage: boolean;
        localResult: LocalResult | null;
        error: string | null;
        isProcessing: boolean;
      };
    }).__STACK_COUNTER_TEST_STATE = {
      count: stack.length,
      analysisMode,
      uploadedImage: Boolean(uploadedImage),
      localResult,
      error,
      isProcessing
    };
  }, [analysisMode, error, isProcessing, localResult, stack.length, uploadedImage]);

  const updateRoiHandle = useCallback((clientY: number, handle: 'start' | 'end') => {
    const image = overlayImageRef.current;
    if (!image) return;

    const rect = image.getBoundingClientRect();
    if (rect.height <= 0) return;

    const ratio = clamp((clientY - rect.top) / rect.height, 0, 1);

    setLocalParams(prev => {
      if (handle === 'start') {
        return {
          ...prev,
          roiStart: clamp(ratio, 0, prev.roiEnd - ROI_HANDLE_GAP)
        };
      }

      return {
        ...prev,
        roiEnd: clamp(ratio, prev.roiStart + ROI_HANDLE_GAP, 1)
      };
    });
  }, []);

  useEffect(() => {
    if (!draggingHandle) {
      return;
    }

    const handlePointerMove = (event: PointerEvent) => {
      updateRoiHandle(event.clientY, draggingHandle);
    };

    const stopDragging = () => {
      setDraggingHandle(null);
    };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', stopDragging);
    window.addEventListener('pointercancel', stopDragging);

    return () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', stopDragging);
      window.removeEventListener('pointercancel', stopDragging);
    };
  }, [draggingHandle, updateRoiHandle]);

  const analyzeImage = async (
    localImageData: string,
    aiImageData: string = localImageData,
    refinedLocalImageData?: string
  ) => {
    setIsProcessing(true);
    setError(null);
    setUploadedImage(localImageData);
    
    try {
      const analyzeLocalData = async (dataUrl: string) => {
        const img = new Image();
        await new Promise((resolve, reject) => {
          img.onload = resolve;
          img.onerror = reject;
          img.src = dataUrl;
        });

        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        if (!ctx) throw new Error("Could not get canvas context");

        ctx.drawImage(img, 0, 0);
        const imageData = ctx.getImageData(0, 0, img.width, img.height);

        return analyzeImageLocal(
          imageData,
          [localParams.roiStart, localParams.roiEnd],
          localParams.distance,
          localParams.prominence
        );
      };

      const commitLocalResult = (result: ReturnType<typeof analyzeImageLocal>, sourceImageData: string) => {
        setUploadedImage(sourceImageData);

        if (result.correctedImageData) {
          const corrCanvas = document.createElement('canvas');
          corrCanvas.width = result.correctedImageData.width;
          corrCanvas.height = result.correctedImageData.height;
          const corrCtx = corrCanvas.getContext('2d');
          if (corrCtx) {
            corrCtx.putImageData(result.correctedImageData, 0, 0);
            setCorrectedImageUrl(corrCanvas.toDataURL('image/jpeg', 0.92));
          }
        }

        setLocalResult({
          peaks: result.peaks,
          intersectionPoints: result.intersectionPoints,
          y1: result.y1,
          y2: result.y2,
          boundaryY1: result.boundaryY1,
          boundaryY2: result.boundaryY2,
          height: result.height,
          width: result.width,
          centerX: result.centerX,
          deskewAngle: result.deskewAngle,
          qualityScore: result.qualityScore,
          boundarySource: result.boundarySource,
          quarterTurnAngle: result.quarterTurnAngle,
          selectedBandRatio: result.selectedBandRatio,
          candidateBands: result.candidateBands
        });
        updateStackCount(result.count);
        setViewMode('image');
      };

      let localRes = await analyzeLocalData(localImageData);
      let finalLocalImageData = localImageData;

      // Phase 1.1: Increased resolution from 1024px to 1920px
      // High-layer images (≥40 layers) need higher resolution to preserve fine texture details
      // and avoid undercount errors. Spacing increases from 4-8px to 6-12px at 1920px.
      const roiHeight = localRes.y2 - localRes.y1;
      const roiCoverage = localRes.height > 0 ? roiHeight / localRes.height : 0;
      const localPeakSpacings = localRes.peaks.slice(1).map((value, index) => value - localRes.peaks[index]);
      const sortedLocalSpacings = [...localPeakSpacings].sort((a, b) => a - b);
      const medianLocalSpacing = sortedLocalSpacings.length > 0
        ? sortedLocalSpacings[Math.floor(sortedLocalSpacings.length / 2)]
        : 0;
      const localQuality = localRes.qualityScore ?? 0;
      const shouldRefineDense =
        localRes.boundarySource !== 'red_lines' &&
        localRes.count >= 45 &&
        localQuality >= 0.78 &&
        medianLocalSpacing > 0 &&
        medianLocalSpacing <= 9;
      const shouldRefineWideTexture = localRes.boundarySource === 'texture' && roiCoverage >= 0.72;
      const shouldRefineDeskewFallback =
        localRes.boundarySource === 'ratio' && Math.abs(localRes.deskewAngle ?? 0) >= 8;

      if (
        refinedLocalImageData &&
        (shouldRefineDense || shouldRefineWideTexture || shouldRefineDeskewFallback)
      ) {
        const refinedRes = await analyzeLocalData(refinedLocalImageData);
        const refinedImprovesWideTexture =
          shouldRefineWideTexture && refinedRes.count <= localRes.count - 4;
        const refinedImprovesDeskewFallback =
          shouldRefineDeskewFallback && refinedRes.boundarySource !== 'ratio';
        const refinedQuality = refinedRes.qualityScore ?? 0;
        const countDelta = refinedRes.count - localRes.count;
        const refinedImprovesDenseLocalGapCase =
          analysisMode === 'local' &&
          shouldRefineDense &&
          (localRes.selectedBandRatio ?? 0.5) !== 0.5 &&
          localRes.count >= 60 &&
          localQuality <= 0.62 &&
          countDelta >= 8 &&
          refinedQuality >= localQuality - 0.12;
        const refinedAllowsDenseDecrease =
          countDelta < 0 &&
          (
            (
              analysisMode !== 'local' &&
              refinedQuality >= localQuality - 0.08
            ) ||
            (
              analysisMode === 'local' &&
              Math.abs(countDelta) <= 6 &&
              refinedQuality >= localQuality - 0.03
            ) ||
            (
              analysisMode === 'local' &&
              refinedQuality >= localQuality + 0.05
            )
          );
        const refinedImprovesDense =
          shouldRefineDense &&
          (
            refinedImprovesDenseLocalGapCase ||
            (
              Math.abs(countDelta) >= 4 &&
              (
                (
                  refinedAllowsDenseDecrease
                ) ||
                (
                  countDelta > 0 &&
                  (
                    (countDelta <= 8 && refinedQuality >= localQuality - 0.01) ||
                    refinedQuality >= localQuality + 0.02
                  )
                )
              )
            )
          );

        if (refinedImprovesWideTexture || refinedImprovesDeskewFallback || refinedImprovesDense) {
          localRes = refinedRes;
          finalLocalImageData = refinedLocalImageData;
        }
      }

      // If red lines are detected, OR if we are already in local mode, use the local result
      if (localRes.boundarySource === 'red_lines' || analysisMode === 'local') {
        if (localRes.boundarySource === 'red_lines' && analysisMode !== 'local') {
          setAnalysisMode('local'); // Auto-switch to local mode
        }

        commitLocalResult(localRes, finalLocalImageData);
        setIsProcessing(false);
        return;
      }

      const promptText = t.promptText;
      const systemInstruction = t.systemInstruction + 
        (lastManualCount ? t.manualCountNote.replace('{count}', lastManualCount.toString()) : "") +
        "Count each individual layer by looking at the distinct horizontal lines on the vertical edges. Be extremely precise and do not skip any layers. Return ONLY the final integer count.";

      let count = 0;

      if (modelConfig.type === 'gemini') {
        const apiKey = modelConfig.apiKey?.trim() || process.env.GEMINI_API_KEY;
        if (!apiKey) throw new Error("Gemini API Key is missing. Please configure it in Settings.");
        
        const ai = new GoogleGenAI({ apiKey });
        const response = await ai.models.generateContent({
          model: modelConfig.modelName || "gemini-3.1-pro-preview",
          contents: [
            {
              parts: [
                { text: promptText },
                {
                  inlineData: {
                    mimeType: "image/jpeg",
                    data: aiImageData.split(',')[1]
                  }
                }
              ]
            }
          ],
          config: {
            systemInstruction,
            temperature: modelConfig.temperature ?? 1,
            thinkingConfig: {
              includeThoughts: true
            }
          }
        });

        const resultText = response.text?.trim() || "0";
        count = parseInt(resultText.replace(/[^0-9]/g, ''), 10);
      } else {
        // Custom API (OpenAI-compatible format)
        if (!modelConfig.endpoint) throw new Error("Custom endpoint URL is required");
        
        let finalEndpoint = modelConfig.endpoint.trim();
        const isOpenRouter = finalEndpoint.includes('openrouter.ai');
        const isNvidia = finalEndpoint.includes('nvidia.com') || finalEndpoint.includes('localhost') || finalEndpoint.includes('127.0.0.1') || modelConfig.modelName?.toLowerCase().includes('nvidia') || modelConfig.modelName?.toLowerCase().includes('nemotron');
        const isNvidiaCv = isNvidia && (finalEndpoint.includes('/cv/') || modelConfig.modelName?.toLowerCase().includes('ocr') || modelConfig.modelName?.toLowerCase().includes('cv'));

        try {
          const urlObj = new URL(finalEndpoint);
          if (isOpenRouter) {
            // For OpenRouter, ensure it uses /api/v1/chat/completions
            if (urlObj.pathname === '/' || urlObj.pathname === '/api/v1' || urlObj.pathname === '/api/v1/') {
              const basePath = finalEndpoint.replace(/\/$/, '');
              finalEndpoint = urlObj.pathname === '/' ? `${basePath}/api/v1/chat/completions` : `${basePath}/chat/completions`;
            }
          } else if (isNvidiaCv) {
            // For NVIDIA CV models, if it's a base URL, append /v1/infer
            if (urlObj.pathname === '/' || urlObj.pathname === '/v1' || urlObj.pathname === '/v1/') {
              finalEndpoint = finalEndpoint.replace(/\/$/, '') + '/v1/infer';
            }
          } else {
            // Only append /chat/completions if it looks like a base URL (e.g., ends with /v1, /api/v1 or has no specific path)
            if (!finalEndpoint.endsWith('/chat/completions') &&
                (urlObj.pathname === '/v1' || urlObj.pathname === '/v1/' || urlObj.pathname === '/api/v1' || urlObj.pathname === '/api/v1/' || urlObj.pathname === '/')) {
              finalEndpoint = finalEndpoint.replace(/\/$/, '') + '/chat/completions';
            }
          }
        } catch (e) {
          // Fallback if URL parsing fails
          if (!isNvidiaCv && !finalEndpoint.endsWith('/chat/completions')) {
            finalEndpoint = finalEndpoint.replace(/\/$/, '') + '/chat/completions';
          }
        }
        
        const apiKey = modelConfig.apiKey?.trim() || '';
        if (!apiKey) throw new Error("API Key is required for Custom API mode.");
        
        let payload: any;
        const modelName = modelConfig.modelName || (isNvidia ? (finalEndpoint.split('/').pop() || 'nvidia/nemotron-ocr-v1') : 'gpt-4o');
        // Nemotron-3 Super does not support image input (text-only)
        const isMultimodal = !modelName.includes('nemotron-3-super') && !modelName.includes('qwen3.5-122b');

        if (isNvidiaCv) {
          // NVIDIA CV-specific API format (e.g., nemotron-ocr-v1)
          // Requires an "input" wrapper as a list
          payload = {
            input: [
              {
                type: 'image_url',
                url: aiImageData
              }
            ],
            merge_levels: ['paragraph']
          };
        } else {
          // Standard OpenAI-compatible Chat format
          payload = {
            model: modelName,
            messages: [
              {
                role: 'user',
                content: isMultimodal ? [
                  { type: 'text', text: (isNvidia && !isOpenRouter) ? `${systemInstruction}\n\n${promptText}` : promptText },
                  {
                    type: 'image_url',
                    image_url: { url: aiImageData }
                  }
                ] : ((isNvidia && !isOpenRouter) ? `${systemInstruction}\n\n${promptText}` : promptText)
              }
            ],
            temperature: modelConfig.temperature ?? 1,
            max_tokens: 1024
          };

          if (!isNvidia || isOpenRouter) {
            payload.messages.unshift({ role: 'system', content: systemInstruction });
          }

          // Special handling for Nemotron-3 Super with thinking capabilities
          if (modelName.includes('nemotron-3-super')) {
            if (!isOpenRouter) {
              payload.chat_template_kwargs = { enable_thinking: true };
              payload.reasoning_budget = 16384;
            }
            payload.max_tokens = 16384;
            payload.top_p = 0.95;
          }

          // Special handling for Gemma 3
          if (modelName.includes('gemma-3')) {
            payload.temperature = modelConfig.temperature ?? 0.20;
            payload.top_p = 0.70;
            payload.max_tokens = 512;
          }

          // Special handling for Qwen 3.5 122b with thinking capabilities
          if (modelName.includes('qwen3.5-122b')) {
            payload.chat_template_kwargs = { enable_thinking: true };
            payload.max_tokens = 16384;
            payload.temperature = modelConfig.temperature ?? 0.60;
            payload.top_p = 0.95;
          }
        }

        const response = await fetch('/api/proxy/llm', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            endpoint: finalEndpoint,
            apiKey: apiKey,
            extraHeaders: isOpenRouter ? {
              'HTTP-Referer': 'https://github.com/google/gemini-cli',
              'X-Title': 'Stack-Counter'
            } : undefined,
            body: payload
          })
        });

        let data: any;
        try {
          data = await response.json();
        } catch {
          throw new Error(`Invalid JSON response from API (status ${response.status})`);
        }

        console.error('LLM error response:', JSON.stringify(data, null, 2));

        if (!response.ok || data?.error) {
          const errorObj = data?.error || data || {};
          let errorMessage = errorObj.message || errorObj.detail || errorObj.type || (typeof data === 'string' ? data : null);

          if (response.status === 422) {
            errorMessage = `API Error 422: Invalid request format. For NVIDIA CV models, ensure the URL is correct. Details: ${JSON.stringify(errorObj)}`;
          } else if (errorObj.type === 'access_terminated_error') {
            errorMessage = "API Access Terminated: Your API key or account has been disabled by the provider. Please check your account status.";
          } else if (errorObj.type === 'insufficient_quota' || errorObj.code === 'insufficient_quota') {
            errorMessage = "Insufficient Quota: Your API account has run out of credits or reached its limit.";
          } else if (errorMessage && typeof errorMessage === 'string' && errorMessage.includes('No endpoints found')) {
            errorMessage = `OpenRouter Error: "No endpoints found". This usually happens for free models if your OpenRouter settings have "Request No Training" enabled. Please go to OpenRouter Settings > Training, Logging, & Privacy and enable "Enable free endpoints that may train on inputs".`;
          } else if (errorMessage === 'Provider returned error' && errorObj.metadata?.raw) {
            errorMessage = `Provider returned error: ${JSON.stringify(errorObj.metadata.raw)}`;
          }

          throw new Error(errorMessage || `API Error: ${response.status}`);
        }

        console.log('LLM response data:', JSON.stringify(data, null, 2));

        let resultText = "";
        if (isNvidiaCv) {
          // NVIDIA CV API usually returns { "content": "..." } or { "text": "..." }
          // Or a nested structure: { "data": [ { "text_detections": [ { "text_prediction": { "text": "..." } } ] } ] }
          if (data?.data && Array.isArray(data.data)) {
            let combinedText = "";
            for (const detection of data.data) {
              if (detection.text_detections && Array.isArray(detection.text_detections)) {
                for (const textDet of detection.text_detections) {
                  if (textDet.text_prediction && textDet.text_prediction.text) {
                    combinedText += textDet.text_prediction.text + " ";
                  }
                }
              }
            }
            resultText = combinedText.trim();
          } else {
            resultText = data?.content || data?.text || data?.description || "";
          }
        } else {
          // Standard Chat API
          if (!data || !Array.isArray(data.choices)) {
            console.error('Unexpected LLM response structure:', data);
            throw new Error('Unexpected response from LLM API: missing choices array');
          }
          const message = data.choices[0]?.message;
          resultText = message?.content || "";

          // If there is reasoning content, log it for debugging
          if (message?.reasoning_content) {
            console.log("Nemotron Reasoning:", message.reasoning_content);
          }
        }
        
        count = parseInt(resultText.trim().replace(/[^0-9]/g, ''), 10);
      }
      
      if (!isNaN(count)) {
        updateStackCount(count);
      } else {
        setError(t.errorDetermining);
      }
    } catch (err: any) {
      console.error("AI Analysis Error:", err);
      let errorMessage = err.message || t.analysisFailed;
      if (errorMessage.includes('Rpc failed due to xhr error')) {
        errorMessage = language === 'zh' ? '图片体积仍然过大，请尝试截取局部图片或使用更低分辨率的照片。' : 'Image size is still too large. Please try cropping the image or using a lower resolution photo.';
      }
      setError(errorMessage);
    } finally {
      setIsProcessing(false);
      setIsCameraActive(false);
    }
  };

  const startCamera = () => {
    setIsCameraActive(true);
  };

  useEffect(() => {
    let stream: MediaStream | null = null;

    const initCamera = async () => {
      if (isCameraActive) {
        // Wait a tiny bit for the modal to mount and videoRef to be available
        await new Promise(resolve => setTimeout(resolve, 100));
        
        if (!videoRef.current) return;

        try {
          const constraints: MediaStreamConstraints = {
            video: currentCameraId 
              ? { deviceId: { exact: currentCameraId }, width: { ideal: 1920 }, height: { ideal: 1080 } }
              : { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } }
          };

          stream = await navigator.mediaDevices.getUserMedia(constraints);
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
          }

          // After getting stream (and permissions), enumerate devices to get labels
          const devices = await navigator.mediaDevices.enumerateDevices();
          const videoDevices = devices.filter(device => device.kind === 'videoinput');
          setCameras(videoDevices);
          
          // If we didn't have a specific camera ID, set it to the currently active one
          if (!currentCameraId && stream) {
            const activeTrack = stream.getVideoTracks()[0];
            const activeDevice = videoDevices.find(d => d.label === activeTrack.label);
            if (activeDevice) {
              setCurrentCameraId(activeDevice.deviceId);
            } else if (videoDevices.length > 0) {
              setCurrentCameraId(videoDevices[0].deviceId);
            }
          }
        } catch (err) {
          console.error("Camera Error:", err);
          setError(t.errorCamera);
          setIsCameraActive(false);
        }
      }
    };

    initCamera();

    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [isCameraActive, currentCameraId, t.errorCamera]);

  const switchCamera = () => {
    if (cameras.length > 1) {
      const currentIndex = cameras.findIndex(c => c.deviceId === currentCameraId);
      const nextIndex = (currentIndex + 1) % cameras.length;
      setCurrentCameraId(cameras[nextIndex].deviceId);
    }
  };

  const stopCamera = () => {
    setIsCameraActive(false);
  };

  const captureImage = async () => {
    if (videoRef.current && canvasRef.current && overlayBoxRef.current && videoContainerRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      const overlayRect = overlayBoxRef.current.getBoundingClientRect();
      const containerRect = videoContainerRef.current.getBoundingClientRect();

      if (context) {
        const vw = video.videoWidth;
        const vh = video.videoHeight;
        const cw = containerRect.width;
        const ch = containerRect.height;

        // Calculate scaling applied by object-cover
        const scaleCover = Math.max(cw / vw, ch / vh);
        const displayW = vw * scaleCover;
        const displayH = vh * scaleCover;

        // Calculate dimensions after zoom
        const zoomedDisplayW = displayW * zoom;
        const zoomedDisplayH = displayH * zoom;
        
        // Calculate offset of the zoomed video relative to the container
        const zoomedOffsetX = (cw - zoomedDisplayW) / 2;
        const zoomedOffsetY = (ch - zoomedDisplayH) / 2;

        // Calculate overlay box position relative to the container
        const boxX = overlayRect.left - containerRect.left;
        const boxY = overlayRect.top - containerRect.top;
        const boxW = overlayRect.width;
        const boxH = overlayRect.height;

        // Map box coordinates to the zoomed video coordinates
        const boxXInVideo = boxX - zoomedOffsetX;
        const boxYInVideo = boxY - zoomedOffsetY;

        // Scale back to the intrinsic video resolution
        const scaleToIntrinsic = vw / zoomedDisplayW;
        const sourceX = Math.max(0, boxXInVideo * scaleToIntrinsic);
        const sourceY = Math.max(0, boxYInVideo * scaleToIntrinsic);
        const sourceW = Math.min(vw - sourceX, boxW * scaleToIntrinsic);
        const sourceH = Math.min(vh - sourceY, boxH * scaleToIntrinsic);

        canvas.width = sourceW;
        canvas.height = sourceH;

        context.drawImage(
          video,
          sourceX, sourceY, sourceW, sourceH,
          0, 0, sourceW, sourceH
        );

        const base64Data = canvas.toDataURL('image/jpeg', 0.85);
        stopCamera();
        const compressedDataUrl = await compressImage(base64Data);
        const localAnalysisDataUrl = await compressImage(
          base64Data,
          LOCAL_ANALYSIS_MAX_DIMENSION,
          LOCAL_ANALYSIS_MAX_DIMENSION,
          0.7,
          'image/png'
        );
        analyzeImage(localAnalysisDataUrl, compressedDataUrl, base64Data);
      }
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      try {
        setIsProcessing(true);
        setError(null);
        let processedFile = file;
        
        // Convert HEIC/HEIF to JPEG
        if (file.name.toLowerCase().endsWith('.heic') || file.name.toLowerCase().endsWith('.heif')) {
          try {
            // Dynamically import to avoid Vite build/SSR issues with WASM/Workers
            const heic2any = (await import('heic2any')).default;
            const convertedBlob = await heic2any({
              blob: file,
              toType: 'image/jpeg',
              quality: 0.8
            });
            
            // heic2any can return a Blob or Blob[]
            const blobToUse = Array.isArray(convertedBlob) ? convertedBlob[0] : convertedBlob;
            processedFile = new File([blobToUse], file.name.replace(/\.heic$|\.heif$/i, '.jpg'), {
              type: 'image/jpeg'
            });
          } catch (heicError: any) {
            console.warn("HEIC Conversion Error, attempting native fallback:", heicError);
            // Fallback: The browser might support it natively (e.g., Safari), or it might be a misnamed JPEG.
            // We will just proceed with the original file and let the Image object try to load it.
            processedFile = file;
          }
        }

        const reader = new FileReader();
        reader.onloadend = async () => {
          try {
            const compressedDataUrl = await compressImage(reader.result as string);
            const localAnalysisDataUrl = await compressImage(
              reader.result as string,
              LOCAL_ANALYSIS_MAX_DIMENSION,
              LOCAL_ANALYSIS_MAX_DIMENSION,
              0.7,
              'image/png'
            );
            await analyzeImage(localAnalysisDataUrl, compressedDataUrl, reader.result as string);
          } catch (err: any) {
            console.error("Image compression/analysis error:", err);
            setError(`${t.analysisFailed} (${err?.message || 'Unknown error'})`);
            setIsProcessing(false);
          } finally {
            // Clear the input value so the same file can be uploaded again
            if (fileInputRef.current) {
              fileInputRef.current.value = "";
            }
          }
        };
        reader.onerror = () => {
          throw new Error("Failed to read file");
        };
        reader.readAsDataURL(processedFile);
      } catch (error: any) {
        console.error("Error processing file:", error);
        setError(`${t.analysisFailed} (${error?.message || String(error)})`);
        setIsProcessing(false);
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
      }
    }
  };

  // Auto-scroll to top of stack
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = 0;
    }
  }, [stack.length]);

  return (
    <div className="min-h-screen bg-zinc-50 text-zinc-900 font-sans selection:bg-emerald-100">
      <div className="max-w-md sm:max-w-lg md:max-w-xl mx-auto h-screen flex flex-col p-6">
        
        {/* Header */}
        <header className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-2 min-w-0">
            <div className="p-2 bg-zinc-900 rounded-lg text-white">
              <Layers size={20} />
            </div>
            <h1 className="text-xl font-bold tracking-tight whitespace-nowrap">{t.title}</h1>
          </div>
          <div className="flex gap-2">
            <button 
              onClick={() => setAnalysisMode(prev => prev === 'ai' ? 'local' : 'ai')}
              data-testid="mode-toggle"
              data-mode={analysisMode}
              className={`p-2 rounded-lg transition-colors flex items-center gap-1 ${analysisMode === 'local' ? 'bg-emerald-100 text-emerald-600' : 'bg-zinc-100 text-zinc-600 hover:bg-zinc-200'}`}
              title={analysisMode === 'local' ? t.fastMode : t.aiMode}
            >
              <Zap size={20} />
              <span className="text-[10px] font-bold uppercase hidden sm:inline">{analysisMode === 'local' ? 'Local' : 'AI'}</span>
            </button>
            <button 
              onClick={toggleLanguage}
              className="p-2 bg-zinc-100 rounded-lg text-zinc-600 hover:bg-zinc-200 transition-colors flex items-center gap-1"
              title={language === 'en' ? 'Switch to Chinese' : '切换至英文'}
            >
              <Languages size={20} />
              <span className="text-[10px] font-bold uppercase">{language}</span>
            </button>
            <button 
              onClick={() => setIsSettingsOpen(true)}
              className="p-2 bg-zinc-100 rounded-lg text-zinc-600 hover:bg-zinc-200 transition-colors"
              title={t.modelSettings}
            >
              <Settings size={20} />
            </button>
            <button 
              onClick={() => fileInputRef.current?.click()}
              className="p-2 bg-zinc-100 rounded-lg text-zinc-600 hover:bg-zinc-200 transition-colors"
              title={t.uploadPhoto}
            >
              <Upload size={20} />
            </button>
            <button 
              onClick={startCamera}
              className="p-2 bg-zinc-100 rounded-lg text-zinc-600 hover:bg-zinc-200 transition-colors"
              title={t.scanCamera}
            >
              <Camera size={20} />
            </button>
          </div>
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileUpload} 
            accept="image/*,.heic,.heif" 
            data-testid="file-input"
            className="hidden" 
          />
        </header>

        {/* Main Display */}
        <div className="flex-1 flex flex-col min-h-0 bg-white rounded-3xl border border-zinc-200 shadow-sm overflow-hidden relative">
          
          {uploadedImage && (
            <div className="absolute top-4 left-4 z-50 flex items-center gap-3">
              <div className="flex bg-white/80 backdrop-blur-md border border-zinc-200 rounded-lg overflow-hidden shadow-sm">
                <button
                  onClick={() => setViewMode('stack')}
                  className={`px-3 py-1.5 text-xs font-bold flex items-center gap-1 transition-colors ${viewMode === 'stack' ? 'bg-zinc-900 text-white' : 'text-zinc-500 hover:bg-zinc-100'}`}
                >
                  <Layers size={14} />
                  <span className="hidden sm:inline">{t.viewStack}</span>
                </button>
                <button
                  onClick={() => setViewMode('image')}
                  className={`px-3 py-1.5 text-xs font-bold flex items-center gap-1 transition-colors ${viewMode === 'image' ? 'bg-zinc-900 text-white' : 'text-zinc-500 hover:bg-zinc-100'}`}
                >
                  <ImageIcon size={14} />
                  <span className="hidden sm:inline">{t.viewImage}</span>
                </button>
              </div>
              {localResult && (
                <button
                  onClick={exportRoiTestImage}
                  className="px-3 py-1.5 text-xs font-bold flex items-center gap-2 bg-white/80 backdrop-blur-md border border-zinc-200 rounded-lg shadow-sm text-zinc-700 hover:bg-white transition-colors"
                >
                  <Save size={14} />
                  <span className="hidden sm:inline">{t.exportTestImage}</span>
                </button>
              )}
            </div>
          )}

          {/* Stack Visualization */}
          {viewMode === 'stack' ? (
            <div 
              ref={scrollRef}
              className="flex-1 overflow-y-auto p-8 flex flex-col-reverse gap-1 scroll-smooth no-scrollbar"
              style={{ scrollbarWidth: 'none' }}
            >
              <AnimatePresence initial={false}>
                {stack.map((item, index) => (
                  <motion.div
                    key={item.id}
                    initial={{ opacity: 0, y: 20, scale: 0.8 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.5, transition: { duration: 0.15 } }}
                    className={`h-12 w-full rounded-lg ${item.color} shadow-sm border-b-4 border-black/10 flex items-center justify-center text-white font-mono font-bold text-lg`}
                  >
                    {stack.length - index}
                  </motion.div>
                ))}
              </AnimatePresence>
              
              {stack.length === 0 && !isProcessing && !uploadedImage && (
                <div className="h-full flex flex-col items-center justify-center text-zinc-300 gap-4">
                  <div className="w-24 h-24 border-2 border-dashed border-zinc-200 rounded-2xl flex items-center justify-center">
                    <Plus size={32} />
                  </div>
                  <p className="text-sm font-medium">{t.stackEmpty}</p>
                </div>
              )}

              {isProcessing && (
                <div className="h-full flex flex-col items-center justify-center text-emerald-500 gap-4">
                  <motion.div 
                    animate={{ rotate: 360 }}
                    transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                  >
                    <Loader2 size={48} />
                  </motion.div>
                  <p className="text-sm font-bold animate-pulse">{analysisMode === 'local' ? t.localAnalyzing : t.aiAnalyzing}</p>
                </div>
              )}
            </div>
          ) : (
            <div className="flex-1 relative bg-zinc-900 overflow-hidden flex items-center justify-center">
              {(correctedImageUrl || uploadedImage) && (
                <div className="relative w-full h-full flex items-center justify-center p-4">
                  <div className="relative inline-block max-w-full max-h-full">
                    <img
                      ref={overlayImageRef}
                      src={correctedImageUrl || uploadedImage || ''}
                      alt="Analyzed"
                      className="max-w-full max-h-[calc(100vh-12rem)] object-contain select-none"
                      draggable={false}
                      style={{ opacity: localResult ? 0.82 : 1 }}
                    />
                    {localResult && (
                      <svg
                        width="100%"
                        height="100%"
                        viewBox={`0 0 ${localResult.width} ${localResult.height}`}
                        preserveAspectRatio="none"
                        className="absolute inset-0 touch-none"
                      >
                        <defs>
                          <filter id="dot-glow">
                            <feGaussianBlur stdDeviation="3" result="blur"/>
                            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                          </filter>
                        </defs>

                        {/* ROI region tint */}
                        <rect
                          x="0" y={localResult.boundaryY1 ?? localResult.y1}
                          width={localResult.width} height={(localResult.boundaryY2 ?? localResult.y2) - (localResult.boundaryY1 ?? localResult.y1)}
                          fill="rgba(239,68,68,0.06)"
                        />

                        {/* ROI top boundary */}
                        <line
                          x1="0" y1={localResult.boundaryY1 ?? localResult.y1} x2={localResult.width} y2={localResult.boundaryY1 ?? localResult.y1}
                          stroke="rgba(239,68,68,0.85)" strokeWidth="1.5" strokeDasharray="6,4"
                        />
                        {/* ROI bottom boundary */}
                        <line
                          x1="0" y1={localResult.boundaryY2 ?? localResult.y2} x2={localResult.width} y2={localResult.boundaryY2 ?? localResult.y2}
                          stroke="rgba(239,68,68,0.85)" strokeWidth="1.5" strokeDasharray="6,4"
                        />

                        {/* Vertical centre line (SVG mirror of the drawn red line in the image) */}
                        <line
                          x1={localResult.centerX} y1={localResult.boundaryY1 ?? localResult.y1}
                          x2={localResult.centerX} y2={localResult.boundaryY2 ?? localResult.y2}
                          stroke="rgba(255,60,60,0.55)" strokeWidth="2"
                        />

                        {/* Intersection dots — one per counted layer */}
                        {localResult.intersectionPoints.map((pt, i) => (
                          <g key={i} filter="url(#dot-glow)">
                            <circle
                              cx={pt.x} cy={pt.y}
                              r="5"
                              fill="rgba(16,185,129,0.25)"
                              stroke="rgba(16,185,129,0.7)"
                              strokeWidth="1"
                            />
                            <circle
                              cx={pt.x} cy={pt.y}
                              r="2.5"
                              fill="#10b981"
                            />
                          </g>
                        ))}

                        {/* Drag handle — ROI start */}
                        <g
                          className="cursor-ns-resize"
                          onPointerDown={(event) => {
                            event.preventDefault();
                            setDraggingHandle('start');
                            updateRoiHandle(event.clientY, 'start');
                          }}
                        >
                          <line x1="0" y1={localResult.boundaryY1 ?? localResult.y1} x2={localResult.width} y2={localResult.boundaryY1 ?? localResult.y1} stroke="transparent" strokeWidth="14" />
                          <circle cx={localResult.width * 0.92} cy={localResult.boundaryY1 ?? localResult.y1} r="7" fill="#ef4444" stroke="white" strokeWidth="1.5" />
                        </g>

                        {/* Drag handle — ROI end */}
                        <g
                          className="cursor-ns-resize"
                          onPointerDown={(event) => {
                            event.preventDefault();
                            setDraggingHandle('end');
                            updateRoiHandle(event.clientY, 'end');
                          }}
                        >
                          <line x1="0" y1={localResult.boundaryY2 ?? localResult.y2} x2={localResult.width} y2={localResult.boundaryY2 ?? localResult.y2} stroke="transparent" strokeWidth="14" />
                          <circle cx={localResult.width * 0.92} cy={localResult.boundaryY2 ?? localResult.y2} r="7" fill="#ef4444" stroke="white" strokeWidth="1.5" />
                        </g>
                      </svg>
                    )}

                    {/* Deskew angle badge */}
                    {localResult?.deskewAngle !== undefined && localResult.deskewAngle !== 0 && (
                      <div className="absolute top-2 left-2 bg-black/70 backdrop-blur-sm text-white text-xs font-mono px-2 py-1 rounded-lg flex items-center gap-1.5 pointer-events-none select-none">
                        <svg width="12" height="12" viewBox="0 0 12 12" fill="none" className="shrink-0">
                          <path d="M2 10 L10 2 M7 2 L10 2 L10 5" stroke="#ff4444" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                        <span>
                          {language === 'zh' ? '已校正' : 'Deskewed'}&nbsp;
                          <span className="text-red-400">{localResult.deskewAngle > 0 ? '+' : ''}{localResult.deskewAngle.toFixed(1)}°</span>
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Counter Overlay */}
          <div className="absolute top-4 right-4 bg-white/80 backdrop-blur-md border border-zinc-200 rounded-2xl p-4 shadow-xl flex flex-col items-center min-w-[100px] z-50">
            <span className="text-xs font-bold text-zinc-400 uppercase tracking-widest mb-1">{t.total}</span>
            <div className="flex flex-col items-center gap-1">
              {isAdjusting ? (
                <div className="flex flex-col items-center gap-2">
                  <input 
                    type="number"
                    value={adjustValue}
                    onChange={(e) => setAdjustValue(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleAdjustSubmit()}
                    autoFocus
                    className="w-16 text-center text-2xl font-black bg-zinc-100 border border-zinc-300 rounded-lg outline-none focus:ring-2 focus:ring-emerald-500"
                  />
                  <div className="flex gap-2">
                    <button 
                      onClick={handleAdjustSubmit}
                      className="p-1 text-emerald-600 hover:bg-emerald-50 rounded transition-colors"
                    >
                      <Check size={16} />
                    </button>
                    <button 
                      onClick={() => setIsAdjusting(false)}
                      className="p-1 text-zinc-400 hover:bg-zinc-100 rounded transition-colors"
                    >
                      <X size={16} />
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  <motion.span 
                    key={stack.length}
                    initial={{ scale: 1.2, color: '#10b981' }}
                    animate={{ scale: 1, color: '#18181b' }}
                    data-testid="total-count"
                    className="text-4xl font-black tabular-nums"
                  >
                    {stack.length}
                  </motion.span>
                  <button 
                    onClick={() => {
                      setAdjustValue(stack.length.toString());
                      setIsAdjusting(true);
                    }}
                    className="text-[10px] font-bold text-emerald-600 hover:underline px-2 py-1 cursor-pointer"
                  >
                    {t.adjust}
                  </button>
                </>
              )}
            </div>
          </div>

          {/* Error Message */}
          <AnimatePresence>
            {error && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="absolute bottom-4 left-4 right-4 bg-rose-50 border border-rose-200 text-rose-600 p-3 rounded-xl text-xs font-medium flex items-center justify-between"
              >
                <span>{error}</span>
                <button onClick={() => setError(null)}><X size={14} /></button>
              </motion.div>
            )}
          </AnimatePresence>

          <div className="hidden" data-testid="local-result-json">
            {JSON.stringify({
              count: stack.length,
              analysisMode,
              uploadedImage: Boolean(uploadedImage),
              error,
              isProcessing,
              localResult
            })}
          </div>
        </div>

        {/* Controls */}
        <div className="mt-8 flex justify-center">
          <button
            onClick={reset}
            disabled={stack.length === 0 || isProcessing}
            className="w-full flex items-center justify-center gap-2 p-4 rounded-2xl bg-zinc-100 text-zinc-600 hover:bg-zinc-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-bold"
            title={t.resetStack}
          >
            <RotateCcw size={24} />
            <span>{t.resetStack}</span>
          </button>
        </div>

        {/* Footer */}
        <footer className="mt-8 text-center">
          <p className="text-xs text-zinc-400 font-medium uppercase tracking-tighter">
            {t.footer} • v{__APP_VERSION__}
          </p>
        </footer>
      </div>

      {/* Camera Modal */}
      <AnimatePresence>
        {isCameraActive && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black flex flex-col"
          >
            <div className="p-4 flex justify-between items-center text-white relative z-10">
              <h2 className="font-bold">{t.scanStack}</h2>
              <div className="flex gap-4">
                {cameras.length > 1 && (
                  <button onClick={switchCamera} className="p-2 bg-white/10 rounded-full hover:bg-white/20 transition-colors">
                    <SwitchCamera size={24} />
                  </button>
                )}
                <button onClick={stopCamera} className="p-2 bg-white/10 rounded-full hover:bg-white/20 transition-colors">
                  <X size={24} />
                </button>
              </div>
            </div>
            
            <div ref={videoContainerRef} className="flex-1 relative overflow-hidden flex items-center justify-center bg-black">
              <video 
                ref={videoRef} 
                autoPlay 
                playsInline 
                muted
                className="w-full h-full object-cover transition-transform duration-200"
                style={{ transform: `scale(${zoom})` }}
              />
              {/* Scan Overlay */}
              <div className="absolute inset-0 border-[40px] border-black/40 pointer-events-none flex items-center justify-center">
                <div ref={overlayBoxRef} className="w-full max-w-[280px] aspect-[3/4] border-2 border-emerald-400/50 rounded-2xl relative">
                  <div className="absolute top-0 left-0 w-8 h-8 border-t-4 border-l-4 border-emerald-400 rounded-tl-lg" />
                  <div className="absolute top-0 right-0 w-8 h-8 border-t-4 border-r-4 border-emerald-400 rounded-tr-lg" />
                  <div className="absolute bottom-0 left-0 w-8 h-8 border-b-4 border-l-4 border-emerald-400 rounded-bl-lg" />
                  <div className="absolute bottom-0 right-0 w-8 h-8 border-b-4 border-r-4 border-emerald-400 rounded-br-lg" />
                  
                  {/* Scanning Line */}
                  <motion.div 
                    animate={{ top: ['0%', '100%', '0%'] }}
                    transition={{ repeat: Infinity, duration: 3, ease: "linear" }}
                    className="absolute left-0 right-0 h-0.5 bg-emerald-400 shadow-[0_0_15px_rgba(52,211,153,0.8)]"
                  />
                </div>
              </div>

              {/* Zoom Control */}
              <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex items-center gap-4 bg-black/40 backdrop-blur-md px-6 py-3 rounded-full border border-white/10">
                <Minus size={16} className="text-white/60" />
                <input 
                  type="range" 
                  min="1" 
                  max="3" 
                  step="0.1" 
                  value={zoom}
                  onChange={(e) => setZoom(parseFloat(e.target.value))}
                  className="w-32 accent-emerald-400"
                />
                <Plus size={16} className="text-white/60" />
              </div>
            </div>

            <div className="p-12 flex justify-center items-center bg-black/80">
              <button 
                onClick={captureImage}
                className="w-20 h-20 rounded-full border-4 border-white flex items-center justify-center p-1 active:scale-90 transition-transform"
              >
                <div className="w-full h-full bg-white rounded-full" />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <canvas ref={canvasRef} className="hidden" />

      {/* Settings Modal */}
      <AnimatePresence>
        {isSettingsOpen && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] bg-black/60 backdrop-blur-sm flex items-center justify-center p-6"
          >
            <motion.div 
              initial={{ scale: 0.9, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.9, y: 20 }}
              className="bg-white w-full max-w-sm rounded-3xl shadow-2xl overflow-hidden"
            >
              <div className="flex flex-col max-h-[85vh]">
                <div className="p-6 border-b border-zinc-100 flex justify-between items-center shrink-0">
                  <div className="flex items-center gap-2">
                    <Settings className="text-zinc-400" size={20} />
                    <h2 className="font-bold text-lg">{t.modelSettings}</h2>
                  </div>
                  <button onClick={() => setIsSettingsOpen(false)} className="p-2 hover:bg-zinc-100 rounded-full transition-colors">
                    <X size={20} />
                  </button>
                </div>
  
                <div className="p-6 space-y-6 overflow-y-auto custom-scrollbar">
                  {/* Local CV Settings */}
                  <div className="space-y-4 border-b border-zinc-100 pb-6">
                    <div className="flex justify-between items-center">
                      <h3 className="text-sm font-bold text-zinc-800 flex items-center gap-2">
                        <Zap size={16} className="text-emerald-500" />
                        {t.fastMode} Settings
                      </h3>
                      <button 
                        onClick={() => setLocalParams({ roiStart: 0.12, roiEnd: 0.94, distance: 8, prominence: 10 })}
                        className="text-[10px] font-bold text-zinc-400 hover:text-emerald-600 uppercase tracking-widest flex items-center gap-1 transition-colors"
                      >
                        <RotateCcw size={12} />
                        {t.resetParams}
                      </button>
                    </div>
                    
                    <div className="flex gap-4">
                      <div className="flex-1">
                        <label className="block text-[10px] font-bold text-zinc-400 uppercase tracking-widest mb-1.5 ml-1 flex justify-between">
                          <span>{t.roiStart}</span>
                          <span className="text-emerald-600">{localParams.roiStart}</span>
                        </label>
                        <input 
                          type="range" min="0" max="0.5" step="0.05" 
                          value={localParams.roiStart}
                          onChange={(e) => setLocalParams(prev => ({ ...prev, roiStart: parseFloat(e.target.value) }))}
                          className="w-full accent-emerald-500"
                        />
                      </div>
                      <div className="flex-1">
                        <label className="block text-[10px] font-bold text-zinc-400 uppercase tracking-widest mb-1.5 ml-1 flex justify-between">
                          <span>{t.roiEnd}</span>
                          <span className="text-emerald-600">{localParams.roiEnd}</span>
                        </label>
                        <input 
                          type="range" min="0.5" max="1" step="0.05" 
                          value={localParams.roiEnd}
                          onChange={(e) => setLocalParams(prev => ({ ...prev, roiEnd: parseFloat(e.target.value) }))}
                          className="w-full accent-emerald-500"
                        />
                      </div>
                    </div>

                    <div className="flex gap-4">
                      <div className="flex-1">
                        <label className="block text-[10px] font-bold text-zinc-400 uppercase tracking-widest mb-1.5 ml-1 flex justify-between">
                          <span>{t.peakDistance}</span>
                          <span className="text-emerald-600">{localParams.distance}</span>
                        </label>
                        <input 
                          type="range" min="1" max="20" step="1" 
                          value={localParams.distance}
                          onChange={(e) => setLocalParams(prev => ({ ...prev, distance: parseInt(e.target.value) }))}
                          className="w-full accent-emerald-500"
                        />
                      </div>
                      <div className="flex-1">
                        <label className="block text-[10px] font-bold text-zinc-400 uppercase tracking-widest mb-1.5 ml-1 flex justify-between">
                          <span>{t.peakProminence}</span>
                          <span className="text-emerald-600">{localParams.prominence}</span>
                        </label>
                        <input 
                          type="range" min="1" max="50" step="1" 
                          value={localParams.prominence}
                          onChange={(e) => setLocalParams(prev => ({ ...prev, prominence: parseInt(e.target.value) }))}
                          className="w-full accent-emerald-500"
                        />
                      </div>
                    </div>
                  </div>

                  {/* AI Settings */}
                  <div className="space-y-4">
                    <h3 className="text-sm font-bold text-zinc-800 flex items-center gap-2">
                      <Cpu size={16} className="text-emerald-500" />
                      {t.aiMode} Settings
                    </h3>
                    {/* Type Selector */}
                    <div className="flex p-1 bg-zinc-100 rounded-xl">
                    <button 
                      onClick={() => setModelConfig(prev => ({ ...prev, type: 'gemini' }))}
                      className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-sm font-bold transition-all ${modelConfig.type === 'gemini' ? 'bg-white shadow-sm text-emerald-600' : 'text-zinc-500 hover:text-zinc-700'}`}
                    >
                      <Cpu size={16} />
                      {t.gemini}
                    </button>
                    <button 
                      onClick={() => setModelConfig(prev => ({ ...prev, type: 'custom' }))}
                      className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-sm font-bold transition-all ${modelConfig.type === 'custom' ? 'bg-white shadow-sm text-emerald-600' : 'text-zinc-500 hover:text-zinc-700'}`}
                    >
                      <Globe size={16} />
                      {t.customApi}
                    </button>
                  </div>
  
                  <div className="space-y-4">
                    {modelConfig.type === 'custom' && (
                      <div>
                        <label className="block text-[10px] font-bold text-zinc-400 uppercase tracking-widest mb-1.5 ml-1">{t.endpointUrl}</label>
                        <input 
                          type="text"
                          placeholder={t.placeholderEndpoint}
                          value={modelConfig.endpoint || ''}
                          onChange={(e) => setModelConfig(prev => ({ ...prev, endpoint: e.target.value }))}
                          className="w-full px-4 py-3 bg-zinc-50 border border-zinc-200 rounded-xl outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 transition-all text-sm"
                        />
                      </div>
                    )}
                    
                    <div>
                      <label className="block text-[10px] font-bold text-zinc-400 uppercase tracking-widest mb-1.5 ml-1">
                        {modelConfig.type === 'gemini' ? t.apiKeyOptional : t.apiKey}
                      </label>
                      <input 
                        type="password"
                        placeholder={modelConfig.type === 'gemini' ? t.leaveEmpty : 'sk-...'}
                        value={modelConfig.apiKey || ''}
                        onChange={(e) => setModelConfig(prev => ({ ...prev, apiKey: e.target.value }))}
                        className="w-full px-4 py-3 bg-zinc-50 border border-zinc-200 rounded-xl outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 transition-all text-sm"
                      />
                    </div>
  
                    <div>
                      <label className="block text-[10px] font-bold text-zinc-400 uppercase tracking-widest mb-1.5 ml-1">{t.modelName}</label>
                      <input 
                        type="text"
                        placeholder={modelConfig.type === 'gemini' ? t.placeholderModel : t.placeholderCustomModel}
                        value={modelConfig.modelName || ''}
                        onChange={(e) => setModelConfig(prev => ({ ...prev, modelName: e.target.value }))}
                        className="w-full px-4 py-3 bg-zinc-50 border border-zinc-200 rounded-xl outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 transition-all text-sm"
                      />
                    </div>
  
                    <div>
                      <label className="block text-[10px] font-bold text-zinc-400 uppercase tracking-widest mb-1.5 ml-1 flex justify-between">
                        <span>{t.temperature}</span>
                        <span className="text-emerald-600">{modelConfig.temperature ?? 1}</span>
                      </label>
                      <input 
                        type="range" 
                        min="0" 
                        max="2" 
                        step="0.1" 
                        value={modelConfig.temperature ?? 1}
                        onChange={(e) => setModelConfig(prev => ({ ...prev, temperature: parseFloat(e.target.value) }))}
                        className="w-full accent-emerald-500"
                      />
                      <p className="text-[10px] text-zinc-400 mt-1">
                        {t.tempDesc}
                      </p>
                    </div>
                  </div>
                  </div>
                </div>
  
                <div className="p-6 border-t border-zinc-100 shrink-0">
                  <button 
                    onClick={() => setIsSettingsOpen(false)}
                    className="w-full py-4 bg-zinc-900 text-white rounded-2xl font-bold flex items-center justify-center gap-2 hover:bg-zinc-800 transition-colors shadow-lg shadow-zinc-200"
                  >
                    <Save size={20} />
                    {t.saveConfig}
                  </button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <style dangerouslySetInnerHTML={{ __html: `
        .no-scrollbar::-webkit-scrollbar {
          display: none;
        }
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #e4e4e7;
          border-radius: 10px;
        }
      `}} />
    </div>
  );
}
