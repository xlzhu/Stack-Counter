/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Plus, Minus, RotateCcw, Layers, Camera, Upload, Loader2, X, Check, Settings, Globe, Cpu, Save, Languages } from 'lucide-react';
import { GoogleGenAI } from "@google/genai";

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
    manualCountNote: "Note: For a similar stack, a manual count of {count} was previously recorded. Use this as a reference but verify the current image exactly. "
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
    manualCountNote: "注意：对于类似的堆栈，之前记录的手动计数为 {count}。请将其作为参考，但要准确核实当前图像。"
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

const compressImage = (dataUrl: string, maxWidth = 1024, maxHeight = 1024, initialQuality = 0.7): Promise<string> => {
  return new Promise((resolve) => {
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
    img.onerror = () => resolve(dataUrl);
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
  const [modelConfig, setModelConfig] = useState<ModelConfig>(() => {
    const saved = localStorage.getItem('stack_counter_config');
    if (saved) {
      try {
        return JSON.parse(saved);
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

  const analyzeImage = async (base64Data: string) => {
    setIsProcessing(true);
    setError(null);
    try {
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
                    data: base64Data.split(',')[1]
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
        const isNvidia = finalEndpoint.includes('nvidia.com') || finalEndpoint.includes('localhost') || finalEndpoint.includes('127.0.0.1');
        const isNvidiaCv = isNvidia && (finalEndpoint.includes('/cv/') || modelConfig.modelName?.toLowerCase().includes('ocr') || modelConfig.modelName?.toLowerCase().includes('cv'));

        try {
          const urlObj = new URL(finalEndpoint);
          if (isNvidiaCv) {
            // For NVIDIA CV models, if it's a base URL, append /v1/infer
            if (urlObj.pathname === '/' || urlObj.pathname === '/v1' || urlObj.pathname === '/v1/') {
              finalEndpoint = finalEndpoint.replace(/\/$/, '') + '/v1/infer';
            }
          } else {
            // Only append /chat/completions if it looks like a base URL (e.g., ends with /v1 or has no specific path)
            if (!finalEndpoint.endsWith('/chat/completions') && 
                (urlObj.pathname === '/v1' || urlObj.pathname === '/v1/' || urlObj.pathname === '/')) {
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
        const isMultimodal = !modelName.includes('nemotron-3-super') && !modelName.includes('qwen3.5-122b');
        
        if (isNvidiaCv) {
          // NVIDIA CV-specific API format (e.g., nemotron-ocr-v1)
          // Requires an "input" wrapper as a list
          payload = {
            input: [
              {
                type: 'image_url',
                url: base64Data
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
                  { type: 'text', text: isNvidia ? `${systemInstruction}\n\n${promptText}` : promptText },
                  {
                    type: 'image_url',
                    image_url: { url: base64Data }
                  }
                ] : (isNvidia ? `${systemInstruction}\n\n${promptText}` : promptText)
              }
            ],
            temperature: modelConfig.temperature ?? 1,
            max_tokens: 1024
          };

          if (!isNvidia && isMultimodal) {
            payload.messages.unshift({ role: 'system', content: systemInstruction });
          }

          // Special handling for Nemotron-3 Super with thinking capabilities
          if (modelName.includes('nemotron-3-super')) {
            payload.chat_template_kwargs = { enable_thinking: true };
            payload.reasoning_budget = 16384;
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
            body: payload
          })
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          const errorObj = errorData.error || errorData;
          let errorMessage = errorObj.message || errorObj.detail || errorObj.type || (typeof errorData === 'string' ? errorData : null);
          
          if (response.status === 422) {
            errorMessage = `API Error 422: Invalid request format. For NVIDIA CV models, ensure the URL is correct. Details: ${JSON.stringify(errorObj)}`;
          } else if (errorObj.type === 'access_terminated_error') {
            errorMessage = "API Access Terminated: Your API key or account has been disabled by the provider. Please check your account status.";
          } else if (errorObj.type === 'insufficient_quota' || errorObj.code === 'insufficient_quota') {
            errorMessage = "Insufficient Quota: Your API account has run out of credits or reached its limit.";
          }
          
          throw new Error(errorMessage || `API Error: ${response.status}`);
        }

        const data = await response.json();
        
        let resultText = "";
        if (isNvidiaCv) {
          // NVIDIA CV API usually returns { "content": "..." } or { "text": "..." }
          // Or a nested structure: { "data": [ { "text_detections": [ { "text_prediction": { "text": "..." } } ] } ] }
          if (data.data && Array.isArray(data.data)) {
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
            resultText = data.content || data.text || data.description || "";
          }
        } else {
          // Standard Chat API
          const message = data.choices?.[0]?.message;
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
          stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
              facingMode: 'environment', 
              width: { ideal: 1920 }, 
              height: { ideal: 1080 } 
            } 
          });
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
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
  }, [isCameraActive]);

  const stopCamera = () => {
    setIsCameraActive(false);
  };

  const captureImage = async () => {
    if (videoRef.current && canvasRef.current) {
      const context = canvasRef.current.getContext('2d');
      if (context) {
        canvasRef.current.width = videoRef.current.videoWidth;
        canvasRef.current.height = videoRef.current.videoHeight;
        context.drawImage(videoRef.current, 0, 0);
        const base64Data = canvasRef.current.toDataURL('image/jpeg', 0.85);
        stopCamera();
        const compressedDataUrl = await compressImage(base64Data);
        analyzeImage(compressedDataUrl);
      }
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = async () => {
        const compressedDataUrl = await compressImage(reader.result as string);
        analyzeImage(compressedDataUrl);
        // Clear the input value so the same file can be uploaded again
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
      };
      reader.readAsDataURL(file);
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
      <div className="max-w-md mx-auto h-screen flex flex-col p-6">
        
        {/* Header */}
        <header className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-2">
            <div className="p-2 bg-zinc-900 rounded-lg text-white">
              <Layers size={20} />
            </div>
            <h1 className="text-xl font-bold tracking-tight">{t.title}</h1>
          </div>
          <div className="flex gap-2">
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
            accept="image/*" 
            className="hidden" 
          />
        </header>

        {/* Main Display */}
        <div className="flex-1 flex flex-col min-h-0 bg-white rounded-3xl border border-zinc-200 shadow-sm overflow-hidden relative">
          
          {/* Stack Visualization */}
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
            
            {stack.length === 0 && !isProcessing && (
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
                <p className="text-sm font-bold animate-pulse">{t.aiAnalyzing}</p>
              </div>
            )}
          </div>

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
            {t.footer} • v1.0.3
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
            <div className="p-4 flex justify-between items-center text-white">
              <h2 className="font-bold">{t.scanStack}</h2>
              <button onClick={stopCamera} className="p-2 bg-white/10 rounded-full">
                <X size={24} />
              </button>
            </div>
            
            <div className="flex-1 relative overflow-hidden flex items-center justify-center bg-black">
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
                <div className="w-full max-w-[280px] aspect-[3/4] border-2 border-emerald-400/50 rounded-2xl relative">
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
