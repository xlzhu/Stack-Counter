/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Plus, Minus, RotateCcw, Layers, Camera, Upload, Loader2, X, Check } from 'lucide-react';
import { GoogleGenAI } from "@google/genai";

interface StackItem {
  id: string;
  color: string;
}

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

export default function App() {
  const [stack, setStack] = useState<StackItem[]>([]);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastManualCount, setLastManualCount] = useState<number | null>(null);
  
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
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
      
      const systemInstruction = "You are a precision counting expert. " + 
        (lastManualCount ? `Note: For a similar stack, a manual count of ${lastManualCount} was previously recorded. Use this as a reference but verify the current image exactly. ` : "") +
        "Count each individual layer by looking at the distinct horizontal lines on the vertical edges. Be extremely precise and do not skip any layers. Return ONLY the final integer count.";

      const response = await ai.models.generateContent({
        model: "gemini-3.1-pro-preview",
        contents: [
          {
            parts: [
              { text: "Methodically count the number of thin stacked layers in this image (e.g., stacked trays). These layers are very dense. First, identify the boundaries of the stack. Then, count each individual layer by looking at the distinct horizontal lines on the vertical edges. Be extremely precise and do not skip any layers. Return ONLY the final integer count." },
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
          thinkingConfig: {
            includeThoughts: true
          }
        }
      });

      const resultText = response.text?.trim() || "0";
      const count = parseInt(resultText.replace(/[^0-9]/g, ''), 10);
      
      if (!isNaN(count)) {
        updateStackCount(count);
      } else {
        setError("Could not determine count. Please try again.");
      }
    } catch (err) {
      console.error("AI Analysis Error:", err);
      setError("Analysis failed. Please check your connection or image quality.");
    } finally {
      setIsProcessing(false);
      setIsCameraActive(false);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraActive(true);
      }
    } catch (err) {
      console.error("Camera Error:", err);
      setError("Could not access camera. Please check permissions.");
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
  };

  const captureImage = () => {
    if (videoRef.current && canvasRef.current) {
      const context = canvasRef.current.getContext('2d');
      if (context) {
        canvasRef.current.width = videoRef.current.videoWidth;
        canvasRef.current.height = videoRef.current.videoHeight;
        context.drawImage(videoRef.current, 0, 0);
        const base64Data = canvasRef.current.toDataURL('image/jpeg');
        stopCamera();
        analyzeImage(base64Data);
      }
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        analyzeImage(reader.result as string);
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
            <h1 className="text-xl font-bold tracking-tight">Stack-Counter</h1>
          </div>
          <div className="flex gap-2">
            <button 
              onClick={() => fileInputRef.current?.click()}
              className="p-2 bg-zinc-100 rounded-lg text-zinc-600 hover:bg-zinc-200 transition-colors"
              title="Upload Photo"
            >
              <Upload size={20} />
            </button>
            <button 
              onClick={startCamera}
              className="p-2 bg-zinc-100 rounded-lg text-zinc-600 hover:bg-zinc-200 transition-colors"
              title="Scan with Camera"
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
                <p className="text-sm font-medium">Stack is empty</p>
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
                <p className="text-sm font-bold animate-pulse">AI Analyzing Layers...</p>
              </div>
            )}
          </div>

          {/* Counter Overlay */}
          <div className="absolute top-4 right-4 bg-white/80 backdrop-blur-md border border-zinc-200 rounded-2xl p-4 shadow-xl flex flex-col items-center min-w-[100px] z-50">
            <span className="text-xs font-bold text-zinc-400 uppercase tracking-widest mb-1">Total</span>
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
                    ADJUST
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
            title="Reset"
          >
            <RotateCcw size={24} />
            <span>RESET STACK</span>
          </button>
        </div>

        {/* Footer */}
        <footer className="mt-8 text-center">
          <p className="text-xs text-zinc-400 font-medium uppercase tracking-tighter">
            AI-Powered Layer Detection
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
              <h2 className="font-bold">Scan Stack</h2>
              <button onClick={stopCamera} className="p-2 bg-white/10 rounded-full">
                <X size={24} />
              </button>
            </div>
            
            <div className="flex-1 relative overflow-hidden flex items-center justify-center">
              <video 
                ref={videoRef} 
                autoPlay 
                playsInline 
                className="w-full h-full object-cover"
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

      <style dangerouslySetInnerHTML={{ __html: `
        .no-scrollbar::-webkit-scrollbar {
          display: none;
        }
      `}} />
    </div>
  );
}
