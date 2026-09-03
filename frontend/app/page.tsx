"use client";

import React, { useState, useEffect, useRef } from "react";
import { 
  BrainCircuit, 
  UploadCloud, 
  FileCheck2, 
  Loader2, 
  Send, 
  ShieldCheck, 
  Sparkles, 
  BookOpen,
  CheckCircle2,
  AlertCircle,
  Database,
  Layers
} from "lucide-react";
import ReactMarkdown from "react-markdown";

interface Citation {
  source_id: number;
  chunk_id: string;
  similarity: number;
  preview: string;
}

interface Message {
  id: string;
  sender: "user" | "assistant";
  content: string;
  confidenceScore?: number;
  citations?: Citation[];
  timestamp: string;
}

interface DocumentState {
  id: string;
  filename: string;
  status: "IDLE" | "UPLOADING" | "PROCESSING" | "COMPLETED" | "FAILED";
}

const API_BASE = 
  process.env.NEXT_PUBLIC_API_URL || 
  process.env.NEXT_PUBLIC_API_BASE_URL || 
  "http://localhost:8000";

export default function DocuMindDashboard() {
  const [hasMounted, setHasMounted] = useState(false);
  const [docState, setDocState] = useState<DocumentState>({
    id: "",
    filename: "",
    status: "IDLE",
  });
  
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      sender: "assistant",
      content:
        "Welcome to **DocuMind AI**. Upload a PDF document on the left panel. Documents are indexed asynchronously with Celery, mapped to 768-dim embeddings, and grounded via Supabase pgvector.",
      timestamp: "", 
    },
  ]);

  const [inputQuery, setInputQuery] = useState("");
  const [isInferring, setIsInferring] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  // Eliminate Hydration Mismatches by stamping initial times strictly on client mount
  useEffect(() => {
    setHasMounted(true);
    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === "welcome"
          ? {
              ...msg,
              timestamp: new Date().toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              }),
            }
          : msg
      )
    );
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isInferring]);

  // Polling Hook for Ingestion Lifecycle
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (docState.status === "PROCESSING" && docState.id) {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/api/v1/documents/${docState.id}/status`);
          if (res.ok) {
            const data = await res.json();
            if (data.status === "COMPLETED") {
              setDocState((prev) => ({ ...prev, status: "COMPLETED" }));
              clearInterval(interval);
            } else if (data.status === "FAILED" || data.status === "ERROR") {
              setDocState((prev) => ({ ...prev, status: "FAILED" }));
              clearInterval(interval);
            }
          }
        } catch (err) {
          console.error("Polling error:", err);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [docState.status, docState.id]);

  const handleFileUpload = async (file: File) => {
    if (!file || !file.name.toLowerCase().endsWith(".pdf")) {
      alert("Please provide a valid PDF file.");
      return;
    }

    setDocState({ id: "", filename: file.name, status: "UPLOADING" });

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/api/v1/documents/upload`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || "Upload request rejected");
      }
      const data = await res.json();

      setDocState({
        id: data.document_id,
        filename: file.name,
        status: "PROCESSING",
      });
    } catch (err: any) {
      alert(err.message || "Failed to upload document");
      setDocState({ id: "", filename: "", status: "FAILED" });
    }
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    const query = inputQuery.trim();
    if (!query || docState.status !== "COMPLETED" || isInferring) return;

    const timeString = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

    const userMessage: Message = {
      id: `usr-${Date.now()}`,
      sender: "user",
      content: query,
      timestamp: timeString,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputQuery("");
    setIsInferring(true);

    const botMessageId = `bot-${Date.now()}`;
    const botPlaceholder: Message = {
      id: botMessageId,
      sender: "assistant",
      content: "",
      timestamp: timeString,
    };

    setMessages((prev) => [...prev, botPlaceholder]);

    try {
      // 1. Try real-time SSE Streaming
      const response = await fetch(`${API_BASE}/api/v1/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          document_id: docState.id,
          question: query,
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error("Streaming connection failed");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let streamBuffer = "";
      let accumulatedContent = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        streamBuffer += decoder.decode(value, { stream: true });
        const lines = streamBuffer.split("\n\n");
        streamBuffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const rawData = line.replace("data: ", "").trim();
            if (!rawData) continue;

            try {
              const payload = JSON.parse(rawData);

              if (payload.type === "meta") {
                if (payload.citations && payload.citations.length > 0) {
                  setActiveCitation(payload.citations[0]);
                }
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === botMessageId
                      ? {
                          ...msg,
                          confidenceScore: payload.confidence_score,
                          citations: payload.citations,
                        }
                      : msg
                  )
                );
              } else if (payload.type === "token") {
                accumulatedContent += payload.token;
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === botMessageId
                      ? { ...msg, content: accumulatedContent }
                      : msg
                  )
                );
              } else if (payload.type === "terminal") {
                accumulatedContent = payload.answer;
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === botMessageId
                      ? {
                          ...msg,
                          content: payload.answer,
                          confidenceScore: payload.confidence_score,
                          citations: payload.citations || [],
                        }
                      : msg
                  )
                );
              } else if (payload.type === "error") {
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === botMessageId
                      ? {
                          ...msg,
                          content: `⚠️ **Inference Error**: ${payload.error || "Generation error"}`,
                        }
                      : msg
                  )
                );
              }
            } catch (pErr) {
              console.warn("Unparseable SSE frame:", rawData);
            }
          }
        }
      }
    } catch (streamErr: any) {
      // 2. Fallback to synchronous chat endpoint if streaming encounters issues
      try {
        const fallbackRes = await fetch(`${API_BASE}/api/v1/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            document_id: docState.id,
            question: query,
          }),
        });

        const fallbackData = await fallbackRes.json();
        if (!fallbackRes.ok) throw new Error(fallbackData.detail || "Inference failed");

        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === botMessageId
              ? {
                  ...msg,
                  content: fallbackData.answer,
                  confidenceScore: fallbackData.confidence_score,
                  citations: fallbackData.citations,
                }
              : msg
          )
        );

        if (fallbackData.citations && fallbackData.citations.length > 0) {
          setActiveCitation(fallbackData.citations[0]);
        }
      } catch (finalErr: any) {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === botMessageId
              ? {
                  ...msg,
                  content: `⚠️ **Inference Error**: ${finalErr.message}`,
                }
              : msg
          )
        );
      }
    } finally {
      setIsInferring(false);
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#070A12] text-slate-100 font-sans">
      
      {/* 1. Left Column: Ingestion & System Control (320px) */}
      <aside className="w-80 border-r border-slate-800/80 bg-[#0B0F19]/90 flex flex-col justify-between p-5 shrink-0">
        <div className="space-y-6">
          
          {/* Header */}
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
              <BrainCircuit className="w-5 h-5" />
            </div>
            <div>
              <h1 className="font-bold text-base tracking-tight flex items-center gap-1">
                DocuMind <span className="text-emerald-400">AI</span>
              </h1>
              <p className="text-[10px] font-mono text-slate-500 uppercase tracking-widest">Async RAG Engine</p>
            </div>
          </div>

          {/* Upload Dropzone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setIsDragging(false);
              if (e.dataTransfer.files?.[0]) handleFileUpload(e.dataTransfer.files[0]);
            }}
            onClick={() => fileInputRef.current?.click()}
            className={`border border-dashed rounded-xl p-5 flex flex-col items-center justify-center gap-2.5 cursor-pointer transition-all ${
              isDragging 
                ? "border-emerald-400 bg-emerald-500/10" 
                : "border-slate-800 hover:border-slate-700 bg-slate-900/30 hover:bg-slate-900/60"
            }`}
          >
            <input 
              type="file" 
              ref={fileInputRef} 
              accept=".pdf" 
              className="hidden" 
              onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])} 
            />
            <div className="w-10 h-10 rounded-full bg-slate-800/60 flex items-center justify-center text-slate-400">
              <UploadCloud className="w-5 h-5" />
            </div>
            <div className="text-center">
              <p className="text-xs font-medium text-slate-200">Upload PDF Document</p>
              <p className="text-[11px] text-slate-500 mt-0.5">Drag & drop or browse files</p>
            </div>
          </div>

          {/* Telemetry Panel */}
          {docState.status !== "IDLE" && (
            <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-2.5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 overflow-hidden">
                  <FileCheck2 className="w-4 h-4 text-slate-400 shrink-0" />
                  <span className="text-xs font-medium text-slate-200 truncate max-w-[140px]">{docState.filename}</span>
                </div>
                {docState.status === "UPLOADING" && (
                  <span className="px-2 py-0.5 rounded text-[9px] font-mono font-semibold bg-blue-500/10 text-blue-400 border border-blue-500/20 flex items-center gap-1">
                    <Loader2 className="w-2.5 h-2.5 animate-spin" /> UPLOAD
                  </span>
                )}
                {docState.status === "PROCESSING" && (
                  <span className="px-2 py-0.5 rounded text-[9px] font-mono font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/20 flex items-center gap-1">
                    <Loader2 className="w-2.5 h-2.5 animate-spin" /> VECTORIZING
                  </span>
                )}
                {docState.status === "COMPLETED" && (
                  <span className="px-2 py-0.5 rounded text-[9px] font-mono font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center gap-1">
                    <CheckCircle2 className="w-2.5 h-2.5" /> READY
                  </span>
                )}
                {docState.status === "FAILED" && (
                  <span className="px-2 py-0.5 rounded text-[9px] font-mono font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/20 flex items-center gap-1">
                    <AlertCircle className="w-2.5 h-2.5" /> ERROR
                  </span>
                )}
              </div>

              <div className="text-[10px] font-mono text-slate-400 space-y-1">
                <div className="flex justify-between">
                  <span>Document ID:</span>
                  <span className="text-slate-200">{docState.id ? `${docState.id.slice(0, 8)}...` : "—"}</span>
                </div>
                <div className="flex justify-between">
                  <span>Vector Index:</span>
                  <span className="text-emerald-400">pgvector HNSW</span>
                </div>
              </div>
            </div>
          )}

        </div>

        {/* Stack Specs */}
        <div className="p-3.5 rounded-xl bg-slate-900/30 border border-slate-800/60 text-[11px] text-slate-500 space-y-2">
          <div className="flex items-center gap-2 text-slate-400 font-semibold">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            <span>Anti-Hallucination Guard</span>
          </div>
          <p className="leading-relaxed text-[10px]">
            Inference requires cosine similarity ≥ 0.50. Generation locked at T=0.2.
          </p>
        </div>
      </aside>

      {/* 2. Middle Column: Conversational RAG Stream */}
      <main className="flex-1 flex flex-col justify-between bg-[#070A12] border-r border-slate-800/80">
        
        {/* Top Navbar */}
        <div className="h-14 border-b border-slate-800/80 px-6 flex items-center justify-between bg-[#0B0F19]/40 backdrop-blur-md">
          <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
            <Database className="w-3.5 h-3.5 text-emerald-400" />
            <span>Active Target:</span>
            <span className="text-slate-200 font-semibold">{docState.filename || "No Document Ingested"}</span>
          </div>
          <span className="text-[10px] font-mono px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            gemini-3.6-flash
          </span>
        </div>

        {/* Messages Stream */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.map((msg) => (
            <div 
              key={msg.id} 
              className={`flex gap-3.5 max-w-2xl ${msg.sender === "user" ? "ml-auto justify-end" : ""}`}
            >
              {msg.sender === "assistant" && (
                <div className="w-7 h-7 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 shrink-0 mt-0.5">
                  <Sparkles className="w-3.5 h-3.5" />
                </div>
              )}

              <div className={`space-y-2 ${msg.sender === "user" ? "items-end" : ""}`}>
                <div 
                  className={`p-4 rounded-2xl text-xs leading-relaxed ${
                    msg.sender === "user"
                      ? "bg-emerald-600 text-white rounded-br-none font-medium"
                      : "bg-[#0E1322] border border-slate-800/80 text-slate-200 rounded-bl-none shadow-lg"
                  }`}
                >
                  <ReactMarkdown>{msg.content}</ReactMarkdown>

                  {/* Inline Citation Badges */}
                  {msg.citations && msg.citations.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-slate-800/60 flex flex-wrap gap-2 items-center">
                      <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider">Citations:</span>
                      {msg.citations.map((c) => (
                        <button
                          key={c.chunk_id}
                          onClick={() => setActiveCitation(c)}
                          className="px-2 py-0.5 rounded-md bg-slate-900 border border-slate-700/80 hover:border-emerald-500/50 text-[10px] font-mono text-emerald-400 hover:text-emerald-300 transition-all flex items-center gap-1"
                        >
                          <span>[Source {c.source_id}]</span>
                          <span className="text-slate-500">({Math.round(c.similarity * 100)}%)</span>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
                {hasMounted && msg.timestamp && (
                  <span suppressHydrationWarning className="text-[9px] font-mono text-slate-600 px-1">
                    {msg.timestamp}
                  </span>
                )}
              </div>
            </div>
          ))}

          {isInferring && !messages.some((m) => m.sender === "assistant" && m.content) && (
            <div className="flex gap-3.5 max-w-2xl">
              <div className="w-7 h-7 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 shrink-0 animate-pulse">
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              </div>
              <div className="p-3.5 rounded-2xl bg-[#0E1322] border border-slate-800 text-xs text-slate-400 italic">
                Executing cosine distance search across HNSW index...
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar */}
        <div className="p-4 bg-[#0B0F19]/80 border-t border-slate-800/80 backdrop-blur-md">
          <form onSubmit={handleSendMessage} className="max-w-3xl mx-auto flex items-center gap-2.5">
            <input 
              type="text" 
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              placeholder={
                docState.status === "COMPLETED" 
                  ? "Query document facts, topics, or figures..." 
                  : "Upload and index a document to query..."
              }
              disabled={docState.status !== "COMPLETED" || isInferring}
              className="flex-1 bg-slate-900/90 border border-slate-800 focus:border-emerald-500 rounded-xl px-4 py-3 text-xs text-slate-100 placeholder-slate-500 focus:outline-none disabled:opacity-40 transition-all"
            />
            <button 
              type="submit"
              disabled={docState.status !== "COMPLETED" || !inputQuery.trim() || isInferring}
              className="px-4 py-3 rounded-xl bg-emerald-500 hover:bg-emerald-600 disabled:opacity-40 text-slate-950 font-semibold text-xs transition-all flex items-center gap-1.5 shrink-0 shadow-lg shadow-emerald-500/20"
            >
              <span>Ask</span>
              <Send className="w-3.5 h-3.5" />
            </button>
          </form>
        </div>

      </main>

      {/* 3. Right Column: Provenance & Chunk Inspector (340px) */}
      <aside className="w-80 bg-[#0B0F19]/95 flex flex-col p-5 overflow-y-auto shrink-0 space-y-4">
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-300 uppercase tracking-wider">
          <Layers className="w-4 h-4 text-emerald-400" />
          <span>Vector Provenance Inspector</span>
        </div>

        {activeCitation ? (
          <div className="space-y-4">
            <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-2">
              <div className="flex items-center justify-between font-mono text-[11px]">
                <span className="text-emerald-400 font-semibold">Source [{activeCitation.source_id}]</span>
                <span className="text-slate-400">Score: {(activeCitation.similarity * 100).toFixed(1)}%</span>
              </div>
              <div className="text-[10px] font-mono text-slate-500 break-all">
                Chunk UUID: {activeCitation.chunk_id}
              </div>
            </div>

            <div className="space-y-2">
              <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">Indexed Content:</span>
              <div className="p-4 rounded-xl bg-[#0E1322] border border-slate-800 text-xs text-slate-300 leading-relaxed font-mono whitespace-pre-wrap">
                {activeCitation.preview}
              </div>
            </div>
          </div>
        ) : (
          <div className="h-64 flex flex-col items-center justify-center text-center p-6 border border-dashed border-slate-800 rounded-xl text-slate-500 space-y-2">
            <BookOpen className="w-6 h-6 text-slate-600" />
            <p className="text-xs">Click on any citation badge in a response to inspect its exact database chunk.</p>
          </div>
        )}
      </aside>

    </div>
  );
}