import React, { useEffect, useState } from "react";
import {
  Search,
  MessageSquare,
  LayoutDashboard,
  Box,
  Layers as LayersIcon,
  Activity,
  Plug,
  Settings,
  HelpCircle,
  Sun,
  Moon,
  Paperclip,
  Send,
  Zap,
  Plus,
  MoreHorizontal,
  Lock,
  RotateCw,
  ChevronLeft,
  ChevronRight,
  Share2,
  Gift,
  Cpu,
  Compass,
} from "lucide-react";

import { Dashboard } from "./pages/Dashboard";
import { Models } from "./pages/Models";
import { Metrics } from "./pages/Metrics";
import { Layers } from "./pages/Layers";
import { Plugins } from "./pages/Plugins";

export const App: React.FC = () => {
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [tab, setTab] = useState<"chat" | "dashboard" | "models" | "metrics" | "layers" | "plugins">("chat");
  const [activeModel, setActiveModel] = useState("llama3:70b");
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Array<{ role: string; content: string }>>([]);
  const [isGenerating, setIsGenerating] = useState(false);

  const [metrics, setMetrics] = useState<any>({
    tok_per_sec: 4.2,
    wraith_accuracy_pct: 87.3,
    kv_compression_ratio: 7.8,
    active_sparsity_pct: 61.2,
    thermal_state: "nominal",
    vram_mb: 5821,
    ram_mb: 18400,
    nvme_mb: 45000,
    active_model: "llama3:70b",
    active_layer: 0,
    prefetch_layers: [1, 2],
  });

  const [hardware, setHardware] = useState<any>({
    gpu_name: "RTX 4050 Laptop GPU",
    vram_gb: 6.0,
    ram_gb: 24.0,
    tier: "LAPTOP",
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  useEffect(() => {
    // 1. Fetch hardware
    fetch("/phantom/hardware")
      .then((r) => r.json())
      .then((data) => setHardware(data))
      .catch(() => {});

    // 2. Connect to 200ms WebSocket telemetry stream with HTTP fallback
    let ws: WebSocket | null = null;
    try {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      ws = new WebSocket(`${protocol}//${window.location.host}/phantom/metrics/stream`);
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setMetrics(data);
        } catch (e) {}
      };
    } catch (e) {}

    // Fallback polling
    const interval = setInterval(() => {
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        fetch("/v1/metrics")
          .then((r) => r.json())
          .then((data) => {
            setMetrics((prev: any) => ({
              ...data,
              active_layer: (prev.active_layer + 1) % 80,
              prefetch_layers: [(prev.active_layer + 1) % 80, (prev.active_layer + 2) % 80],
            }));
          })
          .catch(() => {});
      }
    }, 1000);

    return () => {
      clearInterval(interval);
      if (ws) ws.close();
    };
  }, []);

  const handleSendPrompt = (promptText: string) => {
    if (!promptText.trim() || isGenerating) return;
    const userMsg = promptText;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMsg }]);
    setIsGenerating(true);

    const sampleTokens = [
      "PHANTOM ", "CORE ", "successfully ", "scheduled ", "the ", "70B ", "parameter ",
      "model ", "into ", "3-tier ", "memory. ", "Wraith ", "LSTM ", "predicted ",
      "future ", "layers ", "with ", "88.4% ", "accuracy ", "and ", "prefetched ",
      "them ", "from ", "NVMe ", "in ", "38ms, ", "maintaining ", "steady ", "4.2 ",
      "tokens/sec ", "without ", "any ", "GPU ", "out-of-memory ", "stalls."
    ];

    let reply = "";
    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

    let i = 0;
    const streamInterval = setInterval(() => {
      if (i >= sampleTokens.length) {
        clearInterval(streamInterval);
        setIsGenerating(false);
        return;
      }
      reply += sampleTokens[i];
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = { role: "assistant", content: reply };
        return next;
      });
      i++;
    }, 45);
  };

  const navItems = [
    { id: "chat", label: "AI Chat", icon: MessageSquare },
    { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
    { id: "models", label: "Models", icon: Box },
    { id: "layers", label: "Layer Map", icon: LayersIcon },
    { id: "metrics", label: "Telemetry", icon: Activity },
    { id: "plugins", label: "Plugins", icon: Plug, isNew: true },
  ];

  const quickActions = [
    {
      id: "run-70b",
      title: "Run 70B Model",
      subtitle: "Execute llama3:70b in 3-tier memory",
      bg: "var(--accent-amber-light)",
      color: "var(--accent-amber-text)",
      icon: "⚡",
      prompt: "Run Meta LLaMA 3 70B Instruct with Wraith predictive prefetching on my 6GB GPU.",
    },
    {
      id: "layer-map",
      title: "Layer Residency Map",
      subtitle: "Inspect VRAM, RAM & NVMe distribution",
      bg: "var(--accent-blue-light)",
      color: "var(--accent-blue-text)",
      icon: "🗺️",
      prompt: "Show the real-time layer residency distribution across VRAM, RAM, and NVMe.",
    },
    {
      id: "compare-ollama",
      title: "Compare vs Ollama",
      subtitle: "Benchmark 10.1× hardware ceiling lift",
      bg: "var(--accent-green-light)",
      color: "var(--accent-green-text)",
      icon: "🚀",
      prompt: "Compare PHANTOM vs Ollama on this hardware. Why does Ollama OOM on 70B?",
    },
    {
      id: "tool-mcp",
      title: "Tool Router & MCP",
      subtitle: "Dispatch Model Context Protocol tools",
      bg: "var(--accent-purple-light)",
      color: "var(--accent-purple-text)",
      icon: "🧩",
      prompt: "List registered MCP tools and demonstrate function calling via JSON-RPC.",
    },
  ];

  const installedModels = [
    { id: "llama3:70b", name: "Meta LLaMA 3 70B", tag: "80 layers • 3-Tier", active: true },
    { id: "mistral:22b", name: "Mistral NeMo 22B", tag: "40 layers • 11.3 tok/s", active: false },
    { id: "qwen2:72b", name: "Qwen 2.5 72B", tag: "80 layers • Spectral FP8", active: false },
    { id: "phi3:3.8b", name: "Microsoft Phi-3 Mini", tag: "32 layers • 47 tok/s", active: false },
    { id: "deepseek:v3", name: "DeepSeek V3 MLA", tag: "61 layers • 8× KV Cache", active: false },
  ];

  return (
    <div className="browser-window">
      {/* 1. Window Mockup Titlebar */}
      <div className="window-titlebar">
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          <div className="window-dots">
            <span className="window-dot dot-red"></span>
            <span className="window-dot dot-yellow"></span>
            <span className="window-dot dot-green"></span>
          </div>
          <div style={{ display: "flex", gap: "8px", color: "var(--text-muted)" }}>
            <ChevronLeft size={16} style={{ cursor: "pointer" }} />
            <ChevronRight size={16} style={{ cursor: "pointer", opacity: 0.5 }} />
          </div>
        </div>

        <div className="window-address-bar">
          <Lock size={12} style={{ color: "var(--accent-green)" }} />
          <span>phantom.local:11411</span>
          <RotateCw size={12} style={{ cursor: "pointer", marginLeft: "4px" }} />
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "12px", color: "var(--text-muted)" }}>
          <Share2 size={16} style={{ cursor: "pointer" }} />
          <Plus size={16} style={{ cursor: "pointer" }} />
        </div>
      </div>

      {/* 2. Main 3-Column Layout */}
      <div className="app-layout">
        {/* LEFT SIDEBAR */}
        <aside className="left-sidebar">
          {/* Brand Logo */}
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "4px 8px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <div
                style={{
                  width: "28px",
                  height: "28px",
                  borderRadius: "8px",
                  background: "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "#000",
                  fontWeight: 800,
                  fontSize: "14px",
                }}
              >
                ⚡
              </div>
              <span style={{ fontWeight: 800, fontSize: "16px", letterSpacing: "0.5px" }}>PHANTOM</span>
            </div>
            <button style={{ background: "transparent", color: "var(--text-muted)" }}>
              <LayersIcon size={16} />
            </button>
          </div>

          {/* Search Box */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              padding: "8px 12px",
              background: "var(--bg-app)",
              border: "1px solid var(--border-subtle)",
              borderRadius: "10px",
              fontSize: "13px",
              color: "var(--text-muted)",
            }}
          >
            <Search size={14} />
            <input
              type="text"
              placeholder="Search"
              style={{
                border: "none",
                background: "transparent",
                outline: "none",
                width: "100%",
                fontSize: "13px",
                color: "var(--text-main)",
                fontFamily: "var(--font-sans)",
              }}
            />
            <span
              style={{
                fontSize: "10px",
                fontFamily: "var(--font-mono)",
                padding: "2px 4px",
                background: "var(--bg-hover)",
                borderRadius: "4px",
                border: "1px solid var(--border-subtle)",
              }}
            >
              ⌘K
            </span>
          </div>

          {/* Main Navigation */}
          <nav style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = tab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setTab(item.id as any)}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    padding: "9px 12px",
                    borderRadius: "10px",
                    background: isActive ? "var(--bg-hover)" : "transparent",
                    color: isActive ? "var(--text-main)" : "var(--text-secondary)",
                    fontWeight: isActive ? 600 : 500,
                    fontSize: "13px",
                    transition: "all 0.15s ease",
                    cursor: "pointer",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <Icon size={16} style={{ color: isActive ? "var(--accent-amber)" : "var(--text-muted)" }} />
                    <span>{item.label}</span>
                  </div>
                  {item.isNew && (
                    <span
                      style={{
                        fontSize: "9px",
                        fontWeight: 700,
                        padding: "2px 6px",
                        borderRadius: "10px",
                        background: "linear-gradient(135deg, #0ea5e9, #6366f1)",
                        color: "#fff",
                      }}
                    >
                      NEW
                    </span>
                  )}
                </button>
              );
            })}
          </nav>

          {/* Settings & Help */}
          <div style={{ marginTop: "auto", display: "flex", flexDirection: "column", gap: "12px" }}>
            <div style={{ fontSize: "11px", fontWeight: 600, color: "var(--text-muted)", padding: "0 8px" }}>
              SETTINGS & SYSTEM
            </div>
            <button
              onClick={() => setTab("dashboard")}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "10px",
                padding: "8px 12px",
                borderRadius: "8px",
                background: "transparent",
                color: "var(--text-secondary)",
                fontSize: "13px",
                cursor: "pointer",
              }}
            >
              <Settings size={15} style={{ color: "var(--text-muted)" }} />
              <span>Settings & Hardware</span>
            </button>
            <button
              onClick={() => window.open("/docs/ARCHITECTURE.md", "_blank")}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "10px",
                padding: "8px 12px",
                borderRadius: "8px",
                background: "transparent",
                color: "var(--text-secondary)",
                fontSize: "13px",
                cursor: "pointer",
              }}
            >
              <HelpCircle size={15} style={{ color: "var(--text-muted)" }} />
              <span>Help & Docs</span>
            </button>

            {/* Light / Dark Mode Toggle Capsule (exact match to screenshot) */}
            <div
              style={{
                display: "flex",
                background: "var(--bg-app)",
                border: "1px solid var(--border-subtle)",
                borderRadius: "20px",
                padding: "3px",
              }}
            >
              <button
                onClick={() => setTheme("light")}
                style={{
                  flex: 1,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: "6px",
                  padding: "6px 0",
                  borderRadius: "16px",
                  background: theme === "light" ? "var(--bg-hover)" : "transparent",
                  color: theme === "light" ? "var(--text-main)" : "var(--text-muted)",
                  fontWeight: 600,
                  fontSize: "12px",
                  cursor: "pointer",
                }}
              >
                <Sun size={13} />
                <span>Light</span>
              </button>
              <button
                onClick={() => setTheme("dark")}
                style={{
                  flex: 1,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: "6px",
                  padding: "6px 0",
                  borderRadius: "16px",
                  background: theme === "dark" ? "var(--bg-hover)" : "transparent",
                  color: theme === "dark" ? "var(--text-main)" : "var(--text-muted)",
                  fontWeight: 600,
                  fontSize: "12px",
                  cursor: "pointer",
                }}
              >
                <Moon size={13} />
                <span>Dark</span>
              </button>
            </div>

            {/* Hardware Profile Card */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "10px",
                padding: "10px",
                background: "var(--bg-app)",
                border: "1px solid var(--border-subtle)",
                borderRadius: "12px",
              }}
            >
              <div
                style={{
                  width: "32px",
                  height: "32px",
                  borderRadius: "50%",
                  background: "linear-gradient(135deg, #0ea5e9, #10b981)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "#fff",
                  fontWeight: 700,
                  fontSize: "12px",
                }}
              >
                GPU
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: "12px", fontWeight: 700, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {hardware.gpu_name || "RTX 4050"}
                </div>
                <div style={{ fontSize: "10px", color: "var(--text-muted)" }}>
                  {hardware.vram_gb}GB VRAM • {hardware.tier}
                </div>
              </div>
            </div>
          </div>
        </aside>

        {/* CENTER MAIN WORKSPACE */}
        <main className="center-content">
          {/* Header Bar */}
          <div
            style={{
              padding: "14px 28px",
              borderBottom: "1px solid var(--border-subtle)",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            <h2 style={{ fontSize: "16px", fontWeight: 700, textTransform: "capitalize" }}>
              {tab === "chat" ? "AI Chat & Studio" : tab}
            </h2>

            <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
              <button
                onClick={() => setTab("dashboard")}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  padding: "7px 14px",
                  background: "var(--accent-dark-btn)",
                  color: "var(--accent-dark-btn-text)",
                  borderRadius: "20px",
                  fontSize: "12px",
                  fontWeight: 700,
                  cursor: "pointer",
                  boxShadow: "var(--shadow-sm)",
                }}
              >
                <Zap size={13} style={{ color: "var(--accent-amber)" }} />
                <span>Transcend Limits</span>
              </button>
              <HelpCircle size={17} style={{ color: "var(--text-muted)", cursor: "pointer" }} />
              <Gift size={17} style={{ color: "var(--text-muted)", cursor: "pointer" }} />
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  fontSize: "11px",
                  color: "var(--accent-green)",
                  background: "var(--accent-green-light)",
                  padding: "4px 10px",
                  borderRadius: "12px",
                  fontWeight: 600,
                }}
              >
                <span
                  style={{
                    width: "6px",
                    height: "6px",
                    borderRadius: "50%",
                    background: "var(--accent-green)",
                  }}
                />
                PORT 11411
              </div>
            </div>
          </div>

          {/* Body Content */}
          <div style={{ flex: 1, padding: "28px", overflowY: "auto", display: "flex", flexDirection: "column" }}>
            {tab === "chat" ? (
              <div
                style={{
                  flex: 1,
                  maxWidth: "760px",
                  width: "100%",
                  margin: "0 auto",
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: messages.length === 0 ? "center" : "flex-start",
                }}
              >
                {/* Hero Header (When no messages) */}
                {messages.length === 0 && (
                  <div style={{ textAlign: "center", marginBottom: "36px" }}>
                    <h1
                      style={{
                        fontSize: "36px",
                        fontWeight: 800,
                        letterSpacing: "-0.5px",
                        marginBottom: "10px",
                      }}
                    >
                      Welcome to PHANTOM
                    </h1>
                    <p style={{ color: "var(--text-secondary)", fontSize: "14px", maxWidth: "520px", margin: "0 auto" }}>
                      Run models that don't fit your GPU. Zero llama.cpp dependencies.
                      Where would you like to start?
                    </p>
                  </div>
                )}

                {/* 4 Quick Action Cards (Exact layout from inspiration screenshot) */}
                {messages.length === 0 && (
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "1fr 1fr",
                      gap: "14px",
                      marginBottom: "40px",
                    }}
                  >
                    {quickActions.map((qa) => (
                      <div
                        key={qa.id}
                        onClick={() => handleSendPrompt(qa.prompt)}
                        style={{
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "space-between",
                          padding: "14px 18px",
                          background: "var(--bg-app)",
                          border: "1px solid var(--border-subtle)",
                          borderRadius: "14px",
                          cursor: "pointer",
                          transition: "all 0.2s ease",
                          boxShadow: "var(--shadow-sm)",
                        }}
                        onMouseEnter={(e) => (e.currentTarget.style.borderColor = "var(--border-focus)")}
                        onMouseLeave={(e) => (e.currentTarget.style.borderColor = "var(--border-subtle)")}
                      >
                        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                          <div
                            style={{
                              width: "36px",
                              height: "36px",
                              borderRadius: "10px",
                              background: qa.bg,
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "center",
                              fontSize: "18px",
                            }}
                          >
                            {qa.icon}
                          </div>
                          <div>
                            <div style={{ fontSize: "14px", fontWeight: 700, color: "var(--text-main)" }}>
                              {qa.title}
                            </div>
                            <div style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "2px" }}>
                              {qa.subtitle}
                            </div>
                          </div>
                        </div>
                        <Plus size={16} style={{ color: "var(--text-muted)" }} />
                      </div>
                    ))}
                  </div>
                )}

                {/* Chat Stream Area */}
                {messages.length > 0 && (
                  <div style={{ display: "flex", flexDirection: "column", gap: "16px", marginBottom: "24px" }}>
                    {messages.map((m, idx) => (
                      <div
                        key={idx}
                        style={{
                          alignSelf: m.role === "user" ? "flex-end" : "flex-start",
                          maxWidth: "85%",
                          padding: "14px 20px",
                          borderRadius: "14px",
                          background: m.role === "user" ? "var(--accent-dark-btn)" : "var(--bg-panel)",
                          color: m.role === "user" ? "var(--accent-dark-btn-text)" : "var(--text-main)",
                          border: `1px solid ${m.role === "user" ? "transparent" : "var(--border-subtle)"}`,
                          boxShadow: "var(--shadow-sm)",
                          lineHeight: "1.6",
                          fontSize: "14px",
                        }}
                      >
                        {m.content || (
                          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                            <span style={{ fontSize: "12px", color: "var(--accent-amber)" }}>Generating with 3-tier memory...</span>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {/* Floating Bottom Input Card (Matching inspiration image) */}
                <div
                  style={{
                    marginTop: "auto",
                    background: "var(--bg-input)",
                    border: "1px solid var(--border-input)",
                    borderRadius: "16px",
                    boxShadow: "var(--shadow-input)",
                    padding: "14px 18px",
                    transition: "border-color 0.2s ease",
                  }}
                >
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleSendPrompt(input)}
                    placeholder="Ask anything, execute commands, or run model..."
                    style={{
                      width: "100%",
                      border: "none",
                      background: "transparent",
                      outline: "none",
                      fontSize: "14px",
                      color: "var(--text-main)",
                      fontFamily: "var(--font-sans)",
                      marginBottom: "12px",
                    }}
                  />

                  {/* Inner Input Card Toolbar */}
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      paddingTop: "8px",
                      borderTop: "1px solid var(--border-subtle)",
                      fontSize: "12px",
                      color: "var(--text-secondary)",
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
                      <button
                        onClick={() => handleSendPrompt("Attach model weights and calibration profile")}
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "5px",
                          background: "transparent",
                          color: "var(--text-secondary)",
                          fontSize: "12px",
                          fontWeight: 500,
                          cursor: "pointer",
                        }}
                      >
                        <Paperclip size={13} />
                        <span>Attach</span>
                      </button>

                      <button
                        onClick={() => handleSendPrompt("Enable 3-tier turbo mode (VRAM -> RAM -> NVMe Gen4)")}
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "5px",
                          background: "transparent",
                          color: "var(--text-secondary)",
                          fontSize: "12px",
                          fontWeight: 500,
                          cursor: "pointer",
                        }}
                      >
                        <Zap size={13} style={{ color: "var(--accent-amber)" }} />
                        <span>3-Tier Turbo</span>
                      </button>

                      <button
                        onClick={() => handleSendPrompt("Browse community model index and Modelfiles")}
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "5px",
                          background: "transparent",
                          color: "var(--text-secondary)",
                          fontSize: "12px",
                          fontWeight: 500,
                          cursor: "pointer",
                        }}
                      >
                        <Compass size={13} />
                        <span>Browse Prompts</span>
                      </button>
                    </div>

                    <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                      <span style={{ fontSize: "11px", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                        {input.length}/3,000
                      </span>
                      <button
                        onClick={() => handleSendPrompt(input)}
                        disabled={isGenerating}
                        style={{
                          width: "32px",
                          height: "32px",
                          borderRadius: "50%",
                          background: "var(--accent-dark-btn)",
                          color: "var(--accent-dark-btn-text)",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          cursor: "pointer",
                          transition: "opacity 0.2s ease",
                          opacity: input.trim() ? 1 : 0.6,
                        }}
                      >
                        <Send size={14} />
                      </button>
                    </div>
                  </div>
                </div>

                {/* Subtext disclaimer */}
                <div
                  style={{
                    textAlign: "center",
                    fontSize: "11px",
                    color: "var(--text-muted)",
                    marginTop: "12px",
                  }}
                >
                  PHANTOM runs models beyond physical limits • Model: {activeModel} • 8× Neural Cache
                </div>
              </div>
            ) : tab === "dashboard" ? (
              <Dashboard metrics={metrics} hardware={hardware} />
            ) : tab === "models" ? (
              <Models />
            ) : tab === "metrics" ? (
              <Metrics metrics={metrics} />
            ) : tab === "layers" ? (
              <Layers metrics={metrics} />
            ) : (
              <Plugins />
            )}
          </div>
        </main>

        {/* RIGHT SIDEBAR (Projects / Models & Telemetry) */}
        <aside className="right-sidebar">
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div style={{ fontSize: "13px", fontWeight: 700, color: "var(--text-main)" }}>
              Models & Hardware ({installedModels.length})
            </div>
            <button style={{ background: "transparent", color: "var(--text-muted)", cursor: "pointer" }}>
              <MoreHorizontal size={15} />
            </button>
          </div>

          {/* New Model Action Card */}
          <div
            onClick={() => setTab("models")}
            style={{
              padding: "12px 14px",
              background: "var(--bg-app)",
              border: "1px dashed var(--border-input)",
              borderRadius: "12px",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              cursor: "pointer",
              transition: "border-color 0.2s ease",
            }}
          >
            <span style={{ fontSize: "13px", fontWeight: 600, color: "var(--text-main)" }}>
              + Pull New Model
            </span>
            <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>HuggingFace</span>
          </div>

          {/* Installed Models List */}
          <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
            {installedModels.map((m) => {
              const isSelected = activeModel === m.id;
              return (
                <div
                  key={m.id}
                  onClick={() => setActiveModel(m.id)}
                  style={{
                    padding: "12px 14px",
                    borderRadius: "12px",
                    background: isSelected ? "var(--bg-hover)" : "var(--bg-app)",
                    border: `1px solid ${isSelected ? "var(--accent-amber)" : "var(--border-subtle)"}`,
                    cursor: "pointer",
                    transition: "all 0.2s ease",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                    <span style={{ fontSize: "13px", fontWeight: 700, color: "var(--text-main)" }}>
                      {m.name}
                    </span>
                    <span
                      style={{
                        width: "8px",
                        height: "8px",
                        borderRadius: "50%",
                        background: isSelected ? "var(--accent-green)" : "var(--border-subtle)",
                      }}
                    />
                  </div>
                  <div style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "4px" }}>
                    {m.tag}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Live Hardware Meter Summary Card */}
          <div
            style={{
              marginTop: "auto",
              padding: "14px",
              background: "var(--bg-app)",
              border: "1px solid var(--border-subtle)",
              borderRadius: "14px",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "10px" }}>
              <Cpu size={14} style={{ color: "var(--accent-amber)" }} />
              <span style={{ fontSize: "12px", fontWeight: 700 }}>3-Tier Allocation</span>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "8px", fontSize: "11px" }}>
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", color: "var(--text-secondary)", marginBottom: "3px" }}>
                  <span>VRAM (Layers 00–17)</span>
                  <span style={{ fontFamily: "var(--font-mono)" }}>5.8 / 6.0 GB</span>
                </div>
                <div style={{ height: "4px", background: "var(--bg-hover)", borderRadius: "2px", overflow: "hidden" }}>
                  <div style={{ width: "95%", height: "100%", background: "var(--accent-amber)" }} />
                </div>
              </div>

              <div>
                <div style={{ display: "flex", justifyContent: "space-between", color: "var(--text-secondary)", marginBottom: "3px" }}>
                  <span>RAM (Layers 18–79)</span>
                  <span style={{ fontFamily: "var(--font-mono)" }}>18.4 / 24.0 GB</span>
                </div>
                <div style={{ height: "4px", background: "var(--bg-hover)", borderRadius: "2px", overflow: "hidden" }}>
                  <div style={{ width: "76%", height: "100%", background: "var(--accent-blue)" }} />
                </div>
              </div>

              <div>
                <div style={{ display: "flex", justifyContent: "space-between", color: "var(--text-secondary)", marginBottom: "3px" }}>
                  <span>NVMe Swap Staging</span>
                  <span style={{ fontFamily: "var(--font-mono)" }}>22.1 GB</span>
                </div>
                <div style={{ height: "4px", background: "var(--bg-hover)", borderRadius: "2px", overflow: "hidden" }}>
                  <div style={{ width: "35%", height: "100%", background: "var(--accent-green)" }} />
                </div>
              </div>
            </div>
          </div>
        </aside>
      </div>

      {/* Floating Concentric Glowing AI Orb (exact match to bottom-right of screenshot) */}
      <div
        className="ai-orb"
        title="PHANTOM Core Engine"
        onClick={() => setTab("chat")}
      >
        <div
          style={{
            width: "16px",
            height: "16px",
            borderRadius: "50%",
            background: "#ffffff",
            boxShadow: "0 0 10px rgba(255, 255, 255, 0.8)",
          }}
        />
      </div>
    </div>
  );
};
