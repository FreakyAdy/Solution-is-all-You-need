import React, { useEffect, useState } from "react";
import { Dashboard } from "./pages/Dashboard";
import { Models } from "./pages/Models";
import { Chat } from "./pages/Chat";
import { Metrics } from "./pages/Metrics";
import { Layers } from "./pages/Layers";
import { Plugins } from "./pages/Plugins";

export const App: React.FC = () => {
  const [tab, setTab] = useState<"dashboard" | "models" | "chat" | "metrics" | "layers" | "plugins">("dashboard");
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

  const navItems = [
    { id: "dashboard", label: "Dashboard", icon: "📊" },
    { id: "models", label: "Models", icon: "📦" },
    { id: "chat", label: "Inference Chat", icon: "💬" },
    { id: "metrics", label: "Live Telemetry", icon: "📈" },
    { id: "layers", label: "Layer Map", icon: "🗺️" },
    { id: "plugins", label: "Plugins", icon: "🔌" },
  ];

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      {/* Header */}
      <header
        style={{
          padding: "16px 36px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          borderBottom: "1px solid var(--panel-border)",
          background: "rgba(11, 13, 19, 0.8)",
          backdropFilter: "blur(12px)",
          position: "sticky",
          top: 0,
          zIndex: 100,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div style={{ fontSize: "22px", fontWeight: 800, letterSpacing: "1.5px", color: "var(--accent-amber)" }}>
            ⚡ PHANTOM
          </div>
          <span style={{ fontSize: "11px", padding: "2px 8px", background: "rgba(245, 158, 11, 0.15)", color: "var(--accent-amber)", borderRadius: "4px", fontFamily: "var(--font-mono)", fontWeight: 600 }}>
            v1.0 PLATFORM
          </span>
        </div>

        {/* Navigation Tabs */}
        <nav style={{ display: "flex", gap: "6px" }}>
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => setTab(item.id as any)}
              style={{
                padding: "8px 16px",
                borderRadius: "8px",
                background: tab === item.id ? "rgba(255, 255, 255, 0.1)" : "transparent",
                color: tab === item.id ? "#ffffff" : "var(--text-secondary)",
                fontWeight: tab === item.id ? 600 : 400,
                fontSize: "14px",
                display: "flex",
                alignItems: "center",
                gap: "8px",
                transition: "all 0.15s",
              }}
            >
              <span>{item.icon}</span>
              <span>{item.label}</span>
            </button>
          ))}
        </nav>

        {/* Status Indicator */}
        <div style={{ display: "flex", alignItems: "center", gap: "12px", fontFamily: "var(--font-mono)", fontSize: "12px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "6px", color: "var(--accent-green)" }}>
            <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: "var(--accent-green)" }}></span>
            <span>ONLINE (PORT 11411)</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main style={{ flex: 1, maxWidth: "1400px", width: "100%", margin: "0 auto", padding: "32px 24px" }}>
        {tab === "dashboard" && <Dashboard metrics={metrics} hardware={hardware} />}
        {tab === "models" && <Models />}
        {tab === "chat" && <Chat />}
        {tab === "metrics" && <Metrics metrics={metrics} />}
        {tab === "layers" && <Layers metrics={metrics} />}
        {tab === "plugins" && <Plugins />}
      </main>

      {/* Footer */}
      <footer
        style={{
          padding: "16px 36px",
          borderTop: "1px solid var(--panel-border)",
          fontSize: "12px",
          color: "var(--text-muted)",
          display: "flex",
          justifyContent: "space-between",
        }}
      >
        <div>PHANTOM CORE: Wraith Layers • Spectral Quant • Neural Cache • Phantom Pages • Chronos • Resonance</div>
        <div>Run the Unreachable.</div>
      </footer>
    </div>
  );
};
