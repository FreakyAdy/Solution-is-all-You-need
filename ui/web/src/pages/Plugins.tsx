import React, { useState } from "react";

export const Plugins: React.FC = () => {
  const [plugins, setPlugins] = useState([
    { id: "rag-connector", name: "RAG Connector", active: true, desc: "Retrieval-augmented generation using local vector stores." },
    { id: "tool-router", name: "Tool Router", active: true, desc: "Function calling and MCP tool server routing middleware." },
    { id: "context-cache", name: "Context Cache", active: true, desc: "Prefix KV caching with Neural Cache 8× compression." },
  ]);

  const toggle = (id: string) => {
    setPlugins((prev) =>
      prev.map((p) => (p.id === id ? { ...p, active: !p.active } : p))
    );
  };

  return (
    <div className="glass-panel" style={{ padding: "28px" }}>
      <h2 style={{ fontSize: "20px", fontWeight: 700, marginBottom: "20px", color: "var(--accent-amber)" }}>
        INSTALLED PLUGINS & MIDDLEWARE
      </h2>
      <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
        {plugins.map((p) => (
          <div
            key={p.id}
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              padding: "20px",
              background: "rgba(0,0,0,0.25)",
              borderRadius: "12px",
              border: "1px solid var(--panel-border)",
            }}
          >
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <span style={{ fontSize: "18px" }}>🔌</span>
                <span style={{ fontWeight: 700, fontSize: "16px" }}>{p.name}</span>
                <span style={{
                  fontSize: "11px",
                  padding: "2px 8px",
                  borderRadius: "4px",
                  background: p.active ? "rgba(16, 185, 129, 0.15)" : "rgba(255,255,255,0.06)",
                  color: p.active ? "var(--accent-green)" : "var(--text-muted)",
                  fontFamily: "var(--font-mono)"
                }}>
                  {p.active ? "ENABLED" : "DISABLED"}
                </span>
              </div>
              <p style={{ color: "var(--text-secondary)", fontSize: "14px", marginTop: "6px" }}>{p.desc}</p>
            </div>
            <button
              onClick={() => toggle(p.id)}
              style={{
                padding: "8px 18px",
                borderRadius: "6px",
                background: p.active ? "rgba(239, 68, 68, 0.15)" : "rgba(16, 185, 129, 0.15)",
                border: `1px solid ${p.active ? "#ef4444" : "var(--accent-green)"}`,
                color: p.active ? "#ef4444" : "var(--accent-green)",
                fontWeight: 600,
              }}
            >
              {p.active ? "Disable" : "Enable"}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};
