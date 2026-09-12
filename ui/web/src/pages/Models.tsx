import React, { useState } from "react";

export const Models: React.FC = () => {
  const [search, setSearch] = useState("");
  const [pulling, setPulling] = useState<string | null>(null);
  const [pullProgress, setPullProgress] = useState(0);

  const installedModels = [
    { id: "llama3:70b", name: "Meta LLaMA 3 70B Instruct", size: "38.4 GB", quant: "SPECTRAL", speed: "4.2 tok/s", layers: 80 },
    { id: "mistral:22b", name: "Mistral NeMo 22B", size: "12.8 GB", quant: "SPECTRAL", speed: "11.3 tok/s", layers: 40 },
    { id: "phi3:3.8b", name: "Microsoft Phi-3 Mini 3.8B", size: "2.2 GB", quant: "SPECTRAL", speed: "47.2 tok/s", layers: 32 },
  ];

  const handlePull = (modelId: string) => {
    setPulling(modelId);
    setPullProgress(0);
    const interval = setInterval(() => {
      setPullProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setPulling(null);
          return 100;
        }
        return prev + 10;
      });
    }, 400);
  };

  return (
    <div>
      <div className="glass-panel" style={{ padding: "28px", marginBottom: "24px" }}>
        <h2 style={{ fontSize: "20px", fontWeight: 700, marginBottom: "16px", color: "var(--accent-amber)" }}>
          MODEL LIFECYCLE & REGISTRY
        </h2>
        <div style={{ display: "flex", gap: "12px" }}>
          <input
            type="text"
            placeholder="Search HuggingFace Hub or enter model ID (e.g. llama3:70b, mistral:22b)..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{
              flex: 1,
              padding: "12px 18px",
              background: "rgba(0,0,0,0.4)",
              border: "1px solid var(--panel-border)",
              borderRadius: "8px",
              color: "#fff",
              fontFamily: "var(--font-sans)",
              fontSize: "14px",
              outline: "none",
            }}
          />
          <button
            onClick={() => handlePull(search || "qwen2:72b")}
            style={{
              padding: "12px 24px",
              background: "var(--accent-amber)",
              color: "#000",
              fontWeight: 700,
              borderRadius: "8px",
              display: "flex",
              alignItems: "center",
              gap: "8px",
            }}
          >
            Pull Model
          </button>
        </div>

        {pulling && (
          <div style={{ marginTop: "20px", padding: "16px", background: "rgba(0,0,0,0.3)", borderRadius: "8px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "13px", marginBottom: "6px" }}>
              <span>Pulling and converting <strong>{pulling}</strong>...</span>
              <span style={{ fontFamily: "var(--font-mono)" }}>{pullProgress}%</span>
            </div>
            <div style={{ height: "6px", background: "rgba(255,255,255,0.06)", borderRadius: "3px", overflow: "hidden" }}>
              <div style={{ width: `${pullProgress}%`, height: "100%", background: "var(--accent-amber)", transition: "width 0.3s" }}></div>
            </div>
          </div>
        )}
      </div>

      <div className="glass-panel" style={{ padding: "28px" }}>
        <h3 style={{ fontSize: "18px", fontWeight: 700, marginBottom: "16px" }}>INSTALLED MODELS</h3>
        <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          {installedModels.map((m) => (
            <div
              key={m.id}
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "16px 20px",
                background: "rgba(0,0,0,0.25)",
                borderRadius: "10px",
                border: "1px solid var(--panel-border)",
              }}
            >
              <div>
                <div style={{ fontWeight: 700, fontSize: "16px", color: "var(--text-primary)" }}>{m.id}</div>
                <div style={{ color: "var(--text-secondary)", fontSize: "13px", marginTop: "4px" }}>
                  {m.name} • {m.layers} Layers
                </div>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "24px", fontFamily: "var(--font-mono)", fontSize: "13px" }}>
                <div><span style={{ color: "var(--text-muted)" }}>SIZE:</span> {m.size}</div>
                <div><span style={{ color: "var(--text-muted)" }}>QUANT:</span> {m.quant}</div>
                <div><span style={{ color: "var(--text-muted)" }}>SPEED:</span> {m.speed}</div>
                <button
                  style={{
                    padding: "8px 16px",
                    background: "rgba(16, 185, 129, 0.15)",
                    border: "1px solid var(--accent-green)",
                    color: "var(--accent-green)",
                    borderRadius: "6px",
                    fontWeight: 600,
                  }}
                >
                  Run
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
