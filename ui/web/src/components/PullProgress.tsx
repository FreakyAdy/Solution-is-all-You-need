import React from "react";

export interface PullProgressProps {
  modelId: string;
  stage: "downloading" | "dequantizing" | "spectral_quant" | "wraith_calib" | "serializing" | "completed";
  progressPct: number;
  speedMbps?: number;
  etaSeconds?: number;
  currentLayer?: number;
  totalLayers?: number;
  detailMessage?: string;
  onCancel?: () => void;
}

const STAGES = [
  { key: "downloading", label: "1. Download GGUF", desc: "Resumable chunked transfer" },
  { key: "dequantizing", label: "2. SIMD Dequant", desc: "Pure NumPy bit-unpacking" },
  { key: "spectral_quant", label: "3. Spectral Requant", desc: "DCT FP8 MLP compression" },
  { key: "wraith_calib", label: "4. Wraith Pre-warm", desc: "Predictor transition profiling" },
  { key: "serializing", label: "5. Write .phantomw", desc: "Direct zero-copy layer packing" },
];

export const PullProgress: React.FC<PullProgressProps> = ({
  modelId,
  stage = "downloading",
  progressPct = 42,
  speedMbps = 45.8,
  etaSeconds = 120,
  currentLayer = 34,
  totalLayers = 80,
  detailMessage = "Processing layer tensors with Spectral DCT...",
  onCancel,
}) => {
  const formatEta = (seconds: number) => {
    if (seconds < 60) return `${seconds}s`;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
  };

  const getStageIndex = (s: string) => {
    switch (s) {
      case "downloading": return 0;
      case "dequantizing": return 1;
      case "spectral_quant": return 2;
      case "wraith_calib": return 3;
      case "serializing": return 4;
      case "completed": return 5;
      default: return 0;
    }
  };

  const currentIdx = getStageIndex(stage);

  return (
    <div
      className="glass-panel"
      style={{
        padding: "24px",
        marginTop: "16px",
        marginBottom: "20px",
        border: "1px solid rgba(245, 158, 11, 0.4)",
        background: "rgba(18, 22, 34, 0.85)",
        borderRadius: "12px",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <div
            style={{
              width: "10px",
              height: "10px",
              borderRadius: "50%",
              backgroundColor: "var(--accent-amber)",
              boxShadow: "0 0 10px var(--accent-amber)",
              animation: "pulse 1.5s infinite",
            }}
          />
          <h4 style={{ margin: 0, fontSize: "16px", fontWeight: 700, color: "var(--text-primary)" }}>
            CONVERSION PIPELINE: <span style={{ color: "var(--accent-amber)" }}>{modelId}</span>
          </h4>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          {speedMbps && (
            <span style={{ fontSize: "13px", fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>
              ⚡ {speedMbps.toFixed(1)} MB/s
            </span>
          )}
          {etaSeconds !== undefined && etaSeconds > 0 && (
            <span style={{ fontSize: "13px", fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
              ETA: {formatEta(etaSeconds)}
            </span>
          )}
          {onCancel && (
            <button
              onClick={onCancel}
              style={{
                padding: "4px 12px",
                fontSize: "12px",
                background: "rgba(239, 68, 68, 0.15)",
                color: "#ef4444",
                border: "1px solid rgba(239, 68, 68, 0.3)",
                borderRadius: "6px",
                cursor: "pointer",
              }}
            >
              Cancel
            </button>
          )}
        </div>
      </div>

      {/* Pipeline Stage Badges */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(5, 1fr)",
          gap: "8px",
          marginBottom: "16px",
        }}
      >
        {STAGES.map((stg, idx) => {
          const isDone = idx < currentIdx || stage === "completed";
          const isCurrent = idx === currentIdx && stage !== "completed";
          return (
            <div
              key={stg.key}
              style={{
                padding: "8px 10px",
                borderRadius: "6px",
                background: isCurrent
                  ? "rgba(245, 158, 11, 0.2)"
                  : isDone
                  ? "rgba(16, 185, 129, 0.15)"
                  : "rgba(255, 255, 255, 0.03)",
                border: isCurrent
                  ? "1px solid var(--accent-amber)"
                  : isDone
                  ? "1px solid rgba(16, 185, 129, 0.4)"
                  : "1px solid rgba(255, 255, 255, 0.06)",
                transition: "all 0.3s ease",
              }}
            >
              <div
                style={{
                  fontSize: "12px",
                  fontWeight: 600,
                  color: isCurrent ? "var(--accent-amber)" : isDone ? "var(--accent-green)" : "var(--text-muted)",
                }}
              >
                {isDone ? "✓ " : ""}{stg.label}
              </div>
              <div style={{ fontSize: "10px", color: "var(--text-muted)", marginTop: "2px" }}>
                {stg.desc}
              </div>
            </div>
          );
        })}
      </div>

      {/* Progress Bar */}
      <div
        style={{
          width: "100%",
          height: "8px",
          backgroundColor: "rgba(255, 255, 255, 0.05)",
          borderRadius: "4px",
          overflow: "hidden",
          position: "relative",
          marginBottom: "10px",
        }}
      >
        <div
          style={{
            width: `${Math.min(100, Math.max(0, progressPct))}%`,
            height: "100%",
            background: "linear-gradient(90deg, #f59e0b 0%, #10b981 100%)",
            transition: "width 0.3s ease",
            boxShadow: "0 0 10px rgba(245, 158, 11, 0.5)",
          }}
        />
      </div>

      {/* Status Details */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "12px" }}>
        <span style={{ color: "var(--text-secondary)" }}>
          {detailMessage || (totalLayers ? `Layer ${currentLayer}/${totalLayers}` : "Processing...")}
        </span>
        <span style={{ fontFamily: "var(--font-mono)", fontWeight: 700, color: "var(--accent-amber)" }}>
          {progressPct.toFixed(1)}%
        </span>
      </div>
    </div>
  );
};
