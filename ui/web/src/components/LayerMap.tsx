import React, { useState } from "react";

interface LayerMapProps {
  totalLayers?: number;
  vramLayers?: number[];
  ramLayers?: number[];
  nvmeLayers?: number[];
  activeLayer?: number;
  prefetchLayers?: number[];
  onPinLayer?: (layerId: number, tier: string) => void;
}

export const LayerMap: React.FC<LayerMapProps> = ({
  totalLayers = 80,
  vramLayers = Array.from({ length: 18 }, (_, i) => i),
  ramLayers = Array.from({ length: 37 }, (_, i) => i + 18),
  nvmeLayers = Array.from({ length: 25 }, (_, i) => i + 55),
  activeLayer = 0,
  prefetchLayers = [1, 2],
  onPinLayer,
}) => {
  const [pinned, setPinned] = useState<Record<number, string>>({});
  const [hovered, setHovered] = useState<number | null>(null);

  const handleCellClick = (layerId: number) => {
    const nextTier = pinned[layerId] === "vram" ? "unpinned" : "vram";
    setPinned((prev) => ({ ...prev, [layerId]: nextTier }));
    if (onPinLayer) onPinLayer(layerId, nextTier);
  };

  const getTier = (id: number) => {
    if (pinned[id] === "vram") return "vram";
    if (vramLayers.includes(id)) return "vram";
    if (ramLayers.includes(id)) return "ram";
    return "nvme";
  };

  return (
    <div className="glass-panel" style={{ padding: "28px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
        <div>
          <h3 style={{ fontSize: "18px", fontWeight: 700, color: "var(--text-primary)" }}>
            2D LAYER RESIDENCY HEATMAP ({totalLayers} LAYERS)
          </h3>
          <p style={{ color: "var(--text-muted)", fontSize: "13px", marginTop: "2px" }}>
            Real-time memory hierarchy tracking via 200ms telemetry. Click any layer to pin to VRAM.
          </p>
        </div>
        {/* Legend */}
        <div style={{ display: "flex", gap: "16px", fontSize: "12px", fontFamily: "var(--font-mono)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ width: "10px", height: "10px", borderRadius: "2px", background: "var(--accent-amber)" }}></span>
            <span>VRAM (Hot)</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ width: "10px", height: "10px", borderRadius: "2px", background: "var(--accent-blue)" }}></span>
            <span>RAM (Warm)</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ width: "10px", height: "10px", borderRadius: "2px", background: "var(--accent-slate)" }}></span>
            <span>NVMe (Cold)</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ width: "10px", height: "10px", borderRadius: "2px", background: "var(--accent-green)" }}></span>
            <span>Executing</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ width: "10px", height: "10px", borderRadius: "2px", background: "#eab308" }}></span>
            <span>Prefetch</span>
          </div>
        </div>
      </div>

      {/* Grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(16, 1fr)",
          gap: "8px",
          margin: "20px 0",
        }}
      >
        {Array.from({ length: totalLayers }, (_, i) => {
          const tier = getTier(i);
          const isActive = activeLayer === i;
          const isPrefetch = prefetchLayers.includes(i);
          const isPinned = pinned[i] === "vram";

          let bg = "var(--accent-slate)";
          let color = "#94a3b8";
          let border = "1px solid transparent";

          if (tier === "vram") {
            bg = "#d97706";
            color = "#ffffff";
          } else if (tier === "ram") {
            bg = "#2563eb";
            color = "#ffffff";
          }

          if (isActive) {
            bg = "#10b981";
            color = "#ffffff";
          } else if (isPrefetch) {
            bg = "#ca8a04";
            color = "#ffffff";
          }

          if (isPinned) {
            border = "2px solid #ffffff";
          }

          return (
            <div
              key={i}
              onClick={() => handleCellClick(i)}
              onMouseEnter={() => setHovered(i)}
              onMouseLeave={() => setHovered(null)}
              className={`cell ${isActive ? "pulsing-active" : isPrefetch ? "blinking-prefetch" : ""}`}
              style={{
                aspectRatio: "1",
                borderRadius: "6px",
                background: bg,
                color: color,
                border: border,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                fontSize: "11px",
                fontFamily: "var(--font-mono)",
                fontWeight: 600,
                cursor: "pointer",
                position: "relative",
                transition: "all 0.15s ease",
              }}
              title={`Layer ${i} (${tier.toUpperCase()}) - Click to pin`}
            >
              {i.toString().padStart(2, "0")}
              {isPinned && <span style={{ fontSize: "8px", lineHeight: 1 }}>📌</span>}
            </div>
          );
        })}
      </div>

      {/* Hover Info Drawer */}
      <div
        style={{
          padding: "12px 18px",
          background: "rgba(0,0,0,0.25)",
          borderRadius: "8px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          fontFamily: "var(--font-mono)",
          fontSize: "12px",
          color: "var(--text-secondary)",
        }}
      >
        <div>
          {hovered !== null ? (
            <span>
              Inspecting <strong>Layer {hovered.toString().padStart(2, "0")}</strong> | Tier:{" "}
              <span style={{ color: "var(--accent-amber)" }}>{getTier(hovered).toUpperCase()}</span> | Type:{" "}
              {hovered % 2 === 0 ? "Self-Attention + GQA" : "MLP (Spectral FP8 Compressed)"} | Size: ~245 MB
            </span>
          ) : (
            <span>Hover over any layer cell to view memory profiling & sparsity metrics.</span>
          )}
        </div>
        <div style={{ color: "var(--accent-green)" }}>● Telemetry Stream Active (200ms)</div>
      </div>
    </div>
  );
};
