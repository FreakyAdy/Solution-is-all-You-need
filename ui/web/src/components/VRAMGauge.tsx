import React from "react";

interface VRAMGaugeProps {
  vramUsedMb: number;
  vramTotalMb: number;
  ramUsedMb: number;
  ramTotalMb: number;
  nvmeUsedMb: number;
  nvmeTotalMb: number;
}

export const VRAMGauge: React.FC<VRAMGaugeProps> = ({
  vramUsedMb = 5821,
  vramTotalMb = 6144,
  ramUsedMb = 18400,
  ramTotalMb = 32768,
  nvmeUsedMb = 45000,
  nvmeTotalMb = 512000,
}) => {
  const vramPct = Math.min(100, (vramUsedMb / vramTotalMb) * 100);
  const ramPct = Math.min(100, (ramUsedMb / ramTotalMb) * 100);
  const nvmePct = Math.min(100, (nvmeUsedMb / nvmeTotalMb) * 100);

  return (
    <div className="glass-panel" style={{ padding: "24px" }}>
      <h3 style={{ fontSize: "16px", fontWeight: 700, marginBottom: "20px", color: "var(--text-primary)" }}>
        MEMORY HIERARCHY ALLOCATION
      </h3>

      {/* VRAM */}
      <div style={{ marginBottom: "16px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "13px", marginBottom: "6px" }}>
          <span style={{ fontWeight: 600, color: "var(--accent-amber)" }}>GPU VRAM (Tier 1 Hot)</span>
          <span style={{ fontFamily: "var(--font-mono)" }}>
            {(vramUsedMb / 1024).toFixed(1)} / {(vramTotalMb / 1024).toFixed(1)} GB ({vramPct.toFixed(0)}%)
          </span>
        </div>
        <div style={{ height: "8px", background: "rgba(255,255,255,0.06)", borderRadius: "4px", overflow: "hidden" }}>
          <div style={{ width: `${vramPct}%`, height: "100%", background: "var(--accent-amber)" }}></div>
        </div>
      </div>

      {/* RAM */}
      <div style={{ marginBottom: "16px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "13px", marginBottom: "6px" }}>
          <span style={{ fontWeight: 600, color: "var(--accent-blue)" }}>SYSTEM RAM (Tier 2 Warm)</span>
          <span style={{ fontFamily: "var(--font-mono)" }}>
            {(ramUsedMb / 1024).toFixed(1)} / {(ramTotalMb / 1024).toFixed(1)} GB ({ramPct.toFixed(0)}%)
          </span>
        </div>
        <div style={{ height: "8px", background: "rgba(255,255,255,0.06)", borderRadius: "4px", overflow: "hidden" }}>
          <div style={{ width: `${ramPct}%`, height: "100%", background: "var(--accent-blue)" }}></div>
        </div>
      </div>

      {/* NVMe */}
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "13px", marginBottom: "6px" }}>
          <span style={{ fontWeight: 600, color: "var(--text-secondary)" }}>NVMe SWAP (Tier 3 Cold)</span>
          <span style={{ fontFamily: "var(--font-mono)" }}>
            {(nvmeUsedMb / 1024).toFixed(1)} / {(nvmeTotalMb / 1024).toFixed(1)} GB ({nvmePct.toFixed(0)}%)
          </span>
        </div>
        <div style={{ height: "8px", background: "rgba(255,255,255,0.06)", borderRadius: "4px", overflow: "hidden" }}>
          <div style={{ width: `${nvmePct}%`, height: "100%", background: "var(--accent-slate)" }}></div>
        </div>
      </div>
    </div>
  );
};
