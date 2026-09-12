import React from "react";
import { VRAMGauge } from "../components/VRAMGauge";
import { ThermalMonitor } from "../components/ThermalMonitor";

export const Metrics: React.FC<{ metrics: any }> = ({ metrics }) => {
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px", marginBottom: "24px" }}>
        <VRAMGauge
          vramUsedMb={metrics?.vram_mb || 5821}
          vramTotalMb={6144}
          ramUsedMb={metrics?.ram_mb || 18400}
          ramTotalMb={32768}
          nvmeUsedMb={metrics?.nvme_mb || 45000}
          nvmeTotalMb={512000}
        />
        <ThermalMonitor
          temperatureC={67}
          powerWatts={95}
          throttleActive={false}
          state="nominal"
        />
      </div>

      <div className="glass-panel" style={{ padding: "28px" }}>
        <h3 style={{ fontSize: "18px", fontWeight: 700, marginBottom: "16px" }}>
          CORE INNOVATION TELEMETRY (200ms STREAM)
        </h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "16px", fontFamily: "var(--font-mono)" }}>
          <div style={{ padding: "16px", background: "rgba(0,0,0,0.25)", borderRadius: "8px" }}>
            <div style={{ color: "var(--text-muted)", fontSize: "12px" }}>WRAITH PREDICTOR</div>
            <div style={{ fontSize: "28px", fontWeight: 800, color: "var(--accent-amber)", marginTop: "6px" }}>
              87.3%
            </div>
            <div style={{ fontSize: "12px", color: "var(--text-secondary)", marginTop: "4px" }}>
              Prefetch Latency: &lt;1.0ms
            </div>
          </div>

          <div style={{ padding: "16px", background: "rgba(0,0,0,0.25)", borderRadius: "8px" }}>
            <div style={{ color: "var(--text-muted)", fontSize: "12px" }}>NEURAL CACHE</div>
            <div style={{ fontSize: "28px", fontWeight: 800, color: "var(--accent-blue)", marginTop: "6px" }}>
              7.8×
            </div>
            <div style={{ fontSize: "12px", color: "var(--text-secondary)", marginTop: "4px" }}>
              Cosine Error: &lt;1.8%
            </div>
          </div>

          <div style={{ padding: "16px", background: "rgba(0,0,0,0.25)", borderRadius: "8px" }}>
            <div style={{ color: "var(--text-muted)", fontSize: "12px" }}>SPARSITY ROUTING</div>
            <div style={{ fontSize: "28px", fontWeight: 800, color: "var(--accent-green)", marginTop: "6px" }}>
              61.2%
            </div>
            <div style={{ fontSize: "12px", color: "var(--text-secondary)", marginTop: "4px" }}>
              Inactive Neurons Skipped
            </div>
          </div>

          <div style={{ padding: "16px", background: "rgba(0,0,0,0.25)", borderRadius: "8px" }}>
            <div style={{ color: "var(--text-muted)", fontSize: "12px" }}>NVMe PAGING SPEED</div>
            <div style={{ fontSize: "28px", fontWeight: 800, color: "var(--text-primary)", marginTop: "6px" }}>
              3.5 GB/s
            </div>
            <div style={{ fontSize: "12px", color: "var(--text-secondary)", marginTop: "4px" }}>
              Avg Layer Load: 42ms
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
