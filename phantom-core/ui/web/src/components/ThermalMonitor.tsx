import React from "react";

interface ThermalMonitorProps {
  temperatureC?: number;
  powerWatts?: number;
  throttleActive?: boolean;
  state?: string;
}

export const ThermalMonitor: React.FC<ThermalMonitorProps> = ({
  temperatureC = 67,
  powerWatts = 95,
  throttleActive = false,
  state = "nominal",
}) => {
  const isElevated = temperatureC >= 75;
  const isCritical = temperatureC >= 85;

  const statusColor = isCritical ? "#ef4444" : isElevated ? "var(--accent-amber)" : "var(--accent-green)";

  return (
    <div className="glass-panel" style={{ padding: "24px" }}>
      <h3 style={{ fontSize: "16px", fontWeight: 700, marginBottom: "16px", color: "var(--text-primary)" }}>
        THERMAL & RESONANCE SAMPLER
      </h3>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px", fontFamily: "var(--font-mono)" }}>
        <div style={{ padding: "12px", background: "rgba(0,0,0,0.25)", borderRadius: "8px" }}>
          <div style={{ fontSize: "11px", color: "var(--text-muted)" }}>GPU CORE TEMP</div>
          <div style={{ fontSize: "24px", fontWeight: 800, color: statusColor, marginTop: "4px" }}>
            {temperatureC}°C
          </div>
          <div style={{ fontSize: "11px", color: "var(--text-secondary)", marginTop: "2px" }}>
            Status: {state.toUpperCase()}
          </div>
        </div>

        <div style={{ padding: "12px", background: "rgba(0,0,0,0.25)", borderRadius: "8px" }}>
          <div style={{ fontSize: "11px", color: "var(--text-muted)" }}>POWER DRAW</div>
          <div style={{ fontSize: "24px", fontWeight: 800, color: "var(--text-primary)", marginTop: "4px" }}>
            {powerWatts} W
          </div>
          <div style={{ fontSize: "11px", color: throttleActive ? "#ef4444" : "var(--accent-green)", marginTop: "2px" }}>
            Throttle: {throttleActive ? "ACTIVE" : "INACTIVE"}
          </div>
        </div>
      </div>
      <p style={{ fontSize: "12px", color: "var(--text-muted)", marginTop: "14px", lineHeight: "1.4" }}>
        Resonance Sampler dynamically adjusts top-p and temperature penalties to guarantee generation fidelity.
      </p>
    </div>
  );
};
