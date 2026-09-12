import React from "react";
import { CeilingLift } from "../components/CeilingLift";
import { LayerMap } from "../components/LayerMap";
import { VRAMGauge } from "../components/VRAMGauge";
import { ThermalMonitor } from "../components/ThermalMonitor";
import { CompareOllama } from "../components/CompareOllama";

export const Dashboard: React.FC<{ metrics: any; hardware: any }> = ({ metrics, hardware }) => {
  return (
    <div>
      <CeilingLift hardware={hardware} metrics={metrics} />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px", marginBottom: "24px" }}>
        <VRAMGauge
          vramUsedMb={metrics?.vram_mb || 5821}
          vramTotalMb={hardware?.vram_gb ? hardware.vram_gb * 1024 : 6144}
          ramUsedMb={metrics?.ram_mb || 18400}
          ramTotalMb={hardware?.ram_gb ? hardware.ram_gb * 1024 : 32768}
          nvmeUsedMb={metrics?.nvme_mb || 45000}
          nvmeTotalMb={512000}
        />
        <ThermalMonitor
          temperatureC={67}
          powerWatts={95}
          throttleActive={metrics?.throttle_active || false}
          state={metrics?.thermal_state || "nominal"}
        />
      </div>
      <CompareOllama currentVramGb={hardware?.vram_gb || 6.0} />
      <LayerMap
        activeLayer={metrics?.active_layer || 0}
        prefetchLayers={metrics?.prefetch_layers || [1, 2]}
      />
    </div>
  );
};
