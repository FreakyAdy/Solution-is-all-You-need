import React from "react";
import { LayerMap } from "../components/LayerMap";

export const Layers: React.FC<{ metrics: any }> = ({ metrics }) => {
  return (
    <div>
      <LayerMap
        activeLayer={metrics?.active_layer || 0}
        prefetchLayers={metrics?.prefetch_layers || [1, 2]}
      />
    </div>
  );
};
