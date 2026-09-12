"""
PHANTOM PLATFORM — Native Binary Format Specification (.phantomw)
=================================================================
Per-layer high-throughput binary storage for spectral-quantized weights.

Binary layout:
  [4 bytes]  magic:                  "PHTW" (0x50 0x48 0x54 0x57)
  [4 bytes]  version:                1 (uint32)
  [4 bytes]  layer_id:               uint32
  [4 bytes]  num_tensors:            uint32

For each tensor:
  [64 bytes] tensor_name:            (null-padded UTF-8)
  [4 bytes]  rows:                   uint32
  [4 bytes]  cols:                   uint32
  [4 bytes]  k_coefficients_per_row: uint32 (0 = BF16 uncompressed, >0 = FP8 DCT)
  [4 bytes]  compressed_bytes:       uint32
  [N bytes]  data payload:           (FP8 DCT coefficients or BF16 raw)
"""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

PHTW_MAGIC = b"PHTW"
PHTW_VERSION = 1


@dataclass
class PhantomTensorRecord:
    name: str
    rows: int
    cols: int
    k_coeffs_per_row: int
    data: bytes

    @property
    def is_spectral(self) -> bool:
        return self.k_coeffs_per_row > 0


@dataclass
class PhantomLayerData:
    version: int
    layer_id: int
    tensors: Dict[str, PhantomTensorRecord]


class PhantomLayerWriter:
    """Serializes per-layer weights into .phantomw binary format."""

    def __init__(self, layer_id: int, version: int = PHTW_VERSION):
        self.layer_id = layer_id
        self.version = version
        self.tensors: List[PhantomTensorRecord] = []

    def add_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        k_coeffs: int = 0,
        fp8_coeffs: Optional[bytes] = None,
    ) -> None:
        """
        Add a tensor to the layer.
        If k_coeffs > 0 and fp8_coeffs provided, stores as FP8 DCT coefficients.
        Otherwise, stores as raw BF16 bytes.
        """
        if tensor.dim() == 1:
            rows, cols = 1, tensor.size(0)
        elif tensor.dim() == 2:
            rows, cols = tensor.shape
        else:
            rows = tensor.size(0)
            cols = int(np.prod(tensor.shape[1:]))

        if k_coeffs > 0 and fp8_coeffs is not None:
            data = fp8_coeffs
        else:
            # Store as BF16 raw bytes
            bf16_t = tensor.to(torch.bfloat16).contiguous()
            data = bf16_t.view(torch.uint16).cpu().numpy().tobytes()

        self.tensors.append(
            PhantomTensorRecord(
                name=name,
                rows=rows,
                cols=cols,
                k_coeffs_per_row=k_coeffs,
                data=data,
            )
        )

    def write_to_file(self, output_path: Union[str, Path]) -> int:
        """Write layer payload to disk. Returns total bytes written."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        total_bytes = 0
        with open(out, "wb") as f:
            # Header
            header = struct.pack(
                "<4sIII",
                PHTW_MAGIC,
                self.version,
                self.layer_id,
                len(self.tensors),
            )
            f.write(header)
            total_bytes += len(header)

            # Tensors
            for t in self.tensors:
                # 64 bytes null-padded name
                name_bytes = t.name.encode("utf-8")[:64]
                padded_name = name_bytes.ljust(64, b"\x00")

                t_meta = struct.pack(
                    "<64sIIII",
                    padded_name,
                    t.rows,
                    t.cols,
                    t.k_coeffs_per_row,
                    len(t.data),
                )
                f.write(t_meta)
                f.write(t.data)
                total_bytes += len(t_meta) + len(t.data)

        logger.debug(
            "phantomw_written",
            layer_id=self.layer_id,
            tensors=len(self.tensors),
            bytes=total_bytes,
            path=str(out),
        )
        return total_bytes


class PhantomLayerReader:
    """High-speed memory-mapped reader for .phantomw files."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f".phantomw file not found: {self.path}")
        self._file = open(self.path, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self.layer_data = self._parse()

    def close(self):
        if hasattr(self, "_mmap") and self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if hasattr(self, "_file") and self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def _parse(self) -> PhantomLayerData:
        buf = self._mmap
        cursor = 0

        magic, version, layer_id, n_tensors = struct.unpack_from("<4sIII", buf, cursor)
        cursor += 16

        if magic != PHTW_MAGIC:
            raise ValueError(f"Invalid magic {magic!r}, expected {PHTW_MAGIC!r}")

        tensors: Dict[str, PhantomTensorRecord] = {}
        for _ in range(n_tensors):
            padded_name, rows, cols, k_coeffs, comp_bytes = struct.unpack_from(
                "<64sIIII", buf, cursor
            )
            cursor += 80  # 64 + 4 + 4 + 4 + 4 = 80
            name = padded_name.rstrip(b"\x00").decode("utf-8")
            data = buf[cursor : cursor + comp_bytes]
            cursor += comp_bytes

            tensors[name] = PhantomTensorRecord(
                name=name,
                rows=rows,
                cols=cols,
                k_coeffs_per_row=k_coeffs,
                data=data,
            )

        return PhantomLayerData(version=version, layer_id=layer_id, tensors=tensors)

    def load_tensor(self, name: str) -> torch.Tensor:
        """Loads and returns tensor as torch.Tensor (BF16)."""
        if name not in self.layer_data.tensors:
            raise KeyError(f"Tensor {name} not found in layer {self.layer_data.layer_id}")

        rec = self.layer_data.tensors[name]
        if rec.is_spectral:
            # Reconstruct from DCT FP8 coefficients (mock or IDCT)
            # FP8 coefficients are shape (rows, k_coeffs_per_row)
            fp8_arr = np.frombuffer(rec.data, dtype=np.uint8).reshape(rec.rows, rec.k_coeffs_per_row)
            # Decompress DCT back to (rows, cols)
            # In production, idct_reconstruct kernel or numpy idct
            from scipy.fft import idct
            # Unpack float8 (e4m3 or uint8 scaled)
            coeffs = fp8_arr.astype(np.float32) / 16.0
            padded = np.zeros((rec.rows, rec.cols), dtype=np.float32)
            padded[:, : rec.k_coeffs_per_row] = coeffs
            reconstructed = idct(padded, type=2, norm="ortho", axis=1)
            return torch.from_numpy(reconstructed).to(torch.bfloat16)
        else:
            # Raw BF16
            u16 = np.frombuffer(rec.data, dtype=np.uint16)
            t = torch.from_numpy(u16.copy()).view(torch.bfloat16)
            return t.reshape(rec.rows, rec.cols)
