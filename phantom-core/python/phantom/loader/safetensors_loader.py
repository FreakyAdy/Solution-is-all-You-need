"""
PHANTOM PLATFORM — Safetensors Loader
======================================
Fast memory-mapped reader for HuggingFace safetensors weights.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import structlog

logger = structlog.get_logger(__name__)

DTYPE_MAP = {
    "F32": (np.float32, torch.float32),
    "F16": (np.float16, torch.float16),
    "BF16": (np.uint16, torch.bfloat16),
    "I64": (np.int64, torch.int64),
    "I32": (np.int32, torch.int32),
    "I16": (np.int16, torch.int16),
    "I8": (np.int8, torch.int8),
    "U8": (np.uint8, torch.uint8),
    "BOOL": (np.bool_, torch.bool),
}


class SafetensorsLoader:
    """Memory-mapped reader for .safetensors files."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Safetensors file not found: {self.path}")
        self._file = open(self.path, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self.header, self.data_offset = self._read_header()

    def __del__(self):
        if hasattr(self, "_mmap") and self._mmap is not None:
            self._mmap.close()
        if hasattr(self, "_file") and self._file is not None:
            self._file.close()

    def _read_header(self) -> Tuple[Dict[str, Any], int]:
        header_len = struct.unpack("<Q", self._mmap[:8])[0]
        header_json = self._mmap[8 : 8 + header_len].decode("utf-8")
        header = json.loads(header_json)
        return header, 8 + header_len

    def get_tensor_names(self) -> List[str]:
        return [k for k in self.header.keys() if k != "__metadata__"]

    def load_tensor(self, name: str, target_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        if name not in self.header:
            raise KeyError(f"Tensor {name} not in safetensors header")
        
        info = self.header[name]
        dtype_str = info["dtype"]
        shape = tuple(info["shape"])
        start, end = info["data_offsets"]

        np_dtype, torch_dtype = DTYPE_MAP.get(dtype_str, (np.float32, torch.float32))
        raw_bytes = self._mmap[self.data_offset + start : self.data_offset + end]

        if dtype_str == "BF16":
            arr = np.frombuffer(raw_bytes, dtype=np.uint16)
            tensor = torch.from_numpy(arr.copy()).view(torch.bfloat16).reshape(shape)
        else:
            arr = np.frombuffer(raw_bytes, dtype=np_dtype)
            tensor = torch.from_numpy(arr.copy()).reshape(shape).to(torch_dtype)

        return tensor.to(target_dtype)

    def iter_tensors(
        self, target_dtype: torch.dtype = torch.bfloat16
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        for name in self.get_tensor_names():
            yield name, self.load_tensor(name, target_dtype)
