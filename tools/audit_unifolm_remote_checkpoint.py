#!/usr/bin/env python3
"""Inspect a remote PyTorch ZIP checkpoint using only HTTP ranges and pickle metadata.

No tensor storage is downloaded or materialized.  The unpickler accepts only
the four globals present in the official UniFoLM checkpoint descriptor.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import zipfile

from huggingface_hub import hf_hub_url, model_info
import requests


INTERFACE_KEYS = (
    "action_model.state_encoder.layer1.weight",
    "action_model.action_encoder.layer1.weight",
    "action_model.action_decoder.layer2.weight",
    "action_model.action_decoder.layer2.bias",
)


class RemoteRangeReader(io.RawIOBase):
    def __init__(self, url: str) -> None:
        self.session = requests.Session()
        response = self.session.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
        response.raise_for_status()
        self.final_url = response.url
        self.size = int(response.headers["Content-Range"].split("/")[-1])
        self.position = 0
        self.bytes_downloaded = len(response.content)

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.position = offset
        elif whence == io.SEEK_CUR:
            self.position += offset
        elif whence == io.SEEK_END:
            self.position = self.size + offset
        else:
            raise ValueError(f"unknown whence {whence}")
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self.position
        if size == 0:
            return b""
        start = self.position
        stop = min(self.size - 1, start + size - 1)
        response = self.session.get(
            self.final_url, headers={"Range": f"bytes={start}-{stop}"}, timeout=120
        )
        response.raise_for_status()
        expected = stop - start + 1
        if len(response.content) != expected:
            raise IOError(f"short range read: expected {expected}, got {len(response.content)}")
        self.position += len(response.content)
        self.bytes_downloaded += len(response.content)
        return response.content


class StorageTypeStub:
    def __init__(self, name: str) -> None:
        self.name = name


class StorageStub:
    def __init__(self, storage_type: StorageTypeStub, key: str, location: str, numel: int) -> None:
        self.storage_type = storage_type.name
        self.key = str(key)
        self.location = location
        self.numel = int(numel)


class TensorStub:
    def __init__(
        self,
        storage: StorageStub,
        offset: int,
        shape: tuple[int, ...],
        stride: tuple[int, ...],
        requires_grad: bool,
    ) -> None:
        self.storage = storage
        self.offset = int(offset)
        self.shape = tuple(int(value) for value in shape)
        self.stride = tuple(int(value) for value in stride)
        self.requires_grad = bool(requires_grad)

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


def rebuild_tensor(
    storage: StorageStub,
    offset: int,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    requires_grad: bool,
    backward_hooks: object,
    *metadata: object,
) -> TensorStub:
    del backward_hooks, metadata
    return TensorStub(storage, offset, shape, stride, requires_grad)


class MetadataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> object:
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return rebuild_tensor
        if module == "torch" and name in {"BFloat16Storage", "FloatStorage"}:
            return StorageTypeStub(name)
        raise pickle.UnpicklingError(f"refusing unexpected global {module}.{name}")

    def persistent_load(self, persistent_id: object) -> StorageStub:
        tag, storage_type, key, location, numel = persistent_id
        if tag != "storage" or not isinstance(storage_type, StorageTypeStub):
            raise pickle.UnpicklingError(f"unexpected persistent id {persistent_id!r}")
        return StorageStub(storage_type, key, location, numel)


def prefix_count(state_dict: dict[str, TensorStub], depth: int) -> dict[str, int]:
    counts: collections.Counter[str] = collections.Counter()
    for key, tensor in state_dict.items():
        counts[".".join(key.split(".")[:depth])] += tensor.numel
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="unitreerobotics/UnifoLM-VLA-Base")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--filename", default="checkpoints/pytorch_model.pt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    info = model_info(args.repo, revision=args.revision, files_metadata=True)
    sibling = next(item for item in info.siblings if item.rfilename == args.filename)
    lfs = sibling.lfs
    reader = RemoteRangeReader(hf_hub_url(args.repo, args.filename, revision=args.revision))
    with zipfile.ZipFile(reader) as archive:
        entries = archive.infolist()
        pickle_entry = next(entry for entry in entries if entry.filename.endswith("/data.pkl"))
        pickle_data = archive.read(pickle_entry.filename)
    state_dict = MetadataOnlyUnpickler(io.BytesIO(pickle_data)).load()
    if not isinstance(state_dict, collections.OrderedDict):
        raise TypeError(f"expected OrderedDict, got {type(state_dict)}")
    if not all(isinstance(value, TensorStub) for value in state_dict.values()):
        raise TypeError("checkpoint descriptor contains a non-tensor value")

    dtype_tensor_counts: collections.Counter[str] = collections.Counter()
    dtype_parameter_counts: collections.Counter[str] = collections.Counter()
    for tensor in state_dict.values():
        dtype_tensor_counts[tensor.storage.storage_type] += 1
        dtype_parameter_counts[tensor.storage.storage_type] += tensor.numel
    total_parameters = sum(tensor.numel for tensor in state_dict.values())
    interface = {
        key: {
            "shape": list(state_dict[key].shape),
            "dtype": state_dict[key].storage.storage_type,
            "parameters": state_dict[key].numel,
        }
        for key in INTERFACE_KEYS
    }
    native_dimension_tied = sum(item["parameters"] for item in interface.values())
    target_shapes = {
        INTERFACE_KEYS[0]: (1024, 28),
        INTERFACE_KEYS[1]: (1536, 28),
        INTERFACE_KEYS[2]: (28, 1024),
        INTERFACE_KEYS[3]: (28,),
    }
    new_parameters = sum(math.prod(shape) for shape in target_shapes.values())
    reusable_parameters = total_parameters - native_dimension_tied
    target_total = reusable_parameters + new_parameters
    depth_four = prefix_count(state_dict, 4)
    visual_parameters = depth_four["qwen_vl_interface.model.model.visual"]
    language_parameters = (
        depth_four["qwen_vl_interface.model.model.language_model"]
        + depth_four["qwen_vl_interface.model.lm_head.weight"]
    )
    payload = {
        "audit_method": "HTTP byte ranges + ZIP central directory + allowlisted metadata-only unpickle",
        "tensor_storage_downloaded": False,
        "repo": info.id,
        "revision": info.sha,
        "filename": args.filename,
        "file_size_bytes": sibling.size,
        "lfs_sha256": getattr(lfs, "sha256", None),
        "zip_entry_count": len(entries),
        "pickle_entry_size_bytes": pickle_entry.file_size,
        "pickle_sha256": hashlib.sha256(pickle_data).hexdigest(),
        "http_bytes_downloaded": reader.bytes_downloaded,
        "state_dict_tensor_count": len(state_dict),
        "dtype_tensor_counts": dict(dtype_tensor_counts),
        "dtype_parameter_counts": dict(dtype_parameter_counts),
        "total_parameters": total_parameters,
        "prefix_parameter_counts_depth_1": prefix_count(state_dict, 1),
        "visual_encoder_parameters": visual_parameters,
        "language_model_plus_lm_head_parameters": language_parameters,
        "action_model_parameters": prefix_count(state_dict, 1)["action_model"],
        "native_interface": interface,
        "native_dimension_tied_parameters": native_dimension_tied,
        "target_28d_interface_shapes": {key: list(value) for key, value in target_shapes.items()},
        "new_28d_specific_parameters": new_parameters,
        "net_parameter_increase": new_parameters - native_dimension_tied,
        "reusable_pretrained_parameters": reusable_parameters,
        "target_total_parameters": target_total,
        "pretrained_parameter_reuse_fraction_of_native": reusable_parameters / total_parameters,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
