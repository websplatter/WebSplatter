#!/usr/bin/env python3
"""
Convert INRIA-format 3DGS PLY files to glTF 2.0 with KHR_gaussian_splatting extension.

Usage:
    python ply2gltf.py --input scene.ply --output scene.glb
    python ply2gltf.py --input scene.ply --output scene.gltf
    python ply2gltf.py --input scene.ply --output scene.glb --compression=spz

Dependencies:
    - numpy
    - spz (only when --compression=spz, install from https://github.com/nianticlabs/spz)
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np


# ---------- PLY parser ----------

# Map PLY type names to struct format + byte size
PLY_TYPE_MAP = {
    "char": ("b", 1),
    "int8": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "int16": ("h", 2),
    "ushort": ("H", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}


def parse_ply(filepath: str) -> dict[str, np.ndarray]:
    """Parse a binary_little_endian PLY file into a dict of property_name -> numpy array."""
    with open(filepath, "rb") as f:
        # Read ASCII header
        header_lines: list[str] = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        # Parse header
        vertex_count = 0
        properties: list[tuple[str, str]] = []  # (name, ply_type)
        in_vertex = False

        for line in header_lines:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                if parts[1] != "binary_little_endian":
                    raise ValueError(f"Unsupported PLY format: {parts[1]} (only binary_little_endian supported)")
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    raise ValueError("List properties not supported")
                properties.append((parts[2], parts[1]))

        if vertex_count == 0:
            raise ValueError("No vertices found in PLY file")

        # Build struct format for one vertex
        fmt = "<"
        for _, ply_type in properties:
            if ply_type not in PLY_TYPE_MAP:
                raise ValueError(f"Unsupported PLY type: {ply_type}")
            fmt += PLY_TYPE_MAP[ply_type][0]

        vertex_size = struct.calcsize(fmt)

        # Read binary data
        raw_data = f.read(vertex_count * vertex_size)
        if len(raw_data) < vertex_count * vertex_size:
            raise ValueError(
                f"PLY file truncated: expected {vertex_count * vertex_size} bytes, got {len(raw_data)}"
            )

    # Unpack into numpy arrays
    result: dict[str, np.ndarray] = {}
    all_data = np.frombuffer(raw_data, dtype=np.dtype([(name, f"<{PLY_TYPE_MAP[t][0]}") for name, t in properties]))
    for name, _ in properties:
        result[name] = all_data[name].astype(np.float32)

    print(f"  Parsed {vertex_count} splats, {len(properties)} properties per vertex")
    return result


# ---------- Data extraction ----------


def sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    pos = x >= 0
    result = np.empty_like(x)
    result[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    result[~pos] = exp_x / (1.0 + exp_x)
    return result


def extract_splat_data(ply: dict[str, np.ndarray], max_sh_degree: int) -> dict:
    """Extract and transform PLY data for glTF KHR_gaussian_splatting.

    Returns a dict with numpy arrays ready for glTF packing.
    """
    n = len(ply["x"])

    # Position: direct copy
    positions = np.column_stack([ply["x"], ply["y"], ply["z"]]).astype(np.float32)

    # Opacity: PLY stores pre-sigmoid -> glTF stores post-sigmoid [0,1]
    opacity = sigmoid(ply["opacity"]).astype(np.float32)

    # Scale: both formats use log-space, direct copy
    scale = np.column_stack([ply["scale_0"], ply["scale_1"], ply["scale_2"]]).astype(np.float32)

    # Rotation: PLY is [rot_0=w, rot_1=x, rot_2=y, rot_3=z] -> glTF is [x, y, z, w]
    rot_w = ply["rot_0"]
    rot_x = ply["rot_1"]
    rot_y = ply["rot_2"]
    rot_z = ply["rot_3"]
    rotation = np.column_stack([rot_x, rot_y, rot_z, rot_w]).astype(np.float32)
    # Normalize quaternions
    norms = np.linalg.norm(rotation, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    rotation /= norms

    # SH degree 0 (DC): f_dc_0, f_dc_1, f_dc_2
    sh_dc = np.column_stack([ply["f_dc_0"], ply["f_dc_1"], ply["f_dc_2"]]).astype(np.float32)

    # Determine available SH degree from f_rest fields
    f_rest_names = sorted(
        [k for k in ply if k.startswith("f_rest_")],
        key=lambda k: int(k.split("_")[-1]),
    )
    num_rest = len(f_rest_names)
    if num_rest > 0:
        coefficients_per_channel = num_rest // 3
        available_degree = round((coefficients_per_channel + 1) ** 0.5 - 1)
    else:
        coefficients_per_channel = 0
        available_degree = 0

    final_sh_degree = min(max_sh_degree, available_degree)

    # Higher-degree SH coefficients
    # PLY f_rest layout: f_rest_{offset + coefficientsPerChannel * channel}
    # where offset increments per degree: degree 1 has 3 terms, degree 2 has 5, degree 3 has 7
    sh_higher: dict[int, list[np.ndarray]] = {}  # degree -> list of VEC3 arrays (one per coefficient)

    if final_sh_degree >= 1 and coefficients_per_channel > 0:
        offset_in_channel = 0
        for d in range(1, final_sh_degree + 1):
            term_count = 2 * d + 1
            coefs: list[np.ndarray] = []
            for j in range(term_count):
                # For each coefficient, gather R, G, B from separate f_rest fields
                r_idx = offset_in_channel + j
                g_idx = offset_in_channel + j + coefficients_per_channel
                b_idx = offset_in_channel + j + coefficients_per_channel * 2
                r_name = f"f_rest_{r_idx}"
                g_name = f"f_rest_{g_idx}"
                b_name = f"f_rest_{b_idx}"
                vec3 = np.column_stack([ply[r_name], ply[g_name], ply[b_name]]).astype(np.float32)
                coefs.append(vec3)
            sh_higher[d] = coefs
            offset_in_channel += term_count

    print(f"  SH degree: {final_sh_degree} (available: {available_degree}, requested: {max_sh_degree})")

    return {
        "num_points": n,
        "positions": positions,
        "opacity": opacity,
        "scale": scale,
        "rotation": rotation,
        "sh_dc": sh_dc,
        "sh_higher": sh_higher,
        "sh_degree": final_sh_degree,
    }


# ---------- glTF builder (uncompressed) ----------

FLOAT32 = 5126  # glTF componentType for float32


def build_gltf_uncompressed(data: dict) -> tuple[dict, bytes]:
    """Build glTF JSON + BIN buffer from splat data (no compression)."""
    n = data["num_points"]
    sh_degree = data["sh_degree"]

    # Collect all binary chunks, track buffer views and accessors
    bin_parts: list[bytes] = []
    buffer_views: list[dict] = []
    accessors: list[dict] = []
    attributes: dict[str, int] = {}
    byte_offset = 0

    def add_accessor(
        name: str, arr: np.ndarray, acc_type: str, comp_type: int = FLOAT32
    ) -> int:
        nonlocal byte_offset
        idx = len(accessors)
        blob = arr.astype(np.float32).tobytes()
        # Pad to 4-byte alignment
        padding = (4 - len(blob) % 4) % 4
        blob_padded = blob + b"\x00" * padding

        buffer_views.append(
            {"buffer": 0, "byteOffset": byte_offset, "byteLength": len(blob)}
        )
        accessor_def: dict = {
            "bufferView": len(buffer_views) - 1,
            "componentType": comp_type,
            "count": n,
            "type": acc_type,
        }
        # Add min/max for POSITION (required by glTF spec)
        if name == "POSITION":
            accessor_def["min"] = arr.min(axis=0).tolist()
            accessor_def["max"] = arr.max(axis=0).tolist()
        accessors.append(accessor_def)

        bin_parts.append(blob_padded)
        byte_offset += len(blob_padded)
        return idx

    # POSITION
    attributes["POSITION"] = add_accessor("POSITION", data["positions"], "VEC3")

    # KHR_gaussian_splatting:OPACITY
    attributes["KHR_gaussian_splatting:OPACITY"] = add_accessor(
        "OPACITY", data["opacity"].reshape(-1, 1), "SCALAR"
    )

    # KHR_gaussian_splatting:SCALE
    attributes["KHR_gaussian_splatting:SCALE"] = add_accessor(
        "SCALE", data["scale"], "VEC3"
    )

    # KHR_gaussian_splatting:ROTATION
    attributes["KHR_gaussian_splatting:ROTATION"] = add_accessor(
        "ROTATION", data["rotation"], "VEC4"
    )

    # SH Degree 0 coefficient
    attributes["KHR_gaussian_splatting:SH_DEGREE_0_COEF_0"] = add_accessor(
        "SH_DC", data["sh_dc"], "VEC3"
    )

    # Higher degree SH coefficients
    for d in range(1, sh_degree + 1):
        if d not in data["sh_higher"]:
            break
        coefs = data["sh_higher"][d]
        for j, vec3 in enumerate(coefs):
            attr_name = f"KHR_gaussian_splatting:SH_DEGREE_{d}_COEF_{j}"
            attributes[attr_name] = add_accessor(f"SH_{d}_{j}", vec3, "VEC3")

    # Assemble binary buffer
    bin_data = b"".join(bin_parts)

    # Build glTF JSON
    gltf = {
        "asset": {"version": "2.0", "generator": "ply2gltf.py"},
        "extensionsUsed": ["KHR_gaussian_splatting"],
        "buffers": [{"byteLength": len(bin_data)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": attributes,
                        "mode": 0,  # POINTS
                        "extensions": {
                            "KHR_gaussian_splatting": {}
                        },
                    }
                ]
            }
        ],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }

    return gltf, bin_data


# ---------- glTF builder (SPZ compression) ----------


def build_gltf_spz(data: dict) -> tuple[dict, bytes]:
    """Build glTF JSON + BIN buffer with SPZ compression."""
    try:
        import spz
    except ImportError:
        print(
            "Error: 'spz' package is required for --compression=spz\n"
            "Install from: https://github.com/nianticlabs/spz\n"
            "  git clone https://github.com/nianticlabs/spz && cd spz && pip install .",
            file=sys.stderr,
        )
        sys.exit(1)

    n = data["num_points"]
    sh_degree = data["sh_degree"]

    # Create SPZ GaussianCloud
    cloud = spz.GaussianCloud()
    cloud.sh_degree = sh_degree
    cloud.antialiased = False

    # Positions: flat float32 array (n*3)
    cloud.positions = data["positions"].flatten().astype(np.float32)

    # Scales: log-space, flat float32 array (n*3)
    cloud.scales = data["scale"].flatten().astype(np.float32)

    # Rotations: XYZW quaternion, flat float32 array (n*4)
    cloud.rotations = data["rotation"].flatten().astype(np.float32)

    # Alphas: SPZ expects pre-sigmoid values
    # We converted to post-sigmoid for glTF, but SPZ wants pre-sigmoid
    # Inverse sigmoid: logit(p) = ln(p / (1-p))
    opacity_post = np.clip(data["opacity"], 1e-7, 1.0 - 1e-7)
    alphas_pre = np.log(opacity_post / (1.0 - opacity_post)).astype(np.float32)
    cloud.alphas = alphas_pre

    # Colors: SPZ stores raw SH DC coefficients (same as f_dc_0/1/2)
    cloud.colors = data["sh_dc"].flatten().astype(np.float32)

    # SH higher degree coefficients: coefficient-major order
    # SPZ format: [coef0_R, coef0_G, coef0_B, coef1_R, coef1_G, coef1_B, ...]
    if sh_degree >= 1 and data["sh_higher"]:
        sh_list: list[np.ndarray] = []
        for d in range(1, sh_degree + 1):
            if d not in data["sh_higher"]:
                break
            for vec3 in data["sh_higher"][d]:
                sh_list.append(vec3)  # Each is (n, 3) = [R, G, B]
        if sh_list:
            # Stack: (num_total_coefs, n, 3) -> transpose to (n, num_total_coefs, 3) -> flatten
            sh_all = np.stack(sh_list, axis=1)  # (n, num_coefs, 3)
            cloud.sh = sh_all.flatten().astype(np.float32)

    # Compress to SPZ bytes
    pack_options = spz.PackOptions()
    # Use UNSPECIFIED to avoid coordinate conversion (data is already in PLY coords)
    pack_options.from_coord = spz.CoordinateSystem.UNSPECIFIED

    # Save SPZ to bytes (write to temp file and read back if needed)
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".spz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        spz.save_spz(cloud, pack_options, tmp_path)
        with open(tmp_path, "rb") as f:
            spz_bytes = f.read()
    finally:
        os.unlink(tmp_path)

    print(f"  SPZ compressed: {len(spz_bytes)} bytes ({len(spz_bytes) / (1024*1024):.2f} MB)")

    # Pad SPZ blob to 4-byte alignment
    padding = (4 - len(spz_bytes) % 4) % 4
    bin_data = spz_bytes + b"\x00" * padding

    # Build accessor stubs (no bufferView, just metadata for fallback)
    accessors: list[dict] = []
    attributes: dict[str, int] = {}

    def add_stub_accessor(acc_type: str) -> int:
        idx = len(accessors)
        accessors.append({"componentType": FLOAT32, "count": n, "type": acc_type})
        return idx

    attributes["POSITION"] = add_stub_accessor("VEC3")
    attributes["KHR_gaussian_splatting:OPACITY"] = add_stub_accessor("SCALAR")
    attributes["KHR_gaussian_splatting:SCALE"] = add_stub_accessor("VEC3")
    attributes["KHR_gaussian_splatting:ROTATION"] = add_stub_accessor("VEC4")
    attributes["KHR_gaussian_splatting:SH_DEGREE_0_COEF_0"] = add_stub_accessor("VEC3")

    for d in range(1, sh_degree + 1):
        term_count = 2 * d + 1
        for j in range(term_count):
            attr_name = f"KHR_gaussian_splatting:SH_DEGREE_{d}_COEF_{j}"
            attributes[attr_name] = add_stub_accessor("VEC3")

    # Build glTF JSON
    gltf = {
        "asset": {"version": "2.0", "generator": "ply2gltf.py"},
        "extensionsUsed": [
            "KHR_gaussian_splatting",
            "KHR_gaussian_splatting_compression_spz_2",
        ],
        "buffers": [{"byteLength": len(spz_bytes)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(spz_bytes)}
        ],
        "accessors": accessors,
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": attributes,
                        "mode": 0,  # POINTS
                        "extensions": {
                            "KHR_gaussian_splatting": {
                                "extensions": {
                                    "KHR_gaussian_splatting_compression_spz_2": {
                                        "bufferView": 0
                                    }
                                }
                            }
                        },
                    }
                ]
            }
        ],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }

    return gltf, bin_data


# ---------- Output writers ----------


def write_glb(gltf: dict, bin_data: bytes, filepath: str) -> None:
    """Write GLB binary container."""
    json_str = json.dumps(gltf, separators=(",", ":"))
    json_bytes = json_str.encode("utf-8")
    # Pad JSON to 4-byte alignment with spaces
    json_padding = (4 - len(json_bytes) % 4) % 4
    json_bytes_padded = json_bytes + b" " * json_padding
    # Pad BIN to 4-byte alignment with null bytes
    bin_padding = (4 - len(bin_data) % 4) % 4
    bin_data_padded = bin_data + b"\x00" * bin_padding

    total_length = 12 + (8 + len(json_bytes_padded)) + (8 + len(bin_data_padded))

    with open(filepath, "wb") as f:
        # GLB header
        f.write(struct.pack("<III", 0x46546C67, 2, total_length))  # magic, version, length
        # JSON chunk
        f.write(struct.pack("<II", len(json_bytes_padded), 0x4E4F534A))  # length, 'JSON'
        f.write(json_bytes_padded)
        # BIN chunk
        f.write(struct.pack("<II", len(bin_data_padded), 0x004E4942))  # length, 'BIN\0'
        f.write(bin_data_padded)

    print(f"  Written GLB: {filepath} ({total_length / (1024*1024):.2f} MB)")


def write_gltf(gltf: dict, bin_data: bytes, filepath: str) -> None:
    """Write glTF JSON + separate .bin file."""
    bin_filename = Path(filepath).stem + ".bin"
    bin_filepath = Path(filepath).parent / bin_filename

    # Update buffer URI to point to .bin file
    gltf["buffers"][0]["uri"] = bin_filename

    with open(filepath, "w") as f:
        json.dump(gltf, f, indent=2)

    with open(bin_filepath, "wb") as f:
        f.write(bin_data)

    print(f"  Written glTF: {filepath}")
    print(f"  Written BIN:  {bin_filepath} ({len(bin_data) / (1024*1024):.2f} MB)")


# ---------- Main ----------


def main():
    parser = argparse.ArgumentParser(
        description="Convert INRIA-format 3DGS PLY to glTF 2.0 with KHR_gaussian_splatting"
    )
    parser.add_argument("--input", "-i", required=True, help="Input PLY file path")
    parser.add_argument("--output", "-o", required=True, help="Output file path (.glb or .gltf)")
    parser.add_argument(
        "--compression",
        choices=["none", "spz"],
        default="none",
        help="Compression method (default: none)",
    )
    parser.add_argument(
        "--sh-degree",
        type=int,
        default=3,
        help="Maximum SH degree to include (default: 3)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    ext = Path(args.output).suffix.lower()
    if ext not in (".glb", ".gltf"):
        print(f"Error: output must be .glb or .gltf, got: {ext}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing PLY: {args.input}")
    ply = parse_ply(args.input)

    print("Extracting splat data...")
    data = extract_splat_data(ply, args.sh_degree)

    if args.compression == "spz":
        print("Building glTF with SPZ compression...")
        gltf, bin_data = build_gltf_spz(data)
    else:
        print("Building glTF (uncompressed)...")
        gltf, bin_data = build_gltf_uncompressed(data)

    print(f"Writing output ({ext})...")
    if ext == ".glb":
        write_glb(gltf, bin_data, args.output)
    else:
        write_gltf(gltf, bin_data, args.output)

    print("Done!")


if __name__ == "__main__":
    main()
