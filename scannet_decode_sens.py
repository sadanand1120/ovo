import argparse
import os
import shutil
import struct
import zlib
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm


COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {-1: "unknown", 0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


def write_labels(output_file: Path, labels: np.ndarray) -> None:
    with open(output_file, "w") as handle:
        handle.write("\n".join(str(int(label)) for label in labels))


def read_ply_vertex_labels(mesh_path: Path) -> np.ndarray:
    scalar_types = {
        "char": "i1",
        "uchar": "u1",
        "int8": "i1",
        "uint8": "u1",
        "short": "<i2",
        "ushort": "<u2",
        "int16": "<i2",
        "uint16": "<u2",
        "int": "<i4",
        "uint": "<u4",
        "int32": "<i4",
        "uint32": "<u4",
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "float64": "<f8",
    }

    with open(mesh_path, "rb") as handle:
        if handle.readline().decode("ascii").strip() != "ply":
            raise ValueError(f"Invalid PLY file: {mesh_path}")

        vertex_count = None
        vertex_dtype = []
        in_vertex = False

        while True:
            line = handle.readline().decode("ascii").strip()
            if line.startswith("format ") and "binary_little_endian" not in line:
                raise ValueError(f"Unsupported PLY format in {mesh_path}: {line}")
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
                in_vertex = True
            elif line.startswith("element ") and not line.startswith("element vertex "):
                in_vertex = False
            elif in_vertex and line.startswith("property "):
                parts = line.split()
                if parts[1] == "list":
                    raise ValueError(f"Unsupported list property in vertex block: {line}")
                vertex_dtype.append((parts[2], scalar_types[parts[1]]))
            elif line == "end_header":
                break

        if vertex_count is None:
            raise ValueError(f"Missing vertex block in {mesh_path}")

        vertices = np.fromfile(handle, dtype=np.dtype(vertex_dtype), count=vertex_count)
    if "label" not in vertices.dtype.names:
        raise ValueError(f"PLY does not contain vertex labels: {mesh_path}")
    return np.asarray(vertices["label"])


class RGBDFrame:
    def load(self, file_handle) -> None:
        self.camera_to_world = np.asarray(
            struct.unpack("f" * 16, file_handle.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        self.timestamp_color = struct.unpack("Q", file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack("Q", file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.color_data = file_handle.read(self.color_size_bytes)
        self.depth_data = file_handle.read(self.depth_size_bytes)

    def decompress_depth(self, compression_type: str) -> bytes:
        if compression_type == "zlib_ushort":
            return zlib.decompress(self.depth_data)
        if compression_type == "raw_ushort":
            return self.depth_data
        raise ValueError(f"Unsupported depth compression: {compression_type}")

    def decompress_color(self, compression_type: str) -> np.ndarray:
        if compression_type in {"jpeg", "png"}:
            color = cv2.imdecode(np.frombuffer(self.color_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if color is None:
                raise ValueError("Failed to decode compressed color frame")
            return color
        if compression_type == "raw":
            raise ValueError("Unsupported raw color format in .sens file")
        raise ValueError(f"Unsupported color compression: {compression_type}")


class SensorData:
    def __init__(self, filename: Path):
        self.version = 4
        self.load(filename)

    def load(self, filename: Path) -> None:
        with open(filename, "rb") as handle:
            version = struct.unpack("I", handle.read(4))[0]
            if version != self.version:
                raise ValueError(f"Unsupported .sens version {version}, expected {self.version}")

            strlen = struct.unpack("Q", handle.read(8))[0]
            self.sensor_name = handle.read(strlen).decode("utf-8", errors="ignore")
            self.intrinsic_color = np.asarray(
                struct.unpack("f" * 16, handle.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_color = np.asarray(
                struct.unpack("f" * 16, handle.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.intrinsic_depth = np.asarray(
                struct.unpack("f" * 16, handle.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_depth = np.asarray(
                struct.unpack("f" * 16, handle.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack("i", handle.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack("i", handle.read(4))[0]]
            self.color_width = struct.unpack("I", handle.read(4))[0]
            self.color_height = struct.unpack("I", handle.read(4))[0]
            self.depth_width = struct.unpack("I", handle.read(4))[0]
            self.depth_height = struct.unpack("I", handle.read(4))[0]
            self.depth_shift = struct.unpack("f", handle.read(4))[0]
            num_frames = struct.unpack("Q", handle.read(8))[0]

            self.frames = []
            for _ in range(num_frames):
                frame = RGBDFrame()
                frame.load(handle)
                self.frames.append(frame)


def save_matrix(matrix: np.ndarray, path: Path) -> None:
    with open(path, "w") as handle:
        for row in matrix:
            np.savetxt(handle, row[np.newaxis], fmt="%f")


def ensure_free_space(target: Path, min_free_gb: float) -> None:
    free_bytes = shutil.disk_usage(target).free
    free_gb = free_bytes / (1024 ** 3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Stopping decode: free space on {target} dropped to {free_gb:.1f} GB, below {min_free_gb:.1f} GB"
        )


def decode_scene(scene_dir: Path, output_scene_dir: Path, frame_skip: int, min_free_gb: float) -> None:
    sens_files = list(scene_dir.glob("*.sens"))
    if len(sens_files) != 1:
        raise ValueError(f"Expected exactly one .sens file in {scene_dir}, found {len(sens_files)}")

    output_scene_dir.mkdir(parents=True, exist_ok=True)
    color_dir = output_scene_dir / "color"
    depth_dir = output_scene_dir / "depth"
    pose_dir = output_scene_dir / "pose"
    intrinsic_dir = output_scene_dir / "intrinsic"
    color_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)
    pose_dir.mkdir(exist_ok=True)
    intrinsic_dir.mkdir(exist_ok=True)

    sensor_data = SensorData(sens_files[0])
    save_matrix(sensor_data.intrinsic_color, intrinsic_dir / "intrinsic_color.txt")
    save_matrix(sensor_data.extrinsic_color, intrinsic_dir / "extrinsic_color.txt")
    save_matrix(sensor_data.intrinsic_depth, intrinsic_dir / "intrinsic_depth.txt")
    save_matrix(sensor_data.extrinsic_depth, intrinsic_dir / "extrinsic_depth.txt")

    frame_indices = range(0, len(sensor_data.frames), frame_skip)
    total_frames = len(range(0, len(sensor_data.frames), frame_skip))
    print(f"Decoding {scene_dir.name}: {total_frames} frames")

    for export_idx, frame_idx in enumerate(tqdm(frame_indices, total=total_frames, desc=scene_dir.name)):
        if export_idx % 25 == 0:
            ensure_free_space(output_scene_dir, min_free_gb)

        frame = sensor_data.frames[frame_idx]
        color = frame.decompress_color(sensor_data.color_compression_type)
        depth = np.frombuffer(
            frame.decompress_depth(sensor_data.depth_compression_type), dtype=np.uint16
        ).reshape(sensor_data.depth_height, sensor_data.depth_width)

        if not cv2.imwrite(str(color_dir / f"{frame_idx}.jpg"), color):
            raise RuntimeError(f"Failed to write color frame {frame_idx} for {scene_dir.name}")
        if not cv2.imwrite(str(depth_dir / f"{frame_idx}.png"), depth):
            raise RuntimeError(f"Failed to write depth frame {frame_idx} for {scene_dir.name}")
        save_matrix(frame.camera_to_world, pose_dir / f"{frame_idx}.txt")

    ensure_free_space(output_scene_dir, min_free_gb)


def write_semantic_gt(scans_root: Path, output_root: Path, scene_name: str, link_pcds: bool) -> None:
    mesh_path = scans_root / scene_name / f"{scene_name}_vh_clean_2.labels.ply"
    if not mesh_path.exists():
        raise FileNotFoundError(f"Missing labeled mesh: {mesh_path}")

    semantic_gt_dir = output_root / "semantic_gt"
    semantic_gt_dir.mkdir(parents=True, exist_ok=True)
    write_labels(semantic_gt_dir / f"{scene_name}.txt", read_ply_vertex_labels(mesh_path))

    if link_pcds:
        link_path = output_root / scene_name / f"{scene_name}_vh_clean_2.labels.ply"
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        os.symlink(mesh_path.resolve(), link_path)


def write_vertex_normals_gt(scans_root: Path, output_root: Path, scene_name: str) -> None:
    mesh_path = scans_root / scene_name / f"{scene_name}_vh_clean_2.labels.ply"
    if not mesh_path.exists():
        raise FileNotFoundError(f"Missing labeled mesh: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    np.save(output_root / scene_name / f"{scene_name}_vh_clean_2.vertex_normals.npy", normals)


def extract_filtered_2d_gt(scans_root: Path, output_root: Path, scene_name: str, min_free_gb: float) -> None:
    scene_root = scans_root / scene_name
    output_scene_dir = output_root / scene_name
    zip_names = [
        f"{scene_name}_2d-label-filt.zip",
        f"{scene_name}_2d-instance-filt.zip",
    ]
    for zip_name in zip_names:
        zip_path = scene_root / zip_name
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing filtered 2D GT zip: {zip_path}")
        print(f"Extracting {zip_name} ...")
        with ZipFile(zip_path) as handle:
            members = [member for member in handle.namelist() if member.lower().endswith(".png")]
            for member in tqdm(members, desc=zip_name, unit="file"):
                ensure_free_space(output_scene_dir, min_free_gb)
                handle.extract(member, path=output_scene_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode ScanNet .sens files into OVO-ready RGB-D folders.")
    parser.add_argument("--scans_root", required=True, type=Path, help="Directory containing ScanNet scene folders.")
    parser.add_argument("--output_root", required=True, type=Path, help="Directory to write decoded scene folders into.")
    parser.add_argument("--scenes", nargs="*", default=None, help="Optional scene names to decode. Defaults to all scenes under scans_root.")
    parser.add_argument("--frame_skip", type=int, default=1, help="Export every Nth frame.")
    parser.add_argument("--link_pcds", action="store_true", help="Link each ground-truth mesh into the decoded scene folder.")
    parser.add_argument("--write_semantic_gt", action="store_true", help="Write semantic_gt/<scene>.txt from each labeled ScanNet mesh.")
    parser.add_argument("--extract_2d_gt_filt", action="store_true", help="Extract *_2d-label-filt.zip and *_2d-instance-filt.zip into each decoded scene folder.")
    parser.add_argument(
        "--min_free_gb",
        type=float,
        default=100.0,
        help="Abort before writing more data if free space drops below this threshold.",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    ensure_free_space(args.output_root, args.min_free_gb)

    if args.scenes:
        scenes = [args.scans_root / scene for scene in args.scenes]
    else:
        scenes = sorted(path for path in args.scans_root.iterdir() if path.is_dir())

    for scene_dir in scenes:
        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
        decode_scene(scene_dir, args.output_root / scene_dir.name, args.frame_skip, args.min_free_gb)
        write_vertex_normals_gt(args.scans_root, args.output_root, scene_dir.name)
        if args.write_semantic_gt or args.link_pcds:
            write_semantic_gt(args.scans_root, args.output_root, scene_dir.name, args.link_pcds)
        if args.extract_2d_gt_filt:
            extract_filtered_2d_gt(args.scans_root, args.output_root, scene_dir.name, args.min_free_gb)


if __name__ == "__main__":
    main()
