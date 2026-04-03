import argparse
import shutil
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None, desc=None):
        total = total if total is not None else len(iterable)
        for index, item in enumerate(iterable, start=1):
            if index == 1 or index % 25 == 0 or index == total:
                label = f"{desc}: " if desc else ""
                print(f"{label}{index}/{total}")
            yield item


COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {-1: "unknown", 0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode ScanNet .sens files into OVO-ready RGB-D folders.")
    parser.add_argument("--scans_root", required=True, type=Path, help="Directory containing ScanNet scene folders.")
    parser.add_argument("--output_root", required=True, type=Path, help="Directory to write decoded scene folders into.")
    parser.add_argument("--scenes", nargs="*", default=None, help="Optional scene names to decode. Defaults to all scenes under scans_root.")
    parser.add_argument("--frame_skip", type=int, default=1, help="Export every Nth frame.")
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


if __name__ == "__main__":
    main()
