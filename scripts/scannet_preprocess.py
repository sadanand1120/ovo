import os
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm


def write_labels(output_file: str, pcd_labels: np.ndarray) -> None:
    n_vtx = pcd_labels.shape[0]
    labels_list = [str(int(pcd_labels[i].item())) for i in range(n_vtx)]
    with open(output_file, "w") as f:
        f.write('\n'.join(labels_list))


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
            elif line.startswith("element vertex "):
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

def main(args):
    scannet_data_path = Path(args.data_path)
    scans_path = scannet_data_path / "scans"
    decoded_path = scannet_data_path / "data"/ "val"
    scenes = os.listdir(decoded_path)
    scenes = [scene for scene in scenes if scene[:5] == "scene"]
    out_path = decoded_path / "semantic_gt"

    os.makedirs(out_path, exist_ok=True)

    for scene in tqdm(scenes):
        mesh_path = scans_path / scene  / f"{scene}_vh_clean_2.labels.ply"
        if args.link_pcds:
            link_path = decoded_path / scene / f"{scene}_vh_clean_2.labels.ply"
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            os.symlink(mesh_path.resolve(), link_path)

        gt_labels = read_ply_vertex_labels(mesh_path)
        write_labels(out_path / f"{scene}.txt", gt_labels)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Path to directory containing scans and data folder')
    parser.add_argument('--link_pcds', action='store_true', help='If set, creates a symbolic link to gt pointclouds in data/val/scene*/.')
    args = parser.parse_args()
    main(args)
