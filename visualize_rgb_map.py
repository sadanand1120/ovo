import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import open_clip
import torch
from tqdm.auto import tqdm

CLIP_MODEL_NAME = "ViT-L-14-336-quickgelu"
CLIP_PRETRAINED = "openai"
DEFAULT_PCA_SAMPLE_SIZE = 100_000
DEFAULT_CHUNK_SIZE = 200_000
DEFAULT_POINT_SIZE = 1.0


def resolve_ply_path(input_path: str) -> Path:
    path = Path(input_path)
    return path if path.suffix == ".ply" else path / "rgb_map.ply"


def show_point_cloud(pcd: o3d.geometry.PointCloud, point_size: float) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    vis.run()
    vis.destroy_window()


def fit_pca_projection(features_mm: np.memmap, sample_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_points = features_mm.shape[0]
    sample_size = min(sample_size, n_points)
    sample_idx = np.random.default_rng(0).choice(n_points, size=sample_size, replace=False)
    sample = torch.from_numpy(np.array(features_mm[sample_idx], copy=True)).float()
    mean = sample.mean(dim=0, keepdim=True)
    centered = sample - mean
    _, _, v = torch.pca_lowrank(centered, niter=5)
    proj_v = v[:, :3]
    low_rank = centered @ proj_v
    low_rank_min = torch.quantile(low_rank, 0.01, dim=0)
    low_rank_max = torch.quantile(low_rank, 0.99, dim=0)
    return mean, proj_v, low_rank_min, low_rank_max


def apply_pca_colormap_chunked(features_mm: np.memmap, sample_size: int, chunk_size: int) -> np.ndarray:
    mean, proj_v, low_rank_min, low_rank_max = fit_pca_projection(features_mm, sample_size)
    denom = (low_rank_max - low_rank_min).clamp_min(1e-6)
    colors = np.empty((features_mm.shape[0], 3), dtype=np.float32)
    for start in tqdm(range(0, features_mm.shape[0], chunk_size), desc="PCA color", unit="chunk"):
        end = min(start + chunk_size, features_mm.shape[0])
        chunk = torch.from_numpy(np.array(features_mm[start:end], copy=True)).float()
        low_rank = (chunk - mean) @ proj_v
        colors[start:end] = ((low_rank - low_rank_min) / denom).clamp_(0.0, 1.0).numpy()
    return colors


def l2_normalize_embeddings(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def compute_similarity_scores(clip_features: torch.Tensor, pos_embed: torch.Tensor, neg_embed: torch.Tensor | None = None, softmax_temp: float = 0.1) -> torch.Tensor:
    if pos_embed.ndim == 1:
        pos_embed = pos_embed.unsqueeze(0)

    clip_features = l2_normalize_embeddings(clip_features.float())
    pos_embed = l2_normalize_embeddings(pos_embed.float())
    clip_features = clip_features.to(dtype=pos_embed.dtype)

    if neg_embed is None:
        return (clip_features @ pos_embed.T).squeeze(-1)

    neg_embed = l2_normalize_embeddings(neg_embed.float())
    text_embs = torch.cat([pos_embed, neg_embed], dim=0)
    raw_sims = clip_features @ text_embs.T
    pos_sims, neg_sims = raw_sims[..., :1], raw_sims[..., 1:]
    pos_sims = pos_sims.broadcast_to(neg_sims.shape)
    paired_sims = torch.cat([pos_sims, neg_sims], dim=-1)
    probs = (paired_sims / max(float(softmax_temp), 1e-6)).softmax(dim=-1)[..., :1]
    torch.nan_to_num_(probs, nan=0.0)
    sims, _ = probs.min(dim=-1)
    return sims


def similarity_colormap(scores: torch.Tensor) -> torch.Tensor:
    lo = torch.quantile(scores, 0.01)
    hi = torch.quantile(scores, 0.99)
    scores = ((scores - lo) / (hi - lo).clamp_min(1e-6)).clamp_(0.0, 1.0)
    return torch.stack((scores, 1.0 - (scores - 0.5).abs() * 2.0, 1.0 - scores), dim=-1)


def parse_texts(raw: str | None) -> list[str]:
    return [text.strip() for text in (raw or "").split(",") if text.strip()]


def compute_similarity_scores_chunked(
    features_mm: np.memmap,
    pos_embed: torch.Tensor,
    neg_embed: torch.Tensor | None,
    softmax_temp: float,
    device: str,
    chunk_size: int,
) -> np.ndarray:
    scores = np.empty((features_mm.shape[0],), dtype=np.float32)
    for start in tqdm(range(0, features_mm.shape[0], chunk_size), desc="Similarity", unit="chunk"):
        end = min(start + chunk_size, features_mm.shape[0])
        chunk = torch.from_numpy(np.array(features_mm[start:end], copy=True)).to(device)
        scores[start:end] = compute_similarity_scores(chunk, pos_embed, neg_embed, softmax_temp).cpu().numpy()
    return scores


@torch.inference_mode()
def encode_text_queries(device: str, positives: list[str], negatives: list[str]) -> tuple[torch.Tensor, torch.Tensor | None]:
    model = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=device)[0].eval()
    tokenize = open_clip.get_tokenizer(CLIP_MODEL_NAME)

    pos_tokens = tokenize(positives).to(device)
    pos_embed = model.encode_text(pos_tokens).float()
    pos_embed = l2_normalize_embeddings(pos_embed).mean(dim=0, keepdim=True)
    pos_embed = l2_normalize_embeddings(pos_embed)

    neg_embed = None
    if negatives:
        neg_tokens = tokenize(negatives).to(device)
        neg_embed = l2_normalize_embeddings(model.encode_text(neg_tokens).float())
    return pos_embed, neg_embed


def main(args):
    ply_path = resolve_ply_path(args.input_path)
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if args.mode == "normals":
        normals = np.asarray(pcd.normals)
        if normals.size == 0:
            raise ValueError("Point cloud has no normals.")
        pcd.colors = o3d.utility.Vector3dVector(((normals + 1.0) * 0.5).clip(0.0, 1.0))
    elif args.mode == "feat":
        feat_path = ply_path.with_name("clip_feats.npy")
        feats = np.load(feat_path, mmap_mode="r")
        colors = apply_pca_colormap_chunked(feats, args.pca_sample_size, args.chunk_size)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    elif args.mode == "feature-similarity":
        positives = parse_texts(args.positive)
        if not positives:
            raise ValueError("--positive is required for mode=feature-similarity")
        negatives = parse_texts(args.negative)
        feat_path = ply_path.with_name("clip_feats.npy")
        feats = np.load(feat_path, mmap_mode="r")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pos_embed, neg_embed = encode_text_queries(device, positives, negatives)
        scores = torch.from_numpy(
            compute_similarity_scores_chunked(feats, pos_embed, neg_embed, args.softmax_temp, device, args.chunk_size)
        )
        colors = similarity_colormap(scores).numpy()
        pcd.colors = o3d.utility.Vector3dVector(colors)
        print({"positive": positives, "negative": negatives, "softmax_temp": args.softmax_temp, "score_min": float(scores.min()), "score_max": float(scores.max())})
    print(f"Loaded {ply_path} with {len(pcd.points)} points")
    stats_path = ply_path.with_name("stats.json")
    if stats_path.exists():
        print(json.dumps(json.loads(stats_path.read_text()), indent=2))
    if not args.no_window:
        show_point_cloud(pcd, args.point_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a saved RGB pointcloud map.")
    parser.add_argument("input_path", help="Path to rgb_map.ply or its containing directory.")
    parser.add_argument("--mode", choices=["rgb", "normals", "feat", "feature-similarity"], default="rgb")
    parser.add_argument("--positive", default="")
    parser.add_argument("--negative", default="")
    parser.add_argument("--softmax_temp", type=float, default=0.1)
    parser.add_argument("--pca_sample_size", type=int, default=DEFAULT_PCA_SAMPLE_SIZE)
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--point_size", type=float, default=DEFAULT_POINT_SIZE)
    parser.add_argument("--no_window", action="store_true", help="Only load and print map info.")
    main(parser.parse_args())
