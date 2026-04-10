from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import open3d as o3d
import open_clip
import torch
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from orientation_utils import axis_vectors_to_rotation_and_scale, load_instance_orientations, orientation_path_for_map_dir


DEFAULT_CLIP_MODEL_NAME = "ViT-L-14-336-quickgelu"
DEFAULT_CLIP_PRETRAINED = "openai"
DEFAULT_CLIP_FEATURE_FILE = "clip_feats.npy"
DEFAULT_MIN_COMPONENT_SIZE = 2000
DEFAULT_CLIP_CHUNK_SIZE = 200_000
DEFAULT_MAX_POINTS = 5_000_000
DEFAULT_POINT_SCORE_THRESHOLD = 0.24
DEFAULT_MIN_POINT_FRACTION = 0.05
DEFAULT_MIN_POINT_COUNT = 50
DEFAULT_SOFTMAX_TEMP = 0.01

WholeObjOrPoints = Literal["wholeobj", "points"]
RefReduce = Literal["any", "all"]
Side = Literal["either", "+y", "-y"]
SemanticScoring = Literal["cosine", "softmax"]


def resolve_ply_path(input_path: str | Path) -> Path:
    path = Path(input_path)
    return path if path.suffix == ".ply" else path / "rgb_map.ply"


def downsample_cache_tag(min_component_size: int, max_points: int, seed: int) -> str:
    return f"dsl_mcs{int(min_component_size)}_maxpts{int(max_points)}_seed{int(seed)}"


def downsample_cache_paths(map_dir: Path, min_component_size: int, max_points: int, seed: int) -> dict[str, Path]:
    tag = downsample_cache_tag(min_component_size, max_points, seed)
    return {
        "ply": map_dir / f"rgb_map.{tag}.ply",
        "clip": map_dir / f"clip_feats.{tag}.npy",
        "labels": map_dir / f"instance_labels.{tag}.npy",
        "meta": map_dir / f"downsample.{tag}.json",
    }


def resolve_instance_labels(map_dir: Path, n_points: int, min_component_size: int) -> np.ndarray:
    label_path = map_dir / "instance_labels.npy"
    if not label_path.exists():
        raise FileNotFoundError(f"Missing instance labels at {label_path}")
    labels = np.load(label_path)
    if labels.shape[0] != n_points:
        raise ValueError(f"Instance label count mismatch: expected {n_points}, found {labels.shape[0]}")
    valid = labels >= 0
    if valid.any() and min_component_size > 1:
        uniq, counts = np.unique(labels[valid], return_counts=True)
        keep = uniq[counts >= int(min_component_size)]
        labels = labels.copy()
        labels[~np.isin(labels, keep)] = -1
        valid = labels >= 0
    if valid.any():
        _, relabeled = np.unique(labels[valid], return_inverse=True)
        labels[valid] = relabeled.astype(np.int32, copy=False)
    return labels.astype(np.int32, copy=False)


def select_downsample_indices(labels: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    n_points = int(labels.shape[0])
    if n_points <= max_points:
        return np.arange(n_points, dtype=np.int64)

    valid = labels >= 0
    protected = np.empty((0,), dtype=np.int64)
    if valid.any():
        valid_indices = np.flatnonzero(valid)
        _, first_positions = np.unique(labels[valid], return_index=True)
        protected = valid_indices[first_positions].astype(np.int64, copy=False)
    if protected.shape[0] > max_points:
        raise ValueError(
            f"Cannot downsample to {max_points} points while preserving {protected.shape[0]} labeled instances"
        )

    keep_count = min(max_points, n_points)
    extra_count = keep_count - protected.shape[0]
    if extra_count <= 0:
        return np.sort(protected)

    protected_mask = np.zeros((n_points,), dtype=bool)
    protected_mask[protected] = True
    remaining = np.flatnonzero(~protected_mask)
    if remaining.shape[0] <= extra_count:
        extra = remaining
    else:
        rng = np.random.default_rng(seed)
        extra = rng.choice(remaining, size=extra_count, replace=False)
    return np.sort(np.concatenate([protected, extra.astype(np.int64, copy=False)], axis=0))


def write_downsample_cache(
    cache_paths: dict[str, Path],
    points: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray,
    clip_features: np.ndarray,
    instance_labels: np.ndarray,
    *,
    source_point_count: int,
    min_component_size: int,
    max_points: int,
    seed: int,
) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    pcd.colors = o3d.utility.Vector3dVector((colors / 255.0).astype(np.float64, copy=False))
    pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64, copy=False))
    if not o3d.io.write_point_cloud(str(cache_paths["ply"]), pcd):
        raise IOError(f"Failed to write downsampled point cloud cache to {cache_paths['ply']}")
    np.save(cache_paths["clip"], clip_features)
    np.save(cache_paths["labels"], instance_labels)
    cache_paths["meta"].write_text(
        json.dumps(
            {
                "source_point_count": int(source_point_count),
                "cached_point_count": int(points.shape[0]),
                "min_component_size": int(min_component_size),
                "max_points": int(max_points),
                "downsample_seed": int(seed),
            },
            indent=2,
        )
    )


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    out = np.zeros_like(x, dtype=np.float32)
    valid = norms[:, 0] > 1e-8
    out[valid] = x[valid] / norms[valid]
    return out


def l2_normalize_tensor(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def _as_tuple_str(values: Sequence[str] | None, fallback: tuple[str, ...] = ()) -> tuple[str, ...]:
    if values is None:
        return fallback
    if isinstance(values, str):
        return (values,)
    return tuple(str(v) for v in values)


@dataclass(frozen=True)
class SemanticSpec:
    label: str
    positive_texts: tuple[str, ...] | None = None
    negative_texts: tuple[str, ...] = ()
    templates: tuple[str, ...] = ("{}",)
    scoring: SemanticScoring | None = None
    softmax_temp: float = DEFAULT_SOFTMAX_TEMP
    point_score_threshold: float = DEFAULT_POINT_SCORE_THRESHOLD
    min_point_fraction: float = DEFAULT_MIN_POINT_FRACTION
    min_point_count: int = DEFAULT_MIN_POINT_COUNT

    def __post_init__(self) -> None:
        object.__setattr__(self, "positive_texts", _as_tuple_str(self.positive_texts, (self.label,)))
        object.__setattr__(self, "negative_texts", _as_tuple_str(self.negative_texts))
        object.__setattr__(self, "templates", _as_tuple_str(self.templates, ("{}",)))
        if self.scoring is None:
            scoring = "softmax" if self.negative_texts else "cosine"
            object.__setattr__(self, "scoring", scoring)
        if self.scoring not in ("cosine", "softmax"):
            raise ValueError(f"Unsupported semantic scoring mode: {self.scoring}")
        if self.softmax_temp <= 0:
            raise ValueError("softmax_temp must be > 0")
        if not (0.0 <= self.min_point_fraction <= 1.0):
            raise ValueError("min_point_fraction must be in [0, 1]")
        if self.min_point_count < 0:
            raise ValueError("min_point_count must be >= 0")


@dataclass
class VirtualObject:
    scene: "FeatureFieldScene" = field(repr=False)
    point_indices: np.ndarray
    centroid: np.ndarray
    rotation: np.ndarray | None = None
    instance_ids: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int32))

    def __len__(self) -> int:
        return int(self.point_indices.shape[0])

    def __bool__(self) -> bool:
        return len(self) > 0


@dataclass
class PointSelection:
    scene: "FeatureFieldScene" = field(repr=False)
    point_indices: np.ndarray
    instance_ids: np.ndarray
    scores: np.ndarray
    default_rotation: np.ndarray | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return int(self.point_indices.shape[0])

    def __bool__(self) -> bool:
        return len(self) > 0

    def objectify(self) -> VirtualObject:
        if not self:
            raise ValueError("Cannot objectify an empty point selection")
        points = self.scene.points[self.point_indices]
        # This centroid is only the local-frame origin for the synthetic object.
        # Geometric relations still use the full selected support points.
        centroid = points.mean(axis=0).astype(np.float32, copy=False)
        rotation = None if self.default_rotation is None else self.default_rotation.copy()
        instance_ids = np.unique(self.instance_ids[self.instance_ids >= 0]).astype(np.int32, copy=False)
        return VirtualObject(
            scene=self.scene,
            point_indices=self.point_indices.copy(),
            centroid=centroid,
            rotation=rotation,
            instance_ids=instance_ids,
        )


@dataclass
class ObjSelection:
    scene: "FeatureFieldScene" = field(repr=False)
    instance_ids: np.ndarray
    scores: np.ndarray
    matched_point_fraction: np.ndarray
    matched_point_count: np.ndarray

    def __len__(self) -> int:
        return int(self.instance_ids.shape[0])

    def __bool__(self) -> bool:
        return len(self) > 0

    def __contains__(self, item: object) -> bool:
        ids = set(self.instance_ids.tolist())
        if isinstance(item, (int, np.integer)):
            return int(item) in ids
        if isinstance(item, ObjSelection):
            return len(item) > 0 and set(item.instance_ids.tolist()).issubset(ids)
        return False

    def to_points(self) -> PointSelection:
        return self.scene._instances_to_point_selection(self.instance_ids, self.scores)


@dataclass
class _ReferenceGeometry:
    point_indices: np.ndarray
    centroid: np.ndarray
    rotation: np.ndarray | None
    bounds_min: np.ndarray | None
    bounds_max: np.ndarray | None
    exclude_instance_ids: np.ndarray
    tree: cKDTree


class FeatureFieldScene:
    def __init__(
        self,
        run_dir: str | Path,
        *,
        min_component_size: int = DEFAULT_MIN_COMPONENT_SIZE,
        clip_device: str = "cuda",
        clip_chunk_size: int = DEFAULT_CLIP_CHUNK_SIZE,
        max_points: int | None = None,
        downsample_seed: int = 0,
    ) -> None:
        self.run_dir = resolve_ply_path(run_dir).parent
        self.ply_path = self.run_dir / "rgb_map.ply"
        self.min_component_size = int(min_component_size)
        self.clip_chunk_size = int(clip_chunk_size)
        requested_device = str(clip_device)
        self.clip_device = requested_device if requested_device == "cpu" or torch.cuda.is_available() else "cpu"
        cache_paths = None if max_points is None else downsample_cache_paths(
            self.run_dir,
            self.min_component_size,
            int(max_points),
            int(downsample_seed),
        )

        use_cache = cache_paths is not None and all(path.exists() for key, path in cache_paths.items() if key != "meta")
        if use_cache:
            print(f"[dsl] loading cached downsample from {cache_paths['ply'].name}", flush=True)
            self.ply_path = cache_paths["ply"]
            pcd = o3d.io.read_point_cloud(str(self.ply_path))
        else:
            pcd = o3d.io.read_point_cloud(str(self.ply_path))
        self.points = np.asarray(pcd.points, dtype=np.float32)
        self.colors = np.rint(np.asarray(pcd.colors, dtype=np.float32) * 255.0).clip(0.0, 255.0).astype(np.float32)
        self.normals = l2_normalize_rows(np.asarray(pcd.normals, dtype=np.float32))
        if self.points.shape[0] == 0:
            raise ValueError(f"Empty point cloud: {self.ply_path}")
        if self.normals.shape[0] != self.points.shape[0]:
            raise ValueError(f"{self.ply_path} has no stored normals")

        clip_path = cache_paths["clip"] if use_cache else self.run_dir / DEFAULT_CLIP_FEATURE_FILE
        self.clip_features = np.load(clip_path, mmap_mode="r")
        if self.clip_features.shape[0] != self.points.shape[0]:
            raise ValueError("CLIP feature count does not match point count")

        if use_cache:
            self.instance_labels = np.load(cache_paths["labels"]).astype(np.int32, copy=False)
            if self.instance_labels.shape[0] != self.points.shape[0]:
                raise ValueError("Cached instance label count does not match cached point count")
        else:
            self.instance_labels = resolve_instance_labels(self.run_dir, self.points.shape[0], self.min_component_size)
            if max_points is not None and self.points.shape[0] > int(max_points):
                source_point_count = int(self.points.shape[0])
                keep = select_downsample_indices(self.instance_labels, int(max_points), int(downsample_seed))
                print(
                    f"[dsl] downsampling scene from {self.points.shape[0]} to {keep.shape[0]} points",
                    flush=True,
                )
                self.points = self.points[keep]
                self.colors = self.colors[keep]
                self.normals = self.normals[keep]
                self.clip_features = np.array(self.clip_features[keep], copy=True)
                self.instance_labels = self.instance_labels[keep]
                assert cache_paths is not None
                print(f"[dsl] writing downsample cache to {cache_paths['ply'].name}", flush=True)
                write_downsample_cache(
                    cache_paths,
                    self.points,
                    self.colors,
                    self.normals,
                    self.clip_features,
                    self.instance_labels,
                    source_point_count=source_point_count,
                    min_component_size=self.min_component_size,
                    max_points=int(max_points),
                    seed=int(downsample_seed),
                )
        self.n_instances = int(self.instance_labels.max()) + 1 if (self.instance_labels >= 0).any() else 0
        self.instance_ids = np.arange(self.n_instances, dtype=np.int32)
        self.instance_point_indices = [np.flatnonzero(self.instance_labels == inst_id).astype(np.int64, copy=False) for inst_id in self.instance_ids.tolist()]
        self.instance_point_count = np.asarray([idx.shape[0] for idx in self.instance_point_indices], dtype=np.int32)
        # All DSL geometry is anchored on the current point support, not on the
        # centroid stored in instance_orientations.json. That file centroid is
        # visualization-only.
        self.instance_centroids = np.asarray(
            [self.points[idx].mean(axis=0) for idx in self.instance_point_indices],
            dtype=np.float32,
        ) if self.n_instances > 0 else np.zeros((0, 3), dtype=np.float32)

        self.instance_rotations: list[np.ndarray | None] = [None] * self.n_instances
        self.instance_bounds_min = np.zeros((self.n_instances, 3), dtype=np.float32)
        self.instance_bounds_max = np.zeros((self.n_instances, 3), dtype=np.float32)
        self._load_orientations()

        stats = {}
        stats_path = self.run_dir / "stats.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
        self.clip_model_name = str(stats.get("clip_model_name", DEFAULT_CLIP_MODEL_NAME))
        self.clip_pretrained = str(stats.get("clip_pretrained", DEFAULT_CLIP_PRETRAINED))

        self._clip_model = None
        self._clip_tokenizer = None
        self._text_embed_cache: dict[tuple[tuple[str, ...], tuple[str, ...]], torch.Tensor] = {}
        self._semantic_score_cache: dict[SemanticSpec, np.ndarray] = {}
        self._semantic_points_cache: dict[SemanticSpec, PointSelection] = {}
        self._color_score_cache: dict[tuple[tuple[float, float, float], float], tuple[np.ndarray, np.ndarray]] = {}

    @classmethod
    def load(
        cls,
        run_dir: str | Path,
        *,
        min_component_size: int = DEFAULT_MIN_COMPONENT_SIZE,
        clip_device: str = "cuda",
        clip_chunk_size: int = DEFAULT_CLIP_CHUNK_SIZE,
        max_points: int | None = None,
        downsample_seed: int = 0,
    ) -> "FeatureFieldScene":
        return cls(
            run_dir,
            min_component_size=min_component_size,
            clip_device=clip_device,
            clip_chunk_size=clip_chunk_size,
            max_points=max_points,
            downsample_seed=downsample_seed,
        )

    def _load_orientations(self) -> None:
        orientation_path = orientation_path_for_map_dir(self.run_dir)
        if not orientation_path.exists():
            return
        orientation_data = load_instance_orientations(orientation_path)
        file_min_component_size = int(orientation_data.get("min_component_size", self.min_component_size))
        if file_min_component_size != self.min_component_size:
            raise ValueError(
                f"Orientation file min_component_size={file_min_component_size} does not match requested {self.min_component_size}"
            )
        for entry in orientation_data["instances"]:
            inst_id = int(entry["instance_id"])
            if inst_id < 0 or inst_id >= self.n_instances:
                continue
            # Intentionally ignore entry["centroid"] here. The annotation-file
            # centroid is only for orientation visualization; relation geometry
            # must stay tied to the current instance point support.
            rotation, _ = axis_vectors_to_rotation_and_scale(entry["axes"])
            self.instance_rotations[inst_id] = rotation.astype(np.float32, copy=False)
        for inst_id in self.instance_ids.tolist():
            rotation = self.instance_rotations[inst_id]
            if rotation is None:
                continue
            idx = self.instance_point_indices[inst_id]
            local = (self.points[idx] - self.instance_centroids[inst_id][None, :]) @ rotation
            self.instance_bounds_min[inst_id] = local.min(axis=0)
            self.instance_bounds_max[inst_id] = local.max(axis=0)

    def _ensure_clip_model(self) -> None:
        if self._clip_model is None:
            print(
                f"[dsl] loading CLIP model {self.clip_model_name} ({self.clip_pretrained}) on {self.clip_device}",
                flush=True,
            )
            self._clip_model = open_clip.create_model_and_transforms(
                self.clip_model_name,
                pretrained=self.clip_pretrained,
                device=self.clip_device,
            )[0].eval()
            self._clip_tokenizer = open_clip.get_tokenizer(self.clip_model_name)
            print("[dsl] CLIP model ready", flush=True)

    @torch.inference_mode()
    def _encode_texts(self, texts: tuple[str, ...], templates: tuple[str, ...]) -> torch.Tensor:
        key = (texts, templates)
        cached = self._text_embed_cache.get(key)
        if cached is not None:
            return cached
        self._ensure_clip_model()
        embeds = []
        assert self._clip_model is not None
        assert self._clip_tokenizer is not None
        for text in texts:
            prompts = [template.format(text) for template in templates]
            tokens = self._clip_tokenizer(prompts).to(self.clip_device)
            text_embed = self._clip_model.encode_text(tokens).float()
            text_embed = l2_normalize_tensor(text_embed).mean(dim=0, keepdim=True)
            embeds.append(l2_normalize_tensor(text_embed)[0])
        out = torch.stack(embeds, dim=0)
        self._text_embed_cache[key] = out
        return out

    def _semantic_spec(self, query: str | SemanticSpec) -> SemanticSpec:
        return query if isinstance(query, SemanticSpec) else SemanticSpec(label=str(query))

    def _semantic_query_embeds(self, spec: SemanticSpec) -> tuple[torch.Tensor, torch.Tensor | None]:
        pos_embeds = self._encode_texts(spec.positive_texts, spec.templates)
        pos_embed = l2_normalize_tensor(pos_embeds.mean(dim=0, keepdim=True))
        neg_embed = self._encode_texts(spec.negative_texts, spec.templates) if spec.negative_texts else None
        return pos_embed, neg_embed

    def _semantic_chunk_scores(
        self,
        chunk: torch.Tensor,
        spec: SemanticSpec,
        pos_embed: torch.Tensor,
        neg_embed: torch.Tensor | None,
    ) -> torch.Tensor:
        if spec.scoring == "softmax":
            if neg_embed is None or neg_embed.shape[0] == 0:
                raise ValueError("softmax scoring requires negative_texts")
            text_embs = torch.cat([pos_embed, neg_embed], dim=0)
            raw_sims = chunk @ text_embs.T
            pos_sims, neg_sims = raw_sims[..., :1], raw_sims[..., 1:]
            pos_sims = pos_sims.broadcast_to(neg_sims.shape)
            paired = torch.cat([pos_sims, neg_sims], dim=-1)
            probs = (paired / float(spec.softmax_temp)).softmax(dim=-1)[..., :1]
            torch.nan_to_num_(probs, nan=0.0)
            return probs.min(dim=-1).values
        return (chunk @ pos_embed.T).squeeze(-1)

    @torch.inference_mode()
    def prefetch_semantic_scores(self, *queries: str | SemanticSpec) -> None:
        pending: list[SemanticSpec] = []
        seen: set[SemanticSpec] = set()
        for query in queries:
            spec = self._semantic_spec(query)
            if spec in seen or spec in self._semantic_score_cache:
                continue
            pending.append(spec)
            seen.add(spec)
        if not pending:
            return

        print(
            "[dsl] prefetching semantic queries: " + ", ".join(f"'{spec.label}'" for spec in pending),
            flush=True,
        )
        device = self.clip_device
        query_data = []
        for spec in pending:
            pos_embed, neg_embed = self._semantic_query_embeds(spec)
            pos_compute = pos_embed.to(device)
            neg_compute = None if neg_embed is None else neg_embed.to(device)
            if device == "cuda":
                pos_compute = pos_compute.to(dtype=torch.float16)
                if neg_compute is not None:
                    neg_compute = neg_compute.to(dtype=torch.float16)
            query_data.append((spec, pos_compute, neg_compute))

        score_buffers = {spec: np.empty((self.points.shape[0],), dtype=np.float32) for spec in pending}
        for start in tqdm(
            range(0, self.points.shape[0], self.clip_chunk_size),
            desc="Semantic batch",
            unit="chunk",
        ):
            end = min(start + self.clip_chunk_size, self.points.shape[0])
            chunk = torch.from_numpy(np.array(self.clip_features[start:end], copy=True)).float()
            chunk = torch.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0).to(device)
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    chunk = l2_normalize_tensor(chunk)
                    for spec, pos_compute, neg_compute in query_data:
                        score = self._semantic_chunk_scores(chunk, spec, pos_compute, neg_compute)
                        score_buffers[spec][start:end] = score.float().cpu().numpy().astype(np.float32, copy=False)
            else:
                chunk = l2_normalize_tensor(chunk)
                for spec, pos_compute, neg_compute in query_data:
                    score = self._semantic_chunk_scores(chunk, spec, pos_compute, neg_compute)
                    score_buffers[spec][start:end] = score.float().cpu().numpy().astype(np.float32, copy=False)

        for spec in pending:
            self._semantic_score_cache[spec] = score_buffers[spec]
        print("[dsl] finished semantic prefetch", flush=True)

    @torch.inference_mode()
    def _semantic_point_scores(self, query: str | SemanticSpec) -> np.ndarray:
        spec = self._semantic_spec(query)
        cached = self._semantic_score_cache.get(spec)
        if cached is not None:
            return cached
        print(f"[dsl] cache miss for semantic query '{spec.label}'", flush=True)
        self.prefetch_semantic_scores(spec)
        return self._semantic_score_cache[spec]

    def _make_point_selection(
        self,
        point_indices: np.ndarray,
        scores: np.ndarray,
        default_rotation: np.ndarray | None = None,
    ) -> PointSelection:
        point_indices = np.asarray(point_indices, dtype=np.int64)
        scores = np.asarray(scores, dtype=np.float32)
        if point_indices.shape[0] != scores.shape[0]:
            raise ValueError("point_indices and scores must have the same length")
        if point_indices.size == 0:
            return PointSelection(
                scene=self,
                point_indices=np.empty((0,), dtype=np.int64),
                instance_ids=np.empty((0,), dtype=np.int32),
                scores=np.empty((0,), dtype=np.float32),
                default_rotation=None if default_rotation is None else default_rotation.copy(),
            )
        order = np.argsort(point_indices, kind="stable")
        point_indices = point_indices[order]
        scores = scores[order]
        unique, starts = np.unique(point_indices, return_index=True)
        if unique.shape[0] != point_indices.shape[0]:
            max_scores = np.maximum.reduceat(scores, starts)
            point_indices = unique
            scores = max_scores.astype(np.float32, copy=False)
        instance_ids = self.instance_labels[point_indices]
        return PointSelection(
            scene=self,
            point_indices=point_indices,
            instance_ids=instance_ids.astype(np.int32, copy=False),
            scores=scores.astype(np.float32, copy=False),
            default_rotation=None if default_rotation is None else default_rotation.astype(np.float32, copy=False),
        )

    def _instances_to_point_selection(self, instance_ids: np.ndarray, scores: np.ndarray | None = None) -> PointSelection:
        instance_ids = np.asarray(instance_ids, dtype=np.int32)
        if instance_ids.size == 0:
            return self._make_point_selection(np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32))
        point_chunks = [self.instance_point_indices[int(inst_id)] for inst_id in instance_ids.tolist()]
        point_indices = np.concatenate(point_chunks, axis=0)
        if scores is None:
            point_scores = np.ones((point_indices.shape[0],), dtype=np.float32)
        else:
            scores = np.asarray(scores, dtype=np.float32)
            point_scores = np.concatenate(
                [np.full((self.instance_point_indices[int(inst_id)].shape[0],), float(score), dtype=np.float32) for inst_id, score in zip(instance_ids.tolist(), scores.tolist())],
                axis=0,
            )
        default_rotation = None
        if instance_ids.size == 1:
            default_rotation = self.instance_rotations[int(instance_ids[0])]
        return self._make_point_selection(point_indices, point_scores, default_rotation)

    def _lift_points_to_objects(
        self,
        points: PointSelection,
        *,
        min_point_fraction: float,
        min_point_count: int,
    ) -> ObjSelection:
        valid = points.instance_ids >= 0
        if not valid.any():
            return ObjSelection(
                scene=self,
                instance_ids=np.empty((0,), dtype=np.int32),
                scores=np.empty((0,), dtype=np.float32),
                matched_point_fraction=np.empty((0,), dtype=np.float32),
                matched_point_count=np.empty((0,), dtype=np.int32),
            )
        instance_ids = points.instance_ids[valid]
        counts = np.bincount(instance_ids, minlength=self.n_instances).astype(np.int32, copy=False)
        score_sum = np.bincount(instance_ids, weights=points.scores[valid], minlength=self.n_instances).astype(np.float32, copy=False)
        fraction = counts.astype(np.float32) / np.maximum(self.instance_point_count.astype(np.float32), 1.0)
        keep = (counts >= int(min_point_count)) & (fraction >= float(min_point_fraction))
        selected = np.flatnonzero(keep).astype(np.int32, copy=False)
        if selected.size == 0:
            return ObjSelection(
                scene=self,
                instance_ids=np.empty((0,), dtype=np.int32),
                scores=np.empty((0,), dtype=np.float32),
                matched_point_fraction=np.empty((0,), dtype=np.float32),
                matched_point_count=np.empty((0,), dtype=np.int32),
            )
        scores = (score_sum[selected] / counts[selected].clip(min=1)).astype(np.float32, copy=False)
        order = np.argsort(-scores, kind="stable")
        return ObjSelection(
            scene=self,
            instance_ids=selected[order],
            scores=scores[order],
            matched_point_fraction=fraction[selected][order].astype(np.float32, copy=False),
            matched_point_count=counts[selected][order].astype(np.int32, copy=False),
        )

    def _semantic_points(self, query: str | SemanticSpec) -> PointSelection:
        spec = self._semantic_spec(query)
        cached = self._semantic_points_cache.get(spec)
        if cached is not None:
            return cached
        scores = self._semantic_point_scores(spec)
        keep = scores >= float(spec.point_score_threshold)
        selection = self._make_point_selection(np.flatnonzero(keep), scores[keep])
        self._semantic_points_cache[spec] = selection
        return selection

    def _color_points(self, rgb: Sequence[float], *, rgb_l2_threshold: float) -> PointSelection:
        rgb_key = tuple(float(x) for x in rgb)
        cache_key = (rgb_key, float(rgb_l2_threshold))
        cached = self._color_score_cache.get(cache_key)
        if cached is None:
            rgb_query = np.asarray(rgb_key, dtype=np.float32).reshape(1, 3)
            dists = np.linalg.norm(self.colors - rgb_query, axis=1)
            keep = dists <= float(rgb_l2_threshold)
            score = 1.0 - (dists[keep] / max(float(rgb_l2_threshold), 1e-8))
            cached = (np.flatnonzero(keep).astype(np.int64, copy=False), score.astype(np.float32, copy=False))
            self._color_score_cache[cache_key] = cached
        return self._make_point_selection(cached[0], cached[1])

    def _reference_geometries(self, obj_ref: int | ObjSelection | VirtualObject) -> list[_ReferenceGeometry]:
        refs: list[_ReferenceGeometry] = []
        if isinstance(obj_ref, (int, np.integer)):
            obj_ref = ObjSelection(
                scene=self,
                instance_ids=np.asarray([int(obj_ref)], dtype=np.int32),
                scores=np.asarray([1.0], dtype=np.float32),
                matched_point_fraction=np.asarray([1.0], dtype=np.float32),
                matched_point_count=np.asarray([self.instance_point_count[int(obj_ref)]], dtype=np.int32),
            )

        if isinstance(obj_ref, ObjSelection):
            for inst_id in obj_ref.instance_ids.tolist():
                point_indices = self.instance_point_indices[int(inst_id)]
                refs.append(
                    _ReferenceGeometry(
                        point_indices=point_indices,
                        centroid=self.instance_centroids[int(inst_id)],
                        rotation=self.instance_rotations[int(inst_id)],
                        bounds_min=self.instance_bounds_min[int(inst_id)] if self.instance_rotations[int(inst_id)] is not None else None,
                        bounds_max=self.instance_bounds_max[int(inst_id)] if self.instance_rotations[int(inst_id)] is not None else None,
                        exclude_instance_ids=np.asarray([int(inst_id)], dtype=np.int32),
                        tree=cKDTree(self.points[point_indices]),
                    )
                )
            return refs

        if isinstance(obj_ref, VirtualObject):
            point_indices = np.asarray(obj_ref.point_indices, dtype=np.int64)
            bounds_min = bounds_max = None
            if obj_ref.rotation is not None:
                local = (self.points[point_indices] - obj_ref.centroid[None, :]) @ obj_ref.rotation
                bounds_min = local.min(axis=0).astype(np.float32, copy=False)
                bounds_max = local.max(axis=0).astype(np.float32, copy=False)
            refs.append(
                _ReferenceGeometry(
                    point_indices=point_indices,
                    centroid=obj_ref.centroid.astype(np.float32, copy=False),
                    rotation=None if obj_ref.rotation is None else obj_ref.rotation.astype(np.float32, copy=False),
                    bounds_min=bounds_min,
                    bounds_max=bounds_max,
                    exclude_instance_ids=np.asarray(obj_ref.instance_ids, dtype=np.int32),
                    tree=cKDTree(self.points[point_indices]),
                )
            )
            return refs

        raise TypeError(f"Unsupported reference object type: {type(obj_ref)!r}")

    def _target_points(self, target: str | SemanticSpec | ObjSelection | PointSelection) -> PointSelection:
        if isinstance(target, PointSelection):
            return target
        if isinstance(target, ObjSelection):
            return target.to_points()
        if isinstance(target, (str, SemanticSpec)):
            return self._semantic_points(target)
        raise TypeError(f"Unsupported target type: {type(target)!r}")

    def object_class(self, query: str | SemanticSpec) -> ObjSelection:
        spec = self._semantic_spec(query)
        return self._lift_points_to_objects(
            self._semantic_points(spec),
            min_point_fraction=spec.min_point_fraction,
            min_point_count=spec.min_point_count,
        )

    def color_appearance(
        self,
        rgb: Sequence[float],
        *,
        rgb_l2_threshold: float = 35.0,
        min_point_fraction: float = DEFAULT_MIN_POINT_FRACTION,
        min_point_count: int = DEFAULT_MIN_POINT_COUNT,
    ) -> ObjSelection:
        return self._lift_points_to_objects(
            self._color_points(rgb, rgb_l2_threshold=rgb_l2_threshold),
            min_point_fraction=min_point_fraction,
            min_point_count=min_point_count,
        )

    def _apply_relation(
        self,
        obj_ref: int | ObjSelection | VirtualObject,
        target: str | SemanticSpec | ObjSelection | PointSelection,
        *,
        relation: str,
        ref_reduce: RefReduce = "any",
        distance_threshold: float | None = None,
        min_gap: float = 0.0,
        max_gap: float = 0.0,
        lateral_margin: float = 0.0,
        vertical_margin: float = 0.0,
        footprint_margin: float = 0.0,
        front_back_margin: float = 0.0,
        side: Side = "either",
    ) -> PointSelection:
        refs = self._reference_geometries(obj_ref)
        if not refs:
            return self._make_point_selection(np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32))
        target_points = self._target_points(target)
        if not target_points:
            return target_points

        candidate_ids = target_points.point_indices
        candidate_points = self.points[candidate_ids]
        candidate_instances = target_points.instance_ids
        base_scores = target_points.scores

        ref_masks = []
        ref_scores = []
        for ref in refs:
            allowed = np.ones((candidate_ids.shape[0],), dtype=bool)
            if ref.exclude_instance_ids.size > 0:
                allowed &= ~np.isin(candidate_instances, ref.exclude_instance_ids)
            if not allowed.any():
                ref_masks.append(np.zeros_like(allowed))
                ref_scores.append(np.zeros_like(base_scores))
                continue

            points_allowed = candidate_points[allowed]
            if relation == "close_to":
                assert distance_threshold is not None
                # Distances are nearest-neighbor distances to the full reference
                # support, never to a stored centroid.
                dists, _ = ref.tree.query(points_allowed, k=1, workers=-1)
                rel_mask = dists <= float(distance_threshold)
                rel_score = np.clip(1.0 - dists / max(float(distance_threshold), 1e-8), 0.0, 1.0).astype(np.float32, copy=False)
            else:
                if ref.rotation is None or ref.bounds_min is None or ref.bounds_max is None:
                    raise ValueError("Orientation-aware relation requested on a reference object without orientation")
                # The centroid only defines the local coordinate origin. Gap
                # thresholds are measured against the oriented support envelope
                # (bounds_min / bounds_max) computed from the full point set.
                local = (points_allowed - ref.centroid[None, :]) @ ref.rotation
                min_b = ref.bounds_min
                max_b = ref.bounds_max
                rel_mask = np.zeros((local.shape[0],), dtype=bool)
                rel_score = np.zeros((local.shape[0],), dtype=np.float32)

                if relation == "in_front":
                    gap = local[:, 0] - max_b[0]
                    rel_mask = (
                        (gap >= min_gap)
                        & (gap <= max_gap)
                        & (local[:, 1] >= min_b[1] - lateral_margin)
                        & (local[:, 1] <= max_b[1] + lateral_margin)
                        & (local[:, 2] >= min_b[2] - vertical_margin)
                        & (local[:, 2] <= max_b[2] + vertical_margin)
                    )
                    rel_score = np.clip(1.0 - gap / max(max_gap, 1e-8), 0.0, 1.0)
                elif relation == "behind":
                    gap = min_b[0] - local[:, 0]
                    rel_mask = (
                        (gap >= min_gap)
                        & (gap <= max_gap)
                        & (local[:, 1] >= min_b[1] - lateral_margin)
                        & (local[:, 1] <= max_b[1] + lateral_margin)
                        & (local[:, 2] >= min_b[2] - vertical_margin)
                        & (local[:, 2] <= max_b[2] + vertical_margin)
                    )
                    rel_score = np.clip(1.0 - gap / max(max_gap, 1e-8), 0.0, 1.0)
                elif relation == "on_top":
                    gap = local[:, 2] - max_b[2]
                    rel_mask = (
                        (gap >= min_gap)
                        & (gap <= max_gap)
                        & (local[:, 0] >= min_b[0] - footprint_margin)
                        & (local[:, 0] <= max_b[0] + footprint_margin)
                        & (local[:, 1] >= min_b[1] - footprint_margin)
                        & (local[:, 1] <= max_b[1] + footprint_margin)
                    )
                    rel_score = np.clip(1.0 - gap / max(max_gap, 1e-8), 0.0, 1.0)
                elif relation == "below":
                    gap = min_b[2] - local[:, 2]
                    rel_mask = (
                        (gap >= min_gap)
                        & (gap <= max_gap)
                        & (local[:, 0] >= min_b[0] - footprint_margin)
                        & (local[:, 0] <= max_b[0] + footprint_margin)
                        & (local[:, 1] >= min_b[1] - footprint_margin)
                        & (local[:, 1] <= max_b[1] + footprint_margin)
                    )
                    rel_score = np.clip(1.0 - gap / max(max_gap, 1e-8), 0.0, 1.0)
                elif relation == "on_the_side":
                    pos_gap = local[:, 1] - max_b[1]
                    neg_gap = min_b[1] - local[:, 1]
                    pos_mask = (
                        (pos_gap >= min_gap)
                        & (pos_gap <= max_gap)
                        & (local[:, 0] >= min_b[0] - front_back_margin)
                        & (local[:, 0] <= max_b[0] + front_back_margin)
                        & (local[:, 2] >= min_b[2] - vertical_margin)
                        & (local[:, 2] <= max_b[2] + vertical_margin)
                    )
                    neg_mask = (
                        (neg_gap >= min_gap)
                        & (neg_gap <= max_gap)
                        & (local[:, 0] >= min_b[0] - front_back_margin)
                        & (local[:, 0] <= max_b[0] + front_back_margin)
                        & (local[:, 2] >= min_b[2] - vertical_margin)
                        & (local[:, 2] <= max_b[2] + vertical_margin)
                    )
                    if side == "+y":
                        rel_mask = pos_mask
                        rel_score = np.clip(1.0 - pos_gap / max(max_gap, 1e-8), 0.0, 1.0)
                    elif side == "-y":
                        rel_mask = neg_mask
                        rel_score = np.clip(1.0 - neg_gap / max(max_gap, 1e-8), 0.0, 1.0)
                    else:
                        rel_mask = pos_mask | neg_mask
                        rel_score = np.maximum(
                            np.clip(1.0 - pos_gap / max(max_gap, 1e-8), 0.0, 1.0),
                            np.clip(1.0 - neg_gap / max(max_gap, 1e-8), 0.0, 1.0),
                        )
                else:
                    raise ValueError(f"Unsupported relation: {relation}")
                rel_score = rel_score.astype(np.float32, copy=False)

            full_mask = np.zeros((candidate_ids.shape[0],), dtype=bool)
            full_score = np.zeros((candidate_ids.shape[0],), dtype=np.float32)
            allowed_idx = np.flatnonzero(allowed)
            chosen = allowed_idx[rel_mask]
            full_mask[chosen] = True
            full_score[chosen] = rel_score[rel_mask] * base_scores[chosen]
            ref_masks.append(full_mask)
            ref_scores.append(full_score)

        if ref_reduce == "all":
            combined_mask = np.logical_and.reduce(ref_masks)
            combined_score = np.minimum.reduce(ref_scores)
        else:
            combined_mask = np.logical_or.reduce(ref_masks)
            combined_score = np.maximum.reduce(ref_scores)
        chosen = np.flatnonzero(combined_mask)
        return self._make_point_selection(candidate_ids[chosen], combined_score[chosen])

    def close_to(
        self,
        obj_ref: int | ObjSelection | VirtualObject,
        target: str | SemanticSpec | ObjSelection | PointSelection,
        *,
        distance_threshold: float = 0.20,
        wholeobj_or_points: WholeObjOrPoints = "wholeobj",
        wholeobj_min_point_fraction: float = DEFAULT_MIN_POINT_FRACTION,
        wholeobj_min_point_count: int = DEFAULT_MIN_POINT_COUNT,
        ref_reduce: RefReduce = "any",
    ) -> ObjSelection | PointSelection:
        points = self._apply_relation(
            obj_ref,
            target,
            relation="close_to",
            ref_reduce=ref_reduce,
            distance_threshold=distance_threshold,
        )
        if wholeobj_or_points == "points":
            return points
        return self._lift_points_to_objects(
            points,
            min_point_fraction=wholeobj_min_point_fraction,
            min_point_count=wholeobj_min_point_count,
        )

    def in_front(
        self,
        obj_ref: int | ObjSelection | VirtualObject,
        target: str | SemanticSpec | ObjSelection | PointSelection,
        *,
        min_gap: float = 0.0,
        max_gap: float = 0.40,
        lateral_margin: float = 0.10,
        vertical_margin: float = 0.10,
        wholeobj_or_points: WholeObjOrPoints = "wholeobj",
        wholeobj_min_point_fraction: float = DEFAULT_MIN_POINT_FRACTION,
        wholeobj_min_point_count: int = DEFAULT_MIN_POINT_COUNT,
        ref_reduce: RefReduce = "any",
    ) -> ObjSelection | PointSelection:
        points = self._apply_relation(
            obj_ref,
            target,
            relation="in_front",
            ref_reduce=ref_reduce,
            min_gap=min_gap,
            max_gap=max_gap,
            lateral_margin=lateral_margin,
            vertical_margin=vertical_margin,
        )
        if wholeobj_or_points == "points":
            return points
        return self._lift_points_to_objects(
            points,
            min_point_fraction=wholeobj_min_point_fraction,
            min_point_count=wholeobj_min_point_count,
        )

    def behind(
        self,
        obj_ref: int | ObjSelection | VirtualObject,
        target: str | SemanticSpec | ObjSelection | PointSelection,
        *,
        min_gap: float = 0.0,
        max_gap: float = 0.40,
        lateral_margin: float = 0.10,
        vertical_margin: float = 0.10,
        wholeobj_or_points: WholeObjOrPoints = "wholeobj",
        wholeobj_min_point_fraction: float = DEFAULT_MIN_POINT_FRACTION,
        wholeobj_min_point_count: int = DEFAULT_MIN_POINT_COUNT,
        ref_reduce: RefReduce = "any",
    ) -> ObjSelection | PointSelection:
        points = self._apply_relation(
            obj_ref,
            target,
            relation="behind",
            ref_reduce=ref_reduce,
            min_gap=min_gap,
            max_gap=max_gap,
            lateral_margin=lateral_margin,
            vertical_margin=vertical_margin,
        )
        if wholeobj_or_points == "points":
            return points
        return self._lift_points_to_objects(
            points,
            min_point_fraction=wholeobj_min_point_fraction,
            min_point_count=wholeobj_min_point_count,
        )

    def on_top(
        self,
        obj_ref: int | ObjSelection | VirtualObject,
        target: str | SemanticSpec | ObjSelection | PointSelection,
        *,
        min_gap: float = 0.0,
        max_gap: float = 0.25,
        footprint_margin: float = 0.05,
        wholeobj_or_points: WholeObjOrPoints = "wholeobj",
        wholeobj_min_point_fraction: float = DEFAULT_MIN_POINT_FRACTION,
        wholeobj_min_point_count: int = DEFAULT_MIN_POINT_COUNT,
        ref_reduce: RefReduce = "any",
    ) -> ObjSelection | PointSelection:
        points = self._apply_relation(
            obj_ref,
            target,
            relation="on_top",
            ref_reduce=ref_reduce,
            min_gap=min_gap,
            max_gap=max_gap,
            footprint_margin=footprint_margin,
        )
        if wholeobj_or_points == "points":
            return points
        return self._lift_points_to_objects(
            points,
            min_point_fraction=wholeobj_min_point_fraction,
            min_point_count=wholeobj_min_point_count,
        )

    def below(
        self,
        obj_ref: int | ObjSelection | VirtualObject,
        target: str | SemanticSpec | ObjSelection | PointSelection,
        *,
        min_gap: float = 0.0,
        max_gap: float = 0.25,
        footprint_margin: float = 0.05,
        wholeobj_or_points: WholeObjOrPoints = "wholeobj",
        wholeobj_min_point_fraction: float = DEFAULT_MIN_POINT_FRACTION,
        wholeobj_min_point_count: int = DEFAULT_MIN_POINT_COUNT,
        ref_reduce: RefReduce = "any",
    ) -> ObjSelection | PointSelection:
        points = self._apply_relation(
            obj_ref,
            target,
            relation="below",
            ref_reduce=ref_reduce,
            min_gap=min_gap,
            max_gap=max_gap,
            footprint_margin=footprint_margin,
        )
        if wholeobj_or_points == "points":
            return points
        return self._lift_points_to_objects(
            points,
            min_point_fraction=wholeobj_min_point_fraction,
            min_point_count=wholeobj_min_point_count,
        )

    def on_the_side(
        self,
        obj_ref: int | ObjSelection | VirtualObject,
        target: str | SemanticSpec | ObjSelection | PointSelection,
        *,
        side: Side = "either",
        min_gap: float = 0.0,
        max_gap: float = 0.30,
        front_back_margin: float = 0.10,
        vertical_margin: float = 0.10,
        wholeobj_or_points: WholeObjOrPoints = "wholeobj",
        wholeobj_min_point_fraction: float = DEFAULT_MIN_POINT_FRACTION,
        wholeobj_min_point_count: int = DEFAULT_MIN_POINT_COUNT,
        ref_reduce: RefReduce = "any",
    ) -> ObjSelection | PointSelection:
        points = self._apply_relation(
            obj_ref,
            target,
            relation="on_the_side",
            ref_reduce=ref_reduce,
            side=side,
            min_gap=min_gap,
            max_gap=max_gap,
            front_back_margin=front_back_margin,
            vertical_margin=vertical_margin,
        )
        if wholeobj_or_points == "points":
            return points
        return self._lift_points_to_objects(
            points,
            min_point_fraction=wholeobj_min_point_fraction,
            min_point_count=wholeobj_min_point_count,
        )

    def _surface_selection(
        self,
        obj_ref: int | ObjSelection | VirtualObject,
        *,
        surface: str,
        surface_band: float,
        side: Side = "either",
    ) -> PointSelection:
        refs = self._reference_geometries(obj_ref)
        if not refs:
            return self._make_point_selection(np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32))
        selections = []
        rotations = []
        for ref in refs:
            if ref.rotation is None or ref.bounds_min is None or ref.bounds_max is None:
                raise ValueError("Surface query requires an oriented reference object")
            local = (self.points[ref.point_indices] - ref.centroid[None, :]) @ ref.rotation
            if surface == "front":
                mask = local[:, 0] >= ref.bounds_max[0] - surface_band
            elif surface == "back":
                mask = local[:, 0] <= ref.bounds_min[0] + surface_band
            elif surface == "top":
                mask = local[:, 2] >= ref.bounds_max[2] - surface_band
            elif surface == "bottom":
                mask = local[:, 2] <= ref.bounds_min[2] + surface_band
            elif surface == "sides":
                pos_mask = local[:, 1] >= ref.bounds_max[1] - surface_band
                neg_mask = local[:, 1] <= ref.bounds_min[1] + surface_band
                if side == "+y":
                    mask = pos_mask
                elif side == "-y":
                    mask = neg_mask
                else:
                    mask = pos_mask | neg_mask
            else:
                raise ValueError(f"Unsupported surface type: {surface}")
            selected_idx = ref.point_indices[mask]
            selections.append(self._make_point_selection(selected_idx, np.ones((selected_idx.shape[0],), dtype=np.float32), ref.rotation))
            rotations.append(ref.rotation)
        return union(*selections) if len(selections) > 1 else selections[0]

    def front(self, obj_ref: int | ObjSelection | VirtualObject, *, surface_band: float = 0.03) -> PointSelection:
        return self._surface_selection(obj_ref, surface="front", surface_band=surface_band)

    def back(self, obj_ref: int | ObjSelection | VirtualObject, *, surface_band: float = 0.03) -> PointSelection:
        return self._surface_selection(obj_ref, surface="back", surface_band=surface_band)

    def top(self, obj_ref: int | ObjSelection | VirtualObject, *, surface_band: float = 0.03) -> PointSelection:
        return self._surface_selection(obj_ref, surface="top", surface_band=surface_band)

    def bottom(self, obj_ref: int | ObjSelection | VirtualObject, *, surface_band: float = 0.03) -> PointSelection:
        return self._surface_selection(obj_ref, surface="bottom", surface_band=surface_band)

    def sides(
        self,
        obj_ref: int | ObjSelection | VirtualObject,
        *,
        side: Side = "either",
        surface_band: float = 0.03,
    ) -> PointSelection:
        return self._surface_selection(obj_ref, surface="sides", surface_band=surface_band, side=side)


_ACTIVE_SCENE: FeatureFieldScene | None = None


def set_active_scene(scene: FeatureFieldScene) -> FeatureFieldScene:
    global _ACTIVE_SCENE
    _ACTIVE_SCENE = scene
    return scene


def load_scene(
    run_dir: str | Path,
    *,
    min_component_size: int = DEFAULT_MIN_COMPONENT_SIZE,
    clip_device: str = "cuda",
    clip_chunk_size: int = DEFAULT_CLIP_CHUNK_SIZE,
    max_points: int | None = None,
    downsample_seed: int = 0,
) -> FeatureFieldScene:
    scene = FeatureFieldScene.load(
        run_dir,
        min_component_size=min_component_size,
        clip_device=clip_device,
        clip_chunk_size=clip_chunk_size,
        max_points=max_points,
        downsample_seed=downsample_seed,
    )
    return set_active_scene(scene)


def active_scene() -> FeatureFieldScene:
    if _ACTIVE_SCENE is None:
        raise RuntimeError("No active scene. Call load_scene(...) first.")
    return _ACTIVE_SCENE


def union(*selections: PointSelection | ObjSelection) -> PointSelection | ObjSelection:
    if not selections:
        raise ValueError("union requires at least one selection")
    first = selections[0]
    if any(type(sel) is not type(first) for sel in selections[1:]):
        raise TypeError("union requires all selections to have the same type")
    if isinstance(first, PointSelection):
        scene = first.scene
        if any(sel.scene is not scene for sel in selections[1:]):
            raise ValueError("All point selections in union must belong to the same scene")
        point_indices = np.concatenate([sel.point_indices for sel in selections], axis=0)
        scores = np.concatenate([sel.scores for sel in selections], axis=0)
        rotations = [sel.default_rotation for sel in selections if sel.default_rotation is not None]
        default_rotation = rotations[0] if len(rotations) == 1 else None
        return scene._make_point_selection(point_indices, scores, default_rotation)
    if isinstance(first, ObjSelection):
        scene = first.scene
        if any(sel.scene is not scene for sel in selections[1:]):
            raise ValueError("All object selections in union must belong to the same scene")
        instance_ids = np.concatenate([sel.instance_ids for sel in selections], axis=0)
        scores = np.concatenate([sel.scores for sel in selections], axis=0)
        fractions = np.concatenate([sel.matched_point_fraction for sel in selections], axis=0)
        counts = np.concatenate([sel.matched_point_count for sel in selections], axis=0)
        if instance_ids.size == 0:
            return ObjSelection(
                scene=scene,
                instance_ids=np.empty((0,), dtype=np.int32),
                scores=np.empty((0,), dtype=np.float32),
                matched_point_fraction=np.empty((0,), dtype=np.float32),
                matched_point_count=np.empty((0,), dtype=np.int32),
            )
        order = np.argsort(instance_ids, kind="stable")
        instance_ids = instance_ids[order]
        scores = scores[order]
        fractions = fractions[order]
        counts = counts[order]
        unique, starts = np.unique(instance_ids, return_index=True)
        score_max = np.maximum.reduceat(scores, starts).astype(np.float32, copy=False)
        frac_max = np.maximum.reduceat(fractions, starts).astype(np.float32, copy=False)
        count_max = np.maximum.reduceat(counts, starts).astype(np.int32, copy=False)
        sort_order = np.argsort(-score_max, kind="stable")
        return ObjSelection(
            scene=scene,
            instance_ids=unique[sort_order].astype(np.int32, copy=False),
            scores=score_max[sort_order],
            matched_point_fraction=frac_max[sort_order],
            matched_point_count=count_max[sort_order],
        )
    raise TypeError(f"Unsupported selection type for union: {type(first)!r}")


def object_class(query: str | SemanticSpec) -> ObjSelection:
    return active_scene().object_class(query)


def color_appearance(
    rgb: Sequence[float],
    *,
    rgb_l2_threshold: float = 35.0,
    min_point_fraction: float = DEFAULT_MIN_POINT_FRACTION,
    min_point_count: int = DEFAULT_MIN_POINT_COUNT,
) -> ObjSelection:
    return active_scene().color_appearance(
        rgb,
        rgb_l2_threshold=rgb_l2_threshold,
        min_point_fraction=min_point_fraction,
        min_point_count=min_point_count,
    )


def close_to(
    obj_ref: int | ObjSelection | VirtualObject,
    target: str | SemanticSpec | ObjSelection | PointSelection,
    *,
    distance_threshold: float = 0.20,
    wholeobj_or_points: WholeObjOrPoints = "wholeobj",
    wholeobj_min_point_fraction: float = DEFAULT_MIN_POINT_FRACTION,
    wholeobj_min_point_count: int = DEFAULT_MIN_POINT_COUNT,
    ref_reduce: RefReduce = "any",
) -> ObjSelection | PointSelection:
    return active_scene().close_to(
        obj_ref,
        target,
        distance_threshold=distance_threshold,
        wholeobj_or_points=wholeobj_or_points,
        wholeobj_min_point_fraction=wholeobj_min_point_fraction,
        wholeobj_min_point_count=wholeobj_min_point_count,
        ref_reduce=ref_reduce,
    )


def in_front(
    obj_ref: int | ObjSelection | VirtualObject,
    target: str | SemanticSpec | ObjSelection | PointSelection,
    *,
    min_gap: float = 0.0,
    max_gap: float = 0.40,
    lateral_margin: float = 0.10,
    vertical_margin: float = 0.10,
    wholeobj_or_points: WholeObjOrPoints = "wholeobj",
    wholeobj_min_point_fraction: float = DEFAULT_MIN_POINT_FRACTION,
    wholeobj_min_point_count: int = DEFAULT_MIN_POINT_COUNT,
    ref_reduce: RefReduce = "any",
) -> ObjSelection | PointSelection:
    return active_scene().in_front(
        obj_ref,
        target,
        min_gap=min_gap,
        max_gap=max_gap,
        lateral_margin=lateral_margin,
        vertical_margin=vertical_margin,
        wholeobj_or_points=wholeobj_or_points,
        wholeobj_min_point_fraction=wholeobj_min_point_fraction,
        wholeobj_min_point_count=wholeobj_min_point_count,
        ref_reduce=ref_reduce,
    )


def behind(
    obj_ref: int | ObjSelection | VirtualObject,
    target: str | SemanticSpec | ObjSelection | PointSelection,
    *,
    min_gap: float = 0.0,
    max_gap: float = 0.40,
    lateral_margin: float = 0.10,
    vertical_margin: float = 0.10,
    wholeobj_or_points: WholeObjOrPoints = "wholeobj",
    wholeobj_min_point_fraction: float = DEFAULT_MIN_POINT_FRACTION,
    wholeobj_min_point_count: int = DEFAULT_MIN_POINT_COUNT,
    ref_reduce: RefReduce = "any",
) -> ObjSelection | PointSelection:
    return active_scene().behind(
        obj_ref,
        target,
        min_gap=min_gap,
        max_gap=max_gap,
        lateral_margin=lateral_margin,
        vertical_margin=vertical_margin,
        wholeobj_or_points=wholeobj_or_points,
        wholeobj_min_point_fraction=wholeobj_min_point_fraction,
        wholeobj_min_point_count=wholeobj_min_point_count,
        ref_reduce=ref_reduce,
    )


def on_top(
    obj_ref: int | ObjSelection | VirtualObject,
    target: str | SemanticSpec | ObjSelection | PointSelection,
    *,
    min_gap: float = 0.0,
    max_gap: float = 0.25,
    footprint_margin: float = 0.05,
    wholeobj_or_points: WholeObjOrPoints = "wholeobj",
    wholeobj_min_point_fraction: float = DEFAULT_MIN_POINT_FRACTION,
    wholeobj_min_point_count: int = DEFAULT_MIN_POINT_COUNT,
    ref_reduce: RefReduce = "any",
) -> ObjSelection | PointSelection:
    return active_scene().on_top(
        obj_ref,
        target,
        min_gap=min_gap,
        max_gap=max_gap,
        footprint_margin=footprint_margin,
        wholeobj_or_points=wholeobj_or_points,
        wholeobj_min_point_fraction=wholeobj_min_point_fraction,
        wholeobj_min_point_count=wholeobj_min_point_count,
        ref_reduce=ref_reduce,
    )


def below(
    obj_ref: int | ObjSelection | VirtualObject,
    target: str | SemanticSpec | ObjSelection | PointSelection,
    *,
    min_gap: float = 0.0,
    max_gap: float = 0.25,
    footprint_margin: float = 0.05,
    wholeobj_or_points: WholeObjOrPoints = "wholeobj",
    wholeobj_min_point_fraction: float = DEFAULT_MIN_POINT_FRACTION,
    wholeobj_min_point_count: int = DEFAULT_MIN_POINT_COUNT,
    ref_reduce: RefReduce = "any",
) -> ObjSelection | PointSelection:
    return active_scene().below(
        obj_ref,
        target,
        min_gap=min_gap,
        max_gap=max_gap,
        footprint_margin=footprint_margin,
        wholeobj_or_points=wholeobj_or_points,
        wholeobj_min_point_fraction=wholeobj_min_point_fraction,
        wholeobj_min_point_count=wholeobj_min_point_count,
        ref_reduce=ref_reduce,
    )


def on_the_side(
    obj_ref: int | ObjSelection | VirtualObject,
    target: str | SemanticSpec | ObjSelection | PointSelection,
    *,
    side: Side = "either",
    min_gap: float = 0.0,
    max_gap: float = 0.30,
    front_back_margin: float = 0.10,
    vertical_margin: float = 0.10,
    wholeobj_or_points: WholeObjOrPoints = "wholeobj",
    wholeobj_min_point_fraction: float = DEFAULT_MIN_POINT_FRACTION,
    wholeobj_min_point_count: int = DEFAULT_MIN_POINT_COUNT,
    ref_reduce: RefReduce = "any",
) -> ObjSelection | PointSelection:
    return active_scene().on_the_side(
        obj_ref,
        target,
        side=side,
        min_gap=min_gap,
        max_gap=max_gap,
        front_back_margin=front_back_margin,
        vertical_margin=vertical_margin,
        wholeobj_or_points=wholeobj_or_points,
        wholeobj_min_point_fraction=wholeobj_min_point_fraction,
        wholeobj_min_point_count=wholeobj_min_point_count,
        ref_reduce=ref_reduce,
    )


def front(obj_ref: int | ObjSelection | VirtualObject, *, surface_band: float = 0.03) -> PointSelection:
    return active_scene().front(obj_ref, surface_band=surface_band)


def back(obj_ref: int | ObjSelection | VirtualObject, *, surface_band: float = 0.03) -> PointSelection:
    return active_scene().back(obj_ref, surface_band=surface_band)


def top(obj_ref: int | ObjSelection | VirtualObject, *, surface_band: float = 0.03) -> PointSelection:
    return active_scene().top(obj_ref, surface_band=surface_band)


def bottom(obj_ref: int | ObjSelection | VirtualObject, *, surface_band: float = 0.03) -> PointSelection:
    return active_scene().bottom(obj_ref, surface_band=surface_band)


def sides(
    obj_ref: int | ObjSelection | VirtualObject,
    *,
    side: Side = "either",
    surface_band: float = 0.03,
) -> PointSelection:
    return active_scene().sides(obj_ref, side=side, surface_band=surface_band)


__all__ = [
    "FeatureFieldScene",
    "SemanticSpec",
    "ObjSelection",
    "PointSelection",
    "VirtualObject",
    "load_scene",
    "set_active_scene",
    "active_scene",
    "union",
    "object_class",
    "color_appearance",
    "close_to",
    "in_front",
    "behind",
    "on_top",
    "below",
    "on_the_side",
    "front",
    "back",
    "top",
    "bottom",
    "sides",
]
