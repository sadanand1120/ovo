from pathlib import Path


# Shared repo paths and dataset config locations.
CONFIG_DIR = Path("configs")
INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/output")

DEFAULT_CONFIG_PATH = CONFIG_DIR / "ovo.yaml"
REPLICA_CONFIG_PATH = CONFIG_DIR / "replica.yaml"

DATASET_CONFIG_NAMES = {
    "Replica": "replica",
    "ScanNet": "scannet",
}

# Default output roots and dataset asset locations.
DEFAULT_RGB_MAP_OUTPUT_ROOT = OUTPUT_DIR / "rgb_maps"
DEFAULT_RGB_CACHE_OUTPUT_ROOT = OUTPUT_DIR / "rgb_caches"
DEFAULT_TOPDOWN_VIS_OUTPUT_ROOT = OUTPUT_DIR / "topdown_vis"
DEFAULT_OVO_EVAL_OUTPUT_ROOT = OUTPUT_DIR / "ovo_style_eval"
DEFAULT_REPLICA_ROOT = INPUT_DIR / "Replica"
DEFAULT_OVO_SEMANTIC_GT_ROOT = DEFAULT_REPLICA_ROOT / "semantic_gt"

# Standard artifact filenames used across map build/cache/eval.
TIMING_PATH = "timing.json"
STATS_PATH = "stats.json"
CLIP_FEATURE_FILE = "clip_feats.npy"
INSTANCE_LABEL_FILE = "instance_labels.npy"
INSTANCE_GID_SLOTS_FILE = "instance_gid_slots.npy"
INSTANCE_SUPPORT_FILE = "instance_seed_hits.npy"
POINT_BIRTH_FRAME_FILE = "point_birth_frame.npy"
CACHE_MANIFEST_FILE = "cache_manifest.json"
FRAME_CACHE_DIR = "frame_cache"
FRAME_CACHE_FILE_TEMPLATE = "{frame_id:06d}.npz"

# Shared CLIP backbone and preprocessing defaults.
CLIP_MODEL_NAME = "ViT-L-14-336-quickgelu"
CLIP_PRETRAINED = "openai"
CLIP_LOAD_SIZE = 1024
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_GLOBAL_PATCH_THRESHOLD = 0.07

# Core map-building defaults.
DEFAULT_MAP_EVERY = 8
DEFAULT_MAX_TOTAL_POINTS = int(7e6)
DEFAULT_MATCH_DISTANCE_TH = 0.03
DEFAULT_MIN_COMPONENT_SIZE = 2000

# Shared semantic/OVO evaluation defaults.
DEFAULT_FEATURE_PROB_TH = 0.0
DEFAULT_OVO_SCORE_TH = 0.0
DEFAULT_EVAL_CHUNK_SIZE = 100_000

# Shared visualization defaults.
DEFAULT_PCA_SAMPLE_SIZE = 100_000
DEFAULT_VIS_CHUNK_SIZE = 200_000
DEFAULT_VIS_POINT_SIZE = 1.0
DEFAULT_GT_POINT_SIZE = 3.0
VIEW_SAVE_KEY = ord("V")

# Topdown-video rendering defaults.
DEFAULT_TOPDOWN_VIDEO_FPS = 8
DEFAULT_TOPDOWN_POINT_DILATE = 3
TOPDOWN_VIDEO_DIR_NAME = "topdown_videos"
TOPDOWN_FRUSTUM_DEPTH = 0.6
TOPDOWN_FRUSTUM_COLOR = (40, 40, 220)
TOPDOWN_FRUSTUM_THICKNESS = 2

# Text templates and aggregation modes for feature/OVO scoring.
FEATURE_TEXT_TEMPLATE = "{}"
FEATURE_SOFTMAX_TEMP = 0.01
OVO_TEXT_TEMPLATE = "This is a photo of a {}"
OVO_TEXT_TEMPLATE_OPTIMAL = "{}"
FEATURE_INSTANCE_AGG = "mean_probs"
OVO_FEATURE_AGG = "mean_l2norm_point_features_then_cosine"
OVO_FEATURE_AGG_OPTIMAL = "mean_point_cosine_scores_after_l2norm"
