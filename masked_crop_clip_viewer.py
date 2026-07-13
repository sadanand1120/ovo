from __future__ import annotations

import argparse
import csv
import io
import json
import socket
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

from map_runtime.defaults import (
    CLIP_MEAN,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    CLIP_STD,
    FEATURE_SOFTMAX_TEMP,
    FEATURE_TEXT_TEMPLATE,
)


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Masked Crop CLIP Probe</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #16191f;
      --muted: #626b7a;
      --line: #d9dee8;
      --accent: #0d6efd;
      --gt: #d94a38;
      --bar: #5277c7;
      --overlay: rgba(255, 72, 48, 0.42);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      width: min(1440px, calc(100vw - 32px));
      margin: 18px auto;
      display: grid;
      grid-template-columns: minmax(520px, 1fr) 420px;
      gap: 16px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }
    .topbar {
      display: grid;
      grid-template-columns: 1fr 220px;
      gap: 12px;
      align-items: end;
      margin-bottom: 12px;
    }
    label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      margin-bottom: 6px;
    }
    input[type="range"] { width: 100%; }
    select, button {
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
      color: var(--text);
      font: inherit;
      padding: 7px 9px;
    }
    button {
      border-color: var(--accent);
      background: var(--accent);
      color: white;
      font-weight: 700;
      cursor: pointer;
      margin-top: 10px;
    }
    button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }
    .stage {
      position: relative;
      width: 100%;
      background: #111;
      border-radius: 6px;
      overflow: hidden;
      line-height: 0;
    }
    .stage img {
      width: 100%;
      display: block;
    }
    #overlay {
      position: absolute;
      inset: 0;
      pointer-events: none;
      image-rendering: auto;
    }
    .meta {
      color: var(--muted);
      font-size: 13px;
      min-height: 18px;
      margin-top: 8px;
    }
    .thumb {
      width: 100%;
      min-height: 120px;
      display: grid;
      place-items: center;
      background: #f1f3f6;
      border: 1px dashed var(--line);
      border-radius: 6px;
      overflow: hidden;
      color: var(--muted);
      font-size: 13px;
      margin-top: 12px;
    }
    .thumb img {
      max-width: 100%;
      display: block;
    }
    .bars {
      margin-top: 12px;
      display: grid;
      gap: 8px;
    }
    .bar-row {
      display: grid;
      grid-template-columns: 130px 1fr 92px;
      gap: 8px;
      align-items: center;
      font-size: 13px;
    }
    .bar-label {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .bar-track {
      height: 18px;
      border-radius: 4px;
      background: #eef1f6;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      min-width: 1px;
      background: var(--bar);
    }
    .bar-fill.gt { background: var(--gt); }
    .bar-val {
      color: var(--muted);
      font-variant-numeric: tabular-nums;
      text-align: right;
    }
    .status {
      color: var(--muted);
      font-size: 13px;
      min-height: 20px;
      margin-top: 10px;
    }
    .pill {
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      margin-left: 6px;
      color: var(--muted);
      font-size: 12px;
    }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      .topbar { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<main>
  <section class="panel">
    <div class="topbar">
      <div>
        <label for="frameSlider">Image Index <span id="frameReadout" class="pill"></span></label>
        <input id="frameSlider" type="range" min="0" max="0" value="0">
      </div>
      <div>
        <label for="instanceSelect">Visible Instance</label>
        <select id="instanceSelect"></select>
      </div>
    </div>
    <div class="stage">
      <img id="rgb" alt="ScanNet RGB frame">
      <img id="overlay" alt="">
    </div>
    <div id="frameMeta" class="meta"></div>
  </section>
  <aside class="panel">
    <label>Selected Masked Crop</label>
    <div id="cropBox" class="thumb">Select an instance</div>
    <button id="runClip" disabled>Compute Masked-Crop CLIP Match</button>
    <div id="status" class="status"></div>
    <div id="bars" class="bars"></div>
  </aside>
</main>
<script>
const slider = document.getElementById("frameSlider");
const readout = document.getElementById("frameReadout");
const select = document.getElementById("instanceSelect");
const rgb = document.getElementById("rgb");
const overlay = document.getElementById("overlay");
const frameMeta = document.getElementById("frameMeta");
const runClip = document.getElementById("runClip");
const statusBox = document.getElementById("status");
const bars = document.getElementById("bars");
const cropBox = document.getElementById("cropBox");

let frame = null;

async function fetchJson(url) {
  const resp = await fetch(url);
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data.error || resp.statusText);
  }
  return data;
}

function selectedInstanceId() {
  const value = select.value;
  return value === "" ? null : Number(value);
}

function updateSelection() {
  const instId = selectedInstanceId();
  overlay.src = instId === null ? "" : `/api/mask?idx=${slider.value}&instance_id=${instId}&t=${Date.now()}`;
  if (instId === null) {
    cropBox.textContent = "No visible instances";
    runClip.disabled = true;
  } else {
    cropBox.innerHTML = `<img src="/api/crop?idx=${slider.value}&instance_id=${instId}&t=${Date.now()}" alt="masked crop">`;
    runClip.disabled = false;
  }
  bars.innerHTML = "";
  statusBox.textContent = "";
}

function renderInstances(instances) {
  select.innerHTML = "";
  for (const inst of instances) {
    const option = document.createElement("option");
    option.value = inst.id;
    option.textContent = `${inst.id} | ${inst.label} | ${inst.area.toLocaleString()} px`;
    select.appendChild(option);
  }
  if (instances.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No visible instances";
    select.appendChild(option);
  }
}

async function loadFrame(idx) {
  statusBox.textContent = "Loading frame...";
  bars.innerHTML = "";
  frame = await fetchJson(`/api/frame?idx=${idx}`);
  readout.textContent = `frame ${frame.frame_id}`;
  rgb.src = `/api/image?idx=${idx}&t=${Date.now()}`;
  renderInstances(frame.instances);
  const labels = frame.labels.length ? frame.labels.join(", ") : "none";
  frameMeta.textContent = `${frame.width}x${frame.height}; ${frame.instances.length} instances; labels: ${labels}`;
  updateSelection();
}

function renderBars(result) {
  bars.innerHTML = "";
  const maxProb = Math.max(...result.rows.map(row => row.prob), 1e-12);
  for (const row of result.rows) {
    const wrap = document.createElement("div");
    wrap.className = "bar-row";
    const label = document.createElement("div");
    label.className = "bar-label";
    label.title = row.label;
    label.textContent = row.label;
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill" + (row.is_gt ? " gt" : "");
    fill.style.width = `${Math.max(2, 100 * row.prob / maxProb)}%`;
    track.appendChild(fill);
    const val = document.createElement("div");
    val.className = "bar-val";
    val.textContent = `${(100 * row.prob).toFixed(1)}% / ${row.cosine.toFixed(3)}`;
    wrap.append(label, track, val);
    bars.appendChild(wrap);
  }
}

async function computeClip() {
  const instId = selectedInstanceId();
  if (instId === null) return;
  runClip.disabled = true;
  statusBox.textContent = "Computing CLIP image/text embeddings...";
  bars.innerHTML = "";
  try {
    const result = await fetchJson(`/api/clip?idx=${slider.value}&instance_id=${instId}`);
    statusBox.textContent = `GT: ${result.gt_label}; top: ${result.top_label}`;
    renderBars(result);
  } catch (err) {
    statusBox.textContent = err.message;
  } finally {
    runClip.disabled = false;
  }
}

async function init() {
  const meta = await fetchJson("/api/meta");
  slider.max = Math.max(0, meta.frame_count - 1);
  await loadFrame(0);
}

slider.addEventListener("input", () => {
  readout.textContent = `index ${slider.value}`;
});
slider.addEventListener("change", () => loadFrame(slider.value).catch(err => statusBox.textContent = err.message));
select.addEventListener("change", updateSelection);
runClip.addEventListener("click", computeClip);
init().catch(err => statusBox.textContent = err.message);
</script>
</body>
</html>
"""


class ScanNetMaskedCropApp:
    def __init__(
        self,
        scene_dir: Path,
        label_map_path: Path | None,
        device: str,
        text_template: str,
        min_pixels: int,
    ) -> None:
        self.scene_dir = scene_dir
        self.color_dir = scene_dir / "color"
        self.instance_dir = scene_dir / "instance-filt"
        self.label_dir = scene_dir / "label-filt"
        self.device_name = self._resolve_device(device)
        self.text_template = text_template
        self.min_pixels = int(min_pixels)
        self.label_names = self._load_label_names(label_map_path)
        self.frame_ids = self._load_frame_ids()
        self.frame_cache: dict[int, dict] = {}
        self.model_lock = threading.Lock()
        self.model: torch.nn.Module | None = None
        self.tokenizer = None
        self.image_size = 336

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_frame_ids(self) -> list[int]:
        color_ids = {int(path.stem) for path in self.color_dir.glob("*.jpg")}
        instance_ids = {int(path.stem) for path in self.instance_dir.glob("*.png")}
        label_ids = {int(path.stem) for path in self.label_dir.glob("*.png")}
        frame_ids = sorted(color_ids & instance_ids & label_ids)
        if not frame_ids:
            raise FileNotFoundError(f"No matching color/instance/label frames under {self.scene_dir}")
        return frame_ids

    def _load_label_names(self, explicit_path: Path | None) -> dict[int, str]:
        label_map_path = explicit_path or self._find_scannet_label_tsv()
        names: dict[int, str] = {}
        if label_map_path is not None and label_map_path.exists():
            with open(label_map_path, newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    label_id = int(row["id"])
                    if label_id not in names:
                        names[label_id] = self._clean_label(row.get("raw_category") or row.get("category") or "")
        return names

    def _find_scannet_label_tsv(self) -> Path | None:
        suffix = Path("open3d/_ml3d/datasets/_resources/scannet/scannetv2-labels.combined.tsv")
        for entry in sys.path:
            candidate = Path(entry) / suffix
            if candidate.exists():
                return candidate
        return None

    def _clean_label(self, label: str) -> str:
        label = label.strip().replace("_", " ")
        return label or "unknown"

    def frame_path(self, idx: int, subdir: str, suffix: str) -> Path:
        frame_id = self.frame_ids[idx]
        return self.scene_dir / subdir / f"{frame_id}{suffix}"

    def frame_id(self, idx: int) -> int:
        return self.frame_ids[idx]

    def read_image(self, idx: int) -> Image.Image:
        return Image.open(self.frame_path(idx, "color", ".jpg")).convert("RGB")

    def read_instance(self, idx: int) -> np.ndarray:
        return np.asarray(Image.open(self.frame_path(idx, "instance-filt", ".png")))

    def read_label(self, idx: int) -> np.ndarray:
        return np.asarray(Image.open(self.frame_path(idx, "label-filt", ".png")))

    def frame_meta(self, idx: int) -> dict:
        if idx not in self.frame_cache:
            image = self.read_image(idx)
            instance = self.read_instance(idx)
            label = self.read_label(idx)
            instances = self._visible_instances(instance, label)
            labels = []
            for item in instances:
                if item["label"] not in labels:
                    labels.append(item["label"])
            self.frame_cache[idx] = {
                "frame_index": idx,
                "frame_id": self.frame_id(idx),
                "width": image.width,
                "height": image.height,
                "instances": instances,
                "labels": labels,
            }
        return self.frame_cache[idx]

    def _visible_instances(self, instance: np.ndarray, label: np.ndarray) -> list[dict]:
        output = []
        for instance_id in sorted(int(value) for value in np.unique(instance) if int(value) != 0):
            mask = instance == instance_id
            area = int(mask.sum())
            if area < self.min_pixels:
                continue
            label_values, counts = np.unique(label[mask], return_counts=True)
            keep = label_values != 0
            if keep.any():
                label_values = label_values[keep]
                counts = counts[keep]
            label_id = int(label_values[int(counts.argmax())]) if len(label_values) else 0
            ys, xs = np.nonzero(mask)
            output.append(
                {
                    "id": instance_id,
                    "label_id": label_id,
                    "label": self.label_names.get(label_id, f"label {label_id}"),
                    "area": area,
                    "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                }
            )
        output.sort(key=lambda item: (-item["area"], item["id"]))
        return output

    def instance_info(self, idx: int, instance_id: int) -> dict:
        for item in self.frame_meta(idx)["instances"]:
            if int(item["id"]) == int(instance_id):
                return item
        raise ValueError(f"Instance {instance_id} is not visible in frame index {idx}")

    def mask_png(self, idx: int, instance_id: int) -> bytes:
        self.instance_info(idx, instance_id)
        mask = self.read_instance(idx) == int(instance_id)
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[mask] = (255, 72, 48, 115)
        return self._png_bytes(Image.fromarray(rgba, mode="RGBA"))

    def crop_png(self, idx: int, instance_id: int) -> bytes:
        masked = self.masked_crop_array(idx, instance_id)
        crop = Image.fromarray(masked, mode="RGB")
        crop.thumbnail((360, 360), Image.Resampling.BICUBIC)
        return self._png_bytes(crop)

    def masked_crop_array(self, idx: int, instance_id: int) -> np.ndarray:
        info = self.instance_info(idx, instance_id)
        image = np.asarray(self.read_image(idx))
        instance = self.read_instance(idx)
        x0, y0, x1, y1 = info["bbox"]
        crop = image[y0 : y1 + 1, x0 : x1 + 1]
        mask = instance[y0 : y1 + 1, x0 : x1 + 1] == int(instance_id)
        masked = np.zeros_like(crop)
        masked[mask] = crop[mask]
        return masked

    def _png_bytes(self, image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _jpeg_bytes(self, image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=92)
        return buffer.getvalue()

    def image_jpeg(self, idx: int) -> bytes:
        return self._jpeg_bytes(self.read_image(idx))

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        with self.model_lock:
            if self.model is not None:
                return
            model = open_clip.create_model_and_transforms(
                CLIP_MODEL_NAME,
                pretrained=CLIP_PRETRAINED,
                precision="fp32",
            )[0].eval().to(self.device_name)
            self.model = model
            self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
            image_size = getattr(model.visual, "image_size", 336)
            if isinstance(image_size, tuple):
                self.image_size = int(image_size[0])
            else:
                self.image_size = int(image_size)

    def _preprocess_crop(self, crop: np.ndarray) -> torch.Tensor:
        crop_image = Image.fromarray(crop, mode="RGB").resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        tensor = torch.from_numpy(np.asarray(crop_image).copy()).to(self.device_name).float() / 255.0
        tensor = tensor.permute(2, 0, 1)
        mean = tensor.new_tensor(CLIP_MEAN).view(3, 1, 1)
        std = tensor.new_tensor(CLIP_STD).view(3, 1, 1)
        return ((tensor - mean) / std).unsqueeze(0)

    @torch.inference_mode()
    def clip_result(self, idx: int, instance_id: int) -> dict:
        self._ensure_model()
        assert self.model is not None
        assert self.tokenizer is not None

        info = self.instance_info(idx, instance_id)
        labels = self.frame_meta(idx)["labels"]
        crop = self.masked_crop_array(idx, instance_id)

        with self.model_lock:
            image_tensor = self._preprocess_crop(crop)
            image_feature = self.model.encode_image(image_tensor).float()
            image_feature = F.normalize(image_feature, dim=-1)

            phrases = [self.text_template.format(label) for label in labels]
            tokens = self.tokenizer(phrases).to(self.device_name)
            text_features = self.model.encode_text(tokens).float()
            text_features = F.normalize(text_features, dim=-1)

            cosine = (image_feature @ text_features.T).squeeze(0)
            probs = torch.softmax(cosine / FEATURE_SOFTMAX_TEMP, dim=0)

        cosine_np = cosine.detach().cpu().numpy()
        probs_np = probs.detach().cpu().numpy()
        rows = []
        for label, cos, prob in zip(labels, cosine_np, probs_np, strict=True):
            rows.append(
                {
                    "label": label,
                    "cosine": float(cos),
                    "prob": float(prob),
                    "is_gt": label == info["label"],
                }
            )
        rows.sort(key=lambda item: item["prob"], reverse=True)
        return {
            "frame_index": idx,
            "frame_id": self.frame_id(idx),
            "instance_id": int(instance_id),
            "gt_label": info["label"],
            "top_label": rows[0]["label"] if rows else None,
            "rows": rows,
        }


class RequestHandler(BaseHTTPRequestHandler):
    app: ScanNetMaskedCropApp

    def log_message(self, fmt: str, *args) -> None:
        print(fmt % args)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/api/meta":
                self._send_json({"scene": self.app.scene_dir.name, "frame_count": len(self.app.frame_ids)})
            elif parsed.path == "/api/frame":
                idx = self._idx(parsed)
                self._send_json(self.app.frame_meta(idx))
            elif parsed.path == "/api/image":
                idx = self._idx(parsed)
                self._send_bytes(self.app.image_jpeg(idx), "image/jpeg")
            elif parsed.path == "/api/mask":
                idx, instance_id = self._idx_instance(parsed)
                self._send_bytes(self.app.mask_png(idx, instance_id), "image/png")
            elif parsed.path == "/api/crop":
                idx, instance_id = self._idx_instance(parsed)
                self._send_bytes(self.app.crop_png(idx, instance_id), "image/png")
            elif parsed.path == "/api/clip":
                idx, instance_id = self._idx_instance(parsed)
                self._send_json(self.app.clip_result(idx, instance_id))
            else:
                self._send_error(HTTPStatus.NOT_FOUND, f"Unknown route: {parsed.path}")
        except Exception as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def _query(self, parsed) -> dict[str, list[str]]:
        return parse_qs(parsed.query)

    def _idx(self, parsed) -> int:
        query = self._query(parsed)
        idx = int(query.get("idx", ["0"])[0])
        if idx < 0 or idx >= len(self.app.frame_ids):
            raise ValueError(f"Frame index {idx} out of range [0, {len(self.app.frame_ids) - 1}]")
        return idx

    def _idx_instance(self, parsed) -> tuple[int, int]:
        query = self._query(parsed)
        idx = self._idx(parsed)
        if "instance_id" not in query:
            raise ValueError("Missing instance_id")
        return idx, int(query["instance_id"][0])

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(json.dumps(payload).encode("utf-8"), "application/json", status)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status)

    def _send_bytes(self, payload: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(int(status))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def find_free_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        result = sock.connect_ex(("127.0.0.1", preferred))
    if result != 0:
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browser probe for ScanNet masked-object-crop CLIP descriptors.")
    parser.add_argument("--scene_dir", type=Path, default=Path("data/input/ScanNet/scene0000_00"))
    parser.add_argument("--label_map_tsv", type=Path, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--text_template", default=FEATURE_TEXT_TEMPLATE)
    parser.add_argument("--min_pixels", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    app = ScanNetMaskedCropApp(
        scene_dir=scene_dir,
        label_map_path=args.label_map_tsv,
        device=args.device,
        text_template=args.text_template,
        min_pixels=args.min_pixels,
    )
    port = find_free_port(int(args.port))
    RequestHandler.app = app
    server = ThreadingHTTPServer((args.host, port), RequestHandler)
    display_host = "127.0.0.1" if args.host in {"0.0.0.0", ""} else args.host
    print(f"Loaded {scene_dir.name}: {len(app.frame_ids)} frames", flush=True)
    print(
        f"CLIP model will load lazily on first button click: {CLIP_MODEL_NAME} ({CLIP_PRETRAINED}) on {app.device_name}",
        flush=True,
    )
    print(f"Open http://{display_host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
