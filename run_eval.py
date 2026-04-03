from typing import Dict
from datetime import datetime
from pathlib import Path
import argparse
import wandb
import torch
import numpy as np
import time
import yaml
import uuid
import gc
import os
import shutil
import random

from ovo import io_utils, eval_utils
from ovo.ovomapping import OVOSemMap
from ovo.ovo import OVO


DATASET_DIRS = {
    "replica": "Replica",
    "scannet": "ScanNet",
}
CONFIG_DIR = Path("configs")
INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/output")


def canonical_dataset_name(dataset_name: str) -> str:
    return DATASET_DIRS[dataset_name.lower()]


def load_dataset_info(dataset_name: str, dataset_info_file: str) -> tuple[str, Dict]:
    dataset_dir = canonical_dataset_name(dataset_name)
    with open(CONFIG_DIR / f"{dataset_name.lower()}_{dataset_info_file}", "r") as f:
        return dataset_dir, yaml.full_load(f)

def load_representation(scene_path: Path, eval: bool=False, debug_info: bool=False) -> OVO:
    config = io_utils.load_config(scene_path / "config.yaml", inherit=False)
    submap_ckpt = torch.load(scene_path /"ovo_map.ckpt" )
    map_params = submap_ckpt["map_params"]
    config["semantic"]["verbose"] = False 
    ovo = OVO(config["semantic"],None, config["data"]["scene_name"], eval=eval, device=config.get("device", "cuda"))
    ovo.restore_dict(submap_ckpt["ovo_map_params"], debug_info=debug_info)
    return ovo, map_params


def compute_scene_labels(scene_path: Path, dataset_name: str, scene_name: str, data_path:str, dataset_info: Dict) -> None:

    ovo, map_params = load_representation(scene_path, eval=True)
    pcd_pred = map_params["xyz"]
    points_obj_ids = map_params["obj_ids"]

    _, pcd_gt = io_utils.load_scene_data(dataset_name, scene_name, data_path, dataset_info, False)
    classes = dataset_info["class_names"] if dataset_info.get("map_to_reduced", None) is None else dataset_info["class_names_reduced"]
    pred_path = scene_path.parent / dataset_info["dataset"]
    os.makedirs(pred_path, exist_ok=True)
    pred_path = pred_path / (scene_name+".txt")

    # It may happen that all the points associated to an object where prunned, such that the number of unique labels in points_obj_ids, is different from the number of semantic module instances
    print("Computing predicted instances labels ...")

    instances_info = ovo.classify_instances(classes)

    print("Matching instances to ground truth mesh ...")
    mesh_instance_labels, mesh_instances_masks, _ = eval_utils.match_labels_to_vtx(points_obj_ids[:,0], pcd_pred, pcd_gt)
    
    map_id_to_idx = {id: i for i, id in enumerate(ovo.objects.keys())}
    mesh_semantic_labels = instances_info["classes"][np.vectorize(map_id_to_idx.get)(mesh_instance_labels)]
    instances_info["masks"] = mesh_instances_masks.int().numpy()

    print(f"Writing prediction to {pred_path}!")
    io_utils.write_labels(pred_path, mesh_semantic_labels)
    io_utils.write_instances(scene_path.parent, scene_name, instances_info)

    ovo.cpu()
    del ovo


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_scene_config(scene: str, dataset: str, config_path: str, slam_module: str = None, frame_limit: int = None) -> Dict:
    config = io_utils.load_config(config_path)
    if slam_module is not None:
        config["slam"]["slam_module"] = slam_module

    dataset_dir = canonical_dataset_name(dataset)
    config_dataset = io_utils.load_config(CONFIG_DIR / f"{dataset.lower()}.yaml")
    io_utils.update_recursive(config, config_dataset)

    config.setdefault("data", {})
    config["data"]["scene_name"] = scene
    config["data"]["input_path"] = str(INPUT_DIR / dataset_dir / scene)
    if frame_limit is not None:
        config["data"]["frame_limit"] = frame_limit
    return config


def run_scene(scene: str, dataset: str, experiment_name: str, tmp_run: bool = False, slam_module: str = None, frame_limit: int = None, config_path: str = "configs/ovo.yaml") -> None:

    config = build_scene_config(scene, dataset, config_path, slam_module=slam_module, frame_limit=frame_limit)

    output_path = OUTPUT_DIR / canonical_dataset_name(dataset)

    if tmp_run:
        output_path = output_path / "tmp"

    output_path = output_path / experiment_name / scene

    if os.getenv('DISABLE_WANDB') == 'true':
        config["use_wandb"] = False
    elif config["use_wandb"]:
        wandb.init(
            project=config["project_name"],
            config=config,
            dir="data/output/wandb",
            group=config["data"]["scene_name"]
            if experiment_name != ""
            else experiment_name,
            name=f'{config["data"]["scene_name"]}_{time.strftime("%Y%m%d_%H%M%S", time.localtime())}_{str(uuid.uuid4())[:5]}',
        )

    setup_seed(config["seed"])
    gslam = OVOSemMap(config, output_path=output_path)
    gslam.run()

    if tmp_run:
        final_path = OUTPUT_DIR / canonical_dataset_name(dataset) / experiment_name / scene
        shutil.move(output_path, final_path)

    if config["use_wandb"]:
        wandb.finish()
    print("Finished run.✨")

def main(args):
    if args.experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M")
        tmp_run = True
    else:
        assert len(args.experiment_name) > 0, "Experiment name cannot be '' "
        experiment_name = args.experiment_name
        tmp_run = False

    dataset_dir = canonical_dataset_name(args.dataset_name)
    experiment_path = OUTPUT_DIR / dataset_dir / experiment_name

    if args.scenes_list is not None:
        with open(args.scenes_list, "r") as f:
            scenes = f.read().splitlines() 
    else:
        scenes = args.scenes

    if len(scenes) == 0 or args.segment or args.eval:
        _, dataset_info = load_dataset_info(args.dataset_name, args.dataset_info_file)

        if len(scenes) == 0:
            scenes = dataset_info["scenes"]

    for scene in scenes:        
        input_path = INPUT_DIR / dataset_dir / scene
        if args.run:
            t0 = time.time()
            run_scene(scene, args.dataset_name, experiment_name, tmp_run=tmp_run, slam_module=args.slam_module, frame_limit=args.frame_limit, config_path=args.config_path)
            t1 = time.time()
            print(f"Scene {scene} took: {t1-t0:.2f}")
        gc.collect()
 
    if args.segment: 
        data_path = str(INPUT_DIR)
        for scene in scenes:    
            scene_path = experiment_path / scene
            compute_scene_labels(scene_path, args.dataset_name, scene, data_path, dataset_info)

    if args.eval:
        gt_path = input_path.parent / "semantic_gt"
        eval_utils.eval_semantics(experiment_path / dataset_info["dataset"], gt_path, scenes, dataset_info, ignore_background=args.ignore_background)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Arguments to run and evaluate over a dataset')
    parser.add_argument('--dataset_name', help="Dataset used. Choose either `Replica`, `ScanNet`")
    parser.add_argument('--scenes', nargs="+", type=str, default=[], help=" List of scenes from given dataset to run.  If `--scenes_list` is set, this flag will be ignored.")
    parser.add_argument('--scenes_list',type=str, default=None, help="Path to a txt containing a scene name on each line. If set, `--scenes` is ignored. If neither `--scenes` nor `--scenes_list` are set, the scene list will be loaded from `configs/<dataset>_<dataset_info_file>`")
    parser.add_argument('--dataset_info_file',type=str, default="eval.yaml")
    parser.add_argument('--experiment_name', default=None, type=str)
    parser.add_argument('--run', action='store_true', help="If set, compute the final metrics, after running OVO and segmenting.")
    parser.add_argument('--segment', action='store_true', help="If set, use the reconstructed scene to segment the gt point-cloud, after running OVO.")
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--ignore_background', action='store_true',help="If set, does not use background ids from eval_info to compute metrics.")
    parser.add_argument('--slam_module', type=str, default=None, help="Override slam backend, e.g. vanilla or orbslam.")
    parser.add_argument('--frame_limit', type=int, default=None, help="Override number of input frames to process.")
    parser.add_argument('--config_path', type=str, default="configs/ovo.yaml", help="Base OVO config file to load.")
    args = parser.parse_args()
    main(args)
