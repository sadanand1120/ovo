# Data
## Pretrained models
Run the setup helper from the repo root:

```
./download_ckpts.sh
```

It downloads the manual checkpoints into the expected locations:

* SAM2.1 checkpoint:
```
/<ovo_abs_path>/data/input/sam_ckpts/sam2.1_hiera_large.pt
```

* CLIP merging weights predictor:
```
/<ovo_abs_path>/data/input/weights_predictor/base/model.pt
```

The `hparams.yaml` for the predictor is already tracked in the repo. Hugging Face models used by `open_clip` / Perception Encoder are not downloaded by this script and will use the default global Hugging Face cache location on first use.

## Datasets
### Replica
Following OpenNeRF, we use Replica dataset trajectories processed for NICE-SLAM, and their computed semantic GT.
We provide the GT inside `./data/input/replica_gt_semantics` .

The expected dataset structure is:
``` 
/<ovo_path>/data/input/Datasets/Replica/
     -> semantic_gt/
        -> office0.txt
        .
        .
        .
        -> room2.txt
     -> office0/
         -> results/
         -> traj.txt
     -> office0_mesh.ply
     .
     .
     .    
     -> scene0704_01/
     -> room2/
     -> room2_mesh.ply
```

To download the data and link the semantic GT (you can also move or copy), run in the terminal:
```
cd <ovo_abs_path>
mkdir -p ./data/input/Datasets && cd ./data/input/Datasets/
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip && unzip Replica.zip
rm Replica.zip
cd Replica
ln -s /<ovo_abs_path>/data/input/replica_semantic_gt/ semantic_gt
```

### ScanNet
The expected dataset structure is:
``` 
/<ovo_path>/data/input/Datasets/ScanNet/
     -> semantic_gt/
        -> scene0011_00.txt
        .
        .
        .
        -> scene0704_01.txt
     -> scannet200_gt/
        -> scene0011_00.txt
        .
        .
        .
        -> scene0704_01.txt
     -> scene0011_00/
         -> color/
         -> depth/
         -> pose/
         -> intrinsic/
         -> scene0011_00_vh_clean_2.labels.ply
     .
     .
     .    
     -> scene0704_01/
```

* Follow the official repository https://github.com/ScanNet/ScanNet instructions to obtain a download link. Then download and decode the validation split to obtain two folders like:
``` 
/<ScanNet_data_path>/ 
 -> scans/
     -> scene0011_00/
     .
     .
     .
     -> scene0704_01/
 -> data/val/
     -> scene0011_00/
         -> color/
         -> depth/
         -> pose/
         -> intrinsic/
     .
     .
     .    
     -> scene0704_01/
```

* Then, link GT point-clouds to `data` folder, and extract ScanNet20 semantic gt in `data/semantic_gt` running:
```
cd /<ovo_path>/
conda activate ovo
python scripts/scannet_preprocess.py --data_path /<ScanNet_data_path>/  --link_pcds
```

* (Optional) Extract ScanNet200 semantic gt:
  * Follow instructions in `/<ScanNet_repo_path>/BenchmarkScripts/ScanNet200/ReadMe.md` to compute ScanNet200 labels. You should execute something like:
    ```
    cd /<ScanNet_repo_path>/BenchmarkScripts/ScanNet200/
    create -n scannet200 python=3.8
    conda activate scannet200
    pip install -r requirements.txt
    python preprocess_scannet200.py --dataset_root /<ScanNet_data_path>/scans --output_root  /<ScanNet_data_path>/scannet200 --label_map_file /<ScanNet_repo_path>/Tasks/Benchmark/
    ```
    `preprocess_scannet200.py` by default preprocess all scenes in `scans`. If you want to preprocess only the validation split, you can add a `return` after lines 31 and 37 of `preprocess_scannet200.py` .
  * Then, run the following script to extract scannet200 labels into txts.
    ```
    cd /<ovo_path>/
    conda activate ovo
    python scripts/scannet_preprocess.py --data_path /<ScanNet_data_path>/  --scannet200
    ```

* Finaly move or link `data` folder to `/<ovo_path/`:
```
    ln -s /<ScanNet_data_path>/data/val/ /<ovo_path/data/input/Datasets/ScanNet
```
