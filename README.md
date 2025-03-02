<p align="center">
  <img src="docs/teaser_v1.jpg" width="700">
  <h1 align="center">VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation</h1>
  <!-- <h3 align="center">
    <a href="http://naoki.io/">Naoki Yokoyama</a>, <a href="https://faculty.cc.gatech.edu/~sha9/">Sehoon Ha</a>, <a href="https://faculty.cc.gatech.edu/~dbatra/">Dhruv Batra</a>, <a href="https://www.robo.guru/about.html">Jiuguang Wang</a>, <a href="https://bucherb.github.io">Bernadette Bucher</a>
  </h3>
  <p align="center">
    <a href="http://naoki.io/portfolio/vlfm.html">Project Website</a> , <a href="https://arxiv.org/abs/2312.03275">Paper (arXiv)</a>
  </p> -->
  <p align="center">
    <a href="https://github.com/bdaiinstitute/vlfm">
      <img src="https://img.shields.io/badge/License-MIT-yellow.svg" />
    </a>
    <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/built%20with-Python3-red.svg" />
    </a>
  </p>
</p>

# :sparkles: Overview


# :hammer_and_wrench: Installation

### Getting Started

Create the conda environment:
```bash
conda_env_name=vlfm
conda create -n $conda_env_name python=3.9 -y &&
conda activate $conda_env_name
```
If you are using habitat and are doing simulation experiments, install this repo into your env with the following:
```bash
pip install -e .[habitat]
```
If you are using the Spot robot, install this repo into your env with the following:
```bash
pip install -e .[reality]
```
Install all the dependencies:
```bash
git clone git@github.com:IDEA-Research/GroundingDINO.git
git clone git@github.com:WongKinYiu/yolov7.git  # if using YOLOv7
```
Follow the original install directions for GroundingDINO, which can be found here: https://github.com/IDEA-Research/GroundingDINO.

Nothing needs to be done for YOLOv7, but it needs to be cloned into the repo.



* Should install Mobile SAM follow 
```
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

* Prepare weights of Eva-Vit-g for lavis

https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth


* frontier_exploration 
```
 pip install git+https://github.com/naokiyokoyama/frontier_exploration.git

```

* Other pkgs should be installed 

https://github.com/bdaiinstitute/vlfm/issues/32#issuecomment-2151794886

### Installing GroundingDINO (Only if using conda-installed CUDA)
Only attempt if the installation instructions in the GroundingDINO repo do not work.

To install GroundingDINO, you will need `CUDA_HOME` set as an environment variable. If you would like to install a certain version of CUDA that is compatible with the one used to compile your version of pytorch, and you are using conda, you can run the following commands to install CUDA and set `CUDA_HOME`:
```bash
# This example is specifically for CUDA 11.8
mamba install \
    cub \
    thrust \
    cuda-runtime \
    cudatoolkit=11.8 \
    cuda-nvcc==11.8.89 \
    -c "nvidia/label/cuda-11.8.0" \
    -c nvidia &&
ln -s ${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/cuda_runtime/include/*  ${CONDA_PREFIX}/include/ &&
ln -s ${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/cusparse/include/*  ${CONDA_PREFIX}/include/ &&
ln -s ${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/cublas/include/*  ${CONDA_PREFIX}/include/ &&
ln -s ${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/cusolver/include/*  ${CONDA_PREFIX}/include/ &&
export CUDA_HOME=${CONDA_PREFIX}
```

# :dart: Datasets 

## Downloading the HM3D dataset

### Matterport
First, set the following variables during installation (don't need to put in .bashrc):
```bash
MATTERPORT_TOKEN_ID=<FILL IN FROM YOUR ACCOUNT INFO IN MATTERPORT>
MATTERPORT_TOKEN_SECRET=<FILL IN FROM YOUR ACCOUNT INFO IN MATTERPORT>
DATA_DIR=</path/to/vlfm/data>

# Link to the HM3D ObjectNav episodes dataset, listed here:
# https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md#task-datasets
# From the above page, locate the link to the HM3D ObjectNav dataset.
# Verify that it is the same as the next two lines.
HM3D_OBJECTNAV=https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip
```

### Clone and install habitat-lab, then download datasets
*Ensure that the correct conda environment is activated!!*
```bash
# Download HM3D 3D scans (scenes_dataset)
python -m habitat_sim.utils.datasets_download \
  --username $MATTERPORT_TOKEN_ID --password $MATTERPORT_TOKEN_SECRET \
  --uids hm3d_train_v0.2 \
  --data-path $DATA_DIR &&
python -m habitat_sim.utils.datasets_download \
  --username $MATTERPORT_TOKEN_ID --password $MATTERPORT_TOKEN_SECRET \
  --uids hm3d_val_v0.2 \
  --data-path $DATA_DIR &&

# Download HM3D ObjectNav dataset episodes
wget $HM3D_OBJECTNAV &&
unzip objectnav_hm3d_v1.zip &&
mkdir -p $DATA_DIR/datasets/objectnav/hm3d  &&
mv objectnav_hm3d_v1 $DATA_DIR/datasets/objectnav/hm3d/v1 &&
rm objectnav_hm3d_v1.zip
```


## Downloading the MP3D dataset

https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md

https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/m3d/v1/objectnav_mp3d_v1.zip

## Downloading the Gibson dataset

* Get dataset license

https://docs.google.com/forms/d/e/1FAIpQLScWlx5Z1DM1M-wTSXaa6zV8lTFkPmTHW1LqMsoCBDWsTDjBkQ/viewform?usp=send_form


* Download

```bash
python scripts/download_mp.py --task habitat -o ./
```


# :weight_lifting: Downloading weights for various models
The weights for MobileSAM, GroundingDINO, and PointNav must be saved to the `data/` directory. The weights can be downloaded from the following links:
- `mobile_sam.pt`:  https://github.com/ChaoningZhang/MobileSAM
- `groundingdino_swint_ogc.pth`: https://github.com/IDEA-Research/GroundingDINO
- `yolov7-e6e.pt`: https://github.com/WongKinYiu/yolov7
- `pointnav_weights.pth`: included inside the [data](data) subdirectory

# :arrow_forward: Evaluation within Habitat
To run evaluation, various models must be loaded in the background first. This only needs to be done once by running the following command:
```bash
./scripts/launch_vlm_servers.sh
```
(You may need to run `chmod +x` on this file first.)
This command will create a tmux session that will start loading the various models used for VLFM and serving them through `flask`. When you are done, be sure to kill the tmux session to free up your GPU.



To evaluate on HM3D, run the following:
```bash 

BLIP2ITM_PORT=12182 GROUNDING_DINO_PORT=12181 SAM_PORT=12183 YOLOV7_PORT=12184 SSA_PORT=12185 LOG_LEVEL=INFO HF_ENDPOINT=https://hf-mirror.com python -m vlfm.run habitat.dataset.content_scenes='["svBbv1Pavdk"]' habitat_baselines.video_dir="../VLFMExp/video_dir_19_svBbv1Pavdk" habitat_baselines.rl.policy.name="HabitatITMPolicyV19"

```
To evaluate on MP3D, run the following:

```bash 

BLIP2ITM_PORT=12182 GROUNDING_DINO_PORT=12181 SAM_PORT=12183 YOLOV7_PORT=12184 SSA_PORT=12185 LOG_LEVEL=INFO HF_ENDPOINT=https://hf-mirror.com python -m vlfm.run habitat.dataset.content_scenes='["2azQ1b91cZZ"]' habitat_baselines.video_dir="../VLFMExp/mp3d/pv24_2azQ1b91cZZ" habitat_baselines.rl.policy.name="HabitatITMPolicyV24" habitat.dataset.data_path=data/datasets/objectnav/mp3d/val/val.json.gz

```

To evaluate on Gibson, run the following:
```bash 
conda activate gibson


HF_ENDPOINT=https://hf-mirror.com https_proxy=127.0.0.1:7890 SCENE_NAM
E=Corozal POLICY_NAME=SemExpITMPolicyV24 BLIP2ITM_PORT=12182 GROUNDING_DINO_PORT=12181 SAM_PORT=12183 YOLOV7_PORT=12184 S
SA_PORT=12185 python vlfm/semexp_env/eval.py


```

- Collierville
- Corozal
- Darden
- Markleeville
- Wiconisco



# :newspaper: License

VLFM is released under the [MIT License](LICENSE). This code was produced as part of Naoki Yokoyama's internship at the Boston Dynamics AI Institute in Summer 2023 and is provided "as is" without active maintenance. For questions, please contact [Naoki Yokoyama](http://naoki.io) or [Jiuguang Wang](https://www.robo.guru).

# :black_nib: Citation

If you use VLFM in your research, please use the following BibTeX entry.

```
@inproceedings{yokoyama2024vlfm,
  title={VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation},
  author={Naoki Yokoyama and Sehoon Ha and Dhruv Batra and Jiuguang Wang and Bernadette Bucher},
  booktitle={International Conference on Robotics and Automation (ICRA)},
  year={2024}
}



