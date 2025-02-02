<h1 align="center" >
🕹️ Official Experiment Kit for AMOD
</h1>

<h3 align="center">
💬 Here, AMOD refers to our dataset, <u>A</u>rma3 <u>M</u>ilitary <u>O</u>bject <u>D</u>etection (🪖) in optical satellite imagery (🛰️)!
</h3>

<p align="center">
  <a href="#"><img alt="Python3.7+" src="https://img.shields.io/badge/Python-3.7+-blue?logo=python&logoColor=white"></a>
  <a href="#"><img alt="PyTorch1.5~1.10.2" src="https://img.shields.io/badge/PyTorch-≥1.5, ≤1.10-orange?logo=pytorch&logoColor=white"></a>
  <a href="#"><img alt="MMDetection2.28.2" src="https://img.shields.io/badge/MMDetection-2.28.2-red?logo=mmlab&logoColor=white"></a>
  <a href="#"><img alt="MMRotate0.3.4" src="https://img.shields.io/badge/MMRotate-0.3.4-hotpink?logo=mmlab&logoColor=white"></a>
  <a href="#"><img alt="ARMA3" src="https://img.shields.io/badge/Game-ARMA3-red?logo=steam"></a>
  <a href="#"><img alt="MIT" src="https://img.shields.io/badge/License-MIT-green?logo=MIT"></a>
</p>

<p align="center">
  <b>Yechan Kim</b> and
  <b>SooYeon Kim</b>
</p>

### This repo includes:
* Training & test code for AMOD
* **[NOTE]** We only consider single-machine multi-GPU scenarios and do not address cases involving multiple nodes and using Slurm. For explanations related to multi-machine setups and Slurm, please refer to the official documentation of [MMDetection](https://mmdetection.readthedocs.io/en/v2.28.2/) and [MMRotate](https://mmrotate.readthedocs.io/en/v0.3.4/).

### Preliminaries:


* **Step 1**. Create a conda environment with Python 3.8 and activate it.
    ~~~shell
    conda create --name amodexpkit python=3.8 -y
    conda activate amodexpkit
    ~~~

* **Step 2.** Install PyTorch with TorchVision following [official instructions](https://pytorch.org/get-started/locally/). The below is an example. We do not recommend PyTorch 2.x for our code.
    ~~~shell
    pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/torch_stable.html  
    ~~~

* **Step 3.** Install `MMDetection (v2.28.2)` ([v2.28.2](https://mmdetection.readthedocs.io/en/v2.28.2/) is the latest version suited of 2024 to MMRotate).
    ~~~shell
    # ⚠️ No need to clone MMDet (e.g. "git clone -b 2.x https://github.com/open-mmlab/mmdetection; rm -rf mmdetection/.git"). Already cloned! 
    pip install -U openmim==0.3.9
    mim install mmcv-full==1.7.2
    pip install -v -e mmdetection/
    ~~~

* **Step 4.** Install `MMRotate (v0.3.4)` ([v0.3.4](https://mmrotate.readthedocs.io/en/v0.3.4/) is the latest version of 2024). 
    ~~~shell
    # ⚠️ No need to clone MMRot (e.g. "git clone https://github.com/open-mmlab/mmrotate; rm -rf mmrotate/.git"). Already cloned!
    pip install -v -e mmrotate/
    ~~~

    <details>
      <summary> To verify whether MMRotate is installed correctly, you may try the following things: </summary>
    
    * Download config and checkpoint files.
        ~~~shell
        mim download mmrotate --config oriented_rcnn_r50_fpn_1x_dota_le90 --dest .
        ~~~
    * Verify the inference demo.
        ~~~shell
        python mmrotate/demo/image_demo.py \
        mmrotate/demo/demo.jpg oriented_rcnn_r50_fpn_1x_dota_le90.py \
        oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth --out-file result.jpg
        ~~~
    * If **result.jpg** is generated correctly, it means that the environment is set up properly.
    </details>

### Test a model:
You can use the following commands to infer a dataset.
~~~shell
# Single-gpu
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]

# Multi-gpu
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [optional arguments]
~~~

Examples:

🚧 Under construction!


### Train a model:
You can use the following commands to train a model from the dataset.
~~~shell
# Single-gpu
python tools/train.py ${CONFIG_FILE} [optional arguments]

# Multi-gpu
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
~~~

**Optional arguments** are:
* `--no-validate` (not recommended): No validation (evaluation) during the training.
* `--work-dir ${WORK_DIR}`: Override the working directory specified in the config file.
* `--resume-from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
  * Difference between **resume-from** and **load-from**: resume-from loads both the model weights and optimizer status, and the epoch is also inherited from the specified checkpoint. It is usually used for resuming the training process that is interrupted accidentally. load-from only loads the model weights and the training epoch starts from 0. It is usually used for finetuning.

**Launch multiple jobs on a single machine**: If you launch multiple jobs on a single machine, e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs, you need to specify different ports (29500 by default) for each job to avoid communication conflict.
~~~shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
~~~

Examples:

🚧 Under construction!
