<p align="center">
    <img alt="Welcome" src="./mmrotate/Logo.png" />
</p>

<h1 align="center" >
💬 Official Experiment Kit for AMOD
</h1>

<!--
<h3 align="center">
 Here, AMOD refers to our dataset, <u>A</u>rma3 <u>M</u>ilitary <u>O</u>bject <u>D</u>etection (🪖) in optical satellite imagery (🛰️)!
</h3>
-->

<p align="center">
  <a href="#"><img alt="Python3.7+" src="https://img.shields.io/badge/Python-3.7+-blue?logo=python&logoColor=white"></a>
  <a href="#"><img alt="PyTorch1.5~1.10.2" src="https://img.shields.io/badge/PyTorch-≥1.5, ≤1.10-orange?logo=pytorch&logoColor=white"></a>
  <a href="#"><img alt="MMDetection2.28.2" src="https://img.shields.io/badge/MMDetection-2.28.2-red?logo=mmlab&logoColor=white"></a>
  <a href="#"><img alt="MMRotate0.3.4" src="https://img.shields.io/badge/MMRotate-0.3.4-hotpink?logo=mmlab&logoColor=white"></a>
  <a href="#"><img alt="ARMA3" src="https://img.shields.io/badge/Game-ARMA3-red?logo=steam"></a>
  <a href="#"><img alt="Apache-2.0" src="https://img.shields.io/badge/License-Apache 2.0-green?logo=Apache"></a>
</p>

<p align="center">
  <b>Yechan Kim</b> and
  <b>SooYeon Kim</b>
</p>

### This repo includes:
* Dataset link for AMOD! (*To do*)
* Training & test code for AMOD!
* **[NOTE1]** We only consider single-machine multi-GPU scenarios and do not address cases involving multiple nodes and using Slurm. For explanations related to multi-machine setups and Slurm, please refer to the official documentation of [MMDetection](https://mmdetection.readthedocs.io/en/v2.28.2/) and [MMRotate](https://mmrotate.readthedocs.io/en/v0.3.4/).
* **[NOTE2]** Please note that our code is developed and tested on Linux. Windows is not officially supported.

### Announcements:
* Feb 4 2025: You can run mock tests using the small-scale, tailored data we provide. See `data/AMOD_MOCK`!


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

* **Step 3.** Install `MMDetection (v2.28.2)` ([v2.28.2](https://mmdetection.readthedocs.io/en/v2.28.2/) is the latest version suited to MMRotate of 2024).
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

* **Step 5.** Install `Tensorboard` for visualizing learning curves.
    ~~~shell
    pip install tensorboard==2.9.1
    # If AttributeError: module 'setuptools._distutils' has no attribute 'version' ->
    pip install setuptools==59.5.0
    ~~~
 

### Test a model:
You can use the following commands to infer a dataset.
~~~shell
# Single-gpu
python mmrotate/tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]

# Multi-gpu
./mmrotate/tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [optional arguments]
~~~

Examples:

* Test OrientedRCNN with Swin-S pretrained on AMOD
  * Please download our pretrained weights [here](#)!
  * Run the following command to get AP50, AP75 of test split of AMOD:
  ~~~shell
  python mmrotate/tools/test.py my_config/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amod.py \
         [path/to/pretrained_weights/*.pth] --eval mAP --eval-options iou_thr=0.5,0.75
  ~~~
  * <details>
        <summary>If you want to change the data root path through Python arguments?</summary>
      
    * ~~~shell
      DATA_ROOT="data/AMOD_MOCK/"
      python mmrotate/tools/test.py my_config/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amod.py \
               [path/to/pretrained_weights/*.pth] --cfg-options data.test.data_root="$DATA_ROOT" --eval mAP \
               --eval-options iou_thr=0.5,0.75
      ~~~ 
    
        </details>

  * <details>
      <summary>If you want to get a confusion matrix?</summary>
    
    * You have to save the prediction results as a `.pkl` file using `--out` in `test.py`. 
    ~~~shell
    DATA_ROOT="data/AMOD_MOCK/"
    python mmrotate/tools/test.py my_config/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amod.py \
               [path/to/pretrained_weights/*.pth] --cfg-options data.test.data_root="$DATA_ROOT" --eval mAP \
               --eval-options iou_thr=0.5 --out "./test.pkl" 
    ~~~    

    * Then, run the following code:
    ~~~shell
    DATA_ROOT="data/AMOD_MOCK/"
    mkdir ./confusion_matrix_results
    python mmrotate/tools/analysis_tools/confusion_matrix.py \
             my_config/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amod.py \
             ./test.pkl \
             ./confusion_matrix_results --color-theme 'viridis' --show \
             --tp-iou-thr 0.5 \
             --cfg-options data.test.data_root="$DATA_ROOT" 
     ~~~ 
    
      </details>
  
  
### Train a model:
You can use the following commands to train a model from the dataset.
~~~shell
# Single-gpu
python mmrotate/tools/train.py ${CONFIG_FILE} [optional arguments]

# Multi-gpu
./mmrotate/tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
~~~

**Optional arguments** are:
* `--no-validate` (not recommended): No validation (evaluation) during the training.
* `--work-dir ${WORK_DIR}`: Override the working directory specified in the config file.
* `--resume-from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
  * Difference between **resume-from** and **load-from**: resume-from loads both the model weights and optimizer status, and the epoch is also inherited from the specified checkpoint. It is usually used for resuming the training process that is interrupted accidentally. load-from only loads the model weights and the training epoch starts from 0. It is usually used for finetuning.

**Launch multiple jobs on a single machine**: If you launch multiple jobs on a single machine, e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs, you need to specify different ports (29500 by default) for each job to avoid communication conflict.
~~~shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./mmrotate/tools/dist_train.sh ${CONFIG_FILE} 4 [optional arguments]
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./mmrotate/tools/dist_train.sh ${CONFIG_FILE} 4 [optional arguments]
~~~

Examples:

* Train OrientedRCNN with Swin-S on train set of AMOD. 
  * Tip1: Please read carefully both `Preferred` and `Bad` examples! (🚨)
  * Tip2: You can directly modify config file instead of using `--cfg-options` (Do not over-use)!
  * <details>
    <summary> Preferred example (👌): </summary>

    * If you want to train three models with look angles [0,10], [10,20], [50] respectively, on AMOD for 1 epoch with batch size 1? 
    * **Solution**:
      * 1. Copy the file `orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amod.py` located in the my_config directory three times.
      * 2. Rename each file as follows:
        * `orientedrcnn_swinS_fpn_angle0,10_1epoch_le90_amod.py`
        * `orientedrcnn_swinS_fpn_angle10,20_1epoch_le90_amod.py`
        * `orientedrcnn_swinS_fpn_angle50_1epoch_le90_amod.py`
      * 3. Properly modify variables `angles` (line 3) and `runner.max_epochs` (line 19) in each file.
      * 4. Finally, run the below shell code:
    
    ~~~shell
    DATA_ROOT="/media/yechani9/T7\Shield/AMOD_V1_FINAL_OPTICAL/" # AMOD DATA ROOT PATH!!!
    
    python mmrotate/tools/train.py my_config/orientedrcnn_swinS_fpn_angle0,10_1epoch_le90_amod.py \
     --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
                   data.samples_per_gpu=1
    python mmrotate/tools/train.py my_config/orientedrcnn_swinS_fpn_angle10,20_1epoch_le90_amod.py \
     --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
                   data.samples_per_gpu=1
    python mmrotate/tools/train.py my_config/orientedrcnn_swinS_fpn_angle50_1epoch_le90_amod.py \
     --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
                   data.samples_per_gpu=1
    ~~~
    
    * Tip: You can remove `DATA_ROOT` and `--cfg-options ...` in the above shell code if you also properly modify variables `data_root` (in line 4) and `data.samples_per_gpu` (in line 79) in each config file.

  </details>

  * <details>
    <summary> Another example: </summary>

    * If you want to train three models with look angles [0,10], [10,20], [50] respectively, on AMOD for 1 epoch? The below is a bad example!
  
    ~~~shell
    DATA_ROOT="/media/yechani9/T7\Shield/AMOD_V1_FINAL_OPTICAL/"

    ANGLES="0,10"
    python mmrotate/tools/train.py my_config/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amod.py \
     --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
                   data.train.angles="$ANGLES" data.val.angles="$ANGLES" \
                   runner.max_epochs=1 data.samples_per_gpu=1
    # ... almost same shell code be here for look angles [10,20] 
    # ... almost same shell code be here for look angle [50] 
    ~~~

    * Reason: Using `--cfg-options` dynamically with the same config file **_while keeping the same working directory_** can lead to serious issues, such as experiment results being overwritten and difficulties in tracking which modifications were applied to each experiment. This becomes especially problematic when modifying crucial parameters like `ANGLES`, as it makes it nearly impossible to trace back the exact configurations that led to specific results.

    * **Solution**: You have to separate `--work-dir` (working directories) per each experiment.
    
    ~~~shell
    DATA_ROOT="/media/yechani9/T7\Shield/AMOD_V1_FINAL_OPTICAL/"

    ANGLES="0,10"
    python mmrotate/tools/train.py my_config/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amod.py \
     --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
                   data.train.angles="$ANGLES" data.val.angles="$ANGLES" \
                   runner.max_epochs=1 data.samples_per_gpu=1
                   --work-dir work_dirs/orientedrcnn_swinS_fpn$ANGLES
    # ... almost same shell code be here for look angles [10,20] 
    # ... almost same shell code be here for look angle [50] 
    ~~~
  
    or use more advanced shell script (with `for` iteration) like:
    ~~~shell
    DATA_ROOT="/media/yechani9/T7\Shield/AMOD_V1_FINAL_OPTICAL/"
    for ANGLES in 0,10 10,20 50; do
        python mmrotate/tools/train.py my_config/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amod.py \
         --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
                       data.train.angles="$ANGLES" data.val.angles="$ANGLES" \
                       runner.max_epochs=1 data.samples_per_gpu=1
                       --work-dir work_dirs/orientedrcnn_swinS_fpn$ANGLES
    done
    ~~~
    
    Still, we recommend following the previous **good example**!
  </details>

  
