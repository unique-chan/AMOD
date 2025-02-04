2025-02-04

~~~shell
cd ../..
DATA_ROOT="data/AMOD_MOCK/"
python mmrotate/tools/train.py my_config/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amod.py \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=1 data.samples_per_gpu=2
~~~