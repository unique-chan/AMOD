2025-02-06

* my ubuntu / EO, IR, day, night mAP 비교를 위한 실험 
* batch size = 2 , epoch = 30


~~~shell
 
python  ./mmrotate/tools/train.py 
        ./my_config/config_log_sooyeon/orientedrcnn_swinS_fpn_30epochs_le90-multisize-rrandomcrop_amod_EO_day.py

python  ./mmrotate/tools/train.py 
        ./my_config/config_log_sooyeon/orientedrcnn_swinS_fpn_30epochs_le90-multisize-rrandomcrop_amod_EO_night.py

python  ./mmrotate/tools/train.py 
        ./my_config/config_log_sooyeon/orientedrcnn_swinS_fpn_30epochs_le90-multisize-rrandomcrop_amod_IR_day.py

python  ./mmrotate/tools/train.py 
        ./my_config/config_log_sooyeon/orientedrcnn_swinS_fpn_30epochs_le90-multisize-rrandomcrop_amod_IR_night.py

~~~


* my ubuntu / AMOD 데이터 -> 1/2 sample 버전 실험 
* batch size = 4 , epoch = 30


~~~shell
 
python  ./mmrotate/tools/train.py 
        ./my_config/config_log_sooyeon/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_Half-le90_amod.py

~~~


* gpuadmin@172.26.49.150 / AMOD 데이터 -> 3/4 sample 버전 실험 
* batch size = 4 , epoch = 30


~~~shell

chmod +x ./mmrotate/tools/dist_train.sh

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./mmrotate/tools/dist_train.sh /home/gpuadmin/SY/AMOD-ExpKit/my_config/config_log_sooyeon/orientedrcnn_swinS_fpn_30epochs_le90-multisize-rrandomcrop_amod_EO_day.py 4

~~~