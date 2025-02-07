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


2025-02-07

* my ubuntu / AMOD 데이터 -> 3/4 sample 버전 testset mAP 


~~~shell
 

python ./mmrotate/tools/test.py ./my_config/config_log_sooyeon/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_ThreeFourth-le90_amod.py \
       ./work_dirs/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_ThreeFourth-le90_amod/best_mAP_epoch_28.pth \
        --eval mAP --out "./AMOD_all_test/angle0/test.pkl"  --eval-options iou_thr=0.5   --cfg-options data.test.angles=[0,] \
        --work-dir "./AMOD_all_test/angle0"

python ./mmrotate/tools/test.py ./my_config/config_log_sooyeon/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_ThreeFourth-le90_amod.py \
       ./work_dirs/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_ThreeFourth-le90_amod/best_mAP_epoch_28.pth \
        --eval mAP --out "./AMOD_all_test/angle10/test.pkl"  --eval-options iou_thr=0.5   --cfg-options data.test.angles=[10,] \
        --work-dir "./AMOD_all_test/angle10"

python ./mmrotate/tools/test.py ./my_config/config_log_sooyeon/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_ThreeFourth-le90_amod.py \
       ./work_dirs/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_ThreeFourth-le90_amod/best_mAP_epoch_28.pth \
        --eval mAP --out "./AMOD_all_test/angle20/test.pkl"  --eval-options iou_thr=0.5   --cfg-options data.test.angles=[20,] \
        --work-dir "./AMOD_all_test/angle20"

python ./mmrotate/tools/test.py ./my_config/config_log_sooyeon/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_ThreeFourth-le90_amod.py \
       ./work_dirs/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_ThreeFourth-le90_amod/best_mAP_epoch_28.pth \
        --eval mAP --out "./AMOD_all_test/angle30/test.pkl"  --eval-options iou_thr=0.5   --cfg-options data.test.angles=[30,] \
        --work-dir "./AMOD_all_test/angle30"

python ./mmrotate/tools/test.py ./my_config/config_log_sooyeon/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_ThreeFourth-le90_amod.py \
       ./work_dirs/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_ThreeFourth-le90_amod/best_mAP_epoch_28.pth \
        --eval mAP --out "./AMOD_all_test/angle40/test.pkl"  --eval-options iou_thr=0.5   --cfg-options data.test.angles=[40,] \
        --work-dir "./AMOD_all_test/angle40"

python ./mmrotate/tools/test.py ./my_config/config_log_sooyeon/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_ThreeFourth-le90_amod.py \
       ./work_dirs/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_ThreeFourth-le90_amod/best_mAP_epoch_28.pth \
        --eval mAP --out "./AMOD_all_test/angle50/test.pkl"  --eval-options iou_thr=0.5   --cfg-options data.test.angles=[50,] \
        --work-dir "./AMOD_all_test/angle50"

~~~