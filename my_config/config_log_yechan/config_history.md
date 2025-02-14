원격 서버 공간을 내 PC로 마운트 하는 법
~~~shell
sudo apt update
sudo apt install sshfs

mkdir -p ~/희망하는_폴더명

sshfs [원격서버_사용자명]@[원격서버_IP주소]:/원격/데이터/경로  ~/아까만든_폴더명
원격서버 비밀번호 입력
~~~

마운트 후, 우분투 디스크 rw 권한 강제 부여
~~~
sudo chmod a+rwx MLV-2TB
~~~

원격 서버에서 Tensorboard 키고, 내 PC에서 보는 법
~~~shell
# 원격 서버에서 할 일
tensorboard --logdir=path/to/log/dir --port=[원격서버에서 임의 지정한 포트번호=A]
# 내 PC에서 할 일
ssh -NfL  localhost:[내PC에서 임의 지정한 포트번호=B]:localhost:[A] [원격PC-ID]@[원격PC-IP주소]
# 내 PC에서 포트끼리 충돌이 난 경우, 기존 포트 삭제
lsof -i :[포트번호]
################################################################################
# COMMAND   PID   USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
# python3  1234   user   10u  IPv4  12345      0t0  TCP *:http-alt (LISTEN)
################################################################################
# PID 1234가 해당 [포트번호]를 점유하고 있음... 따라서 ->
# kill -9 1234
~~~


데이터 위치
* `mlv` @ `172.26.19.210`: mlv210/ADD데이터셋/AMOD_EO_final
* `yechani7` @ `172.26.19.26`: /media/yechani7/b6a6d52a-b20a-4e5a-a3d1-61770bbc9edc/AMOD_V1_FINAL_OPTICAL

모의 테스트

~~~shell
cd ../..
DATA_ROOT="data/AMOD_MOCK/"
python mmrotate/tools/train.py my_config/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90-multisize-rrandomcrop_amod.py \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=1 data.samples_per_gpu=2
~~~

모의 테스트 2 (Fine-grained)
~~~shell
cd ../..
DATA_ROOT="data/AMOD_MOCK/"
python mmrotate/tools/train.py my_config/config_log_yechan/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amodFG.py \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=1 data.samples_per_gpu=2
~~~

2025-02-04 

* 멀티 GPU 훈련 실험 -> 성공 (172.26.19.26)

~~~shell
DATA_ROOT="/media/yechani7/b6a6d52a-b20a-4e5a-a3d1-61770bbc9edc/AMOD_V1_FINAL_OPTICAL/"
chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1 PORT=29500 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amod.py 2 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=1 data.samples_per_gpu=4
~~~

2025-02-05

* RRandomcrop + Multiscale -> Train/Val -> 성공 (172.26.19.26)

~~~shell
DATA_ROOT="/media/yechani7/b6a6d52a-b20a-4e5a-a3d1-61770bbc9edc/AMOD_V1_FINAL_OPTICAL/"
chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1 PORT=29500 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90-multisize_rrandomcroptest_amod.py 2 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4
~~~

* 멅티 GPU Test 코드 -> 성공 (내 로컬 PC)

~~~shell
#DATA_ROOT="/media/yechani7/b6a6d52a-b20a-4e5a-a3d1-61770bbc9edc/AMOD_V1_FINAL_OPTICAL/"
cd ../..
DATA_ROOT="data/AMOD_MOCK/"
chmod +x ./mmrotate/tools/dist_test.sh
CUDA_VISIBLE_DEVICES=0 PORT=29501 ./mmrotate/tools/dist_test.sh \
  my_config/config_log_yechan/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amod.py \
  my_config/config_log_yechan/best_mAP_epoch_12.pth \
 1 --cfg-options data.test.data_root="$DATA_ROOT" --eval mAP --eval-options iou_thr=0.5,0.75
~~~

* Confusion Matrix 그리기 (내 로컬 PC)
1) test.pkl 파일 저장하기
~~~shell
cd ../..
DATA_ROOT="data/AMOD_MOCK/"
python mmrotate/tools/test.py my_config/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amod.py \
           my_config/config_log_yechan/best_mAP_epoch_12.pth --cfg-options data.test.data_root="$DATA_ROOT" --eval mAP \
           --out "./test.pkl"  --eval-options iou_thr=0.5
~~~

2) 저장한 test.pkl 파일을 활용하여 confusion matrix 그리기
~~~shell
# color-theme? Blues, Greens, Oranges, Reds, viridis, plasma, inferno, magma, cividis, Greys
cd ../..
mkdir ./confusion_matrix_results
DATA_ROOT="data/AMOD_MOCK/"
python mmrotate/tools/analysis_tools/confusion_matrix.py \
         my_config/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amod.py \
         "./test.pkl" \
         "./confusion_matrix_results" --color-theme 'plasma' --show \
         --tp-iou-thr 0.5 \
         --cfg-options data.test.data_root="$DATA_ROOT" 
~~~


* 실험 (172.26.49.151) / 4 GPU / Angle별 실험
~~~shell
DATA_ROOT="/MLV-2TB/AMOD_V1_FINAL_OPTICAL/"
chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/orientedrcnn_swinS_fpn_angle0_30epochs_le90_amod.py 4 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4
~~~
~~~shell
DATA_ROOT="/MLV-2TB/AMOD_V1_FINAL_OPTICAL/"
chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/orientedrcnn_swinS_fpn_angle10_30epochs_le90_amod.py 4 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4
~~~
~~~shell
DATA_ROOT="/MLV-2TB/AMOD_V1_FINAL_OPTICAL/"
chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/orientedrcnn_swinS_fpn_angle20_30epochs_le90_amod.py 4 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4
~~~

~~~shell
DATA_ROOT="/MLV-2TB/AMOD_V1_FINAL_OPTICAL/"
chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/orientedrcnn_swinS_fpn_angle30_30epochs_le90_amod.py 4 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4
~~~
~~~shell
DATA_ROOT="/MLV-2TB/AMOD_V1_FINAL_OPTICAL/"
chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/orientedrcnn_swinS_fpn_angle40_30epochs_le90_amod.py 4 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4
~~~
~~~shell
DATA_ROOT="/MLV-2TB/AMOD_V1_FINAL_OPTICAL/"
chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/orientedrcnn_swinS_fpn_angle50_30epochs_le90_amod.py 4 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4
~~~

2025-02-06

* 172.26.19.26 train 데이터를 1/6로 줄여보자. 결과는? (val/test는 상동)

~~~shell
DATA_ROOT="/media/yechani7/b6a6d52a-b20a-4e5a-a3d1-61770bbc9edc/AMOD_V1_FINAL_OPTICAL/"
chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1 PORT=29500 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_OneSixth-le90_amod.py 2 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4
~~~

* 실험 아직 안돌림... 돌려야 함. (데이터를 1/2, 3/4으로 줄이면?) -> 수연이가 돌리는 중...
~~~shell
DATA_ROOT="/media/yechani7/b6a6d52a-b20a-4e5a-a3d1-61770bbc9edc/AMOD_V1_FINAL_OPTICAL/"
chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1 PORT=29500 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_Half-le90_amod.py 2 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4
~~~

~~~shell
DATA_ROOT="/media/yechani7/b6a6d52a-b20a-4e5a-a3d1-61770bbc9edc/AMOD_V1_FINAL_OPTICAL/"
chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1 PORT=29500 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_ThreeFourth-le90_amod.py 2 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4
~~~

* 172.26.19.172 -> 에폭을 더 늘려볼까? 100 에폭 (LR 업데이트 설정은 그대로...)

~~~shell
DATA_ROOT="/media/t5evo/data/AMOD_V1_FINAL_OPTICAL/"
python mmrotate/tools/train.py  my_config/config_log_yechan/orientedrcnn_swinS_fpn_angle_all_100epochs_le90_amod.py \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=100 data.samples_per_gpu=4
~~~

2025-02-07 ~ Fine-grained Object Detection 실험 결과 공유

* 172.26.19.26
~~~shell
DATA_ROOT="/media/yechani7/b6a6d52a-b20a-4e5a-a3d1-61770bbc9edc/AMOD_V1_FINAL_OPTICAL/"
chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1 PORT=29500 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodFG/orientedrcnn_swinS_fpn_angle50_30epochs_le90_amodFG.py 2 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4

chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1 PORT=29500 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodFG/orientedrcnn_swinS_fpn_angle40_30epochs_le90_amodFG.py 2 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4

chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1 PORT=29500 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodFG/orientedrcnn_swinS_fpn_angle30_30epochs_le90_amodFG.py 2 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4

chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1 PORT=29500 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodFG/orientedrcnn_swinS_fpn_angle0_30epochs_le90_amodFG.py 2 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4

chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1 PORT=29500 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodFG/orientedrcnn_swinS_fpn_angle10_30epochs_le90_amodFG.py 2 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4

chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1 PORT=29500 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodFG/orientedrcnn_swinS_fpn_angle20_30epochs_le90_amodFG.py 2 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4
~~~


* 172.26.49.151
~~~shell
DATA_ROOT="/MLV-2TB/AMOD_V1_FINAL_OPTICAL/"
chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodFG/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amodFG_Half.py  4 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4

chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodFG/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amodFG.py 4 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4

chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodFG/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amodFG_OneSixth.py 4 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4

chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodFG/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amodFG_ThreeFourth.py 4 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4
~~~

2025-02-08

* 172.26.49.26

~~~shell
DATA_ROOT="/media/yechani7/b6a6d52a-b20a-4e5a-a3d1-61770bbc9edc/AMOD_V1_FINAL_OPTICAL/"
chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodC/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amodC.py  2 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4

chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodC/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amodC_Half.py  2 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4

chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodC/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amodC_OneSixth.py  2 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4

chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodC/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amodC_ThreeFourth.py  2 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4


chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodC/orientedrcnn_swinS_fpn_angle0_30epochs_le90_amodC.py  2 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4

chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodC/orientedrcnn_swinS_fpn_angle10_30epochs_le90_amodC.py  2 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4

chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodC/orientedrcnn_swinS_fpn_angle20_30epochs_le90_amodC.py  2 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4

chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodC/orientedrcnn_swinS_fpn_angle30_30epochs_le90_amodC.py  2 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4

chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodC/orientedrcnn_swinS_fpn_angle40_30epochs_le90_amodC.py  2 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4

chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodC/orientedrcnn_swinS_fpn_angle50_30epochs_le90_amodC.py  2 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4
~~~

2025-02-09
* 172.26.49.151

~~~shell
DATA_ROOT="/MLV-2TB/AMOD_V1_FINAL_OPTICAL/"
chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodFGC/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amodFGC.py  4 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4

chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodFGC/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amodFGC_Half.py  4 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4

chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodFGC/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amodFGC_OneSixth.py  4 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4

chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodFGC/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amodFGC_ThreeFourth.py  4 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4


chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodFGC/orientedrcnn_swinS_fpn_angle0_30epochs_le90_amodFGC.py  4 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4

chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodFGC/orientedrcnn_swinS_fpn_angle10_30epochs_le90_amodFGC.py  4 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4

chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodFGC/orientedrcnn_swinS_fpn_angle20_30epochs_le90_amodFGC.py  4 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4

chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodFGC/orientedrcnn_swinS_fpn_angle30_30epochs_le90_amodFGC.py  4 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4

chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodFGC/orientedrcnn_swinS_fpn_angle40_30epochs_le90_amodFGC.py  4 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4

chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodFGC/orientedrcnn_swinS_fpn_angle50_30epochs_le90_amodFGC.py  4 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4
~~~


2025-02-14 (config 파일 오타... 데이터양 관련 재실험)
26번 서버!
~~~
DATA_ROOT="/media/yechani7/b6a6d52a-b20a-4e5a-a3d1-61770bbc9edc/AMOD_V1_FINAL_OPTICAL/"
chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodC/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amodC_Half.py  2 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4

chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodC/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amodC_OneSixth.py  2 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4

chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodC/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amodC_ThreeFourth.py  2 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4
~~~


151번 서버!

~~~
DATA_ROOT="/MLV-2TB/AMOD_V1_FINAL_OPTICAL/"
chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodFGC/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amodFGC_Half.py  4 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4

chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodFGC/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amodFGC_OneSixth.py  4 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4

chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodFGC/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amodFGC_ThreeFourth.py  4 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4


chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodFG/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amodFG_Half.py  4 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4


chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodFG/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amodFG_OneSixth.py 4 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4


chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/amodFG/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amodFG_ThreeFourth.py 4 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4
~~~