원격 서버 공간을 내 PC로 마운트 하는 법
~~~shell
sudo apt update
sudo apt install sshfs

mkdir -p ~/희망하는_폴더명

sshfs [원격서버_사용자명]@[원격서버_IP주소]:/원격/데이터/경로  ~/아까만든_폴더명
원격서버 비밀번호 입력
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

* 멅티 GPU 테스트 코드 -> 성공 (172.26.19.26)

~~~shell
DATA_ROOT="/media/yechani7/b6a6d52a-b20a-4e5a-a3d1-61770bbc9edc/AMOD_V1_FINAL_OPTICAL/"
chmod +x ./mmrotate/tools/dist_train.sh
CUDA_VISIBLE_DEVICES=0,1 PORT=29500 ./mmrotate/tools/dist_train.sh my_config/config_log_yechan/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90-multisize_rrandomcroptest_amod.py 2 \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=4
~~~
