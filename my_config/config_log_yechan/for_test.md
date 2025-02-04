원격 서버 공간을 내 PC로 마운트 하는 법
~~~shell
sudo apt update
sudo apt install sshfs

mkdir -p ~/희망하는_폴더명

sshfs [원격서버_사용자명]@[원격서버_IP주소]:/원격/데이터/경로  ~/아까만든_폴더명
원격서버 비밀번호 입력
~~~

데이터 위치
* `mlv` @ `172.26.19.210`: mlv210/ADD데이터셋/AMOD_EO_final
* `yechani7` @ `172.26.19.26`: /media/yechani7/b6a6d52a-b20a-4e5a-a3d1-61770bbc9edc/AMOD_V1_FINAL_OPTICAL

모의 테스트

~~~shell
cd ../..
DATA_ROOT="data/AMOD_MOCK/"
python mmrotate/tools/train.py my_config/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amod.py \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=1 data.samples_per_gpu=2
~~~

2025-02-04

~~~shell
cd ../..
DATA_ROOT="/home/yechani9/remote_26/AMOD_V1_FINAL_OPTICAL/"
python mmrotate/tools/train.py my_config/orientedrcnn_swinS_fpn_angle0,10,20,30,40,50_30epochs_le90_amod.py \
 --cfg-options data.train.data_root="$DATA_ROOT" data.val.data_root="$DATA_ROOT" \
               runner.max_epochs=30 data.samples_per_gpu=2
~~~