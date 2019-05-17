from subprocess import Popen
import time
while True:
    print("Start service")
    start_time = time.time()
    p = Popen("source ~/.zshrc;cd /home/eyeq/source-code/vincheckin_deep/iq_facial_recognition/src; CUDA_VISIBLE_DEVICES=1 python3 onetime/generic_detection_tracking.py -c rtsp://admin:abcd1234@10.111.16.6:554 -a VHLT -fem /home/eyeq/source-code/vincheckin_deep/iq_facial_recognition/models/am_inception_res_v1_transfer_Vin_5hour_20180701.pb -db -rs", \
       shell=True)
    p.wait()
    print("Service killed!\n Runtime: " + str((time.time() - start_time) / 60) +
          " minutes, Restarting...")
    time.sleep(1)
