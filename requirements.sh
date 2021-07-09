 sudo apt-get install curl
 sudo apt-get install wget
 sudo apt-get install python3-matplotlib
 sudo apt-get install python3-pip libopenblas-dev libopenmpi-dev libomp-dev
 sudo -H pip3 install future
 sudo -H pip3 install --upgrade setuptools
 sudo -H pip3 install Cython
 sudo -H pip3 install gdown
 sudo cp ~/.local/bin/gdown /usr/local/bin/gdown
 gdown https://drive.google.com/uc?id=12UiREE6-o3BthhpjQxCKLtRg3u4ssPqb
 sudo -H pip3 install torch-1.9.0a0+gitd69c22d-cp36-cp36m-linux_aarch64.whl
 rm torch-1.9.0a0+gitd69c22d-cp36-cp36m-linux_aarch64.whl
 pip3 install -r requirements.txt
 pip3 install seaborn --no-deps
 wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt -O yolov5s.pt
 export OPENBLAS_CORETYPE=ARMV8

