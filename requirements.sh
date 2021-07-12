echo 'export OPENBLAS_CORETYPE=ARMV8' >> ~/.bashrc
echo 'export PATH=/home/deepconsc/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/usr/local/cuda/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64' >> ~/.bashrc
source ~/.bashrc
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
gdown https://drive.google.com/uc?id=1tU6YlPjrP605j4z8PMnqwCSoP6sSC91Z
sudo -H pip3 install torchvision-0.10.0a0+300a8a4-cp36-cp36m-linux_aarch64.whl
rm torchvision-0.10.0a0+300a8a4-cp36-cp36m-linux_aarch64.whl
pip3 install -r requirements.txt
pip3 install seaborn --no-deps
sudo pip3 install --global-option=build_ext --global-option="-I/usr/local/cuda/include" --global-option="-L/usr/local/cuda/lib64" pycuda
wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt -O yolov5s.pt
bash manage_swap.sh
