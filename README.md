# Yolo V5 TensorRT Conversion & Deployment on Jetson Nano & Xavier [Ultralytics EXPORT]
[<img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/splash-export-competition.png">](https://github.com/ultralytics/yolov5/discussions/3213)

# JetPack
Firstly you should get latest JetPack *v4.5.1* from NVIDIA and boot it onto Jetson Nano. 
You could find JetPack download options ***[here](https://developer.nvidia.com/embedded/jetpack)***


# Conversion Steps
1. Create project directory and enter it:
   ```
   mkdir yolov5_conversion && cd yolov5_conversion
   ```
2. Clone into required repositories:  
   ```
   git clone -b v5.0 https://github.com/ultralytics/yolov5.git
   git clone https://github.com/deepconsc/yolov5_tensorrt.git
   ```
3. Install requirements for JetPack:
	Weirdly enough the Jetson doesn't come with preinstalled pip, we've to do it manually. Also ***matplotlib*** and ***torch*** would fail via pip installation, so we've to use workaround. The below script will take care of installing all the dependencies we'll need.
	
	Enter ***yolov5_tensorrt*** and run:
   ```
    bash requirements.sh
   ```
   > Note: in case of seeing error **Illegal instruction (core dumped)**, 
   > run the below line again. (You can see this command in last line of requirements.sh) 
	  ```
	  export OPENBLAS_CORETYPE=ARMV8
	  ```
4. Convert the model.
    Default ***yolov5s.pt*** has already been downloaded into the folder. Take this file along with generator.py and put them into ***yolov5*** repository folder. 
    ```
    mv yolov5s.pt generator.py ../yolov5 && cd ../yolov5
    python3 generator.py yolov5s.pt
    ``` 
  > Note: don't mind the Matplotlib warning. This script will take couple of seconds to finish the process.

 5. Move ***.wts*** network file to ***tensorrt_yolov5*** repository folder, build YoloV5 and run conversion on FP16. 
	 ```
	mkdir ../tensorrt_yolov5/build
	mv yolov5s.wts ../tensorrt_yolov5/build && cd ../tensorrt_yolov5/build 
	cmake .. && make
	sudo ./yolov5 -s yolov5s.wts yolov5s.engine s
	``` 

After couple of minutes the TensorRT engine file will be built. The default settings used in the script are:
- Input Size: H=W=640
- Precision: FP16
- Max Batch Size: 1
- Num_classes: 80
## Checking the engine sanity
```
sudo ./yolov5 -d yolov5s.engine sample_images
```
The script will output images with plotted boxes & confidences. 

## NOTE
The ***libmyplugins.so*** file in build will be needed for inference. 

# Inference

For inference you could use the module in ***yolov5s.py***. It's basically a class that:
- Initializes engine
- Preprocesses image
- Handles CPU-> GPU -> CPU ops
- Does inference on image
- Postprocesses the results

You'll find it very much easy to incorporate the class file into your custom codebase, whether it's running inference on image(s) or stream. 

Also you can simply check out ***inference.py*** to see example of TensorRT class inference. 

# Custom model

For custom model conversion there are some factors to take in consideration. 
-  Only YoloV5 S (small) version is supported.
- You should use your own checkpoint that only contains network weights (i.e. stripped optimizer, which is last output of YoloV5 pipeline after training finishes)
 - Change the ***CLASS_NUM*** in ***yololayer.h*** - ```Line 28``` to number of classes your model has before building yolo. If you've already built it, you can just run ```cmake .. && make``` from build folder after changing the class numbers. 
 - Change the ***Input W*** and ***Input H*** according to resolution you've trained the network in ***yololayer.h*** on - ```Lines 29, 30```
 - Change the ***CONF*** and ***NMS*** thresholds according to your preferences in ***yolov5.cpp*** - ```Lines 12, 13```
 - Change the batch size according to your requirements in ***yolov5.cpp*** - ```Line 14```.
 
# INT 8 Conversion & Calibration
 Exporting YoloV5 network to INT8 is pretty much straightforward & easy. 

Before you run last command in ***Step 5*** in conversion steps, you should take in consideration that INT8 needs dataset for calibration.
Good news is - we won't need labels, just images. 
 There's no recommended amount of data samples to use for calibration, but as many - as better. 
 
 In this case, as long as we're exporting the standard yolov5s, trained on COCO dataset, we'll download *val* set of images from coco and do calibration on it. 

1. Enter the ***tensorrt_yolov5*** folder and run. 
 ```
 wget http://images.cocodataset.org/zips/val2017.zip
 unzip val2017.zip
 mv val2017 build/coco_calib
 ```
2. Change the precision in ***yolov5.cpp*** - ```Line 10```
 
 ```#define USE_FP16``` -> ```#define USE_INT8```
 
3. Run the conversion.
```
cmake .. && make
sudo ./yolov5 -s yolov5s.wts yolov5s_int8.engine s
``` 
 It'll do calibration on every image in coco_calib folder, it might take a while. After it's finished, the engine is ready for usage.


## Thanks to the following sources/repositories for scripts & helpful suggestions.
- For TensorRT conversion codebase:
 https://github.com/wang-xinyu/tensorrtx
 - For Torch installation procedures:
 https://qengineering.eu/install-pytorch-on-jetson-nano.html
