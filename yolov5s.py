import ctypes
import os
import shutil
import random
import sys
import threading
import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torchvision
import tqdm 
import logging 

CONF_THRESH = 0.7
IOU_THRESHOLD = 0.4

PLUGIN_LIBRARY = "build/libmyplugins.so"

ctypes.CDLL(PLUGIN_LIBRARY)

class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, checkpoint_path: str, device_num: int):
        self.ctx = cuda.Device(device_num).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)
        self.ENGINE_PATH = checkpoint_path

        with open(self.ENGINE_PATH, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))

            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(cuda_mem))

            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size
        self.warmup()

    def warmup(self):
        """
        description: Warmup function for TensorRT. 
        """
        logging.info('Starting to warm up...')
        for _ in tqdm.tqdm(range(20)):
            standard = np.random.randn(800, 600, 3)
            self.infer((standard + 255).astype(np.uint8))
        logging.info('Warmup has been finished')

    def infer(self, raw_image: np.ndarray):
        """
        description: Entire inference function. Takes raw BGR image (np.ndarray), returns detections.

        param:
            raw_image: np.ndarray (Frame/Image)
        return:
            detection_results: list[list] -> [[x0, y0, x1, y1, confidence], ...]
        """

        self.ctx.push()

        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        batch_image_raw = []
        batch_origin_h = []
        batch_origin_w = []
        batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_h, self.input_w])
        input_image, image_raw, origin_h, origin_w = self.preprocess_image(raw_image)
        batch_image_raw.append(image_raw)
        batch_origin_h.append(origin_h)
        batch_origin_w.append(origin_w)
        np.copyto(batch_input_image[0], input_image)
        batch_input_image = np.ascontiguousarray(batch_input_image)

        np.copyto(host_inputs[0], batch_input_image.ravel())
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        stream.synchronize()
        self.ctx.pop()
        output = host_outputs[0]
        for i in range(1):
            result_boxes, result_scores, result_classid = self.postprocess(
                output[i * 6001: (i + 1) * 6001], batch_origin_h[i], batch_origin_w[i]
            )
        detection_results = []

        for x in range(len(result_boxes)):
            classid = result_classid[x]
            box = result_boxes[x].tolist()
            score = result_scores[x]
            box.append(score)
            detection_results.append(box)

        return detection_results

    def destroy(self):
        self.ctx.pop()
        

    def preprocess_image(self, raw_bgr_image):
        """
        Image Preprocessor: Letterbox, BGR2RGB, NCHW 
        """
        image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)

        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0

        image = cv2.resize(image, (tw, th))
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        )
        image = image.astype(np.float32)
        image /= 255.0
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y

    def postprocess(self, output, origin_h, origin_w):
        num = int(output[0])
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        pred = torch.Tensor(pred).cuda()
        boxes = pred[:, :4]
        scores = pred[:, 4]
        classid = pred[:, 5]

        chunk = scores > CONF_THRESH
        boxes = boxes[chunk, :]
        scores = scores[chunk]
        classid = classid[chunk]

        boxes = self.xywh2xyxy(origin_h, origin_w, boxes)

        indices = torchvision.ops.nms(boxes, scores, iou_threshold=IOU_THRESHOLD).cpu()
        result_boxes = boxes[indices, :].cpu().numpy().astype(np.int)
        result_scores = scores[indices].cpu().numpy()
        result_classid = classid[indices].cpu().numpy()
        return result_boxes, result_scores, result_classid
