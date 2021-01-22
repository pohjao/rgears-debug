from typing import Tuple
import simplejpeg
import cv2
import numpy as np
from redisAI import modelRunnerAddOutput, createModelRunner, createTensorFromBlob, modelRunnerAddInput, modelRunnerRun #type: ignore

def compute_resize_scale(image_shape, min_side=800, max_side=1333):
    (rows, cols, _) = image_shape
    smallest_side = min(rows, cols)
    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side
    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side
    return scale

def resize_image(img, min_side=800, max_side=1333):
    # compute scale to resize the image
    scale = compute_resize_scale(img.shape, min_side=min_side, max_side=max_side)
    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)
    return np.expand_dims(img, axis=0), scale


def preprocess(np_img) -> Tuple[np.ndarray, int]:
    np_img = np_img.astype(np.float32)
    #np_img -= [103.939, 116.779, 123.68]
    return resize_image(np_img)

def run(np_img):
    np_img, _ = preprocess(np_img)
    print(np_img.shape)
    print(np_img.dtype)
    model_key = "mymodel"
    model_runner = createModelRunner(model_key)
    for i in range(3):
        print(i)
        modelRunnerAddOutput(model_runner, f"output{i}")
    input_tensor = createTensorFromBlob("FLOAT", list(np_img.shape), bytearray(np_img.tobytes()))
    modelRunnerAddInput(model_runner, "input", input_tensor)
    #The following call leads to crash
    outputs = modelRunnerRun(model_runner)
    print(len(outputs))

def decode(data) -> np.ndarray:
    return simplejpeg.decode_jpeg(data)

gb = GearsBuilder("KeysReader", "image")
gb.map(lambda elem: decode(elem["value"]))
gb.map(run)
gb.run()