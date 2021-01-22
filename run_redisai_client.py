import redisai
import simplejpeg
import numpy as np
with open("aerial.jpg", mode='rb') as imfile: # b is important -> binary
    np_img = simplejpeg.decode_jpeg(imfile.read()).astype(np.float32)
    np_img = np.expand_dims(np_img, axis=0)
with open("aerial_detector.pb", mode='rb') as modelfile:
    model_data = modelfile.read()
client = redisai.Client(host="localhost", port=6379)
client.tensorset("image", np_img)
client.modelset(
    "mymodel",
    device="CPU",
    data=model_data,
    backend="TF",
    inputs=["input_1"],
    outputs=["filtered_detections/map/TensorArrayStack/TensorArrayGatherV3",
        "filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3",
        "filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3"])
client.modelrun("mymodel",inputs=["image"],outputs=["output1","output2","output3"])
print(client.tensorget("output1", meta_only=True))
