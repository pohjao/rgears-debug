import redisai

with open("aerial.jpg", mode='rb') as imfile: # b is important -> binary
    im_data = imfile.read()
with open("aerial_detector.pb", mode='rb') as modelfile:
    model_data = modelfile.read()
client = redisai.Client(host="localhost", port=6379)
client.set("image", im_data)
client.modelset(
    "mymodel",
    device="CPU",
    data=model_data,
    backend="TF",
    inputs=["input_1"],
    outputs=["filtered_detections/map/TensorArrayStack/TensorArrayGatherV3",
        "filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3",
        "filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3"])
print("Model meta:")
print(client.modelget("mymodel", meta_only=True))
print("Keys:")
print(client.keys("*"))

with open('gear.py', 'rb') as fgear:
    gear =fgear.read()

with open('pip_requirements.txt') as freq:
    requirements = freq.read()

print("Executing gear...")
print(client.execute_command('RG.PYEXECUTE',gear,"REQUIREMENTS","numpy","simplejpeg","opencv-python"))
print(client.execute_command("RG.DUMPEXECUTIONS"))