from typing import List
import PIL.Image
import torch
import torchvision
import gradio as gr

title = "Person detection with YOLO v5"
description = "Person detection, you can twik the corresponding confidence threshold. Good results even when face not visible."
article = "<p style='text-align: center'><a href='https://github.com/scoutant/yolo-persons-gradio' target='_blank' class='footer'>Github Repo</a></p>"

model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
model.classes = [ 0 ] # only considering class 'person' and not the 79 other classes...
model.conf = 0.6 # only considering detection above the threshold.

def inference(img:PIL.Image.Image, threshold):
    if img is None:
        return None,0
    images:List[PIL.Image.Image] = [ img ] # inference operates on a list of images
    model.conf = threshold
    detections:torchvision.Detections = model(images, size=640)
    predictions:torch.Tensor = detections.pred[0] # the predictions for our single image
    detections.render() # bounding boxes and labels added into image
    return detections.imgs[0], predictions.size(dim=0) # image and number of detections

gr.Interface(
    inference,
    [
        gr.inputs.Image(type="pil", label="Input"),
        gr.Slider(minimum=0.5, maximum=0.9, step=0.05, value=0.7, label="Confidence threshold")
      ],
    [
        gr.components.Image(type="pil", label="Output"),
        gr.components.Label(label="nb of persons detected for given confidence threshold")
      ],
    title=title,
    description=description,
    article=article,
    examples=[['data/businessmen-612.jpg'], ['data/businessmen-back.jpg']],
    enable_queue=True,
    allow_flagging="never"
).launch(debug=True)