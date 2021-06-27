import json
import base64

from django.shortcuts import render
from django.http import HttpResponse
import cv2
import numpy as np

from .inference import inference

# Create your views here.
def main(request):
    """
    trash.html 메인 화면
    :param request:
    :return:
    """
    return render(request, 'trash.html')

def trash_recognize(frame):
    classes = {
        0: 'Battery',
        1: 'Clothing',
        2: 'Glass',
        3: 'Metal',
        4: 'Paper',
        5: 'Paperpack',
        6: 'Plastic',
        7: 'Plasticbag',
        8: 'Styrofoam',
    }
    values, indexes = inference(frame)
    data = {
        'values': [round(e, 4) for e in values],
        'trashes': [classes[e] for e in indexes]
    }
    return data

def trashClassify(request):
    if request.method == 'POST':
        image_b64 = request.POST['image']
        image_b64 = image_b64.split(",")[1]
        binary = base64.b64decode(image_b64)
        image = np.asarray(bytearray(binary), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        data = trash_recognize(image)
        return HttpResponse(json.dumps(data))

    return render(request, "null.html")