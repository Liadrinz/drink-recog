import os
import numpy as np

from PIL import Image
from config import IMAGE_ROOT

def load():
    result = []
    tags = []
    for triple in os.walk(IMAGE_ROOT):
        for file in triple[2]:
            if file.endswith('.jpg') or file.endswith('.png'):
                path = triple[0] + file
                result.append(np.array(Image.open(path)))
                tags.append(file.split('.')[0])
    return result, tags
