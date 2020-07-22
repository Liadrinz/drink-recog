import os
import json
import numpy as np

from PIL import Image
from imgaug import augmenters as iaa
from config import conf
from utils import get_label_vector

augmenters = [
    iaa.Affine(
        scale={"x": (0.5, 1.5), "y": (0.5, 1.5)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-45, 45),
        shear=(-16, 16),
    ),
    iaa.PerspectiveTransform(scale=(0.01, 0.2)),
]

def load():
    result = []
    tags = []
    for triple in os.walk(conf('image.root')):
        for file in triple[2]:
            if file.endswith('.jpg') or file.endswith('.png'):
                path = triple[0] + file
                result.append(np.array(Image.open(path).resize(conf('image.size'))))
                tags.append(get_label_vector(file.split('.')[0]))
    return result, tags

def augment_and_shuffle(images, tags, mul):
    seq = iaa.Sequential(augmenters)
    X = []
    Y = []
    tag_idx = 0
    for image in images:
        for _ in range(mul):
            x = seq.augment_image(image).tolist()
            rand_idx = np.random.randint(0, len(X) + 1)
            X.insert(rand_idx, x)
            Y.insert(rand_idx, tags[tag_idx])
        tag_idx += 1
    return X, Y

def generate_aug(which):
    images, tags = load()
    images, tags = augment_and_shuffle(images, tags, conf(f'aug.{which}.factor'))
    dataset = {
        'images': images,
        'tags': tags
    }
    with open(conf(f'aug.{which}.output'), 'wb') as f:
        f.write(json.dumps(dataset).encode('utf-8'))

if __name__ == "__main__":
    generate_aug('train')
    generate_aug('test')