import numpy as np

from imgaug import augmenters as iaa

augmenters = [
    iaa.Affine(
        scale={"x": (0.5, 1.5), "y": (0.5, 1.5)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-45, 45),
        shear=(-16, 16),
    ),
    iaa.PerspectiveTransform(scale=(0.01, 0.2)),
]

def augment_and_shuffle(images, tags, mul):
    seq = iaa.Sequential(augmenters)
    X = []
    Y = []
    tag_idx = 0
    for image in images:
        for _ in range(mul):
            x = seq.augment_image(image)
            X.append(x)
            Y.append(tags[tag_idx])
        tag_idx += 1
    return X, Y