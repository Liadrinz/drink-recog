import matplotlib.pyplot as plt

from loader import load
from aug import augment_and_shuffle

if __name__ == "__main__":
    images, tags = load()
    images, tags = augment_and_shuffle(images, tags, 10)
    plt.ion()
    plt.show()
    for im, tag in zip(images, tags):
        plt.imshow(im)
        print(tag)
        plt.pause(1)