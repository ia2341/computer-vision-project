import cv2
import numpy as np
from PIL import Image

import config


def color_transfer(source, target):
    # convert color space from BGR to L*a*b color space
    # note - OpenCV expects a 32bit float rather than 64bit
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # compute color stats for both images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)
    
    # split the color space
    (l, a, b) = cv2.split(target)
    
    # substarct the means from target image
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar
    
    # scale by the standard deviation
    l = (lStdTar/lStdSrc)*l
    a = (aStdTar/aStdSrc)*a
    b = (bStdTar/bStdSrc)*b
    
    # add the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc
    
    # clipping the pixels between 0 and 255
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)
    
    # merge the channels
    transfer = cv2.merge([l, a, b])
    
    # converting back to BGR
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
    return transfer


def image_stats(image):
    # compute mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)

    lMean, lStd = (np.average(l), np.std(l))
    aMean, aStd = (np.average(a), np.std(a))
    bMean, bStd = (np.average(b), np.std(b))

    return (lMean, lStd, aMean, aStd, bMean, bStd)


def main():
    target = cv2.imread(config.style_path)
    source = cv2.imread(config.content_path)

    # transfer of color
    transfer = color_transfer(source, target)

    im = Image.fromarray(transfer)
    new_style_path = 'output/transfer.jpg'
    im.save(new_style_path)


if __name__ == '__main__':
    main()
