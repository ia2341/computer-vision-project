import cv2
import numpy as np
from PIL import Image

import config


def color_transfer(source, target):
    # convert color space from BGR to L*a*b color space
    # note - OpenCV expects a 32bit float rather than 64bit
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB).astype("float32")
    # compute color stats for both images
    (smean, scov) = image_stats(source)
    (tmean, tcov) = image_stats(target)

    # cholesky
    L_s = np.linalg.cholesky(scov)
    L_t = np.linalg.cholesky(tcov)
    Lt_inv = np.linalg.inv(L_t)
    A = np.matmul(L_s, Lt_inv)
    b_act = smean - (np.matmul(A, tmean))
    
    (r, g, b) = cv2.split(target)
    new_img = target.copy()
    
    m,n = r.shape
    for x in range(m):
      for y in range(n):
        x_val = np.matrix([[r[x][y], g[x][y], b[x][y]]]).transpose()
        x_new = np.matmul(A, x_val) + b_act
        new_img[x][y][0] = x_new[0][0]
        new_img[x][y][1] = x_new[1][0]
        new_img[x][y][2] = x_new[2][0]
    
    
    # converting back to BGR
    transfer = cv2.cvtColor(new_img.astype("uint8"), cv2.COLOR_RGB2BGR)
    return transfer


def image_stats(image):
    # compute mean and standard deviation of each channel
    (r, g, b) = cv2.split(image)

    rMean = np.average(r)
    gMean = np.average(g)
    bMean = np.average(b)
    mean = np.matrix([[rMean, gMean, bMean]]).transpose()
    cov = np.matrix([[0.0, 0.0, 0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
    m,n = r.shape
    for x in range(m):
      for y in range(n):
        x_val = np.matrix([[r[x][y], g[x][y], b[x][y]]]).transpose()
        temp_one = (x_val - mean)
        temp_two = np.matrix(temp_one).transpose()
        sum_val = ((np.matmul(temp_one, temp_two)) / (m*n))
        cov += sum_val

    return (mean, cov)



def main():
    target = cv2.imread(config.style_path)
    source = cv2.imread(config.content_path)

    # transfer of color
    transfer = color_transfer(source, target)

    im = Image.fromarray(transfer)
    new_style_path = 'output/transfer.jpg'
    im.save(new_style_path)


if __name__ == "__main__":
    main()
