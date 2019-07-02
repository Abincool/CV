import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

def imcrop(im, W , H ):
    '''W:List 长
    H:list 宽
    '''
    #im=cv2.imread(im)
    return im[W[0]:W[1],H[0]:H[1]]

def change_color(img,num):
    
    B, G, R = cv2.split(img)

    b_rand = random.randint(-num, num)
    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        B[B > lim] = 255
        B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)
    elif b_rand < 0:
        lim = 0 - b_rand
        B[B < lim] = 0
        B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)

    g_rand = random.randint(-num, num)
    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
    elif g_rand < 0:
        lim = 0 - g_rand
        G[G < lim] = 0
        G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)

    r_rand = random.randint(-num, num)
    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
    elif r_rand < 0:
        lim = 0 - r_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)

    img_merge = cv2.merge((B, G, R))
    #img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img_merge

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(image, table)

def gamma_change(image,gamma=1.0):
    
    image=((image/255)**gamma*255).astype("uint8")
    
    '''
    size=image.shape
    
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size(2)):
                image[i][j][k]=((image[i][j][k]/255.0)**gamma*255).astype("uint8")
    '''
    return image

def histogram(image,label):
    
    if label=='gray':
        im_hist=cv2.equalizeHist(image)
        return im_hist
    
    elif label=='RGB':
        (b, g, r) = cv2.split(image)
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        # 合并每一个通道
        result = cv2.merge((bH, gH, rH))
        return result
    
    elif label=='YUV':
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output
    
        '''
        imgYUV = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        channelsYUV = cv2.split(imgYUV)
        channelsYUV[0] = cv2.equalizeHist(channelsYUV[0])

        channels = cv2.merge(channelsYUV)
        result = cv2.cvtColor(channels, cv2.COLOR_YCrCb2BGR)
        return result
        
        '''
def rotaton(image,angle):
    M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1) # center, angle, scale
    img_rotate = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return img_rotate 
    

def affine(img):
    rows, cols, ch = img.shape
    pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
    pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])

    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def perspective(img):
    height, width, channels = img.shape

    # warp:
    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    return img_warp


def augmentation(img):
    
    if random.randint(0,1)==1:
        img=change_color(img,50)

    if random.randint(0,1)==1:
        img=adjust_gamma(img, gamma=2.0)

    if random.randint(0,1)==1:
        img=histogram(img,'RGB')

    if random.randint(0,1)==1:
        img=rotaton(img,30)

    if random.randint(0,1)==1:
        img=affine(img)

    if random.randint(0,1)==1:
        img=perspective(img)

    return img
            

img=cv2.imread('1447726528788.jpg')

for i in range(20):
    img1=augmentation(img)
    
    cv2.imwrite('image augmentation./'+str(i)+'.jpg',img1)




