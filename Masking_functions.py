import cv2
import numpy as np

#Masking function for content images in GLStyleNet
def content_masking(img): #Input is an image of type ndarray
    dim = (int(img.shape[1]), int(img.shape[0]))
    
    # performing otsu's binarization
    # convert to gray scale first
    gray_pic = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(gray_pic,40,10,10)
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 3)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations = 20)

    #sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
    _, sure_fg = cv2.threshold(dist_transform,0.40*dist_transform.max(),255,0)

    #Defining unknown region 
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1
    
    #combining foreground, background, unknown
    markers[unknown==255] = 0
    markers_c = markers*255
    markers_c = np.where(markers_c==510, 120,markers_c)
    
    #Converting back to 0-255 RGB colours
    img_float32 = np.float32(markers_c)
    backtorgb = cv2.cvtColor(img_float32,cv2.COLOR_GRAY2RGB)

    backtorgb[np.where((backtorgb==[255.,255.,255.]).all(axis=2))] = [150.,0.,0.]
    backtorgb[np.where((backtorgb==[120.,120.,120.]).all(axis=2))] = [0,150.,0.]
    backtorgb[np.where((backtorgb==[0.,0.,0.]).all(axis=2))] = [0,0.,150.]

    return backtorgb

def style_masking(img):
    dim = (int(img.shape[1]), int(img.shape[0]))
    
    gray_pic = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # convert to gray scale first
    gray = cv2.bilateralFilter(gray_pic,40,10,10)

    # performing otsu's binarization
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
   
    # noise removal
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 1)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations = 1)

    # sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
    _, sure_fg = cv2.threshold(dist_transform,0.40*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    _, markers = cv2.connectedComponents(sure_fg)
    
    markers = markers + 1

    #combining foreground, background, unknown
    markers[unknown==255] = 0
    markers_c = markers*255
    markers_c = np.where(markers_c==510, 120,markers_c)

    #Converting back to 0-255 RGB colours
    img_float32 = np.float32(markers_c)
    backtorgb = cv2.cvtColor(img_float32,cv2.COLOR_GRAY2RGB)

    #Assigning colors based on style image mean color value. 
    #Back/foregrounding painted pictures can be reverse of naturalistic images, why
    #Flipping fore and background depending on image mean yields better results.
    if gray_pic.mean(axis=0).mean(axis=0) >= 120:
        backtorgb[np.where((backtorgb==[255.,255.,255.]).all(axis=2))] = [150.,0.,0.]
        backtorgb[np.where((backtorgb==[120.,120.,120.]).all(axis=2))] = [0,150.,0.]
        backtorgb[np.where((backtorgb==[0.,0.,0.]).all(axis=2))] = [0,0.,150.]
    else:
        backtorgb[np.where((backtorgb==[255.,255.,255.]).all(axis=2))] = [0.,0.,150.]
        backtorgb[np.where((backtorgb==[120.,120.,120.]).all(axis=2))] = [0,150.,0.]
        backtorgb[np.where((backtorgb==[0.,0.,0.]).all(axis=2))] = [150,0.,0.]
    
    backtorgb_res = cv2.resize(backtorgb, dim, interpolation = cv2.INTER_AREA)
    
    return backtorgb
def main():
    print("masking")

if __name__ == '__main__':
    print(tf.__version__)
    main()
