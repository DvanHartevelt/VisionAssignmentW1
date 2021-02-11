# In here, image manipulation functions are stored
import numpy as np

def gray_histo(img, printVal=False):
    # expetion testing
    if (len(img.shape) != 2):
        print("non-grayscaled pictures can't be stretched with this function")
        return False

    width = img.shape[1]
    height = img.shape[0]
    print("Width  =", width)
    print("Height =", height)
    histoOutput = [0 for i in range(0, 256)]

    # now a double for-loop, to loop across all pixels of the picture
    for y in range(0, height):
        for x in range(0, width):
            # ok, so indexed in the actual image matrix, height comes first.
            if printVal:
                print("Pixel at pos", [x, y], "has a value of ", img[y, x], ".")
            histoOutput[img[y, x]] += 1

    return histoOutput

def gray_stretching(img, imgHisto=None, lowPercent=1, highPercent=99):
    # exeption testing
    if (lowPercent > highPercent):
        print("lowPercent can't be higher than highpercent")
        return img

    if (lowPercent == 0 and highPercent == 100):
        print("No stretching needed")
        return img

    if (len(img.shape) != 2):
        print("non-grayscaled pictures can't be stretched with this function")
        return img

    # Some used variables
    width = img.shape[1]
    height = img.shape[0]
    nrOfPixels = height * width

    """
    Determining the lower and higher bounds of intensity for stretching
    """
    lowboundPix  = int(nrOfPixels * lowPercent  / 100)
    highboundPix = int(nrOfPixels * highPercent / 100)

    if imgHisto == None:
        imgHisto = gray_histo(img)

    lowboundInt = None
    highboundInt = 255
    runningCount = 0
    for intensity in range(0, 256):
        runningCount += imgHisto[intensity]
        if runningCount >= lowboundPix and lowboundInt == None:
            lowboundInt = intensity
        elif runningCount > highboundPix:
            highboundInt = intensity
            break

    """
    Lineair formulaic stretching of each pixel
    
    This goes by the formula  val_new = scalar * (val_old - bias)
    
    bias and scalar are chosen, so that an oldvalue at lowboundInt will be transformed to be 0
    and an oldvalue at highboundInt will be transformed to become 255
    """
    bias = lowboundInt
    scalar = 255 / (highboundInt - bias)     # this is a float
    print(f"lowboundInt = {lowboundInt}, highboundInt = {highboundInt}.")
    print(f"bias = {bias}, scalar = {scalar}.")

    imgStretched = np.zeros((height, width), np.uint8)
    for y in range(0, height):
        for x in range(0, width):
            # After scalar multiplication, this is a float.
            # so, the value has to be cast as an int again at the end.
            value = scalar * (int(img[y, x]) - bias)

            # some bounding, to keep values between 0 and 255
            if value < 0:
                value = 0
            elif value > 255:
                value = 255

            imgStretched[y, x] = int(value)

    return imgStretched