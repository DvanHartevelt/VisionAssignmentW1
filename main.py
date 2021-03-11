# This script is an assignment for Vision, week one.
# By David van Hartevelt, 10/02/2021
#
# In this assignment, three things are done.
# 1) A picture is imported or taken by a camera, and grayscaled.
# 2) A histogram is made of this picture
# 3) The pictures values are stretched to increase contrast.
#
# All pictures and

from util import image_manipulation_functions as im
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    #Testing the camera
    # print("Taking picture...")
    # try:
    #     cap = cv2.VideoCapture(0)
    #     camSucces, imgCam = cap.read()
    # except:
    #     print("Picture could not be taken.")
    #     camSuccess = False
    #
    # if camSuccess:
    #     cv2.imshow("Camera output", imgCam)
    #     cv2.imwrite("Output/cameraPicture.png", imgCam)
    #     cv2.waitKey(0)


    #importing the foggy street picture
    img = cv2.imread("Resources/foggyPicture.PNG", 0)
    img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))


    """
    Histogram function
    """

    def gray_histo(img):
        """
        This function creates a histogram of a grayscaled picture

        :param img: image matrix
        :return: list of nr of pixels at each value between 0 and 255
        """
        width = img.shape[1]
        height = img.shape[0]

        histo = [0 for i in range(0, 256)]

        # now a double for-loop, to loop across all pixels of the picture
        for y in range(0, height):
            for x in range(0, width):
                # ok, so indexed in the actual image matrix, height comes first.
                # if printVal:
                #     print("Pixel at pos", [x, y], "has a value of ", img[y, x], ".")
                histo[img[y, x]] += 1

        return histo

    histo = gray_histo(img)

    x_axis = np.array([i for i in range(256)])

    plt.bar(x_axis, histo, label="Original histogram")
    plt.title('Histogram')
    plt.xlabel('Grayscale value [0-255]')
    plt.ylabel('Relative intensity [?]')
    plt.legend()
    plt.savefig("Output/Histogram_beforeStretching.png", bbox_inches='tight')

    cv2.imshow("Original picture", img)
    cv2.waitKey(10)

    plt.show()



    """
    Stretching
    """
    def stretching(img, lowerbound, higherbound):
        """
        Stretches image by remapping values from [lowerbound, higherbound] to [0, 255]

        :param img: grayscaled image
        :param lowerbound: uint8
        :param higherbound: uint8
        :return: newImg
        """

        imgNew = np.zeros_like(img)

        width = img.shape[1]
        height = img.shape[0]

        for y in range(height):
            for x in range(width):
                newValue = np.interp(img[y, x], [lowerbound, higherbound], [0, 255])

                if newValue < 0:
                    newValue = 0
                if newValue > 255:
                    newValue = 255

                imgNew[y, x] = newValue

        return imgNew

    # lo = int(input("What is the lowerbound? [uint8]"))
    # hi = int(input("What is the higherbound? [uint8]"))
    lo, hi = 124, 175

    imgStretched = stretching(img, lo, hi)

    histoStretched = gray_histo(imgStretched)

    plt.bar(x_axis, histoStretched, label="Stretched histogram")
    plt.title('Histogram')
    plt.xlabel('Grayscale value [0-255]')
    plt.ylabel('Relative intensity [?]')
    plt.legend()
    plt.savefig("Output/Histogram_afterStretching.png", bbox_inches='tight')

    cv2.imwrite("Output/stretched_foggyPicture.png", imgStretched)

    cv2.imshow("Original picture", img)
    cv2.imshow("Stretched picture", imgStretched)
    cv2.waitKey(10)

    plt.show()

    #showing both histograms in one figure.
    plt.bar(x_axis, histoStretched, label='Stretched image')
    plt.bar(x_axis, histo, label='(original) grayscaled image')

    plt.title('Histograms')
    plt.xlabel('Grayscale value [0-255]')
    plt.ylabel('Relative intensity [?]')
    plt.legend()

    plt.savefig("Output/Histograms_combined.png", bbox_inches='tight')
    plt.show()

    print("Thank you for running this demo.")

    pass

def query_resize(imgOrig):
    smallerPic = input("Would you like to resize the picture to be smaller? [y/n]")
    if (smallerPic.lower() == 'y'):
        print("Image resized to one quarter the size.")
        img = cv2.resize(imgOrig, (int(imgOrig.shape[1]/2), int(imgOrig.shape[0]/2)))
        return True, img
    elif (smallerPic.lower() == 'n'):
        print("Image kept the same size.")
        return False, imgOrig
    else:
        print("Please answer with a [y/n].")
        return query_resize(imgOrig)

def takePicture():
    print("Taking picture...")
    try:
        cap = cv2.VideoCapture(0)
        return cap.read()
    except:
        print("Picture could not be taken.")
        return False, None

def main2():
    print("Importing picture...")
    imgOrig = cv2.imread("Resources/foggyPicture.PNG")
    camSuccess, imgCam = takePicture()

    isResized, img = query_resize(imgOrig)
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    histo1 = im.gray_histo(imgGray, printVal=isResized)

    imgStretched = im.gray_stretching(imgGray, histo1, 1, 99)

    histo2 = im.gray_histo(imgStretched)


    # Display and save all plots and pictures
    cv2.imshow("Original", img)
    cv2.imshow("Grayscaled", imgGray)
    cv2.imwrite("Output/gray_foggyPicture.png", imgGray)
    cv2.imshow("Stretched", imgStretched)
    cv2.imwrite("Output/stretched_foggyPicture.png", imgStretched)
    if camSuccess:
        cv2.imshow("Camera output", imgCam)
        cv2.imwrite("Output/cameraPicture.png", imgCam)

    cv2.waitKey(0)

    x_axis = [i for i in range(0, 256)]

    plt.bar(x_axis, histo2, label='Stretched image')
    plt.bar(x_axis, histo1, label='(original) grayscaled image')

    plt.title('Histograms')
    plt.xlabel('Grayscale value [0-255]')
    plt.ylabel('Relative intensity [?]')
    plt.legend()

    plt.savefig("Output/Histograms_combined.png", bbox_inches='tight')
    plt.show()

    plt.bar(x_axis, histo1, label='(original) grayscaled image')

    plt.title('Histograms')
    plt.xlabel('Grayscale value [0-255]')
    plt.ylabel('Relative intensity [?]')
    plt.legend()

    plt.savefig("Output/Histograms_beforeStretching.png", bbox_inches='tight')
    plt.close()

    plt.bar(x_axis, histo2, label='Stretched image')

    plt.title('Histograms')
    plt.xlabel('Grayscale value [0-255]')
    plt.ylabel('Relative intensity [?]')
    plt.legend()

    plt.savefig("Output/Histograms_afterStretching.png", bbox_inches='tight')

    pass


if __name__ == "__main__":
    main()
