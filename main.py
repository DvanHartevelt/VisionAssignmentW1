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
import matplotlib.pyplot as plt

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

def main():
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
