import cv2
import numpy as np

def color_histogram_of_test_image(test_src_image):

    # load the image
    image = test_src_image

    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        print(hist)
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue
    ar=np.zeros((352,640,3))
    # for i in range(500):
    #     for j in range():
    #         j=feature_data
    ar[::]=[int(x) for x in feature_data.split(",")]
    cv2.imshow("ss",ar)
    cv2.waitKey(10000)
    print(feature_data)
    with open('test.data', 'w') as myfile:
        myfile.write(feature_data)
img=cv2.imread("/home/quest/Abdul sathar/Codes/tensorflow/vehicle_counting_tensorflow-master (2)/utils/color_recognition_module/training_dataset/green/green2.png")
# img=cv2.resize(img,(50,50))
color_histogram_of_test_image(img)