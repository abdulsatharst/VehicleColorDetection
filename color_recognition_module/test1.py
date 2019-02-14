import cv2, numpy as np

from matplotlib import pyplot as plt

# features=[]
#
# img = cv2.imread('/home/quest/Abdul sathar/Codes/Keras/Vehicle color detection/green.png')
# # img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# plt.hist(img.ravel(), 180, [0, 180]);
# plt.show()
#
# print(img.ravel())
# # cv2.imshow("ss",img)
# # cv2.waitKey(1000)
# hist = cv2.calcHist([img], [2], None, [180], [0, 180])
# features.extend(hist)
# elem = np.argmax(hist)
#
# print(elem,hist)
# 1


img = cv2.imread('/home/quest/Abdul sathar/Codes/Keras/Vehicle color detection/joy.jpeg')
img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
color = ('b', 'g', 'r')
# for i, col in enumerate(color):
histr_h = cv2.calcHist([img], [0], None, [180], [0, 180])
histr_s = cv2.calcHist([img], [1], None, [256], [0, 256])
histr_v= cv2.calcHist([img], [2], None, [256], [0, 256])
plt.plot(histr_h, color="r")
plt.plot(histr_s, color="g")
plt.plot(histr_v, color="b")
plt.xlim([0, 256])
print(histr_h,histr_s,histr_v)
elem_h= np.argmax(histr_h)
elem_s= np.argmax(histr_s)
elem_v= np.argmax(histr_v)
print(elem_h,elem_s,elem_v)
plt.show()
