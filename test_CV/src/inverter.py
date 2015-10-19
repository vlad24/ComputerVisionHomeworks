import cv2

name = str(raw_input("filename:"))
img = cv2.imread(name, 0)
dst = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imwrite("inv_" + name, dst)
print "Saved"
cv2.waitKey()
cv2.destroyAllWindows()
