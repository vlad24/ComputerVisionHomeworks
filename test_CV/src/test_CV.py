import cv2
image = cv2.imread('colors.JPG', cv2.IMREAD_COLOR)
blue = 0
green = 1
red = 2
white_pixel = (255, 255,255)



def mouse_callback(self,event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print image[x,y]

def isBlue((b,g,r)):
    return (b > 200) and (g < 254) and (r < 254)


for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if isBlue(image[i,j]):
            image[i,j] = white_pixel
cv2.setMouseCallback('image', mouse_callback)            
cv2.imshow("Without blue", image)
cv2.waitKey()
cv2.destroyAllWindows()
