import cv2

image = cv2.imread('colors.JPG', cv2.IMREAD_COLOR)
blue = 0
green = 1
red = 2
white_pixel = (255, 255,255)
black_pixel = (0, 0, 0)


def mouse_click_callback(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print "({},{}):{}".format(x,y,image[x,y])

def isWhite((b,g,r)):
    return (250 <= b < 255) and (250 <= g < 255) and (250 <= r < 255)

def isBlack((b,g,r)):
    return (b < 50) and (g < 50) and (r < 100)
    
def isBlue((b,g,r)):
    return (max(b,g,r) == b) and (g < b - 10) and (r < b - 10)  

def isGreen((b,g,r)):
    return (max(b,g,r) == g) and (b < g - 5) and (r < g - 5)

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if isBlack(image[i,j]):
            image[i,j] = black_pixel
        else:
            image[i,j] = white_pixel
            
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
new_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)[1]        
new_image = cv2.dilate(new_image, kernel, iterations=1)        
cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_click_callback)
cv2.imshow("image", new_image)
cv2.waitKey()
cv2.destroyAllWindows()
