import cv2
import numpy as np

# Load the image
image = cv2.imread("cat.png")  # Replace with your image file

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply pixelation
def pixelate(img, pixel_size=10):
    height, width = img.shape[:2]
    temp = cv2.resize(img, (width // pixel_size, height // pixel_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

pixelated = pixelate(image, pixel_size=20)


# Apply Gaussian blur
blurred = cv2.GaussianBlur(image, (15, 15), 0)


# Apply Canny edge detection
edges = cv2.Canny(gray, 50, 150)


# Apply thresholding
_, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)


# Apply sharpening
sharpening_kernel = np.array([  [ 0, -1,  0],
                                [-1,  5, -1],
                                [ 0, -1,  0]], dtype=np.float32)
sharpened = cv2.filter2D(image, -1, sharpening_kernel)


# Apply emboss
emboss_kernel = np.array([  [ -2, -1,  0],
                            [-1,  1, 1],
                            [ 0, 1,  2]], dtype=np.float32)
emboss = cv2.filter2D(gray, -1, emboss_kernel)


# Apply outline
outline_kernel = np.array([  [ -1, -1,  -1],
                            [-1,  8, -1],
                            [ -1, -1,  -1]], dtype=np.float32)
outline = cv2.filter2D(gray, -1, outline_kernel)


# Show results
# Enable that you want to see, just erase the '#' or just press 'Ctrl + /'
cv2.imshow("original", image)
# cv2.imshow("gray", gray)
# cv2.imshow("pixelated", pixelated)
# cv2.imshow("blurred", blurred)
# cv2.imshow("edges", edges)
# cv2.imshow("thresholded", thresholded)
# cv2.imshow("sharpened", sharpened)
# cv2.imshow("emboss", emboss)
# cv2.imshow("outline", outline)

# Save your processed image
# Inside the quotation is for the name of the file and then after that choose process that you want to store (See the list of available function above)
cv2.imwrite("sharpening.png", sharpened)

cv2.waitKey(0)
cv2.destroyAllWindows()