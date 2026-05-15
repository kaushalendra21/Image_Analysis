import cv2
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('data/Black_Footed_Albatross_0001_796111.jpg', 0)

# Sobel
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

# Prewitt
kernelx = [[1,1,1],[0,0,0],[-1,-1,-1]]
# prewitt = cv2.filter2D(img, -1, kernelx)

# Roberts
roberts = cv2.Canny(img, 50, 100)

# Canny
canny = cv2.Canny(img, 100, 200)

# Display
titles = ['Original','Sobel','Roberts','Canny']
images = [img,sobelx,roberts,canny]

# for i in range(4):
#     plt.figure(figsize=(10,6))
#     plt.subplot(1,4,i+1)
#     plt.imshow(images[i], cmap='gray')
#     plt.title(titles[i])
#     plt.axis('off')

plt.figure(figsize = (10,6))
plt.subplot(2,2,1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title("Original image")
plt.axis("off")
plt.subplot(2,2,2)
plt.imshow(sobelx,cmap="gray")
plt.title("sobel image")
plt.axis("off")
plt.subplot(2,2,3)
plt.imshow(roberts,cmap="gray")
plt.title("robert image")
plt.axis("off")
plt.subplot(2,2,4)
plt.imshow(canny,cmap="gray")
plt.title("canny image")
plt.axis("off")


plt.show()

