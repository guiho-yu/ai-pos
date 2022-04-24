import cv2
import os
from PIL import Image,ImageFont, ImageDraw
import googletrans
import pytesseract
import numpy as np

translator = googletrans.Translator()
kernel = np.ones((6, 15), np.uint8)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

image = cv2.imread("14240.png")


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
Laplacian = cv2.Laplacian(image, -2)

kernel = np.ones((17, 2), np.uint8)
kernel2 = np.ones((6, 15), np.uint8)

morph = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
cv2.imshow("morph", morph)

thr = cv2.adaptiveThreshold(morph, 255, cv2.ADAPTIVE_THRESH_MEAN_C,  cv2.THRESH_BINARY_INV, 3, 30)
cv2.imshow("thr", thr)

morph2 = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel2)
cv2.imshow("morph2", morph2)

filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)
text = pytesseract.image_to_string(gray, lang=None)
os.remove(filename)
print(text)
#cv2.imshow("Image", image)
trans = translator.translate(text, dest='ko')
print(trans.text)
cv2.waitKey(0)

