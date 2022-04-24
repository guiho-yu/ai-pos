import cv2
import numpy as np
from PIL import Image,ImageFont, ImageDraw
import googletrans
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
translator = googletrans.Translator()


large = cv2.imread('14240.png') #이미지 가져오기
original = large.copy() #원본 이미지
transe = large.copy() #번용용 이미지
small = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY) #이미지 회색조로 변경


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)# 경선찾기

_, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 35))
connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)#번지게 하기

# using RETR_EXTERNAL instead of RETR_CCOMP
contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #특징부분 찾기
mask = np.zeros(bw.shape, dtype=np.uint8)
count = 0
for idx in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[idx])
    mask[y:y+h, x:x+w] = 0

    cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
    r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

    if r > 0.45 and w > 8 and h > 8:
        cv2.rectangle(large, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
        crop = transe[y:y+h, x:x+w] #글자있는 부분 추출
        img = np.full(shape=(h, w, 3), fill_value=255, dtype=np.uint8)
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        font=ImageFont.truetype("batang.ttc",15)
        org=(0,0)
        text = pytesseract.image_to_string(crop, lang=None)
        tran = translator.translate(text, dest='ko')
        draw.text(org,tran.text,font=font,fill=(0,0,0)) #text를 출력
        img = np.array(img)
        transe[y:y + h, x:x + w] = img
        count = count+1

# show image with contours rect
cv2.imshow('original', original)
cv2.imshow('transelate', transe)


cv2.waitKey()


