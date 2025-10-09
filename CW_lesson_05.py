import cv2
import numpy as np

img=cv2.imread('images/cat.jpg')
img_copy=img.copy()
img=cv2.GaussianBlur(img,(7,7),5)

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([0,49,92]) #беремо нижній поріг
upper = np.array([74,255,255]) #беремо верхній поріг
mask=cv2.inRange(img,lower,upper)
img=cv2.bitwise_and(img,img,mask=mask)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area>150:
        perimeter = cv2.arcLength(cnt,True)
        M = cv2.moments(cnt)
        #центр масс
        if M["m00"]!=0:
            cx=int(M["m10"]/M["m00"])
            cy=int(M["m01"]/M["m00"])


        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = round(w/h,2)#допомагає відрізняти співвідношення сторін
        compactness =round((4*np.pi*area)/(perimeter**2),2)
        approx = cv2.approxPolyDP(cnt,0.02*perimeter,True)
        if len(approx)==3:
            shape="tricutnik"
        elif len(approx)==4:
            shape="square"
        elif len(approx)>8:
            shape="oval"
        else:
            shape="inshe"

        cv2.drawContours(img_copy,[cnt],-1,(255,255,255),2)
        cv2.circle(img_copy,(cx,cy),4,(255,0,0),-1)
        cv2.putText(img_copy,f'{shape}',(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,0),2)
        cv2.putText(img_copy, f'A:{int(area)}, P:{int(perimeter)}', (x, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img_copy, f'AR:{aspect_ratio}, C:{compactness}',(x,y-40),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
cv2.imshow('image',img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()