import cv2
import numpy as np

img=cv2.imread('images/Object.jpg')
img=cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))
img_copy=img.copy()
img=cv2.GaussianBlur(img,(7,7),5)
img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lower_rgb=np.array([0,34,48])
upper_rgb=np.array([179,255,255])
lower_black=np.array([0,0,0])
upper_black=np.array([179,88,115])
mask_rgb=cv2.inRange(img,lower_rgb,upper_rgb)
mask_black=cv2.inRange(img,lower_black,upper_black)
mask_total=cv2.bitwise_or(mask_rgb,mask_black)
img=cv2.bitwise_and(img,img,mask=mask_total)
contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area=cv2.contourArea(cnt)
    if area>150:
        perimeter = cv2.arcLength(cnt, True)
        M=cv2.moments(cnt)
        cx=int(M['m10']/M['m00'])
        cy=int(M['m01']/M['m00'])
        x,y,w,h=cv2.boundingRect(cnt)


        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            shape = "rectangle"
        elif len(approx) ==8:
            shape = "oval"
        else:
            shape = "inshe"
        print(len(approx))
        cv2.drawContours(img_copy, [cnt], -1, (255, 255, 255), 2)
        cv2.circle(img_copy,(cx,cy),4,(255,255,255),-1)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(img_copy, f'{shape}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        cv2.putText(img_copy, f'Area:{area}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        cv2.putText(img_copy, f'{x}, {y}', (x-70, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
cv2.putText(img_copy,"Green",(32,26),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1 )
cv2.putText(img_copy,"Black",(258,18),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1 )
cv2.putText(img_copy,"Red",(57,189),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1 )
cv2.putText(img_copy,"Blue",(200,150),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1 )
cv2.imshow( 'result',img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()