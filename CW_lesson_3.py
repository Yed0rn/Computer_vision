import cv2
import numpy as np


img = np.zeros((512,400,3), np.uint8)
#rgb = bgr
#img[:]=242,0,255 залити все

#img[100:150, 200:280]=242,0,255 відступ прописати y,x

cv2.rectangle(img,(100,100),(200,200),(242,0,255),thickness=cv2.FILLED)
cv2.line(img, (100,100),(200,200),(0,187,255),thickness=2)
print(img.shape)
cv2.line(img,(0,img.shape[0]//2),(img.shape[1],img.shape[0]//2),(0,187,255),thickness=1)
cv2.line(img,(img.shape[1]//2,0),(img.shape[1]//2,img.shape[0]),(0,187,255),thickness=1)

cv2.circle(img,(200,200), 20, (0,255,4), thickness=1)
cv2.putText(img,"Ornatskyi Yehor",(200,400),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,4))


cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()