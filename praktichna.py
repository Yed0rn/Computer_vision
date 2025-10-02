import cv2
import numpy as np
img=np.zeros((400,600,3),np.uint8)
foto=cv2.imread("images/face.jpg")
QR_code=cv2.imread("images/QR.jpeg")
size1=cv2.resize(foto,(120,140))
size2=cv2.resize(QR_code,(110,110))
img[:]=228,255,150
cv2.rectangle(img,(10,10),(590,390),(255,189,58),thickness=3)
img[30:170,30:150] = size1
img[220:330,460:570]=size2
cv2.putText(img,"Ornatskyi Yehor",(180,90),cv2.FONT_HERSHEY_DUPLEX,1.25,(0,0,0))
cv2.putText(img,"Computer Vision Student",(180,140),cv2.FONT_HERSHEY_DUPLEX,0.9,(114,114,114))
cv2.putText(img,"Email: e.ornatsky@gmail.com",(180,220),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,75,56))
cv2.putText(img,"Phone: +380501042939",(180,255),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,75,56))
cv2.putText(img,"02/10/2025",(180,290),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,75,56))
cv2.putText(img,"OpenCV Business Card",(160,360),cv2.FONT_HERSHEY_DUPLEX,0.9,(0,0,0))





cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()