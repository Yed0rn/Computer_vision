import cv2

img = cv2.imread("images/face.jpg")

size1=cv2.resize(img,(img.shape[1]//4,img.shape[0]//4))
cv2.imshow('face',size1)
cv2.rectangle(size1,(150,166),(630,940),(0,255,4),3)
cv2.putText(size1,"Ornatskyi Yehor",(330,960),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,4))
cv2.imshow('face',size1)


cv2.waitKey(0)
cv2.destroyAllWindows()