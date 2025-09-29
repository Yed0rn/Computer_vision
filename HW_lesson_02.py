import cv2
image= cv2.imread('images/face.jpg')
size1=cv2.resize(image,(300,500))
gray1=cv2.cvtColor(size1, cv2.COLOR_BGR2GRAY)
cont=cv2.Canny(gray1 ,100,100)
cv2.imshow('face', cont)
image2= cv2.imread('images/mail.jpg')
size2=cv2.resize(image2,(300,500))
gray2=cv2.cvtColor(size2, cv2.COLOR_BGR2GRAY)
cont2=cv2.Canny(gray2 ,90,90)
cv2.imshow('mail', cont2)




cv2.waitKey(0)
cv2.destroyAllWindows()

