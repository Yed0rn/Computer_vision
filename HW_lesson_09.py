import cv2
net=cv2.dnn.readNet("data/MobileNet/mobilenet_deploy.prototxt","data/MobileNet/MobileNet.caffemodel")
#1 Завантажуємо модель MobileNet

#2 Зчитуємо список назв классів

classes=[]
with open("data/MobileNet/synset.txt",'r', encoding="utf-8") as f:
    for line in f:
        line=line.strip()
        if not line:
            continue
        parts=line.split(" ",1)
        name=parts[1] if len(parts)>1 else parts[0]
        classes.append(name)
image=cv2.imread("images/MobileNet/orange.jpg")
blob = cv2.dnn.blobFromImage(
    cv2.resize(image, (224, 224)),
    1.0 / 127.5,
    (224, 224),
    (127.5, 127.5, 127.5)
)

net.setInput(blob)
preds=net.forward()

#6 знаходимо індекс класа з найбільшою вірогіднісю

idx=preds[0].argmax()
label=classes[idx] if idx < len(classes) else "Unknown"
conf = float(preds[0][idx])*100
#8 виводимо результат в консоль
print("Class: ", label)
print("Likelyhood: ", conf)

#9 Підписуємо зображення
text = f'{label}: {int(conf)}%'

cv2.putText(image,text,(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
cv2.imshow("image",image)


image1=cv2.imread("images/MobileNet/pug.jpeg")
blob1 = cv2.dnn.blobFromImage(
    cv2.resize(image1, (224, 224)),
    1.0 / 127.5,
    (224, 224),
    (127.5, 127.5, 127.5)
)

net.setInput(blob1)
preds1=net.forward()

#6 знаходимо індекс класа з найбільшою вірогіднісю

idx1=preds1[0].argmax()
label1=classes[idx1] if idx1 < len(classes) else "Unknown"
conf1 = float(preds1[0][idx1])*100
#8 виводимо результат в консоль
print("Class: ", label1)
print("Likelyhood: ", conf1)

#9 Підписуємо зображення
text1 = f'{label1}: {int(conf1)}%'

cv2.putText(image1,text1,(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
cv2.imshow("image1",image1)


image2=cv2.imread("images/MobileNet/tiger.jpg")
blob2 = cv2.dnn.blobFromImage(
    cv2.resize(image2, (224, 224)),
    1.0 / 127.5,
    (224, 224),
    (127.5, 127.5, 127.5)
)

net.setInput(blob2)
preds2=net.forward()

#6 знаходимо індекс класа з найбільшою вірогіднісю

idx2=preds2[0].argmax()
label2=classes[idx2] if idx < len(classes) else "Unknown"
conf2 = float(preds2[0][idx2])*100

print("Class: ", label2)
print("Likelyhood: ", conf2)


text2= f'{label2}: {int(conf2)}%'

cv2.putText(image2,text2,(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
cv2.imshow("image2",image2)




image3=cv2.imread("images/MobileNet/Horse.png")
blob3 = cv2.dnn.blobFromImage(
    cv2.resize(image3, (224, 224)),
    1.0 / 127.5,
    (224, 224),
    (127.5, 127.5, 127.5)
)

net.setInput(blob3)
preds3=net.forward()

#6 знаходимо індекс класа з найбільшою вірогіднісю

idx3=preds3[0].argmax()
label3=classes[idx3] if idx < len(classes) else "Unknown"
conf3 = float(preds3[0][idx3])*100
#8 виводимо результат в консоль
print("Class: ", label3)
print("Likelyhood: ", conf3)

#9 Підписуємо зображення
text3 = f'{label3}: {int(conf3)}%'

cv2.putText(image3,text3,(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
cv2.imshow("image3",image3)
cv2.waitKey(0)
cv2.destroyAllWindows()

