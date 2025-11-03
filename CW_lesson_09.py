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
image=cv2.imread("images/MobileNet/dog.jfif")
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
cv2.waitKey(0)
cv2.destroyAllWindows()

