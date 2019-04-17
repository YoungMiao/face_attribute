import cv2
import numpy as np
from wide_resnet import WideResNet
from tqdm import tqdm
from keras.models import load_model

# age-gender detect loading model
#depth of network
depth = 16
#width of network
k = 8
#margin around detected face for age-gender estimation
margin = 0.4
weight_file="./models/age_gender_model.hdf5"
# load model and weights
img_size = 64
model = WideResNet(img_size, depth=depth, k=k)()
model.load_weights(weight_file)


# emotion detect loading model
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = {0:'angry',1:'disgust',2:'fear',3:'happy',
                4:'sad',5:'surprise',6:'neutral'}
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_target_size = emotion_classifier.input_shape[1:3]

def AgeAndGenderDetect(image):
    faces = np.empty((1, img_size, img_size, 3))
    faces[0, :, :, :] = cv2.resize(image, (img_size, img_size))

    # predict ages and genders of the detected faces
    results = model.predict(faces)
    predicted_genders = results[0]
    ages = np.arange(0, 101).reshape(101, 1)
    predicted_ages = results[1].dot(ages).flatten()
    label = "{}, {}".format(int(predicted_ages[0]),
                                        0 if predicted_genders[0][0] < 0.5 else 1)
    gender = 0 if predicted_genders[0][0] < 0.5 else 1
    age = int(predicted_ages[0])
    return (gender,age)
    

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def EmotionDetect(image):
    print("start")
    imggray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray_image = np.expand_dims(imggray,axis=2)#224*224*1
    gray_face = cv2.resize(gray_image, (emotion_target_size))
    gray_face = preprocess_input(gray_face, True)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_prediction = emotion_classifier.predict(gray_face)
    emotion_label_arg = np.argmax(emotion_prediction)


    return emotion_label_arg 

if __name__ == '__main__':
    image = cv2.imread("1.png")
    (gender,age) = AgeAndGenderDetect(image)
    emotion = EmotionDetect(image)
    print(gender,age,emotion)
