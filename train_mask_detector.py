import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications import MobileNetV2
from keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
from imutils import paths
import os

#intializing intial learning rate, number of epochs and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

DIRECTORY = r'C:\Users\future\Downloads\Face-Mask-Detection-master\dataset'
CATEGORIES = ['with_mask', 'without_mask']

data = []
labels = []

# read images preprocess them, and built the dataset
for category in CATEGORIES:
    path = os.path.join(DIRECTORY,category)
    for img in os.listdir(path):
        img_path = os.path.join(path,img)
        image = load_img(img_path,target_size=(224,224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

# apply one-hot encoding on labels (categories)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# convert the lists to numpy arrays so they are valid to be the input for DL model
data = np.array(data, dtype='float32') #########  HOOOOOOLAAAA   ############
labels = np.array(labels)

print(f'shape of data list : {data.shape}')
print(f'shape of data list : {labels.shape}')
# split the data to begin the preprocessing
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# construct the training image generator for data agumentation
aug = ImageDataGenerator(
    rotation_range= 20,
    zoom_range= 0.15,
    width_shift_range= 0.2,
    height_shift_range= 0.2,
    shear_range= 0.15,
    horizontal_flip=True,
    fill_mode= "nearest"
)

# we will use mobilenet model for classifying images instead of CNN

# loading the mobilenet model and constructing the base model
baseModel = MobileNetV2(weights= "imagenet" , include_top=False, input_tensor=Input(shape=(224,224,3)))
# imagenet:  weight of pre-trained images models
# include_top: checks if to include fully connected layer or not

# constructing the head model that will be replaced on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel) # pooling
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation='relu')(headModel) #
headModel = Dropout(0.5)(headModel)  # drop that there is not overfitting in the model
headModel = Dense(2, activation='softmax')(headModel) # output layer which classify to 2 categories (with mask, without_mask)

#place the head model on the top of the base model (this will be the actual model we will train)
model = keras.Model(inputs=baseModel.input,outputs=headModel)
#model = tensorflow.keras.Sequential([baseModel, pooling, flatten, dense, dropout, dense])

#loop over all layers in base modeland freeze them so they will not be updated during the first training process as we use the pretrained image models from it
for layer in baseModel.layers:
    layer.trainable = False

# compile our model
optimizer = Adam(lr = INIT_LR, weight_decay= INIT_LR/EPOCHS)
model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])

# fitting the model
H = model.fit(
    aug.flow(X_train,y_train,batch_size=BS),
              steps_per_epoch=len(X_train) // BS,
              validation_data= (X_test,y_test),
              validation_steps= len(X_test) // BS,
              epochs= EPOCHS)

# make predictions
predIndx = model.predict(X_test, batch_size=BS)

# finding index of label corresponding to the largest predicted probability for each image in the testing dataset
predIndx = np.argmax(predIndx,axis=1)

#serialize the model to the disk
model.save("mask_detector.model", save_format="h5")

# plot the training loss & accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,N), H.history['loss'], label="train_loss")
plt.plot(np.arange(0,N), H.history['val_loss'], label="val_loss")
plt.plot(np.arange(0,N), H.history['accuracy'], label="train_accuracy")
plt.plot(np.arange(0,N), H.history['val_accuracy'], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss / Accuracy")
plt.legend(loc = "lower left")
plt.savefig("plot.png")



