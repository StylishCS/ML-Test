#!/usr/bin/env python
# coding: utf-8

# # 1. Setup

# ## 1.1 Install Dependencies

# In[1]:
# from IPython import get_ipython

# get_ipython().system('pip install protobuf==3.20.0')


# ## 1.2 Import Dependencies

# In[2]:


# Import standard dependencies
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt


# In[3]:


# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf


# ## 1.3 Set GPU Growth

# In[4]:


# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)


# ## 1.4 Create Folder Structures

# In[5]:


# Setup paths
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')


# In[6]:


# Make the directories
os.makedirs(POS_PATH)
os.makedirs(NEG_PATH)
os.makedirs(ANC_PATH)


# # 2. Collect Positives and Anchors

# ## 2.1 Untar Labelled Faces in the Wild Dataset

# In[7]:


# http://vis-www.cs.umass.edu/lfw/


# In[8]:


# Uncompress Tar GZ Labelled Faces in the Wild Dataset
get_ipython().system('tar -xf lfw.tgz')


# In[9]:


# Move LFW Images to the following repository data/negative
for directory in os.listdir('lfw'):
    for file in os.listdir(os.path.join('lfw', directory)):
        EX_PATH = os.path.join('lfw', directory, file)
        NEW_PATH = os.path.join(NEG_PATH, file)
        os.replace(EX_PATH, NEW_PATH)


# ## 2.2 Collect Positive and Anchor Classes

# In[10]:


# Import uuid library to generate unique image names
import uuid


# In[11]:


os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))


# In[12]:


# Establish a connection to the webcam
cap = cv2.VideoCapture(0)
while cap.isOpened(): 
    ret, frame = cap.read()
   
    # Cut down frame to 250x250px
    #frame = frame[120:120+250,200:200+250, :]
    
    # Collect anchors 
    if cv2.waitKey(1) & 0XFF == ord('a'):
        # Create the unique file path 
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        # Write out anchor image
        cv2.imwrite(imgname, frame)
    
    # Collect positives
    if cv2.waitKey(1) & 0XFF == ord('p'):
        # Create the unique file path 
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        # Write out positive image
        cv2.imwrite(imgname, frame)
    
    # Show image back to screen
    cv2.imshow('Image Collection', frame)
    
    # Breaking gracefully
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
        
# Release the webcam
cap.release()
# Close the image show frame
cv2.destroyAllWindows()


# In[13]:


plt.imshow(frame[120:120+250,200:200+250, :])


# # 2.x NEW - Data Augmentation

# In[14]:


def data_aug(img):
    data = []
    for i in range(9):
        img = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1,2))
        img = tf.image.stateless_random_contrast(img, lower=0.6, upper=1, seed=(1,3))
        # img = tf.image.stateless_random_crop(img, size=(20,20,3), seed=(1,2))
        img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_saturation(img, lower=0.9,upper=1, seed=(np.random.randint(100),np.random.randint(100)))
            
        data.append(img)
    
    return data


# In[15]:


import os
import uuid


# In[16]:


img_path = os.path.join(ANC_PATH, '01e37373-058e-11ef-864b-1cbfc0784cb7.jpg')
img = cv2.imread(img_path)
augmented_images = data_aug(img)

for image in augmented_images:
    cv2.imwrite(os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1())), image.numpy())


# In[17]:


for file_name in os.listdir(os.path.join(POS_PATH)):
    img_path = os.path.join(POS_PATH, file_name)
    img = cv2.imread(img_path)
    augmented_images = data_aug(img) 
    
    for image in augmented_images:
        cv2.imwrite(os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1())), image.numpy())


# # 3. Load and Preprocess Images

# ## 3.1 Get Image Directories

# In[18]:


anchor = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(3000)
positive = tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(3000)
negative = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(3000)


# In[19]:


dir_test = anchor.as_numpy_iterator()


# In[20]:


print(dir_test.next())


# ## 3.2 Preprocessing - Scale and Resize

# In[21]:


def preprocess(file_path):
    
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image 
    img = tf.io.decode_jpeg(byte_img)
    
    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100,100))
    # Scale image to be between 0 and 1 
    img = img / 255.0

    # Return image
    return img


# In[22]:


img = preprocess('data\\anchor\\64703789-0562-11ef-8d95-1cbfc0784cb7.jpg')


# In[23]:


img.numpy().max() 


# ## 3.3 Create Labelled Dataset

# In[25]:


# (anchor, positive) => 1,1,1,1,1
# (anchor, negative) => 0,0,0,0,0


# In[26]:


positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)


# In[27]:


samples = data.as_numpy_iterator()


# In[28]:


exampple = samples.next()


# In[29]:


exampple


# ## 3.4 Build Train and Test Partition

# In[30]:


def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)


# In[31]:


res = preprocess_twin(*exampple)


# In[32]:


plt.imshow(res[1])


# In[33]:


res[2]


# In[34]:


# Build dataloader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=10000)


# In[35]:


# Training partition
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)


# In[36]:


# Testing partition
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)


# # 4. Model Engineering

# ## 4.1 Build Embedding Layer

# In[37]:


inp = Input(shape=(100,100,3), name='input_image')


# In[38]:


c1 = Conv2D(64, (10,10), activation='relu')(inp)


# In[39]:


m1 = MaxPooling2D(64, (2,2), padding='same')(c1)


# In[40]:


c2 = Conv2D(128, (7,7), activation='relu')(m1)
m2 = MaxPooling2D(64, (2,2), padding='same')(c2)


# In[41]:


c3 = Conv2D(128, (4,4), activation='relu')(m2)
m3 = MaxPooling2D(64, (2,2), padding='same')(c3)


# In[42]:


c4 = Conv2D(256, (4,4), activation='relu')(m3)
f1 = Flatten()(c4)
d1 = Dense(4096, activation='sigmoid')(f1)


# In[43]:


mod = Model(inputs=[inp], outputs=[d1], name='embedding')


# In[44]:


mod.summary()


# In[45]:


def make_embedding(): 
    inp = Input(shape=(100,100,3), name='input_image')
    
    # First block
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    
    # Second block
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
    # Third block 
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    
    # Final embedding block
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')


# In[46]:


embedding = make_embedding()


# In[47]:


embedding.summary()


# ## 4.2 Build Distance Layer

# In[48]:


# Siamese L1 Distance class
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


# In[49]:


l1 = L1Dist()


# ## 4.3 Make Siamese Model

# In[50]:


input_image = Input(name='input_img', shape=(100,100,3))
validation_image = Input(name='validation_img', shape=(100,100,3))


# In[51]:


inp_embedding = embedding(input_image)
val_embedding = embedding(validation_image)


# In[52]:


siamese_layer = L1Dist()


# In[53]:


distances = siamese_layer(inp_embedding, val_embedding)


# In[54]:


classifier = Dense(1, activation='sigmoid')(distances)


# In[55]:


classifier


# In[56]:


siamese_network = Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


# In[57]:


siamese_network.summary()


# In[58]:


def make_siamese_model(): 
    
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100,100,3))
    
    # Validation image in the network 
    validation_image = Input(name='validation_img', shape=(100,100,3))
    
    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    # Classification layer 
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


# In[59]:


siamese_model = make_siamese_model()


# In[60]:


siamese_model.summary()


# # 5. Training

# ## 5.1 Setup Loss and Optimizer

# In[61]:


binary_cross_loss = tf.losses.BinaryCrossentropy()


# In[62]:


opt = tf.keras.optimizers.Adam(1e-4) # 0.0001


# ## 5.2 Establish Checkpoints

# In[63]:


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)


# ## 5.3 Build Train Step Function

# In[64]:


test_batch = train_data.as_numpy_iterator()


# In[65]:


batch_1 = test_batch.next()


# In[66]:


X = batch_1[:2]


# In[67]:


y = batch_1[2]


# In[68]:


y


# In[56]:


get_ipython().run_line_magic('pinfo2', 'tf.losses.BinaryCrossentropy')


# In[69]:


@tf.function
def train_step(batch):
    
    # Record all of our operations 
    with tf.GradientTape() as tape:     
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]
        
        # Forward pass
        yhat = siamese_model(X, training=True)
        # Calculate loss
        loss = binary_cross_loss(y, yhat)
    print(loss)
        
    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    
    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
        
    # Return loss
    return loss


# ## 5.4 Build Training Loop

# In[70]:


# Import metric calculations
from tensorflow.keras.metrics import Precision, Recall


# In[71]:


def train(data, EPOCHS):
    # Loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        
        # Creating a metric object 
        r = Recall()
        p = Precision()
        
        # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train step here
            loss = train_step(batch)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat) 
            progbar.update(idx+1)
        print(loss.numpy(), r.result().numpy(), p.result().numpy())
        
        # Save checkpoints
        if epoch % 10 == 0: 
            checkpoint.save(file_prefix=checkpoint_prefix)


# ## 5.5 Train the model

# In[72]:


EPOCHS = 50


# In[75]:


train(train_data, EPOCHS)


# # 6. Evaluate Model

# ## 6.1 Import Metrics

# In[76]:


# Import metric calculations
from tensorflow.keras.metrics import Precision, Recall


# ## 6.2 Make Predictions

# In[77]:


# Get a batch of test data
test_input, test_val, y_true = test_data.as_numpy_iterator().next()


# In[78]:


y_hat = siamese_model.predict([test_input, test_val])


# In[79]:


# Post processing the results 
[1 if prediction > 0.5 else 0 for prediction in y_hat ]


# In[80]:


y_true


# ## 6.3 Calculate Metrics

# In[81]:


# Creating a metric object 
m = Recall()

# Calculating the recall value 
m.update_state(y_true, y_hat)

# Return Recall Result
m.result().numpy()


# In[82]:


# Creating a metric object 
m = Precision()

# Calculating the recall value 
m.update_state(y_true, y_hat)

# Return Recall Result
m.result().numpy()


# In[83]:


r = Recall()
p = Precision()

for test_input, test_val, y_true in test_data.as_numpy_iterator():
    yhat = siamese_model.predict([test_input, test_val])
    r.update_state(y_true, yhat)
    p.update_state(y_true,yhat) 

print(r.result().numpy(), p.result().numpy())


# ## 6.4 Viz Results

# In[84]:


# Set plot size 
plt.figure(figsize=(10,8))

# Set first subplot
plt.subplot(1,2,1)
plt.imshow(test_input[0])

# Set second subplot
plt.subplot(1,2,2)
plt.imshow(test_val[0])

# Renders cleanly
plt.show()


# # 7. Save Model

# In[85]:


# Save weights
siamese_model.save('siamesemodelv2.h5')


# In[86]:


# Reload model 
siamese_model = tf.keras.models.load_model('siamesemodelv2.h5', 
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})


# In[87]:


# Make predictions with reloaded model
siamese_model.predict([test_input, test_val])


# In[88]:


# View model summary
siamese_model.summary()


# # 8. Real Time Test

# ## 8.1 Verification Function

# In[90]:


os.listdir(os.path.join('application_data', 'verification_images'))


# In[91]:


os.path.join('application_data', 'input_image', 'input_image.jpg')


# In[92]:


for image in os.listdir(os.path.join('application_data', 'verification_images')):
    validation_img = os.path.join('application_data', 'verification_images', image)
    print(validation_img)


# In[93]:


def verify(model, detection_threshold, verification_threshold):
    # Build results array
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'verification_images', image))
        
        # Make Predictions 
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
    
    # Detection Threshold: Metric above which a prediciton is considered positive 
    detection = np.sum(np.array(results) > detection_threshold)
    
    # Verification Threshold: Proportion of positive predictions / total positive samples 
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
    verified = verification > verification_threshold
    
    return results, verified


# ## 8.2 OpenCV Real Time Verification

# In[95]:


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
   # frame = frame[120:120+250,200:200+250, :]
    
    cv2.imshow('Verification', frame)
    
    # Verification trigger
    if cv2.waitKey(10) & 0xFF == ord('v'):
        # Save input image to application_data/input_image folder 
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         h, s, v = cv2.split(hsv)

#         lim = 255 - 10
#         v[v > lim] = 255
#         v[v <= lim] -= 10
        
#         final_hsv = cv2.merge((h, s, v))
#         img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
        # Run verification
        results, verified = verify(siamese_model, 0.7, 0.7)
        print(verified)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[96]:


np.sum(np.squeeze(results) > 0.9)


# In[18]:


results


# In[98]:




