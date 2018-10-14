from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Convolution2D, Cropping2D
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping

class KerasPilot:

    def load(self, model_path):
        self.model = load_model(model_path)
        
class KerasLinear(KerasPilot):
    def __init__(self, model=None, num_outputs=None, *args, **kwargs):
        super(KerasLinear, self).__init__(*args, **kwargs)
        
        if model:
            print ("model: {}".format(model))
            self.model = model
        elif num_outputs is not None:
            print ("model_num_output: {}".format(num_outputs))

            self.model = default_categorical() #default_linear()
        else:
            print ("model and numouput is None")
            self.model = default_categorical() #default_linear()

    def predict (self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        return outputs
    
    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        # print(len(outputs), outputs)
        steering = outputs[0]
        throttle = outputs[1]
        return steering[0][0], throttle[0][0]

def default_linear():
    img_in = Input(shape=(120, 160, 3), name='img_in')
    x = img_in

    # Convolution2D class name is an alias for Conv2D
    x = Cropping2D(cropping=((45,0), (0,0)))(x)
    x = Convolution2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(filters=64, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
    x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)

    x = Flatten(name='flattened')(x)
    x = Dense(units=100, activation='linear')(x)
    x = Dropout(rate=.1)(x)
    x = Dense(units=50, activation='linear')(x)
    x = Dropout(rate=.1)(x)
    # categorical output of the angle
    angle_out = Dense(units=1, activation='linear', name='angle_out')(x)

    # continous output of throttle
    throttle_out = Dense(units=1, activation='linear', name='throttle_out')(x)

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])

    model.compile(optimizer='adam',
                  loss={'angle_out': 'mean_squared_error',
                        'throttle_out': 'mean_squared_error'},
                  loss_weights={'angle_out': 0.5, 'throttle_out': .5})

    return model

def default_categorical():
    from keras.layers import Input, Dense, merge
    from keras.models import Model
    from keras.layers import Cropping2D, Convolution2D, MaxPooling2D, Reshape, BatchNormalization
    from keras.layers import Activation, Dropout, Flatten, Dense
    
    img_in = Input(shape=(120, 160, 3), name='img_in')                      
    x = img_in
    x = Cropping2D(cropping=((45,0), (0,0)))(x)
    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu')(x)       
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu')(x)       
    x = Convolution2D(64, (5,5), strides=(2,2), activation='relu')(x)       
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)       
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)       
    
    # Possibly add MaxPooling (will make it less sensitive to position in image).  Camera angle fixed, so may not to be needed

    x = Flatten(name='flattened')(x)                                        # Flatten to 1D (Fully connected)
    x = Dense(100, activation='relu')(x)                                    # Classify the data into 100 features, make all negatives 0
    x = Dropout(.1)(x)                                                      # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dense(50, activation='relu')(x)                                     # Classify the data into 50 features, make all negatives 0
    x = Dropout(.1)(x)                                                      # Randomly drop out 10% of the neurons (Prevent overfitting)
    #categorical output of the angle
    angle_out = Dense(15, activation='softmax', name='angle_out')(x)        # Connect every input with every output and output 15 hidden units. Use Softmax to give percentage. 15 categories and find best one based off percentage 0.0-1.0
    
    #continous output of throttle
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)      # Reduce to 1 number, Positive number only
    
    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    model.compile(optimizer='adam',
                  loss={'angle_out': 'categorical_crossentropy', 
                        'throttle_out': 'mean_absolute_error'},
                  loss_weights={'angle_out': 0.9, 'throttle_out': .0001})

    return model

def cam(img_path):
    from keras.applications.vgg16 import VGG16
    import matplotlib.image as mpimg
    from keras import backend as K
    import numpy as np

    #import matplotlib.pyplot as plt
    #%matplotlib inline
    K.clear_session()
    #model = default_linear() #kl #VGG16(weights='imagenet')
    img=mpimg.imread(img_path)
    #plt.imshow(img)
    
    from keras.preprocessing import image
    img = image.load_img(img_path)
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    from keras.applications.vgg16 import preprocess_input
    img_arr = preprocess_input(img_arr)
#    img_arr = img_arr.reshape((1,) + img_arr.shape)
#    img_arr.shape
    preds = kl.run(img_arr)
    #preds = model.predict(x)
    predictions = pd.DataFrame(decode_predictions(preds, top=3)[0],columns=['angle_out','throttle_out']).iloc[:,1:]
    argmax = np.argmax(preds[0])
    output = model.output[:, argmax]
    
    last_conv_layer = model.get_layer('dense_1')
    model.summary()
    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    import cv2
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    hif = .8
    superimposed_img = heatmap * hif + img
    output = 'output.jpeg'
    cv2.imwrite(output, superimposed_img)
    #img=mpimg.imread(output)
    #plt.imshow(img)
    #plt.axis('off')
    #plt.title(predictions.loc[0,'category'].upper())
    return None

model_path = "/home/ec2-user/ml/model/car-model.pkl-blue-office-20181011_022147"
model_path = "/home/ec2-user/ml/model/car-model.pkl"
img_path = '/home/ec2-user/ml/inputdata/20181008_070220/10001_cam-image_array_.jpg'

kl = KerasLinear()
kl.load(model_path)

print(kl)
def heatmap(img_path):
    from keras.preprocessing import image
    import pandas as pd
    from keras import backend as K
    import numpy as np
    
    img = image.load_img(img_path)
    img_arr = image.img_to_array(img)
    model = kl.model
    
    preds = kl.run(img_arr)
    preds = kl.predict(img_arr)

    print (preds[0])
    #predictions = pd.DataFrame(decode_predictions(preds, top=3)[0],columns=['angle_out','throttle_out']).iloc[:,1:]
    
    #print (predictions)
    print ("model output:")
    argmax = np.argmax(preds[0])
    print ("argmax = {}".format(argmax))
    output = preds[:argmax]
    
    
    print(output)
    #print (dir(model))
    model._layers_by_depth
    #for layer in model._layers:
    #    print (layer._name)
    #    print (layer._name_scope_name)
    last_conv_layer = model.get_layer('conv2d_5')

    x = K.gradients(output, last_conv_layer.output)
    print (x)
    #model.summary()
    grads = K.gradients(output, last_conv_layer.output)[0]
    print (grads)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    
heatmap(img_path)