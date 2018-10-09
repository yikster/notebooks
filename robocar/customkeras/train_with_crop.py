#!/usr/bin/env python3


import os
import numpy as np
import keras
import tensorflow as tf
import pandas as pd
import glob
import json
import random
import time
import sys
import traceback

from PIL import Image

BATCH_SIZE = 128 * 2 
TRAINING_SPLIT = 0.8
EPOCHS = 20

prefix = '/home/ec2-user/ml/'
input_path = prefix + 'inputdata'
#prefix = '/opt/ml/'
#input_path = prefix + 'input/data'
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')
#param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

model_loc = os.path.join(model_path, 'car-model.pkl')

# This algorithm has a single channel of input data called 'training'. Since we run in
# File mode, the input files are copied to the directory specified here.
#channel_name='training'
channel_name='tub_20181009_115228'
training_path = os.path.join(input_path, channel_name)
training_paths = [training_path, training_path]
INPUT_TENSOR_NAME = "inputs"
SIGNATURE_NAME = "serving_default"
LEARNING_RATE = 0.001

class Tub(object):
    """
    A datastore to store sensor data in a key, value format.

    Accepts str, int, float, image_array, image, and array data types.

    For example:

    #Create a tub to store speed values.
    >>> path = '~/mydonkey/test_tub'
    >>> inputs = ['user/speed', 'cam/image']
    >>> types = ['float', 'image']
    >>> t=Tub(path=path, inputs=inputs, types=types)

    """

    def __init__(self, path, inputs=None, types=None):

        self.path = os.path.expanduser(path)
        print('path_in_tub:', self.path)
        self.meta_path = os.path.join(self.path, 'meta.json')
        self.df = None

        exists = os.path.exists(self.path)

        if exists:
            #load log and meta
            print("Tub exists: {}".format(self.path))
            with open(self.meta_path, 'r') as f:
                self.meta = json.load(f)
            self.current_ix = self.get_last_ix() + 1

        elif not exists and inputs:
            print('Tub does NOT exist. Creating new tub...')
            #create log and save meta
            os.makedirs(self.path)
            self.meta = {'inputs': inputs, 'types': types}
            with open(self.meta_path, 'w') as f:
                json.dump(self.meta, f)
            self.current_ix = 0
            print('New tub created at: {}'.format(self.path))
        else:
            msg = "The tub path you provided doesn't exist and you didnt pass any meta info (inputs & types)" +                   "to create a new tub. Please check your tub path or provide meta info to create a new tub."

            raise AttributeError(msg)

        self.start_time = time.time()


    def get_last_ix(self):
        index = self.get_index()
        return max(index)

    def update_df(self):
        df = pd.DataFrame([self.get_json_record(i) for i in self.get_index(shuffled=False)])
        self.df = df

    def get_df(self):
        if self.df is None:
            self.update_df()
        return self.df


    def get_index(self, shuffled=True):
        files = next(os.walk(self.path))[2]
        record_files = [f for f in files if f[:6]=='record']
        
        def get_file_ix(file_name):
            try:
                name = file_name.split('.')[0]
                num = int(name.split('_')[1])
            except:
                num = 0
            return num

        nums = [get_file_ix(f) for f in record_files]
        
        if shuffled:
            random.shuffle(nums)
        else:
            nums = sorted(nums)
            
        return nums 


    @property
    def inputs(self):
        return list(self.meta['inputs'])

    @property
    def types(self):
        return list(self.meta['types'])

    def get_input_type(self, key):
        input_types = dict(zip(self.inputs, self.types))
        return input_types.get(key)

    def write_json_record(self, json_data):
        path = self.get_json_record_path(self.current_ix)
        try:
            with open(path, 'w') as fp:
                json.dump(json_data, fp)
                #print('wrote record:', json_data)
        except TypeError:
            print('troubles with record:', json_data)
        except FileNotFoundError:
            raise
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    def get_num_records(self):
        import glob
        files = glob.glob(os.path.join(self.path, 'record_*.json'))
        return len(files)


    def make_record_paths_absolute(self, record_dict):
        #make paths absolute
        d = {}
        for k, v in record_dict.items():
            if type(v) == str: #filename
                if '.' in v:
                    v = os.path.join(self.path, v)
            d[k] = v

        return d




    def check(self, fix=False):
        '''
        Iterate over all records and make sure we can load them.
        Optionally remove records that cause a problem.
        '''
        print('Checking tub:%s.' % self.path)
        print('Found: %d records.' % self.get_num_records())
        problems = False
        for ix in self.get_index(shuffled=False):
            try:
                self.get_record(ix)
            except:
                problems = True
                if fix == False:
                    print('problems with record:', self.path, ix)
                else:
                    print('problems with record, removing:', self.path, ix)
                    self.remove_record(ix)
        if not problems:
            print("No problems found.")

    def remove_record(self, ix):
        '''
        remove data associate with a record
        '''
        record = self.get_json_record_path(ix)
        os.unlink(record)

    def put_record(self, data):
        """
        Save values like images that can't be saved in the csv log and
        return a record with references to the saved values that can
        be saved in a csv.
        """
        json_data = {}
        self.current_ix += 1
        
        for key, val in data.items():
            typ = self.get_input_type(key)

            if typ in ['str', 'float', 'int', 'boolean']:
                json_data[key] = val

            elif typ is 'image':
                path = self.make_file_path(key)
                val.save(path)
                json_data[key]=path

            elif typ == 'image_array':
                img = Image.fromarray(np.uint8(val))
                name = self.make_file_name(key, ext='.jpg')
                img.save(os.path.join(self.path, name))
                json_data[key]=name

            else:
                msg = 'Tub does not know what to do with this type {}'.format(typ)
                raise TypeError(msg)

        self.write_json_record(json_data)
        return self.current_ix


    def get_json_record_path(self, ix):
        return os.path.join(self.path, 'record_'+str(ix)+'.json')

    def get_json_record(self, ix):
        path = self.get_json_record_path(ix)
        try:
            with open(path, 'r') as fp:
                json_data = json.load(fp)
        except UnicodeDecodeError:
            raise Exception('bad record: %d. You may want to run `python manage.py check --fix`' % ix)
        except FileNotFoundError:
            raise
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

        record_dict = self.make_record_paths_absolute(json_data)
        return record_dict


    def get_record(self, ix):

        json_data = self.get_json_record(ix)
        data = self.read_record(json_data)
        return data



    def read_record(self, record_dict):
        data={}
        for key, val in record_dict.items():
            typ = self.get_input_type(key)

            #load objects that were saved as separate files
            if typ == 'image_array':
                img = Image.open((val))
                val = np.array(img)

            data[key] = val


        return data


    def make_file_name(self, key, ext='.png'):
        name = '_'.join([str(self.current_ix), key, ext])
        name = name = name.replace('/', '-')
        return name

    def delete(self):
        """ Delete the folder and files for this tub. """
        import shutil
        shutil.rmtree(self.path)

    def shutdown(self):
        pass


    def get_record_gen(self, record_transform=None, shuffle=True, df=None):

        if df is None:
            df = self.get_df()


        while True:
            for row in self.df.iterrows():
                if shuffle:
                    record_dict = df.sample(n=1).to_dict(orient='record')[0]

                if record_transform:
                    record_dict = record_transform(record_dict)

                record_dict = self.read_record(record_dict)

                yield record_dict


    def get_batch_gen(self, keys, record_transform=None, batch_size=128, shuffle=True, df=None):

        record_gen = self.get_record_gen(record_transform, shuffle=shuffle, df=df)

        if keys == None:
            keys = list(self.df.columns)

        while True:
            record_list = []
            for _ in range(batch_size):
                record_list.append(next(record_gen))

            batch_arrays = {}
            for i, k in enumerate(keys):
                arr = np.array([r[k] for r in record_list])
                # if len(arr.shape) == 1:
                #    arr = arr.reshape(arr.shape + (1,))
                batch_arrays[k] = arr

            yield batch_arrays


    def get_train_gen(self, X_keys, Y_keys, batch_size=128, record_transform=None, df=None):

        batch_gen = self.get_batch_gen(X_keys + Y_keys,
                                       batch_size=batch_size, record_transform=record_transform, df=df)

        while True:
            batch = next(batch_gen)
            X = [batch[k] for k in X_keys]
            Y = [batch[k] for k in Y_keys]
            yield X, Y


    def get_train_val_gen(self, X_keys, Y_keys, batch_size=128, record_transform=None, train_frac=.8):
        train_df = train=self.df.sample(frac=train_frac,random_state=200)
        val_df = self.df.drop(train_df.index)

        train_gen = self.get_train_gen(X_keys=X_keys, Y_keys=Y_keys, batch_size=batch_size,
                                       record_transform=record_transform, df=train_df)

        val_gen = self.get_train_gen(X_keys=X_keys, Y_keys=Y_keys, batch_size=batch_size,
                                       record_transform=record_transform, df=val_df)

        return train_gen, val_gen






class TubHandler():
    def __init__(self, path):
        self.path = os.path.expanduser(path)

    def get_tub_list(self,path):
        folders = next(os.walk(path))[1]
        return folders

    def next_tub_number(self, path):
        def get_tub_num(tub_name):
            try:
                num = int(tub_name.split('_')[1])
            except:
                num = 0
            return num

        folders = self.get_tub_list(path)
        numbers = [get_tub_num(x) for x in folders]
        #numbers = [i for i in numbers if i is not None]
        next_number = max(numbers+[0]) + 1
        return next_number

    def create_tub_path(self):
        tub_num = self.next_tub_number(self.path)
        date = datetime.datetime.now().strftime('%y-%m-%d')
        name = '_'.join(['tub',str(tub_num),date])
        tub_path = os.path.join(self.path, name)
        return tub_path

    def new_tub_writer(self, inputs, types):
        tub_path = self.create_tub_path()
        tw = TubWriter(path=tub_path, inputs=inputs, types=types)
        return tw



class TubImageStacker(Tub):
    '''
    A Tub for training a NN with images that are the last three records stacked 
    togther as 3 channels of a single image. The idea is to give a simple feedforward
    NN some chance of building a model based on motion.
    If you drive with the ImageFIFO part, then you don't need this.
    Just make sure your inference pass uses the ImageFIFO that the NN will now expect.
    '''
    
    def rgb2gray(self, rgb):
        '''
        take a numpy rgb image return a new single channel image converted to greyscale
        '''
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    def stack3Images(self, img_a, img_b, img_c):
        '''
        convert 3 rgb images into grayscale and put them into the 3 channels of
        a single output image
        '''
        width, height, _ = img_a.shape

        gray_a = self.rgb2gray(img_a)
        gray_b = self.rgb2gray(img_b)
        gray_c = self.rgb2gray(img_c)
        
        img_arr = np.zeros([width, height, 3], dtype=np.dtype('B'))

        img_arr[...,0] = np.reshape(gray_a, (width, height))
        img_arr[...,1] = np.reshape(gray_b, (width, height))
        img_arr[...,2] = np.reshape(gray_c, (width, height))

        return img_arr

    def get_record(self, ix):
        '''
        get the current record and two previous.
        stack the 3 images into a single image.
        '''
        data = super(TubImageStacker, self).get_record(ix)

        if ix > 1:
            data_ch1 = super(TubImageStacker, self).get_record(ix - 1)
            data_ch0 = super(TubImageStacker, self).get_record(ix - 2)

            json_data = self.get_json_record(ix)
            for key, val in json_data.items():
                typ = self.get_input_type(key)

                #load objects that were saved as separate files
                if typ == 'image':
                    val = self.stack3Images(data_ch0[key], data_ch1[key], data[key])
                    data[key] = val
                elif typ == 'image_array':
                    img = self.stack3Images(data_ch0[key], data_ch1[key], data[key])
                    val = np.array(img)

        return data



class TubTimeStacker(TubImageStacker):
    '''
    A Tub for training N with records stacked through time. 
    The idea here is to force the network to learn to look ahead in time.
    Init with an array of time offsets from the current time.
    '''

    def __init__(self, frame_list, *args, **kwargs):
        '''
        frame_list of [0, 10] would stack the current and 10 frames from now records togther in a single record
        with just the current image returned.
        [5, 90, 200] would return 3 frames of records, ofset 5, 90, and 200 frames in the future.

        '''
        super(TubTimeStacker, self).__init__(*args, **kwargs)
        self.frame_list = frame_list
  
    def get_record(self, ix):
        '''
        stack the N records into a single record.
        Each key value has the record index with a suffix of _N where N is
        the frame offset into the data.
        '''
        data = {}
        for i, iOffset in enumerate(self.frame_list):
            iRec = ix + iOffset
            
            try:
                json_data = self.get_json_record(iRec)
            except FileNotFoundError:
                pass
            except:
                pass

            for key, val in json_data.items():
                typ = self.get_input_type(key)

                #load only the first image saved as separate files
                if typ == 'image' and i == 0:
                    val = Image.open(os.path.join(self.path, val))
                    data[key] = val                    
                elif typ == 'image_array' and i == 0:
                    d = super(TubTimeStacker, self).get_record(ix)
                    data[key] = d[key]
                else:
                    '''
                    we append a _offset to the key
                    so user/angle out now be user/angle_0
                    '''
                    new_key = key + "_" + str(iOffset)
                    data[new_key] = val
        return data


class TubGroup(Tub):
    def __init__(self, tub_paths_arg):
        tub_paths = expand_path_arg(tub_paths_arg)
        
        print('TubGroup:tubpaths:', tub_paths)
        tubs = [Tub(path) for path in tub_paths]
        self.input_types = {}

        record_count = 0
        for t in tubs:
            t.update_df()
            record_count += len(t.df)
            self.input_types.update(dict(zip(t.inputs, t.types)))

        print('joining the tubs {} records together. This could take {} minutes.'.format(record_count,
                                                                                         int(record_count / 300000)))

        self.meta = {'inputs': list(self.input_types.keys()),
                     'types': list(self.input_types.values())}


        self.df = pd.concat([t.df for t in tubs], axis=0, join='inner')
        
def expand_path_arg(path_str):
    path_list = path_str.split(",")
    expanded_paths = []
    for path in path_list:
        paths = expand_path_mask(path)
        expanded_paths += paths
    return expanded_paths


def expand_path_mask(path):
    matches = []
    path = os.path.expanduser(path)
    for file in glob.glob(path):
        if os.path.isdir(file):
            matches.append(os.path.join(os.path.abspath(file)))
    return matches

def linear_bin(a):
    a = a + 1
    b = round(a / (2/14))
    arr = np.zeros(15)
    arr[int(b)] = 1
    return arr

def rt(record):
        record['user/angle'] = linear_bin(record['user/angle'])
        return record
        
def default_categorical():
    from keras.layers import Input, Dense, merge
    from keras.models import Model
    from keras.layers import Cropping2D, Convolution2D, MaxPooling2D, Reshape, BatchNormalization
    from keras.layers import Activation, Dropout, Flatten, Dense
    
    img_in = Input(shape=(120, 160, 3), name='img_in')                      # First layer, input layer, Shape comes from camera.py resolution, RGB
    x = img_in
    x = Cropping2D(cropping=((45,0), (0,0)))(x)
    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu')(x)       # 24 features, 5 pixel x 5 pixel kernel (convolution, feauture) window, 2wx2h stride, relu activation
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu')(x)       # 32 features, 5px5p kernel window, 2wx2h stride, relu activatiion
    x = Convolution2D(64, (5,5), strides=(2,2), activation='relu')(x)       # 64 features, 5px5p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)       # 64 features, 3px3p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)       # 64 features, 3px3p kernal window, 1wx1h stride, relu

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
                  loss_weights={'angle_out': 0.9, 'throttle_out': .001})

    return model



def train():

    print('Starting the training.')
    try:
        # Read in any hyperparameters that the user passed with the training job
        #with open(param_path, 'r') as tc:
        #    trainingParams = json.load(tc)

        #input_files = [ os.path.join(training_path, file) for file in os.listdir(training_path) ]
        input_files = []
        print (training_paths)
        for path in training_paths:
            print(path) 
            input_files.extend([ os.path.join(training_path, file) for file in os.listdir(path) ])
            print(len(input_files))

        for filename in input_files:
            print(filename) 

        if len(input_files) == 0:
            raise ValueError(('There are no files in {}.\n' +
                              'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                              'the data specification in S3 was incorrectly specified or the role specified\n' +
                              'does not have permission to access the data.').format(training_path, channel_name))

        tubgroup = TubGroup(training_path)


        total_records = len(tubgroup.df)
        total_train = int(total_records * TRAINING_SPLIT)
        total_val = total_records - total_train
        steps_per_epoch = total_train // BATCH_SIZE
        X_keys = ['cam/image_array']
        y_keys = ['user/angle', 'user/throttle']
        train_gen, val_gen = tubgroup.get_train_val_gen(X_keys, y_keys, record_transform=rt,
                                                        batch_size=BATCH_SIZE,
                                                        train_frac=TRAINING_SPLIT)
        save_best = keras.callbacks.ModelCheckpoint(model_loc, 
                                                        monitor='val_loss', 
                                                        verbose=1, 
                                                        save_best_only=True, 
                                                        mode='min')

        #stop training if the validation error stops improving.
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                       min_delta=0.0005, 
                                                       patience=2, 
                                                       verbose=1, 
                                                       mode='auto')
        callbacks_list = [save_best]
        callbacks_list.append(early_stop)
        model = default_categorical()
        hist = model.fit_generator(
                            train_gen, 
                            steps_per_epoch=steps_per_epoch, 
                            epochs=EPOCHS, 
                            verbose=1, 
                            validation_data=val_gen,
                            callbacks=callbacks_list, 
                            validation_steps=steps_per_epoch*(1.0 - TRAINING_SPLIT))
        print('Training complete.')
        
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)
    
if __name__ == '__main__':
    train()

    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)



        
