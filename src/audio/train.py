import argparse
import numpy as np

from utils.data import get_data
from models.trainer import Trainer

import tensorflow as tf
import tensorflow_model_optimization as tfmot

RANDOM_STATE = 42

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def main(args):

    if args.resample:
        MFCC_OPTIONS = {
            'frame_length': 640 * 2, 'frame_step': 320 * 2, 'mfcc': True, 'lower_frequency': 20,
            'upper_frequency': 4000, 'num_mel_bins': 40, 'num_coefficients': 10
        }
        resampling_rate = 16000
    else:
        MFCC_OPTIONS = {
            'frame_length': 1764 *2 , 'frame_step': 882 * 2, 'mfcc': True, 'lower_frequency': 20,
            'upper_frequency': 4000, 'num_mel_bins': 32, 'num_coefficients': 20
        }
        resampling_rate = None

    labels = ['Bark', 'Doorbell', 'Drill', 'Glass', 'Hammer', 'Speech']

    train_ds, val_ds = get_data(
        labels=labels,
        mfcc_options=MFCC_OPTIONS,
        resampling=resampling_rate)

    for elem in train_ds:
      input_shape = elem[0].shape.as_list()
      break

    learning_rate = 0.001
    epochs = args.epochs

    model = Trainer(model_name=args.model_name, 
                  n_classes=len(labels),
                  input_shape=input_shape, 
                  alpha=1,
                  pruning=True)    

    model.train_model(train_ds, val_ds, learning_rate, input_shape, epochs)
    model.save_tf('./assets/audio/models_tf')
    model.save_tflite(f'./assets/audio/models_tflite/{args.model_name}.tflite')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--epochs', type=int, default=20)
    parser.add_argument('-r', '--resample', type=bool, default=False)
    parser.add_argument('-T', '--only_train', action='store_true')

    parser.add_argument('-m', '--model_name', required=True, choices=['DS-CNN','VGG','VGGish',
                                                                      'Yamnet','MusicTaggerCNN',
                                                                      'MobileNet13','MobileNet3',
                                                                      'MobileNet2'])

    args = parser.parse_args()

    main(args)



