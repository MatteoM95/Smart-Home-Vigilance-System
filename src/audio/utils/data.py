
from constants.path import SPLIT_BASE_PATH
from utils.signal_generator import SignalGenerator


def get_data(labels, resampling, mfcc_options):

   with open('{}/train_split.txt'.format(SPLIT_BASE_PATH) ,"r") as fp:
      train_files = [line.rstrip() for line in fp.readlines()]
    
   with open('{}/val_split.txt'.format(SPLIT_BASE_PATH) ,"r") as fp:
      val_files = [line.rstrip() for line in fp.readlines()]

   generator = SignalGenerator(labels, sampling_rate=44100, resampling_rate=resampling, **mfcc_options)
   
   train_ds = generator.make_dataset(train_files, True)
   val_ds = generator.make_dataset(val_files, False)

   return train_ds, val_ds