import os
import tensorflow as tf

from scipy import signal


class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step,
                 num_mel_bins=None, lower_frequency=None, upper_frequency=None,
                 num_coefficients=None, mfcc=False, resampling_rate=None, seconds=2):

        self.labels = labels

        self.sampling_rate = sampling_rate
        self.resampling_rate = resampling_rate

        self.frame_length = frame_length
        self.frame_step = frame_step

        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency

        self.num_mel_bins = num_mel_bins
        self.num_coefficients = num_coefficients

        num_spectrogram_bins = (frame_length) // 2 + 1
        rate = self.resampling_rate if self.resampling_rate else self.sampling_rate

        self.seconds = seconds

        if mfcc is True:
            self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                self.num_mel_bins, num_spectrogram_bins, rate,
                self.lower_frequency, self.upper_frequency)
            self.preprocess = self.preprocess_with_mfcc
        else:
            self.preprocess = self.preprocess_with_stft
    
    
    def apply_resampling(self, audio):
        """Resample waveform if required."""
        if self.sampling_rate != self.resampling_rate:
            desired_length = int(round(float(len(audio)) /
                                    self.sampling_rate * self.resampling_rate))
            audio = signal.resample(audio, desired_length)
        return audio
    
    def apply_resampling_old(self, audio):
        audio = signal.resample_poly(audio, 1, self.sampling_rate // self.resampling_rate)
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)
        return audio

    def read(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]
        label_id = tf.argmax(label == self.labels)
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)

        if self.resampling_rate:
            audio = tf.numpy_function(self.apply_resampling, [audio], tf.float32)

        return audio, label_id

    def pad(self, audio):

        if self.resampling_rate is not None:
            rate = self.resampling_rate
        else:
            rate = self.sampling_rate

        zero_padding = tf.zeros([int(self.seconds * rate)] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([int(self.seconds * rate)])

        return audio

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(audio, frame_length=self.frame_length,
                              frame_step=self.frame_step, fft_length=self.frame_length)
        spectrogram = tf.abs(stft)

        return spectrogram

    def get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram,
                                       self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]

        return mfccs

    def preprocess_with_stft(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [32, 32])

        return spectrogram, label

    def read_pad(self, file_path):

        audio, label = self.read(file_path)

        if self.resampling_rate:
            rate = self.resampling_rate
        else:
            rate = self.sampling_rate

        if tf.shape(audio) > int(self.seconds * rate):
            audio = audio[:int(self.seconds * rate)]
        else:
            audio = self.pad(audio)

        return audio, label

    def preprocess_with_mfcc(self, audio, label):

        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)

        return mfccs, label

    def make_dataset(self, files, train):

        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(self.read_pad, num_parallel_calls=4)
        ds = ds.map(self.preprocess, num_parallel_calls=4)

        ds = ds.batch(32)
        ds = ds.cache()

        if train:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds
