import pyaudio
import json
import wave
import time
import os
import tensorflow as tf
import numpy as np

from datetime import datetime
from array import array
from argparse import ArgumentParser
from io import BytesIO
from datetime import datetime
from pytz import timezone
from src.MQTT.DoSomething import DoSomething

import logging
logging.getLogger().setLevel(logging.INFO)


def main(args):

    if args.store_files:
        if not os.path.isdir('audio_files'):
            os.mkdir('audio_files')

    print("\n\n")

    p = pyaudio.PyAudio()

    publisher = DoSomething("Publisher - Sound Recognition")
    publisher.run()

    time.sleep(1)
    logging.info("The mic is running...")

    while True:

        stream = p.open(format=pyaudio.paInt16, channels=1, rate=args.rate, input=True, frames_per_buffer=args.chunk)

        # wait for a trigger
        while(True):
            temp_data = stream.read(args.chunk)
            temp_chunk = array('h',temp_data)
            volume = max(temp_chunk)
        
            if volume >= 1000:
                break

        # record the audio file & stop stream
        tf_audio = record_audio(args, p, stream)
        tf_mfccs = get_mfccs(tf_audio,args.seconds)

        prediction, probability = make_inference(tf_mfccs, args.tflite_path)
        
        if probability >= 0.5:
            publish_outcome(publisher, prediction, probability)


def record_audio(args, p, stream):

    chunks = int((args.rate / args.chunk) * args.seconds)

    frames = []

    stream.start_stream()
    for _ in range(chunks):
        data = stream.read(args.chunk)
        frames.append(data)
    stream.stop_stream()
    stream.close() 

    if args.store_files:
        amsterdam = timezone('Europe/Amsterdam')
        FILENAME = 'audio_files/{}.wav'.format(str(datetime.now(amsterdam)).replace(" ","_"))
        wf = wave.open(FILENAME, 'wb')
    else:
        buffer = BytesIO()
        buffer.seek(0)
        wf = wave.open(buffer, 'wb')

    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(args.rate)
    wf.writeframes(b''.join(frames))    
    wf.close() 
    
    if args.store_files:
        raw_audio = tf.io.read_file(FILENAME)
        tf_audio, _ = tf.audio.decode_wav(raw_audio) 
    else:
        buffer.seek(0)
        tf_audio, _ = tf.audio.decode_wav(buffer.read()) 
        
    
    tf_audio = tf.squeeze(tf_audio, 1)


    return tf_audio


def get_mfccs(tf_audio,seconds):

    frame_length = 1764 * 2
    frame_step = 882 * 2
    num_mel_bins = 32
    num_coefficients = 20

    spectrogram_width = ((int(44100 * seconds) - frame_length) // frame_step) + 1
    num_spectrogram_bins = frame_length // 2 + 1

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, 44100, num_mel_bins, 4000)

    stft = tf.signal.stft(tf_audio, frame_length, frame_step,
            fft_length=frame_length)
    spectrogram = tf.abs(stft)
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs = mfccs[..., :num_coefficients]

    mfccs = tf.reshape(mfccs, [1, spectrogram_width, num_coefficients, 1])

    return mfccs   


def make_inference(tf_mfccs, tflite_path):

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # give the input
    interpreter.set_tensor(input_details[0]["index"], tf_mfccs)
    interpreter.invoke()

    # get the possible predictions and their probabilities
    predictions = interpreter.get_tensor(output_details[0]['index']).squeeze()
    predictions = tf.nn.softmax(tf.convert_to_tensor(predictions)).numpy()
  
    first_prediction = np.argmax(predictions)
    
    return first_prediction, np.max(predictions)


def publish_outcome(publisher, prediction, probability):
    
    timestamp = datetime.now(timezone('Europe/Amsterdam')).strftime("%m-%d-%Y_%H:%M:%S")

    labels = ['Bark', 'Doorbell', 'Drill', 'Glass', 'Hammer', 'Speech']

    body = {
        'timestamp': timestamp,
        'class': labels[int(prediction)], 
        'path': 'assets/storage/last_image.png'
    }

    publisher.myMqttClient.myPublish("/devices/M0001", json.dumps(body))
    

if __name__ == '__main__':
    
    parser = ArgumentParser()
    
    parser.add_argument('--chunk', type=int, default=4410, help='Set number of chunks')
    parser.add_argument('--seconds', type=float, default=2, help='Set the length of the recording (seconds)')
    parser.add_argument('--rate', type=int, default=44100, help='Set the rate')
    parser.add_argument('--tflite_path', type=str, default='assets/audio/models_tflite/model.tflite', help='tflite_path')
    parser.add_argument('--store_files', type=bool, default=False, help='Store the recorded audio files')
    
    args = parser.parse_args()

    main(args)
