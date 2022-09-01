import os
import time
import json
import datetime

from picamera import PiCamera
from src.bot.botds import Bot
from src.MQTT.DoSomething import DoSomething


class Subscriber(DoSomething):

    def notify(self, topic, msg):

        input_json = json.loads(msg)

        timestamp = input_json['timestamp']
        label = input_json['class']
        img_path = input_json['path']

        if label == 'Bark' or label == 'Doorbell': # not an intrusion
            return 

        # keep a window of 5 minutes, if the bot has already sent an alarm
        # in the last 5 minutes, don't do anythings
        if time.time() - self.last_alarm > 300:  

            self.last_alarm = time.time()
            self.bot.send_alarm(timestamp, 'img', label, img_path)



if __name__ == "__main__":
    test = Subscriber("Bot")
    test.run()
    test.myMqttClient.mySubscribe("/devices/M0001")
    test.myMqttClient.mySubscribe("/devices/C0001")

    while True:
        time.sleep(1)