from src.MQTT.MyMQTT import MyMQTT
from src.bot.botmessage import Bot

import time

class DoSomething():
    def __init__(self, clientID):
        # create an instance of MyMQTT class
        self.last_alarm = time.time() - 10000 # enough for the window
        self.bot = Bot(True)
        self.clientID = clientID
        self.myMqttClient = MyMQTT(self.clientID, "test.mosquitto.org", 1883, self)

    def run(self):
        # if needed, perform some other actions befor starting the mqtt communication
        print ("running %s" % (self.clientID))
        self.myMqttClient.start()

    def end(self):
        # if needed, perform some other actions befor ending the software
        print ("ending %s" % (self.clientID))
        self.myMqttClient.stop ()

    def notify(self, topic, msg):
        # manage here your received message. You can perform some error-check here
        print ("received '%s' under topic '%s'" % (msg, topic))