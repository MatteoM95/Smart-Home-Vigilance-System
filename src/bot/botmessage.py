import os
import logging
import pandas as pd
import requests
import numpy as np

from urllib.parse import uses_params
from datetime import datetime
from pytz import timezone
from telegram.ext import Updater, CommandHandler
from telegram.ext.callbackcontext import CallbackContext

from src.bot.bot_settings import * 


class Bot:

    def __init__(self,danger) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

        self.TOKEN = TOKEN

        self.logger = logging.getLogger("LOG")
        self.logger.info("Starting BOT.")
        self.updater = Updater(self.TOKEN)
        self.dispatcher = self.updater.dispatcher
        self.danger = danger
    
    
    def send_start(self, chatbot, update) -> None:
        welcome_message =  "Hello, I am the bot that will keep your home safe!\n\n"
        welcome_message += 'Welcome to the notification centre'
        chatbot.message.reply_text(welcome_message)
    
   
    def send_alarm(self, timestamp, input_type, label, path) -> int:

        reports = []
        current_time = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
        db = self._read_db()
        db.set_index('ids',inplace = True)

        # send image and caption
        for chat_id in CHAT_IDS:
            if db.at[chat_id,"status"] == True:
                self._send_img(open(path, 'rb'),label,chat_id)
        
        # store intrusion on file
        arr = np.genfromtxt('assets/bot/reports.txt',dtype='str')
        reports = np.append(arr,str(current_time))
        np.savetxt('assets/bot/reports.txt', reports, delimiter=" ", fmt="%s")
    

    def _send_img(self,file_opened,label,bot_chatID):

        if label == 'Human':
            emoticon_label = "📷"
        else:
            emoticon_label = "🔈"

        amsterdam = timezone('Europe/Amsterdam')
        timestamp = datetime.now(amsterdam).strftime("%H:%M:%S")

        caption = f"⚠️ *Intrusion Alert* ⚠️\n\n🕚 {timestamp} \n{emoticon_label} {label}"
        
        method = "sendPhoto"
        params = {'chat_id': bot_chatID, 'caption':caption, 'parse_mode':'Markdown'}
        files = {'photo': file_opened}
        url = 'https://api.telegram.org/bot' + TOKEN_MSG + "/"
        response = requests.post(url + method, params, files=files)
        return response.json()


    def _read_db(self):      
        df = pd.read_csv("assets/bot/users.csv")
        return df


    def _update_db(self,df,index,column,value):
        df.at[index, column] = value
        df.to_csv("assets/bot/users.csv")