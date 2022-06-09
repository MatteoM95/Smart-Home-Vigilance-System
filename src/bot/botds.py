import logging
import pandas as pd
import telegram
import os.path
import os
import numpy as np

from src.bot.bot_settings import * 
from urllib.parse import uses_params
from telegram.ext.callbackcontext import CallbackContext
from datetime import datetime
from telegram.ext import Updater, CommandHandler


class Bot:

    def __init__(self) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

        self.TOKEN = TOKEN

        self.logger = logging.getLogger("LOG")
        self.logger.info("Starting BOT.")
        self.updater = Updater(self.TOKEN)
        self.dispatcher = self.updater.dispatcher
        
        enable_handler = CommandHandler("enable", self.send_enable)
        self.dispatcher.add_handler(enable_handler)

        enable_handler = CommandHandler("disable", self.send_disable)
        self.dispatcher.add_handler(enable_handler)

        start_handler = CommandHandler("start", self.send_start)
        self.dispatcher.add_handler(start_handler)

        help_handler = CommandHandler("help", self.send_help)
        self.dispatcher.add_handler(help_handler)
         
        enable_handler = CommandHandler("reports", self.report)
        self.dispatcher.add_handler(enable_handler)


    # message to send when the bot is started
    def send_start(self, chatbot, update) -> None:
        welcome_message =  "*Hello, I am the bot that will keep your home safe* \n\n"
        welcome_message += '🗞 Type: /enable in order to enable the notifications!\n\n'
        welcome_message += '❌ Type: /disable in order to disable the notifications!\n\n'
        welcome_message += '📓 Type: /report in order to see the full report of alarms!\n\n'
        welcome_message += '🆘 Type: /help in order to contact the authors!\n\n'
        chatbot.message.reply_text(welcome_message)
    
   
    def send_enable(self, chatbot, update) -> None:
        # write the chat id in the database
        chat_id = chatbot.message.chat_id
        db = self._read_db()
        db.set_index('ids',inplace = True)

        if chat_id in db.index:
            if db.at[chat_id,'status'] == True:
                enable_message = '✅ The notifications are already enabled'
                chatbot.message.reply_text(enable_message)
            else:
                enable_message = "✅ You'll receive all the notifications"
                self._update_db(db,chat_id,'status',True)
                chatbot.message.reply_text(enable_message)
        else:
            enable_message = "You don't have the permission for this service, contact the authors"
            chatbot.message.reply_text(enable_message)

    def send_disable(self, chatbot, update) -> None:
        # write the chat id in the database
        chat_id = chatbot.message.chat_id
        db = self._read_db()
        db.set_index('ids',inplace = True)

        if chat_id in db.index:
            if db.at[chat_id,'status'] == True:
                enable_message = "❌ You won't receive other notifications"
                chatbot.message.reply_text(enable_message)
                self._update_db(db,chat_id,'status',False)

            else:
                enable_message = "You are not subscribed to the notification service"
                chatbot.message.reply_text(enable_message)

        else:
            enable_message = "You don't have the permission for this service, contact the authors"
            chatbot.message.reply_text(enable_message)                    

    # message to send when /help is received
    def send_help(self, chatbot, update) -> None:
        help_message =  'Authors: @GianlucaLM  @francescodis  @leomaggio \n'
        help_message += 'Feel free to write us!\n'
        chatbot.message.reply_text(help_message, parse_mode = telegram.ParseMode.MARKDOWN)
       
    
    def report(self, chatbot, update) -> None:
        string = ""
        chat_id = chatbot.message.chat_id
        db = self._read_db()
        db.set_index('ids',inplace = True)
        string = "Here is the full report:\n"
        if chat_id in db.index:
            if not os.path.exists("assets/bot/reports.txt"):
                chatbot.message.reply_text("There are no events")
            
            reports = np.genfromtxt('assets/bot/reports.txt',dtype='str')
            for report in reports:
                string += "•" + report + "\n"
            chatbot.message.reply_text(string)

        else: 
            chatbot.message.reply_text("Access denied negato❌")

    # start the bot
    def run(self) -> int:
        self.logger.info("Polling BOT.")
        self.updater.start_polling()

        # Run the BOT until you press Ctrl-C or the process receives SIGINT,
        # SIGTERM or SIGABRT. This should be used most of the time, since
        # start_polling() is non-blocking and will stop the BOT gracefully.
        self.updater.idle()
        return 0
    
    def _read_db(self):      
        df = pd.read_csv("assets/bot/users.csv")
        return df


    def _update_db(self,df,index,column,value):
        df.at[index, column] = value
        df.to_csv("assets/bot/users.csv")

if __name__ == "__main__":
    
    BOT = Bot()
    BOT.run()