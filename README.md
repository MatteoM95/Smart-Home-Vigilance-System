# Smart Home Vigilance System on Raspberry-Pi

### Overview
Final project of the course `Machine Learning for IOT` 2021/2022.


<p align="center">
  <img src="assets/Example/Human_detectionResized.gif" width=75% height=75% />
</p>

### Contents
- [Description](#description)
- [Setup](#setup)
- [How to run it](#howtorun)
- [Demo](#demo)
- [Report](#report)
- [Contributors](#contributors)

---

<a name="description"/>

## General description
The vast majority of modern surveillance solutions involve a camera and motion sensors, and just a few of them **use artificial intelligence algorithms**. In this context, we decided to build an **indoor video surveillance system** capable of **recognizing the presence of a human intrusion**, rather than mere movement. In this way, a photo of the intruder can be taken instantly, eliminating the burden of reviewing the footage.

<a name="setup"/>

## Setup

1. Setup your RaspberryPi
  ```shell
sudo apt update && sudo apt upgrade
sudo apt install -y mosquitto mosquitto-clients
  ```
2. Install Python
```shell
sudo apt install -y python3.7 python3-venv python3.7-venv
```

3. Setup python environement
```shell
python3.7 -m venv py37
source py37/bin/activate
```

4. Download and install tensorflow
```shell
wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.3.0/tensorflow-2.3.0-cp37-none-linux_armv7l.whl
pip install -U pip
pip install tensorflow-2.3.0-cp37-none-linux_armv7l.whl
```

5. Install requirements
```shell
pip install -r assets/files/requirements.txt
```

6. Install microphone dependencies
```
sudo apt install -y libgpiod2
sudo apt install -y libatlas-base-dev
sudo apt install -y libportaudio2
```

7. Setup bot: Fill the following constants in src/bot/bot_settings.py
```
TOKEN = (str) bot's token
TOKEN_MSG = (str) token of the bot that will manage the messages (optional)
CHAT_IDS = (list) list containing all the chat ids of the people that will receive the notifications
```

### Setting up Telegram bot

1. Search for the telegram bot `@SHVigilanceSystem_bot` and `@SHVigilanceNotification_bot`
2. Activate bot to start a chat.
3. Retrieve your chat_id using `@myidbot`, then insert the chat_id [here](assets/bot/users.csv) in users.csv and [here](https://github.com/MatteoM95/Smart-Home-Vigilance-System/blob/main/src/bot/bot_settings.py) in bot_setting.py inside the list.

<a name="howtorun"/>

### How to run it

You need to run four scripts, the camera publisher, the microphone publisher, the subscriber (the agent that will receive the notifications) and the bot.

```
python src/bot/botds.py
python pub_camera.py
python pub_microphone.py
python sub_bot.py

```


<a name="demo" />

### Demo

You can find some audio/video demos [here](assets/Example)



<a name="Report" />


### Report


The project [presentation](assets/Documentation/Slides_SHVS.pdf) or the technical [report](assets/Documentation/Smart_Home_Vigilance_System.pdf) are freely available!


<a name="contributors" />

### Contributors

<a href="https://github.com//MatteoM95/Smart-Home-Vigilance-System/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=MatteoM95/Smart-Home-Vigilance-System" />
</a>
