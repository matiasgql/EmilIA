import datetime
import random
import shutil
import socket
import time
import webbrowser
import pyshorteners as pyshorteners
import requests
import urllib3
import argparse
import io
import re
import os
import public_ip as ip
import speech_recognition as sr
import whisper
import torch
import replicate
import psutil
import openai
import google.cloud.texttospeech as tts
import pywhatkit
import numpy as np
import cv2
import pyautogui
import json
import urllib
import pyowm
from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from sys import platform
from PIL import Image
from PyQt5.QtCore import *
from audioplayer import AudioPlayer
from colored import fg, attr
from pyautogui import press, hotkey
from langchain.chat_models import ChatOpenAI
from langchain.chains import APIChain
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.chains.api import open_meteo_docs
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from googletrans import Translator
from langchain.utilities import OpenWeatherMapAPIWrapper
from transformers import pipeline

CURR_DIR = os.getcwd() + "\\"
client = tts.TextToSpeechClient.from_service_account_file(CURR_DIR + 'key.json')
reset = attr('reset')  # Reset the text Color to Default


# speak
def speak(text: str, color=reset):
    readable_text = text.replace("\n", " ").replace(":", " ")
    response = client.synthesize_speech(
        input=tts.SynthesisInput(text=readable_text),
        voice=tts.VoiceSelectionParams(language_code="es-US", name="es-US-Neural2-A"),
        audio_config=tts.AudioConfig(audio_encoding=tts.AudioEncoding.MP3, pitch=-5.0,
                                     effects_profile_id=["headphone-class-device"]),
    )
    with open(CURR_DIR + "output.mp3", "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)
    print(color + text + reset)
    AudioPlayer(CURR_DIR + "output.mp3").play(block=True)


print("Asignando variables...")

########################################################################
IA_NAME = "emilia"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CURR_DIR + 'key.json'
red = fg('red')
blue = fg('blue')
yellow = fg('yellow')
green = fg('green')
openai.api_key = open(CURR_DIR + "key_openai.json").read()
r = sr.Recognizer()
mes = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre",
       "Diciembre"]
repli = replicate.Client(api_token=open(CURR_DIR + "key_replicate.json").read())
print("Hay disponibilidad de núcleos CUDA" if torch.cuda.is_available() else "No hay disponibilidad de núcleos CUDA")

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="medium", help="Model to use",
                    choices=["tiny", "base", "small", "medium", "large"])
parser.add_argument("--non_english", action='store_true',
                    help="Don't use the english model.")
parser.add_argument("--language", default="English", help="Language to use",
                    choices=["Spanish", "English", "Portuguese", "French", "German", "Italian", "Russian", "Japanese",
                             "Chinese", "Korean", "Arabic", "Vietnamese", "Hind"])
# parser.add_argument("–-device", default="CUDA", help="use CUDA", type=str)
parser.add_argument("--energy_threshold", default=1000,
                    help="Energy level for mic to detect.", type=int)
parser.add_argument("--record_timeout", default=2,
                    help="How real time the recording is in seconds.", type=float)
parser.add_argument("--phrase_timeout", default=3,
                    help="How much empty space between recordings before we "
                         "consider it a new line in the transcription.", type=float)
if 'linux' in platform:
    parser.add_argument("--default_microphone", default='pulse',
                        help="Default microphone name for SpeechRecognition. "
                             "Run this with 'list' to view available Microphones.", type=str)
args = parser.parse_args()

# The last time a recording was retrieved from the queue.
phrase_time = None
# Current raw audio bytes.
last_sample = bytes()
# Thread safe Queue for passing data from the threaded recording callback.
data_queue = Queue()
# We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
recorder = sr.Recognizer()
recorder.energy_threshold = args.energy_threshold
# Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
recorder.dynamic_energy_threshold = False
os.environ["OPENAI_API_KEY"] = open(CURR_DIR + "key_openai.json").read()
os.environ["SERPAPI_API_KEY"] = open(CURR_DIR + "key_serpapi.json").read()
os.environ["OPENWEATHERMAP_API_KEY"] = open(CURR_DIR + "key_owm.json").read()
llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                 temperature=0.2)
llm_2 = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
chain_new = APIChain.from_llm_and_api_docs(llm, open_meteo_docs.OPEN_METEO_DOCS, verbose=True)
python_code = create_python_agent(
    llm=OpenAI(temperature=0, max_tokens=1000),
    tool=PythonREPLTool(),
    verbose=True
)
# Important for linux users.
# Prevents permanent application hang and crash by using the wrong Microphone
if 'linux' in platform:
    mic_name = args.default_microphone
    if not mic_name or mic_name == 'list':
        print("Available microphone devices are: ")
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"Microphone with name \"{name}\" found")
    else:
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            if mic_name in name:
                source = sr.Microphone(sample_rate=16000, device_index=index)
                break
else:
    source = sr.Microphone(sample_rate=16000)

print("Cargando modelo de reconocimiento de voz...")
# Load / Download model
audio_model = whisper.load_model("large")

record_timeout = args.record_timeout
phrase_timeout = args.phrase_timeout
#
temp_file = NamedTemporaryFile().name
transcription = ['', '']

with source:
    recorder.adjust_for_ambient_noise(source)


def record_callback(_, audio: sr.AudioData) -> None:
    """
    Threaded callback function to receive audio data when recordings finish.
    audio: An AudioData containing the recorded bytes.
    """
    # Grab the raw bytes and push it into the thread safe queue.
    datos = audio.get_raw_data()
    data_queue.put(datos)


# Create a background thread that will pass us raw audio bytes.
# We could do this manually but SpeechRecognizer provides a nice helper.
recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

emotion_classifier = pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion')

# Cue the user that we're ready to go.
print("Modelo de reconocimiento de voz cargado...")


# (_______________________________________________\\______________________________\\________________________)


# torch.cuda.empty_cache()
# speak("Cargando modelo de inteligencia artificial conversacional...")

def ChatGPT_conversation(conversation):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=conversation,
    )
    conversation.append({'role': response.choices[0].message.role, 'content': response.choices[0].message.content})
    return conversation


def gpt_response(instruction, model="babbage"):
    response = openai.Completion.create(
        model=model,
        prompt="Eres " + IA_NAME + open("C:\\Users\\Usuario\\PycharmProjects\\EmilIA\\prompt-es"+"-2.txt" if model == "babbage" else "-1.txt").read() + instruction,
        temperature=0.2,
        max_tokens=128,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["Instruccion:"]
    )
    return response.choices[0].text


# speak("Modelo de inteligencia artificial conversacional cargado...")

print("\033[H\033[J", end="")


def screenshot():
    image = pyautogui.screenshot(CURR_DIR + "screenshot.png")
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.imwrite(CURR_DIR + "screenshot.png", image)


def get_horario():
    tiempo = datetime.now()
    hora = str(tiempo.hour)
    minuto = " y " + str(tiempo.minute) if tiempo.minute != 0 else " en punto"
    return ("es la " if int(hora) == 1 else "son las ") + hora + minuto


def degToCompass(num):
    val = int((num / 22.5) + .5)
    arr = ["Norte", "Norte-noreste", "Noreste", "Este-noreste", "Este", "Este-sureste", "Sureste", "Sur-sureste", "Sur",
           "Sur-suroeste", "Suroeste", "Oeste-suroeste", "Oeste", "Oeste-noreste", "Noroeste", "Norte-noroeste"]
    return arr[(val % 16)]


def get_weather():
    response = urllib.request.urlopen('http://ipinfo.io/json')
    data = json.load(response)
    loc = f"{data['city']}, {data['region']}, {data['country']}"
    mgr = pyowm.OWM(os.environ['OPENWEATHERMAP_API_KEY']).weather_manager()
    observation = mgr.weather_at_place(loc)
    w = observation.weather
    cielo_es = json.loads(open(CURR_DIR + "climas.json").read())
    temp = w.temperature('celsius')
    wind = w.wind('km_hour')
    return f", la temperatura en {data['city']} es de {int(round(temp['temp'], 0))}° con una sensación térmica de {int(round(temp['feels_like'], 0))}°," \
           f" hay {cielo_es[str(w.weather_code)]} y el viento va a {int(round(wind['speed'], 0))} km/h en dirección al {degToCompass(wind['deg'])}."


def saludo():
    tiempo = datetime.now()
    hour = int(tiempo.hour)
    clima = get_weather()
    horario = get_horario()
    if 5 <= hour < 12:
        speak('Buenos días Matias, ' + horario + clima)
    elif 12 <= hour < 18:
        speak('Buenas tardes Matias, ' + horario + clima)
    else:
        speak('Buenas noches Matias, ' + horario + clima)


def createImage(prompt: str):
    output = repli.run(
        "prompthero/openjourney:9936c2001faa2194a261c01381f90e65261879985476014a0a37a334593a05eb",
        input={"prompt": prompt}
    )
    return output[0]


def comandos(rec):
    rec = rec.replace('Respuesta: ', '')
    if '0xR0' in rec:
        rec = rec.replace('0xR0', '')
        print(rec)
        pywhatkit.playonyt(rec)
        speak('Reproduciendo' + rec)
    elif '0xA' in rec:
        if 'Y' in rec:
            if 'M' in rec:
                os.system(CURR_DIR + "YT_Music.lnk")
                speak('Abriendo YouTube Music')
            else:
                webbrowser.open("https://www.youtube.com/")
                speak('Abriendo YouTube')
        elif 'S' in rec:
            os.system("spotify")
            speak('Abriendo Spotify')
        elif 'F' in rec:
            webbrowser.open("https://www.facebook.com/")
            speak('Abriendo Facebook')
        elif 'W' in rec:
            os.system(CURR_DIR + "WhatsApp.lnk")
            speak('Abriendo Whatsapp')
        elif 'I' in rec:
            os.system(CURR_DIR + "Instagram.lnk")
            speak('Abriendo Instagram')
        elif 'G' in rec:
            webbrowser.open("https://www.google.com.ar/")
            speak('Abriendo Google')
    elif '0xGH' in rec:
        horario = get_horario()
        speak(horario)
    elif '0xTC' in rec:
        screenshot()
        speak("Captura guardada")
        os.system(CURR_DIR + "screenshot.png")
    elif '0xSD' in rec:
        speak('Esta bien! Apagando sistema en un minuto', red)
        os.system('shutdown -s')
    elif '0xMI' in rec:
        rec = rec.replace('0xMI', '')
        url = createImage(rec)
        print(url)
        speak("Imagen de " + Translator().translate(text=rec, dest="es") + " creada")

        response = requests.get(url, stream=True)
        file_name = "img.png"
        if response.status_code == 200:
            with open(file_name, 'wb') as file:
                shutil.copyfileobj(response.raw, file)
            os.system(CURR_DIR + "img.png")
    elif '0xRS' in rec:
        screenshot()
        img = Image.open(CURR_DIR + "screenshot.png")
        img.load()
        os.system("tesseract " + CURR_DIR + "screenshot.png output")
        speak("Transcribiendo lo que estas viendo")
        os.system("output.txt")
    elif '0xBA' in rec:
        try:
            speak("Tienes: " + str(psutil.sensors_battery().percent) + "%")
        except:
            speak("Lo siento, no he podido obtener el porcentaje de la batería")
    elif '0xSU' in rec:
        try:
            speak("Estas usando un " + str(psutil.cpu_percent()) + "% del procesador y " + str(
                psutil.virtual_memory().percent) + "% de memoria disponible")
        except:
            speak("Lo siento, no he podido obtener el uso del sistema")
    elif '0xPC' in rec:
        rec.replace("0xPC", "")
        python_code.run(rec)
    elif '0xCL' in rec:
        speak(get_weather())


class botChat:
    conversation = []

    def __init__(self):
        self.conversation = [{'role': 'system',
                              'content': f"""You're {IA_NAME.capitalize()} a useful assistant. Answer as concisely as possible. You will also answer taking into account my emotions that I will mention between "[" and "]"."""}]

    def chatbot(self, rec):
        self.conversation.append({'role': 'user', 'content': rec})
        self.conversation = ChatGPT_conversation(self.conversation)
        return ('{0}: {1}\n'.format(self.conversation[-1]['role'].strip(), self.conversation[-1]['content'].strip())
                .replace('\n', '')
                .replace('\'', '')
                .replace('[', '')
                .replace(']', '')
                .replace(':', '')
                .replace('😁', '')
                .replace('assistant ', ''))


def remover_tildes(rec):
    rec = rec.replace('á', 'a')
    rec = rec.replace('é', 'e')
    rec = rec.replace('í', 'i')
    rec = rec.replace('ó', 'o')
    rec = rec.replace('ú', 'u')
    rec = rec.replace('ü', 'u')
    return rec


bot = botChat()
transcription_length = 0
webbrowser.open("https://youtu.be/H_V9b3Bz2Zw?t=0")
saludo()

while True:
    try:
        now = datetime.utcnow()
        # Pull raw recorded audio from the queue.
        if not data_queue.empty():
            phrase_complete = False
            # If enough time has passed between recordings, consider the phrase complete.
            # Clear the current working audio buffer to start over with the new data.
            if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                last_sample = bytes()
                phrase_complete = True
            # This is the last time we received new audio data from the queue.
            phrase_time = now

            # Concatenate our current audio data with the latest audio data.
            while not data_queue.empty():
                data = data_queue.get()
                last_sample += data

            # Use AudioData to convert the raw data to wav data.
            audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
            wav_data = io.BytesIO(audio_data.get_wav_data())

            # Write wav data to the temporary file as bytes.
            with open(temp_file, 'w+b') as f:
                f.write(wav_data.read())

            # Read the transcription.
            result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available())
            text = result['text'].strip()

            # If we detected a pause between recordings, add a new item to our transcription,
            # Otherwise edit the existing one.
            if phrase_complete:
                transcription.append(text)
            else:
                transcription[-1] = str(text)

        elif len(transcription) != transcription_length and 'emilia' in transcription[-2].lower():
            print("\033[H\033[J", end="")
            transcription_length = len(transcription)
            transcription2 = "Emilia " + transcription[-2].split("Emilia")[1].strip()
            print(transcription2)
            try:
                texto = Translator().translate(text=transcription2).text
            except:
                texto = ""
            emotion = Translator().translate(text=emotion_classifier(texto)[0]["label"]).text
            # print("Emoción: ", emotion)
            res = gpt_response(transcription2)
            print(res)
            if "0xDE" in res:
                res = gpt_response(transcription2, "text-davinci-003")
                print(res)
                if "0x00" in res:
                    try:
                        respuesta = agent.run(transcription2)
                        respuesta = respuesta.split("Final Answer: ")[0].strip()
                        respuesta = respuesta.replace("Final Answer: ", "")
                        print("\033[H\033[J", end="")
                        translator = Translator()
                        respuesta = translator.translate(text=respuesta, dest="es")
                        speak(respuesta.text)
                    except:
                        speak("Lo siento, no he podido conseguir la información.")
            elif '0x0F' in res:
                speak(bot.chatbot(f"[{emotion}] {transcription2}"))
            else:
                    comandos(res)
    except KeyboardInterrupt:
        break
