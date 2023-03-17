import datetime
import os
import time
import urllib
import webbrowser
import urllib3
import whisper
from PIL import Image
from GoogleSearch import Search
from urllib.request import urlopen
import openai
import google.cloud.texttospeech as tts
import keyboard
import pywhatkit
import speech_recognition as sr
from PyQt5.QtCore import *
from audioplayer import AudioPlayer
from colored import fg, attr
import win32console
import win32gui
from pyautogui import press, typewrite, hotkey

win32gui.ShowWindow(win32console.GetConsoleWindow(), 1)
########################################################################

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= 'C:\\Users\\Usuario\\PycharmProjects\\EmilIA\\key.json'
http = urllib3.PoolManager()
reset = attr('reset')# Reset the text Color to Default
red = fg('red')
blue = fg('blue')
yellow = fg('yellow')
green = fg('green')
openai.api_key = open("key_openai.json").read()
model= 'gpt-3.5-turbo';
r = sr.Recognizer()
client = tts.TextToSpeechClient.from_service_account_file('C:\\Users\\Usuario\\PycharmProjects\\EmilIA\\key.json')
mes = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]

#(_______________________________________________\\______________________________\\________________________)

#speak
def speak(color,text: str):
    response = client.synthesize_speech(
        input=tts.SynthesisInput(text=text),
        voice=tts.VoiceSelectionParams(language_code="es-US", name="es-US-Neural2-A"),
        audio_config=tts.AudioConfig(audio_encoding=tts.AudioEncoding.MP3, pitch= -5.0, effects_profile_id=["headphone-class-device"]),
    )
    with open("C:\\Users\\Usuario\\PycharmProjects\\EmilIA\\output.mp3", "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)
    print(color + text + reset)
    AudioPlayer("C:\\Users\\Usuario\\PycharmProjects\\EmilIA\\output.mp3").play(block=True)

def ChatGPT_conversation(conversation):
    response = openai.ChatCompletion.create(
        model=model,
        messages=conversation
    )
    conversation.append({'role': response.choices[0].message.role, 'content': response.choices[0].message.content})
    return conversation

conversation = [{'role': 'system', 'content': 'You are Emilia, a helpful assistant . Answer as concisely as possible.'}]
# conversation = ChatGPT_conversation(conversation)
# print('{0}: {1}\n'.format(conversation[-1]['role'].strip(), conversation[-1]['content'].strip()))
# url = 'https://www.google.com/search?q='+str(urllib.parse.quote_plus("Hipopotamo saltando sobre una vaca"))
# req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
# page = urlopen(req).read()
# print(page)

def chatbot(rec, conversacion):
    conversacion.append({'role': 'user', 'content': rec})
    conversation = ChatGPT_conversation(conversacion)
    res = '{0}: {1}\n'.format(conversation[-1]['role'].strip(), conversation[-1]['content'].strip())
    res = res.replace('Emilia', '')
    res = res.replace('\n', '')
    res = res.replace('\'', '')
    res = res.replace('[', '')
    res = res.replace(']', '')
    res = res.replace(':', '')
    res = res.replace('assistant ', '')
    return res

def remover_tildes(rec):
    rec = rec.replace('á', 'a')
    rec = rec.replace('é', 'e')
    rec = rec.replace('í', 'i')
    rec = rec.replace('ó', 'o')
    rec = rec.replace('ú', 'u')
    rec = rec.replace('ü', 'u')
    return rec

class MainThread(QThread):
    def __init__(self):
        super(MainThread, self).__init__()
    def takecommand(self):
        # os.system("python C:\\Users\\Usuario\\PycharmProjects\\EmilIA\\record_audio.py")
        # model = whisper.load_model("large")
        # result = model.transcribe("output.wav")
        # print(result["text"])
        with sr.Microphone() as source:
            print("Escuchando...")
            AudioPlayer("C:\\Users\\Usuario\\PycharmProjects\\EmilIA\\Notificacion\\10.mp3").play(block=True)
            audio = r.listen(source)
            self.query = r.recognize_google(audio, language='es-US')
            print(self.query)
            self.query = remover_tildes(self.query.lower())
        return self.query
        # return result["text"]

Thread = MainThread()

# start greeting
def saludo():
    time = datetime.datetime.now()
    hour = int(time.hour)
    dato = ", el dato curioso del día de hoy es: "+chatbot("Cuentame un dato curioso del "+str(time.day)+" de "+mes[time.month]+", si no, que sea de otro día", conversation)
    if hour == 1:
        horario = 'Es la ' + str(hour) + ' y ' + str(time.minute)
    else:
        horario = 'Son las ' + str(hour) + ' y ' + str(time.minute)
    if hour >= 5 and hour < 12:
        speak(reset, 'Buenas días Matias, '+horario+dato)
    elif hour >= 12 and hour < 18:
        speak(reset, 'Buenas tardes Matias, '+horario+dato)
    else:
        speak(reset, 'Buenas noches Matias, '+horario+dato)

#shutdown
def shutdown():
    speak(yellow, '¿Estas seguro?')
    reply = Thread.takecommand()
    if 'si' in reply or 'obvio' in reply or 'hacelo' in reply or 'hazlo' in reply or 'seguro' in reply:
        speak(red, 'Esta bien! Apagando sistema en un minuto')
        os.system('shutdown -s')
    else:
        speak(green, "Entendido! No te preocupes")
        
def comandos(rec):
    if 'reproduci' in rec:
        music = rec.replace('reproduce', '')
        pywhatkit.playonyt(music)
        speak(reset, 'Reproduciendo' + music)
    elif 'buenos dias' in rec or 'buenas tardes' in rec or 'buenas noches' in rec or 'hola' in rec:
        webbrowser.open("https://open.spotify.com/playlist/22q6F2dh5NcMYDy4QOopTM?si=1a450388b09b4686")
        time.sleep(1.25)
        press('space')
        saludo()
    elif 'abri' in rec:
        if 'youtube' in rec:
            if 'music' in rec:
                webbrowser.open("https://music.youtube.com/")
                speak(reset, 'Abriendo YouTube Music')
            else:
                webbrowser.open("https://www.youtube.com/")
                speak(reset, 'Abriendo YouTube')
        elif 'spotify' in rec:
            webbrowser.open("https://www.spotify.com/")
            speak(reset, 'Abriendo Spotify')
        elif 'facebook' in rec:
            webbrowser.open("https://www.facebook.com/")
            speak(reset, 'Abriendo Facebook')
        elif 'whatsapp' in rec:
            os.system('"C:\\Program Files\\WindowsApps\\5319275A.WhatsAppDesktop_2.2308.6.0_x64__cv1g1gvanyjgm\\Whatsapp.exe"')
            speak(reset, 'Abriendo Whatsapp')
        elif 'instagram' in rec:
            webbrowser.open("https://www.instagram.com/?utm_source=pwa_homescreen")
            speak(reset, 'Abriendo Instagram')
        elif 'google' in rec:
            webbrowser.open("https://www.google.com.ar/")
            speak(reset, 'Abriendo Google')
    elif 'que hora es' in rec:
        tiempo = datetime.datetime.now()
        hour = datetime.datetime.now().hour
        if hour == 1:
            horario = 'Es la ' + str(hour) + ' y ' + str(tiempo.minute)
        else:
            horario = 'Son las ' + str(hour) + ' y ' + str(tiempo.minute)
        speak(reset, horario)
    elif 'toma captura' in rec:
        pywhatkit.take_screenshot("screenshot")
        speak(reset, "Captura guardada")
    elif 'apaga el sistema' in rec or 'apaga la compu' in rec or 'apaga todo' in rec:
        shutdown()
    elif 'quiero hablar' in rec or 'quiero conversar' in rec or 'vamos a hablar' in rec or 'vamos a conversar' in rec:
        speak(blue, chatbot(Thread.takecommand(), conversation))
    elif 'crea una imagen de' in rec or 'crea una imagen de' in rec:
        rec = rec.replace('crea una imagen de', '')
        os.system("C:\\Users\\Usuario\\anaconda3\\condabin\\activate.bat ldm&cd C:\\Users\\Usuario\\stability-sdk\\stablediffusion&python scripts\\txt2img.py --prompt \""+rec+"\" --plms --n_iter 5 --n_samples 1")
        speak(reset, "Imagen de "+rec+" creada")
    elif 'busca lo que estoy viendo en internet' in rec or 'busca' in rec:
        if 'busca lo que estoy viendo en internet' in rec:
            pywhatkit.take_screenshot("screenshot")
            output = Search(file_path="screenshot.png")
            print(output[output])
        elif 'busca' in rec:
            rec = rec.replace('busca ', '')
            pywhatkit.search(rec)
            speak(reset, "Buscando" + rec + " En Google")
    elif 'lee lo que estoy viendo' in rec or 'transcribe lo que estoy viendo' in rec:
        pywhatkit.take_screenshot("screenshot")
        img = Image.open("screenshot.png")
        img.load()
        os.system("tesseract screenshot.png output")
        speak(reset, "Transcribiendo lo que estas viendo")
        os.system("output.txt")
    else:
        chat = chatbot(rec, conversation)
        speak(blue, chat)

while True:
    comandos(Thread.takecommand())
