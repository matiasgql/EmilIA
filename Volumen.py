import cv2
import Seguimientomanos as sm
import numpy as np

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

anchoCam, altoCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, anchoCam)
cap.set(4, altoCam)

detector = sm.detectormanos(maxManos=1, Confdeteccion=1)

dispositivos = AudioUtilities.GetSpeakers()
interfaz = dispositivos.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volumen = cast(interfaz, POINTER(IAudioEndpointVolume))
RangoVol = volumen.GetVolumeRange()
VolMin = RangoVol[0]
VolMax = RangoVol[1]

while True:
    ret, frame = cap.read()
    frame = detector.encontrarmanos(frame)
    lista, bbox = detector.encontrarposicion(frame, dibujar=False)
    if len(lista) != 0:
        x1, y1 = lista[4][1], lista[4][2]
        x2, y2 = lista[8][1], lista[8][2]

        dedos = detector.dedosarriba()

        if dedos[0] == 1 and dedos[1] == 1:
            longitud, frame, linea = detector.distancia(4, 8, frame, r=8, t=2)
            print(longitud)

            vol = np.interp(longitud, [25, 400], [VolMin, VolMax])
            volumen.SetMasterVolumeLevel(vol, None)

            if longitud < 100:
                cv2.circle(frame, (linea[4], linea[5]), 10, (0, 255, 0), cv2.FILLED)

    cv2.imshow("Video", frame)
    t = cv2.waitKey(1)

    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()