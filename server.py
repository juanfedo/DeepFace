import cv2
from flask import Flask, Response
import DeepFace_own
import connection_posgres
from deepface.modules import streaming
import time 

app = Flask(__name__)

# Configura aquí las URL de las cámaras
cameras = [0, 1]  # Ejemplo con dos cámaras (ID de cámara o URL)

# Carga el clasificador pre-entrenado para la detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def generate_frames(camera_index):    
    cap = cv2.VideoCapture(camera_index)
    images = connection_posgres.read_blob()
    imagenes = []
    for imagen in images:
        imagenes.append((imagen[0], imagen[1], imagen[2], imagen[3]))

    streaming.build_facial_recognition_model(model_name="VGG-Face")

    prev_frame_time = 0
    
    # used to record the time at which we processed current frame 
    new_frame_time = 0
    jump = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        jump += 1 
        
        ## Convierte a escala de grises para la detección de rostros
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Dibuja rectángulos alrededor de los rostros detectados
        #for (x, y, w, h) in faces:
        #    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Codifica el frame en formato JPEG
        #ret, buffer = cv2.imencode('.jpg', frame)
        #frame = buffer.tobytes()        

        new_frame_time = time.time() 
    
        if faces is not None and len(faces) > 0:
            frame = DeepFace_own.stream4(frame, imagenes=imagenes)

        # Calculando FPS
    
        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 
    
        fps = int(fps)     

        print ('FPS:' + str(fps))

        # Codifica el frame en formato JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()        

        # Devuelve el frame en formato de stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    return Response(generate_frames(cameras[camera_id]),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Puedes acceder a las diferentes cámaras por su ID, e.g., /video_feed/0, /video_feed/1
    app.run(host='0.0.0.0', port=5000, threaded=True)
