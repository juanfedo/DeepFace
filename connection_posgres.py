import psycopg2 
import numpy as np
from PIL import Image
import io, json 
from deepface.modules import verification

def create_connection(): 
    conn = psycopg2.connect(dbname='deepface', 
                            user='postgres', 
                            password='patojo', 
                            host='localhost', 
                            port='5434') 
    curr = conn.cursor() 
    return conn, curr 

def write_blob(id,nombre, archivo, model_name: str = "VGG-Face", 
                detector_backend: str = "opencv", 
                enforce_detection: bool = False,    
                align: bool = True,
                expand_percentage: int = 0,
                normalization: str = "base",
                anti_spoofing: bool = False,): 

    img1_embeddings, img1_facial_areas = verification.__extract_faces_and_embeddings(
        img_path=archivo,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
        normalization=normalization,
        anti_spoofing=anti_spoofing,
    )

    embeddings = json.dumps(img1_embeddings)
    facial_areas = json.dumps(img1_facial_areas)

    try: 
        drawing = open(archivo, 'rb').read() 
        conn, cursor = create_connection() 
        try:            
            cursor.execute("INSERT INTO personas (id, nombre, imagen, embeddings, facial_areas) " +
                    "VALUES(%s,%s,%s,%s,%s)", 
                    (id,nombre, psycopg2.Binary(drawing),embeddings, facial_areas)) 
            conn.commit() 
        except (Exception, psycopg2.DatabaseError) as error: 
            print("Error while inserting data in table", error) 
        finally: 
            conn.close() 
    finally: 
        pass

def read_blob():
    try:
        conn, cursor = create_connection()
        cursor.execute("SELECT id, nombre, embeddings, facial_areas from personas")
        return cursor.fetchall()        
    except (Exception, psycopg2.DatabaseError) as error: 
            print("Error while read data in table", error) 
    finally: 
            conn.close()

def convert_bytea_to_numpy_array(bytes):
    buffer = io.BytesIO(bytes)
    image = Image.open(buffer)
    return np.asarray(image)
