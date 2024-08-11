import psycopg2 
import numpy as np
from PIL import Image
import io 

def create_connection(): 
    conn = psycopg2.connect(dbname='deepface', 
                            user='postgres', 
                            password='patojo', 
                            host='localhost', 
                            port='5434') 
    curr = conn.cursor() 
    return conn, curr 

def write_blob(id,nombre, archivo): 
    try: 
        drawing = open(archivo, 'rb').read() 
        conn, cursor = create_connection() 
        try:            
            cursor.execute("INSERT INTO personas (id,nombre,imagen) " +
                    "VALUES(%s,%s,%s)", 
                    (id,nombre, psycopg2.Binary(drawing))) 
            conn.commit() 
        except (Exception, psycopg2.DatabaseError) as error: 
            print("Error while inserting data in cartoon table", error) 
        finally: 
            conn.close() 
    finally: 
        pass

def read_blob():
    try:
        conn, cursor = create_connection()
        cursor.execute("SELECT * from personas")
        return cursor.fetchall()        
    except (Exception, psycopg2.DatabaseError) as error: 
            print("Error while inserting data in cartoon table", error) 
    finally: 
            conn.close()

def convert_bytea_to_numpy_array(bytes):
    buffer = io.BytesIO(bytes)
    image = Image.open(buffer)
    return np.asarray(image)
