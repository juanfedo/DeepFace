import DeepFace_own
import connection_posgres

#connection_posgres.write_blob(1,"Hug","D:\\Python\\deepface\\database\\Hug\\hug.jpg") 
#connection_posgres.write_blob(2,"Ryan","D:\\Python\\deepface\\database\\Ryan\\ryan.jpg") 
#connection_posgres.write_blob(3,"Ryan2","D:\\Python\\deepface\\database\\Ryan\\ryan2.jpg") 
#connection_posgres.write_blob(4,"Julian","D:\\Python\\deepface\\database\\Julian\\julian.jpg") 

images = connection_posgres.read_blob()
imagenes = []
for imagen in images:
     imagenes.append((imagen[0], imagen[1], imagen[2], imagen[3]))

#image_data_binary1 = convert_bytea_to_numpy_array(images[1][2])
#image_data_binary2 = convert_bytea_to_numpy_array(images[2][2])

###veri = DeepFace.verify(img1_path = "D:\\Python\\deepface\\fotos\\hug.jpg", imagenes = imagenes)

#dfs = DeepFace.find2(  img_path = "D:\\Python\\deepface\\fotos\\hug.jpg",db_images=imagenes,refresh_database = False)

DeepFace_own.stream3(db_path = "D:\\Python\\deepface\\database", imagenes = imagenes)
print ('Finalizado')