
import DeepFace_own
import connection_posgres

#write_blob(1,"Hug","D:\\Python\\deepface\\fotos\\hug.jpg") 
#write_blob(2,"Ryan","D:\\Python\\deepface\\fotos\\ryan.jpg") 
#write_blob(3,"Ryan2","D:\\Python\\deepface\\fotos\\ryan2.jpg") 
#write_blob(4,"Julian","D:\\Python\\deepface\\database\\Julian\\julian.jpg") 

images = connection_posgres.read_blob()
imagenes = []
for imagen in images:
     imagenes.append((imagen[0], imagen[1], connection_posgres.convert_bytea_to_numpy_array(imagen[2])))

#image_data_binary1 = convert_bytea_to_numpy_array(images[1][2])
#image_data_binary2 = convert_bytea_to_numpy_array(images[2][2])

###veri = DeepFace.verify(img1_path = "D:\\Python\\deepface\\fotos\\hug.jpg", imagenes = imagenes)

#dfs = DeepFace.find2(  img_path = "D:\\Python\\deepface\\fotos\\hug.jpg",db_images=imagenes,refresh_database = False)

#if veri['verified']:
#    print ("Encontrado")
#else:
#    print(veri)

DeepFace_own.stream3(db_path = "D:\\Python\\deepface\\database", imagenes = imagenes)