import os
import random
import shutil

# Ruta de la carpeta que deseas modificar
folder_path = './data/train/sad'

# Lista de archivos en la carpeta
file_list = os.listdir(folder_path)

# Número de imágenes que deseas eliminar
num_images_to_delete = 3000

# Verifica si hay suficientes imágenes para eliminar
if len(file_list) < num_images_to_delete:
    print(f'No hay suficientes imágenes en {folder_path} para eliminar.')
else:
    # Selecciona aleatoriamente las imágenes que se eliminarán
    images_to_delete = random.sample(file_list, num_images_to_delete)

    # Elimina las imágenes seleccionadas
    for file_name in images_to_delete:
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Error al eliminar {file_path}: {e}')

    print(f'Se han eliminado {num_images_to_delete} imágenes de {folder_path}.')