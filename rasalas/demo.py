#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
'''Demo de reconocimiento facial.'''

import argparse
import os
import re

import click
import face_recognition as fr
import numpy as np
import cv2


def _scan_known_people(people_path):
    """Escanea a las personas conocidas desde un directorio."""
    known_names = []
    known_face_encodings = []

    pattern = r'.*\.(jpg|jpeg|png)'
    img_files = [os.path.join(people_path, f) for f in os.listdir(people_path) if re.match(pattern, f, flags=re.I)]

    for file in img_files:
        basename = os.path.splitext(os.path.basename(file))[0]
        img = fr.load_image_file(file)
        encodings = fr.face_encodings(img)

        if len(encodings) > 1:
            click.echo("WARNING: More than one face found in {}. Only considering the first face.".format(file))

        if encodings:
            known_names.append(basename)
            known_face_encodings.append(encodings[0])
        else:
            click.echo("WARNING: No faces found in {}. Ignoring file.".format(file))

    return known_names, known_face_encodings


def live(known_path, scale=0.25, tolerance=0.6):
    """Demo en vivo utilizando la cámara integrada."""
    video_capture = cv2.VideoCapture(0)

    # Carga las imagenes de gente conocida desde un directorio
    known_face_names, known_face_encodings = _scan_known_people(known_path)

    process_this_frame = True
    while True:
        # Obtiene un nuevo frame
        _, frame = video_capture.read()

        # Escala el frame, usando el parámetro 'scale'
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        # Convierte la imagen de BGR (usado por OpenCV) a RGB (usado por face_recognition)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Procesa frames por medio
        if process_this_frame:
            # Encuentra todas las caras en el video y las codifica para ser comparadas
            face_locations = fr.face_locations(rgb_small_frame)
            face_encodings = fr.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # Obtiene las "distancias" entre el rostro del video y los rostros conocidos
                distances = fr.face_distance(known_face_encodings, face_encoding)
                # Si las distancias son superiores a cierta tolerancia, se marcan con np.nan
                distances[distances > tolerance] = np.nan

                if np.all(np.isnan(distances)):
                    # Si no existe ninguna distancia, es un rostro desconocido
                    name = "Unknown"
                else:
                    # Si existe al menos una distancia, obtiene la menor
                    index = np.nanargmin(distances)
                    name = known_face_names[index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Muestra los resultados en el video
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Escala las posiciones de vuelta al tamaño original
            scale_inv = 1 / scale
            top = int(top * scale_inv)
            right = int(right * scale_inv)
            bottom = int(bottom * scale_inv)
            left = int(left * scale_inv)

            # Dibuja una caja roja en la cara
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Dibuja un label con el nombre bajo la cara
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Muestra la imagen resultante
        cv2.imshow('Video', frame)

        # Si 'q' es presionado, termina el ciclo
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera los recursos
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path',
        help='Directorio con las imagenes de personas conocidas.'
    )
    args = parser.parse_args()
    live(args.path)
