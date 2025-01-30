import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

max_distance_vertical = 1  # Variable global vertical
max_distance_horizontal = 1  # Variable global horizontal

# Índices de los landmarks de la boca en MediaPipe Face Mesh
MOUTH_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78
]

SPECIAL_LANDMARKS = [13, 14, 61, 291]  # Landmarks que deben estar en color azul

def calculate_mouth_height(face_landmarks, image_width, image_height):
    global max_distance_vertical
    # Obtener los landmarks de los labios
    top_lip = face_landmarks.landmark[13]  # Punto superior del labio
    bottom_lip = face_landmarks.landmark[14]  # Punto inferior del labio

    # Convertir las coordenadas normalizadas a píxeles
    top_lip_y = int(top_lip.y * image_height)
    bottom_lip_y = int(bottom_lip.y * image_height)

    # Calcular la distancia entre los labios
    mouth_open_distance = abs(bottom_lip_y - top_lip_y)

    # Calcular el porcentaje de apertura
    if mouth_open_distance > max_distance_vertical:
        max_distance_vertical = mouth_open_distance  # Aumenta el valor de la distancia máxima
    mouth_open_percentage = min((mouth_open_distance / max_distance_vertical) * 100, 100)

    return mouth_open_percentage

def calculate_mouth_width(face_landmarks, image_width, image_height):
    global max_distance_horizontal
    # Obtener los landmarks de los labios
    left_lip = face_landmarks.landmark[61]  # Punto izquierdo del labio
    right_lip = face_landmarks.landmark[291]  # Punto derecho del labio

    # Convertir las coordenadas normalizadas a píxeles
    left_lip_x = int(left_lip.x * image_width)
    right_lip_x = int(right_lip.x * image_width)

    # Calcular la distancia entre los labios
    mouth_open_distance = abs(right_lip_x - left_lip_x)

    # Calcular el porcentaje de apertura
    if mouth_open_distance > max_distance_horizontal:
        max_distance_horizontal = mouth_open_distance  # Aumenta el valor de la distancia máxima
    mouth_open_percentage = min((mouth_open_distance / max_distance_horizontal) * 100, 100)

    return mouth_open_percentage

def get_color(percentage):
    # Determinar el color basado en el porcentaje
    if percentage >= 90:
        return (0, 255, 0)  # Verde
    elif 75 <= percentage < 90:
        return (0, 165, 255)  # Naranja
    else:
        return (0, 0, 255)  # Rojo

def draw_mouth_landmarks(image, face_landmarks, color):
    for idx in MOUTH_LANDMARKS:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * image.shape[1])
        y = int(landmark.y * image.shape[0])
        point_color = (255, 0, 0) if idx in SPECIAL_LANDMARKS else color
        cv2.circle(image, (x, y), 1, point_color, -1)

def draw_mouth_connections(image, face_landmarks, connections, color):
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx in MOUTH_LANDMARKS and end_idx in MOUTH_LANDMARKS:
            start_point = face_landmarks.landmark[start_idx]
            end_point = face_landmarks.landmark[end_idx]
            start_coords = (int(start_point.x * image.shape[1]), int(start_point.y * image.shape[0]))
            end_coords = (int(end_point.x * image.shape[1]), int(end_point.y * image.shape[0]))
            cv2.line(image, start_coords, end_coords, color, 1)

def draw_mouth(image, face_landmarks, connections, color):
    draw_mouth_landmarks(image, face_landmarks, color)
    draw_mouth_connections(image, face_landmarks, connections, color)

def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        mouth_height_percentage = calculate_mouth_height(face_landmarks, frame.shape[1], frame.shape[0])
        mouth_width_percentage = calculate_mouth_width(face_landmarks, frame.shape[1], frame.shape[0])

        color = get_color(max(mouth_height_percentage, mouth_width_percentage))

        # Dibujar los landmarks de la boca en la imagen con el color adecuado
        draw_mouth(frame, face_landmarks, mp_face_mesh.FACEMESH_LIPS, color)

        # Mostrar los porcentajes de apertura en la pantalla
        display_text(frame, 'Apertura Vertical', mouth_height_percentage, (50, 50))
        display_text(frame, 'Apertura Lateral', mouth_width_percentage, (50, 100))

        return frame, mouth_height_percentage, mouth_width_percentage

    return frame, None, None

def display_text(frame, text, percentage, position):
    color = get_color(percentage)
    cv2.putText(frame, f'{text}: {percentage:.2f}%', position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)