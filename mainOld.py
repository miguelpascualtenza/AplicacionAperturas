import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)



cap = cv2.VideoCapture(0)

max_distance_vertical = 1 #Variable global vertical
max_distance_horizontal = 1 #Variable global horizontal


def calculate_mouth_height(face_landmarks, image_width, image_height):
    global max_distance_vertical
    # Obtener los landmarks de los labios
    left_lip = face_landmarks.landmark[13]  # Punto superior del labio
    right_lip = face_landmarks.landmark[14]  # Punto inferior del labio

    # Convertir las coordenadas normalizadas a píxeles
    left_lip_x = int(left_lip.y * image_height)
    right_lip_x = int(right_lip.y * image_height)

    # Calcular la distancia entre los labios
    mouth_open_distance = abs(right_lip_x - left_lip_x)

    # Calcular el porcentaje de apertura
    if mouth_open_distance > max_distance_vertical:
        max_distance_vertical = mouth_open_distance #Aumenta el valor de la distancia maxima
    mouth_open_percentage = min((mouth_open_distance / max_distance_vertical) * 100, 100)

    return mouth_open_percentage

def calculate_mouth_width(face_landmarks, image_width, image_height):
    global max_distance_horizontal
    # Obtener los landmarks de los labios
    left_lip = face_landmarks.landmark[61]  # Punto izquierdo del labio
    right_lip = face_landmarks.landmark[291]  # Punto derecho del labio

    # Convertir las coordenadas normalizadas a píxeles
    left_lip_x = int(left_lip.x * image_height)
    right_lip_x = int(right_lip.x * image_height)

    # Calcular la distancia entre los labios
    mouth_open_distance = abs(right_lip_x - left_lip_x)

    # Calcular el porcentaje de apertura
    if mouth_open_distance > max_distance_horizontal:
        max_distance_horizontal = mouth_open_distance #Aumenta el valor de la distancia maxima
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

def display_text(frame, text, percentage, position):
    color = get_color(percentage)
    cv2.putText(frame, f'{text}: {percentage:.2f}%', position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen con MediaPipe Face Mesh
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Dibujar los landmarks en la imagen
            mp.solutions.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
            )
            #Calculo porcentaje de apertura de la boca
            #mouth_open_percentage = calculate_mouth_height(face_landmarks, frame.shape[1], frame.shape[0])


        face_landmarks = results.multi_face_landmarks[0]

        mouth_height_percentage = calculate_mouth_height(face_landmarks, frame.shape[1], frame.shape[0])
        mouth_width_percentage = calculate_mouth_width(face_landmarks, frame.shape[1], frame.shape[0])

        # Mostrar los porcentajes de apertura en la pantalla
        display_text(frame, 'Apertura Vertical', mouth_height_percentage, (50, 50))
        display_text(frame, 'Apertura Lateral', mouth_width_percentage, (50, 100))


    # Mostrar la imagen
    cv2.imshow('Face Mesh', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
