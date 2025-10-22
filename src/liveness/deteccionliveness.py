#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import os

CONFIDENCE_THRESHOLD = 0.35
CONTAINMENT_THRESHOLD = 0.65
IMG_SIZE = 640
DRAW = True

def box_area(box):
    w = max(0.0, box[2] - box[0])
    h = max(0.0, box[3] - box[1])
    return w * h

def intersection_area(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    w = max(0.0, xB - xA)
    h = max(0.0, yB - yA)
    return w * h

def clamp_box_to_frame(box, w, h):
    x1 = int(max(0, min(w-1, box[0])))
    y1 = int(max(0, min(h-1, box[1])))
    x2 = int(max(0, min(w-1, box[2])))
    y2 = int(max(0, min(h-1, box[3])))
    return [x1, y1, x2, y2]

def main(args):
    model_path = args.model
    if not os.path.exists(model_path):
        print(f"ERROR: No se encontró el modelo en {model_path}")
        return

    print("Cargando modelo YOLO desde:", model_path)
    model = YOLO(model_path)

    names = model.model.names
    person_id, phone_id = None, None
    for cid, cname in names.items():
        lname = str(cname).lower()
        if lname == "person":
            person_id = cid
        if lname in ("cell phone", "cellphone", "mobile phone", "phone"):
            phone_id = cid

    if person_id is None or phone_id is None:
        print("ERROR: El modelo no contiene 'person' o 'cell phone'")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(int(args.source)) if str(args.source).isdigit() else cv2.VideoCapture(args.source)

    # ✅ Creamos UNA sola ventana fija
    window_name = "Detección YOLO - Liveness AntiSpoof"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        h_frame, w_frame = frame.shape[:2]

        results = model.predict(frame, imgsz=IMG_SIZE, conf=CONFIDENCE_THRESHOLD, verbose=False)
        r = results[0]

        xyxy = r.boxes.xyxy.cpu().numpy() if len(r.boxes) > 0 else []
        confs = r.boxes.conf.cpu().numpy().flatten() if len(r.boxes) > 0 else []
        clss = r.boxes.cls.cpu().numpy().astype(int).flatten() if len(r.boxes) > 0 else []

        persons_idx = [i for i,c in enumerate(clss) if c == person_id and confs[i] >= CONFIDENCE_THRESHOLD]
        phones_idx  = [i for i,c in enumerate(clss) if c == phone_id  and confs[i] >= CONFIDENCE_THRESHOLD]

        persons = [xyxy[i] for i in persons_idx]
        phones  = [xyxy[i] for i in phones_idx]
        persons_conf = [confs[i] for i in persons_idx]
        phones_conf  = [confs[i] for i in phones_idx]

        multiple_people = len(persons) >= 2
        spoof_pairs = []
        phone_inside_person = False

        for i_p, phone_box in enumerate(phones):
            area_phone = box_area(phone_box)
            if area_phone <= 0: continue
            for i_person, person_box in enumerate(persons):
                inter = intersection_area(phone_box, person_box)
                ratio = inter / area_phone
                if ratio >= CONTAINMENT_THRESHOLD:
                    phone_inside_person = True
                    x1,y1,x2,y2 = clamp_box_to_frame(phone_box, w_frame, h_frame)
                    crop = frame[y1:y2, x1:x2]
                    faces_found = False
                    if crop.size != 0:
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray,1.1,5)
                        faces_found = len(faces) > 0
                    spoof_pairs.append((i_p, i_person, ratio, faces_found))

        liveness_fail = any([f for (_,_,_,f) in spoof_pairs])

        alert_text = "No spoof detectado"
        color_msg = (200,200,200)

        if liveness_fail:
            alert_text = "[ALERTA] LIVENESS FAIL — RECHAZAR"
            color_msg = (0,0,255)
        elif multiple_people:
            alert_text = "[ALERTA] Varias personas detectadas — acceso bloqueado"
            color_msg = (0,0,255)
        elif phone_inside_person:
            alert_text = "Telefono en persona detectado"
            color_msg = (0,165,255)

        for idx, pbox in enumerate(persons):
            x1,y1,x2,y2 = map(int, pbox)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,180,50),2)

        for idx, phbox in enumerate(phones):
            x1,y1,x2,y2 = map(int, phbox)
            color = (50,200,50)
            tag = f"Phone {phones_conf[idx]:.2f}"
            for (i_p, _, _, faces_found) in spoof_pairs:
                if i_p == idx and faces_found:
                    color = (0,0,255)
                    tag = "SPOOF PHONE"
                    break
                elif i_p == idx:
                    color = (0,165,255)
                    tag = "Phone-in-person"
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,tag,(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)

        cv2.putText(frame, alert_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.75,color_msg,2)

        # ✅ Ahora siempre actualiza la MISMA ventana
        cv2.imshow(window_name, frame)

        print(f"Frame {frame_idx}: persons={len(persons)}, phones={len(phones)}, multi={multiple_people}, spoof={liveness_fail}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--model", type=str, default="./yolo11m.pt")
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE)
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--containment", type=float, default=CONTAINMENT_THRESHOLD)
    args = parser.parse_args()
    main(args)
