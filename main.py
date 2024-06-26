from tkinter import *
from PIL import ImageTk, Image
from torchvision import transforms
import numpy as np
import cv2
import torch
import time

from Models.model_v3.build_model import Model
model = torch.load('Models\\model_v3\\model_v3_trained.pt')
DEVICE = torch.device('cuda')
race_dict = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}

root = Tk()
root.geometry("800x600")
root.title("FaceNet - Age & Gender & Race Estimation")
root.config(bg='white')
root.iconbitmap("icon.ico")
main_frame = Frame(root)
main_frame.grid(row=0, column=0)

face_cascade = cv2.CascadeClassifier("Main\\haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
cam_label = Label(main_frame)
cam_label.grid(row=0, column=0)

pipeline = transforms.Compose([
    transforms.Resize([128, 128]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5961, 0.4564, 0.3906], std=[0.2178, 0.1938, 0.1845])
])

#--------------------Functions--------------------
def _to_batch(self, imgs, device):
    batch_list = [x.unsqueeze(0) for x in imgs]
    if self.batch_size > len(batch_list):
        fill_size = self.batch_size - len(batch_list)
        batch_list.append(torch.zeros([fill_size, 3, self._img_size[0], self._img_size[1]]).to(device))
    batch = torch.cat(batch_list, 0).half()
    return batch

# Show Webcam Frame
def show_frame():
    img = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (800, 600))
    img = cv2.flip(img, 1)

    faces = face_cascade.detectMultiScale(img)
    try:
        if len(faces):
            for (x, y, w, h) in faces:
                face_img = img[x:x+w, y:y+h]
                face_img = cv2.resize(face_img, (128, 128))
                face_img = np.asarray([face_img]).astype(np.float32)
                face_img = torch.from_numpy(face_img)
                face_input = face_img.permute(0, 3, 1, 2)
                face_input = face_input.to(DEVICE)

                age, gender, race = model(face_input)
                age = str(round(age.item()**0.5))
                gender = "Male" if gender.item() <= 0.5 else "Female"
                race = torch.argmax(race)
                race = race_dict[race.item()]
                result = age + " " + gender + " " + race

                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)
                cv2.putText(img, result, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    except:
        pass

    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    cam_label.imgtk = imgtk
    cam_label.configure(image=imgtk)
    cam_label.after(10, show_frame)

# Main Loop
show_frame()
root.mainloop()