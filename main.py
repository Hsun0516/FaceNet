from tkinter import *
from tkinter.ttk import *
from ttkbootstrap import Style
from PIL import ImageTk, Image
from torchvision import transforms
import cv2
import torch
import time

face_cascade = cv2.CascadeClassifier("Main\\haarcascade_frontalface_default.xml")

style = Style(theme='cosmo')
style.configure("1.Horizontal.TProgressbar", thickness=3, troughcolor ='white', background='green')
window = style.master
window.title('Test')
window.geometry('800x400')
window.resizable(False, False)
#window.iconbitmap('path')

pipeline = transforms.Compose([
    transforms.Resize([128, 128]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5961, 0.4564, 0.3906], std=[0.2178, 0.1938, 0.1845])
])

cap = cv2.VideoCapture(0)

#--------------------Labels--------------------

# Feature Label
feature_label = Label(window)

# Image Label
img_label = Label(window)
img_label.place(x=10, y=10)

# Age Label
age_label = Label(window, text="年齡:", font=('Arial', 20))
age_label.place(x=410, y=70)

# Gender Label
gender_label = Label(window, text="性別:", font=('Arial', 20))
gender_label.place(x=410, y=110)

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
    _, frame = cap.read()
    img = cv2.flip(frame, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    img = cv2.resize(img, (380, 380))

    faces = face_cascade.detectMultiScale(img)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    img = Image.fromarray(img)

    imgtk = ImageTk.PhotoImage(image=img)
    img_label.imgtk = imgtk
    img_label.configure(image=imgtk)
    img_label.after(10, show_frame)

# Predict Button Function
from Models.model_v2.build_model import Model
model = torch.load('Models\\model_v2\\model_v2.pt')

def predict_button():
    _, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame)
    for (x, y, w, h) in faces:
        frame = frame[y:y+h, x:x+w]

    #input_data
    input_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_data = Image.fromarray(input_data, 'RGB')
    input_data = pipeline(input_data)

    # Predict
    age, gender, race = model(torch.tensor([input_data]))

    # Progress Bar
    progressbar = Progressbar(window, length=375, mode="determinate", style="1.Horizontal.TProgressbar")
    progressbar.place(x=410, y=60)
    while True:
        if progressbar["value"] < progressbar["maximum"]:
            progressbar["value"] += 10
            window.update()
            time.sleep(0.001)
        else:
            break

    # Show Predict Result
    if gender >= 0.5:
        probability = round(gender*100, 2)
        gender_label['text'] = f"性別: 男 ({probability}%)"
    else:
        probability = round((1-gender)*100, 2)
        gender_label['text'] = f"性別: 女 ({probability}%)"
    age_label['text'] = "年齡: " + str(round(age))
    

    # Show Aligned Image
    img_show = cv2.resize(frame, (100,100))
    img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGBA)
    img_show = cv2.flip(img_show, 1)
    img_show = Image.fromarray(img_show)
    imgtk_show = ImageTk.PhotoImage(image=img_show)
    feature_label.imgtk = imgtk_show
    feature_label.configure(image=imgtk_show)
    feature_label.place(x=410, y=170)

#--------------------Buttons--------------------
# Predict Button
button = Button(window, text="Analyze", width=50, command=predict_button)
button.place(x=410, y=30)


# Main Loop
show_frame()
window.mainloop()