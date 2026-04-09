import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
 
model = tf.keras.models.load_model('pneumonia_cnn_model.h5')
selected_path = None
 
def preprocess(path):
   img = Image.open(path).convert('RGB').resize((150,150))
   arr = np.array(img) / 255.0
   return np.expand_dims(arr, axis=0)
 
def browse():
   global selected_path
   p = filedialog.askopenfilename(
       filetypes=[('Images','*.jpg *.jpeg *.png *.bmp')])
   if p:
       selected_path = p
       img = Image.open(p).resize((250,250))
       tk_img = ImageTk.PhotoImage(img)
       panel.config(image=tk_img); panel.image = tk_img
       lbl_result.config(text='Image loaded. Click Predict.', fg='black')
 
def predict():
   if not selected_path:
       lbl_result.config(text='Please browse first.', fg='red'); return
   prob = model.predict(preprocess(selected_path))[0][0]
   if prob > 0.5:
       txt, conf, col = 'PNEUMONIA DETECTED', prob*100, '#CC0000'
   else:
       txt, conf, col = 'NORMAL', (1-prob)*100, '#006400'
   lbl_result.config(
       text=f'Prediction : {txt}\nConfidence : {conf:.2f}%', fg=col)
 
root = tk.Tk()
root.title('Pneumonia Detection System'); root.geometry('500x600')
tk.Label(root, text='Pneumonia Detection System',
        font=('Helvetica',16,'bold')).pack(pady=10)
panel = Label(root, bg='#DDD', width=250, height=250); panel.pack(pady=10)
frm = Frame(root); frm.pack(pady=5)
Button(frm, text='Browse X-Ray', command=browse,
      bg='#3A86FF', fg='white').pack(side='left', padx=10)
Button(frm, text='Predict', command=predict,
      bg='#FF006E', fg='white', font=('Helvetica',11,'bold')).pack(side='left', padx=10)
lbl_result = Label(root, text='', font=('Helvetica',14,'bold'))
lbl_result.pack(pady=20)
root.mainloop()
