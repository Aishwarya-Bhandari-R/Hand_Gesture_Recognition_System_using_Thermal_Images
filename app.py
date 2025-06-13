import os
import cv2

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing.image import img_to_array
import pickle
from flask import Flask, render_template, url_for, request
import numpy as np
from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing.image import load_img

import sqlite3
import shutil

import pygame
import time
from gtts import gTTS
from mutagen.mp3 import MP3




app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/userlog.html')
def userlogg():
    return render_template('userlog.html')



@app.route('/graph.html', methods=['GET', 'POST'])
def graph():
    
    images = ['http://127.0.0.1:5000/static/accuracy_plot.png',
              'http://127.0.0.1:5000/static/loss_plot.png',
              'http://127.0.0.1:5000/static/confusion_matrix.png']
    content=['Accuracy Graph',
             'Loss Graph',
             'Confusion Matrix']

            
    
        
    return render_template('graph.html',images=images,content=content)
    


@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
 
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
        
        

        shutil.copy("test/"+fileName, dst)
        image = cv2.imread("test/"+fileName)
        
        
        
        model=load_model('gesture_classifier.h5')
        path='static/images/'+fileName


        # Load the class names
        with open('class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
        
        # Function to preprocess the input image
        def preprocess_input_image(path):
            img = load_img(path, target_size=(150,150))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize the image
            return img_array

        # Function to make predictions on a single image
        def predict_single_image(path):
            input_image = preprocess_input_image(path)
            prediction = model.predict(input_image)
            print(prediction)
            predicted_class_index = np.argmax(prediction)
            predicted_class = class_names[predicted_class_index]
            confidence = prediction[0][predicted_class_index]

            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
                
            return predicted_class, confidence 

        predicted_class, confidence = predict_single_image(path)
        #predicted_class, confidence = predict_single_image(path, model, class_names)
        print(predicted_class, confidence)
        if predicted_class == 'call':
            str_label = "call"
            
            
             
           
        elif predicted_class == 'closed_fist':
            str_label = "closed_fist"
            

        elif predicted_class == 'down':
            str_label = "down"
            
            

        elif predicted_class == 'hi':
            str_label = "hi"


            
        elif predicted_class == 'left':
            str_label = "left"
            

        elif predicted_class == 'one':
            str_label = "one"
            
            

        elif predicted_class == 'rock':
            str_label = "rock"


            
        elif predicted_class == 'shot':
            str_label = "shot"
            

        elif predicted_class == 'super':
            str_label = "super"
            
            

        elif predicted_class == 'Thumbs_down':
            str_label = "Thumbs_down"


            
        elif predicted_class == 'Thumbs_up':
            str_label = "Thumbs_up"
            

        elif predicted_class == 'victory':
            str_label = "victory"

        myobj = gTTS(text=predicted_class, lang='en', slow =False)
        myobj.save("voice.mp3")
        song = MP3("voice.mp3")
        pygame.mixer.init()
        pygame.mixer.music.load('voice.mp3')
        pygame.mixer.music.play()
        time.sleep(song.info.length)
        pygame.quit()
    
            

        

            
       
           
            
        accuracy = f"The predicted image is {str_label} with a confidence of {confidence:.2%}"
       
            

       


        return render_template('results.html', status=str_label,accuracy=accuracy,ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName)
        
    return render_template('userlog.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
