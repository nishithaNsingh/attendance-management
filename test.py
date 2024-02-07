import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras.models import load_model
import numpy as np
import pandas as pd
from datetime import date
import os

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX

model = load_model('keras_model.h5')

def get_className(classNo):
    if classNo == 0:
        return "shraddha"
    elif classNo == 1:
        return "mona"

# Load or create the attendance Excel file
attendance_file = 'attendance.xlsx'

# Check if the file already exists
if os.path.exists(attendance_file):
    os.chmod(attendance_file, 0o666)  # Set read and write permissions
    attendance_df = pd.read_excel(attendance_file, index_col=0)
else:
    attendance_df = pd.DataFrame(columns=['Date'])

while True:
    success, imgOriginal = cap.read()
    faces = facedetect.detectMultiScale(imgOriginal, 1.3, 5)

    for x, y, w, h in faces:
        crop_img = imgOriginal[y:y+h, x:x+w]
        img = cv2.resize(crop_img, (224, 224))
        img = img.reshape(1, 224, 224, 3)
        prediction = model.predict(img)
        classIndex = np.argmax(prediction)
        probabilityValue = np.amax(prediction)

        if classIndex == 0 or classIndex == 1:  # Adjust based on your class labels
            name = get_className(classIndex)

            # Capture the current date
            current_date = date.today().strftime("%Y-%m-%d")

            # Check if the name is already in the attendance DataFrame
            if name not in attendance_df.columns:
                attendance_df[name] = ''

            # Check if the date column exists
            if current_date not in attendance_df.index:
                attendance_df.loc[current_date] = ''

            # Mark attendance for the current date and name
            attendance_df.at[current_date, name] = 'Present'

            cv2.rectangle(imgOriginal, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(imgOriginal, (x, y-40), (x+w, y), (0, 255, 0), -2)
            cv2.putText(imgOriginal, name, (x, y-10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(imgOriginal, f'{round(probabilityValue*100, 2)}%', (180, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Result", imgOriginal)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Save the updated attendance to the Excel file
attendance_df.to_excel(attendance_file)

cap.release()
cv2.destroyAllWindows()
