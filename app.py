from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model = YOLO('model.pt')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('home'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    if filename.lower().endswith(('.mp4', '.avi', '.mov')):
        cap = cv2.VideoCapture(filepath)
        frame_count = 0
        results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count > 30:
                break

            cv2.imwrite('temp.jpg', frame)
            prediction = model('temp.jpg', verbose=False)
            pred_class = prediction[0].names[int(prediction[0].probs.top1)]
            results.append(pred_class)
            frame_count += 1

        cap.release()
        os.remove('temp.jpg')

        from collections import Counter
        final_pred = Counter(results).most_common(1)[0][0]
        return render_template('result.html', prediction=final_pred, filename=filename)

    else:
        prediction = model(filepath, verbose=False)
        pred_class = prediction[0].names[int(prediction[0].probs.top1)]
        return render_template('result.html', prediction=pred_class, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
