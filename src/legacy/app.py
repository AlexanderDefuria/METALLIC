from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS
import final_model2
import final_model
from flask_cors import cross_origin
import queue
import threading
import uuid
import time


app = Flask(__name__)
# app.config.from_object(Config) keep a good habit
CORS(app)

task_queue = queue.Queue()
results = {}
def daemon():
    while True:
        task = task_queue.get(block=True)
        try:
            # time.sleep(10)
            task_id = task['id']
            recommendations = final_model.run_model(**task['args'])
            results[task_id] = recommendations
            print(f"Task completed with result: {recommendations}")
        finally:
            task_queue.task_done()
            filepath = task['args']['filename']
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"Deleted file at {filepath}")

threading.Thread(target=daemon, daemon=True).start()

@app.route('/runmodel', methods=['POST','GET'])
@cross_origin()
def run_model_route():
    # Check if the 'file' part is present in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    metric = request.form['metric']
    print(metric)
    classifier = request.form['classifier']
    print(classifier)
    no_resampling_methods = int(request.form['no_resampling_methods'])
    print(no_resampling_methods)

    filename = secure_filename(file.filename)
    print(filename)
    filepath = os.path.join('./', filename)
    
    # Save the file temporarily
    file.save(filepath)

    task_id = str(uuid.uuid4())

    task = {
        'id': task_id,
        'args': {
            'filename': filepath,
            'metric': metric,
            'classifier': classifier,
            'no_resampling_methods': no_resampling_methods
        },
        'filepath': filepath  
    }
    task_queue.put(task)
    return jsonify({"task_id": task_id}), 202  

@app.route('/results/<task_id>', methods=['GET'])
def get_results(task_id):
    if task_id in results:
        # return jsonify({"recommendations": results.pop(task_id)})  # Use pop to clear completed results
        return jsonify({"recommendations": results[task_id]})  
    else:
        return jsonify({"status": "Task is still processing"}), 202
    

# CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})
CORS(app, resources={r"*": {"origins": "https://meta-recommendation-system-gold.vercel.app"}})
if __name__ == '__main__':
    # app.run(debug=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
