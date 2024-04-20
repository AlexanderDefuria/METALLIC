from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS
import pandas as pd
import final_model2
import final_model
from flask_cors import cross_origin
import numpy as np


app = Flask(__name__)
# app.config.from_object(Config) keep a good habit
CORS(app)
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
    # filepath = os.path.join('C:/Ronald/uOttawa/CSI 6900/Metallic-main/Recommendation_system/test_dataset', filename)
    # filepath = os.path.join('./Recommendation_system/test_dataset', filename)
    filepath = os.path.join('./', filename)
    # filepath = os.path.join('./test_dataset', filename)
    
    # Save the file temporarily
    file.save(filepath)
    
    try:
        print("Processing...")
        # recommendations = final_model2.run_model(filepath, metric, classifier, no_resampling_methods)
        # recommendations = final_model2.run_model(filename, metric, classifier, no_resampling_methods)
        recommendations = final_model.run_model(filename, metric, classifier, no_resampling_methods)
        print("Success!")
        print(recommendations)
    except Exception as e:
        # If there's an error during processing, log it and return an error response
        print(e)  # Log to console or a file as appropriate
        return jsonify({"error": "Failed to process the file"}), 500

    finally:
        # Ensure the file is removed after processing, even if there's an error
        if os.path.exists(filepath):
            os.remove(filepath)

    print(jsonify({"recommendations": recommendations}))
    return jsonify({"recommendations": recommendations})
    

CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})
# CORS(app, resources={r"/api/*": {"origins": "https://meta-recommendation-system-gold.vercel.app"}})
# CORS(app, resources={r"*": {"origins": "https://meta-recommendation-system-gold.vercel.app"}})
if __name__ == '__main__':
    app.run(debug=True)
    # port = int(os.environ.get('PORT', 5000))
    # app.run(host='0.0.0.0', port=port)
