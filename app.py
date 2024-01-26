from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import copy
from scripts.cyclone_detection import detectCyclone
from scripts.intensity_estimation import intensityModel
from scripts.cyclone_categorization import categorizeCyclone
from scripts.intensity_estimation import estimateIntensity
from scripts.base64_converter import getBase64String
from scripts.coast_analysis import detectCoastalLine
from scripts.coast_analysis import extractKinematics
from scripts.coast_analysis import drawCoastLine
from scripts.data_analysis import getDataAnalysisData

app = Flask(__name__)
CORS(app)
session = {}

def isValidFile(file_name):
    accepted_files = ['jpg', 'jpeg', 'png']
    return '.' in file_name and file_name.rsplit('.', 1)[1].lower() in accepted_files

@app.route('/upload_image', methods=['POST'])
def uploadImage():
    print(request.files)
    file = request.files['cyclone_image']
    file_name = secure_filename(file.filename)

    if isValidFile(file_name):

        img_bytes = file.read()
        image_array = np.frombuffer(img_bytes, np.uint8)
        session['original_image'] = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        session['roi_image'], cyclone_corr = detectCyclone(img_bytes)

        if cyclone_corr is None:
            return jsonify({"error": "No Instance of Cyclone Found"})
        
        session['intensity'] = estimateIntensity(session['intensity_model'], session['roi_image'])

        # cv2.imwrite('./static/original_images/image.jpg', image)
        
        start_point = (cyclone_corr[0], cyclone_corr[1])
        end_point = (cyclone_corr[2], cyclone_corr[3])

        session['bounding_box_image'] = copy.deepcopy(session['original_image'])
        session['bounding_box_image'] = cv2.rectangle(session['bounding_box_image'], end_point, start_point, (0, 0, 255), 2)
        session['confidence_score'] = format(cyclone_corr[-1], ".2f")

        (w, h), _ = cv2.getTextSize(f'Confidence: {format(cyclone_corr[-1], ".2f")} Intensity: {format(session["intensity"], ".2f")}', cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        session['bounding_box_image'] = cv2.rectangle(session['bounding_box_image'], (cyclone_corr[0], cyclone_corr[1] - 20), (cyclone_corr[0] + w, cyclone_corr[1]), (0, 0, 255), -1)

        session['bounding_box_image'] = cv2.putText(session['bounding_box_image'], f'Confidence: {format(cyclone_corr[-1], ".2f")} Intensity: {format(session["intensity"], ".2f")}', (cyclone_corr[0], cyclone_corr[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)
        # cv2.imwrite('./static/cyclone_detected_images/image.jpg', session['bounding_box_image'])

        session['original_base64_image'] = getBase64String(session['original_image'][:, :, ::-1])
        session['bounding_box_base64_image'] = getBase64String(session['bounding_box_image'][:, :, ::-1])
        session['roi_base64_image'] = getBase64String(session['roi_image'][:, :, ::-1])

        landfall, coast_line, dist = detectCoastalLine(session['original_image'], cyclone_corr)
        time = extractKinematics(dist, session['intensity'])

        session['coastline_image'] = drawCoastLine(session['original_image'], cyclone_corr, landfall, coast_line)
        session['coastline_base64_image'] = getBase64String(session['coastline_image'][:, :, ::-1])

        int_min, int_max = getDataAnalysisData(file_name)
        
        file_name = 'image.jpg'
        category = categorizeCyclone(session['intensity'])
        status = 'Detected'


        data = {
            "original_image": session['original_base64_image'],
            "roi_image": session['roi_base64_image'],
            "bounding_box_image": session['bounding_box_base64_image'],
            "coastline_image": session['coastline_base64_image'],
            "coastline": coast_line,
            "time": time,
            "category": category,
            "distance": round(dist * 4, 2),
            "intensity": session["intensity"],
            "confidence_score": session['confidence_score'],
            "int_min": int_min,
            "int_max": int_max
        }

        return jsonify(data)

    return {}

if __name__ == '__main__':
    session['intensity_model'] = intensityModel()
    app.run(debug=True)