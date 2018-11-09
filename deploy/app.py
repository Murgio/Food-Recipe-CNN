# coding=utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datetime import datetime

from flask import Flask, redirect, url_for, request, render_template, stream_with_context, Response, send_from_directory
from werkzeug.utils import secure_filename

from predict_food import PredictFood
from chefkoch_parser import ChefkochParser
from log_food import *
#from gevent.wsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
#app.config['MEDIA_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '/Extracting-food-preferences-master/notebooks/input')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

#@app.route('/images/<path:filename>')
#def custom_static(filename):
#    print(filename)
#    return send_from_directory(app.config['MEDIA_FOLDER'], filename)

@app.route('/predict', methods=['GET', 'POST'])
def streamed_response():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # [['category', incep_confidence, recipe_id, image_index, image_path], [], ...]
        start = datetime.now()

        result_list = predict_food_result.model_predict(file_path)

        end = datetime.now()
        print('FOUND THE FOOD')
        food_time = end-start
        # max 1 second
        print('TOOK ME: ', food_time.microseconds*(1/1000000))
        log(result_list, food_time)
        parser.update_result_list(result_list[:7])
        #result = ' '.join([str(i) for i in result])
        food = parser.food_list_html()
        return ''.join(map(str, food))
    #def upload():
    #        for i, id in enumerate(ids):
    #            yield food_list_html(id_food=id, i=i)
    #return Response(stream_with_context(upload()))
    return None

if __name__ == '__main__':
    parser = ChefkochParser()
    predict_food_result = PredictFood(k=18)
    app.run(host='0.0.0.0', port=5000, debug=False)

    # Serve the app with gevent
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
