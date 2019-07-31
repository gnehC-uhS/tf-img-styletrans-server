import os
from flask import Flask, render_template, send_file, request
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename

from model import ffwd

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

# webapp
app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/style_transfer', methods=["POST"])
def style_transfer():
    """
        Take the input image and style transfer it
    """
    # check if the post request has the file part
    if 'content' not in request.files:
        return BadRequest("Content File not present in request")
    if 'style' not in request.files:
        return BadRequest("Style File not present in request")

    content_image = request.files['content']
    style_image = request.files['style']

    
    if content_image.filename == '' or style_image.filename == '':
        return BadRequest("File name is not present in request")
    if (not allowed_file(content_image.filename)) or (not allowed_file(style_image.filename)):
        return BadRequest("Invalid file type")
    if content_image and style_image and allowed_file(style_image.filename) and allowed_file(content_image.filename):
        content_filename = secure_filename(content_image.filename)
        style_filename = secure_filename(style_image.filename)

        content_filepath = os.path.join('./content_images/', content_filename)
        style_filepath = os.path.join('./style_images/', style_filename)
        output_filepath = os.path.join('./out_images/', content_filename)

        content_image.save(content_filepath)
        style_image.save(style_filepath)

        ffwd(content_filepath, style_filepath, output_filepath, epochs = 1, steps_per_epoch = 10)
        return send_file(output_filepath, mimetype='image/jpg')


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run()
