import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory
from flask_cors import CORS, cross_origin
import subprocess

UPLOAD_VIDEO_FOLDER = 'video'
UPLOAD_IMAGE_FOLDER = 'image'
DOWNLOAD_IMAGE_FOLDER = 'image-pred'
DOWNLOAD_VIDEO_FOLDER = 'video'
CONDA_ENV = 'lane'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'mp4', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_VIDEO_FOLDER'] = UPLOAD_VIDEO_FOLDER
app.config['UPLOAD_IMAGE_FOLDER'] = UPLOAD_IMAGE_FOLDER
app.config['DOWNLOAD_VIDEO_FOLDER'] = DOWNLOAD_VIDEO_FOLDER
app.config['DOWNLOAD_IMAGE_FOLDER'] = DOWNLOAD_IMAGE_FOLDER
app.config['CONDA_ENV'] = CONDA_ENV


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload/video', methods=['GET', 'POST'])
@cross_origin()
def upload_video():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_VIDEO_FOLDER'], filename))
            return redirect(url_for('download_video', name=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


def pred_image(file_path):
    extension = file_path.rsplit('.', 1)[-1]
    prefix_folder = file_path.rsplit('\\', 1)[1].rsplit('.', 1)[0]
    commands = (
            "conda activate " + app.config['CONDA_ENV'] + "\n" +
            "python tools/vis/lane_img_dir.py"
            " --image-path=" + os.path.join(app.config['UPLOAD_IMAGE_FOLDER'], prefix_folder) +
            " --image-suffix=" + extension +
            " --save-path=" + app.config['DOWNLOAD_IMAGE_FOLDER'] +
            " --pred --config=configs/lane_detection/bezierlanenet/resnet18_culane-aug1b.py" +
            " --checkpoint=checkpoints/trained_model/resnet18_bezierlanenet_culane_aug1b_20211109.pt\n"
    )
    process = subprocess.Popen("cmd.exe", shell=False, universal_newlines=True,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdin, stdout = process.communicate(commands)
    print("input:\n", stdin)
    print("output:\n", stdout)
    return


@app.route('/upload/image', methods=['GET', 'POST'])
@cross_origin()
def upload_image():
    if request.method == 'POST':
        # check if the post request has the file part
        print(request)
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            prefix_folder = filename.rsplit('.', 1)[0]  # 用文件名在上传文件夹内再创建一个文件夹
            folder_path = os.path.join(app.config['UPLOAD_IMAGE_FOLDER'], prefix_folder)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            file_path = os.path.join(folder_path, filename)
            file.save(file_path)
            pred_image(file_path)
            return url_for('download_image', name=filename)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


@app.route('/download/image/<name>')
def download_image(name):
    return send_from_directory(app.config["DOWNLOAD_IMAGE_FOLDER"], name)


@app.route('/download/video/<name>')
def download_video(name):
    return send_from_directory(app.config["DOWNLOAD_VIDEO_FOLDER"], name)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
