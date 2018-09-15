from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, StreamingHttpResponse
from django.contrib.auth import get_user_model, authenticate
import base64, re, json, time
import matplotlib.pyplot as plt
from io import BytesIO
import tensorflow as tf
from statistics import mode
from skimage.transform import resize
from scipy.spatial import distance
from keras.models import load_model
from face.FaceRec.src.utils.inference import *
from face.FaceRec.src.utils.datasets import get_labels
from face.FaceRec.src.utils.preprocessor import preprocess_input

# model path
detection_model_path = './face/FaceRec/trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = './face/FaceRec/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
facenet_model_path = './face/FaceRec/trained_models/facenet/facenet_keras.h5'
emotion_labels = get_labels('fer2013')

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
facenet_model = load_model(facenet_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
image_size = 160
embeddings = []
emotion_count = 0
emotion_data = {}

# initialization
graph = tf.get_default_graph()
camera = 0
cameraChange = False
video_capture = cv2.VideoCapture(camera)
User = get_user_model()


# Create your views here.

# response of accessing home page /monitor/
# verify user's login state and authority
# change camera between local and RTSP
def home(request):
    error = ''
    global camera
    global cameraChange
    if request.method == 'GET':
        if request.GET.get('logout') is not None:
            request.session.flush()
    elif request.method == 'POST':
        newCamera = request.POST['camera']
        if newCamera == '0':
            newCamera = 0
        if newCamera != camera:
            cap = cv2.VideoCapture(newCamera)
            if cap.read()[1] is not None:
                camera = newCamera
                cameraChange = True
            else:
                error = 'Address error'
    username = request.session.get('username')
    if username:
        if User.objects.get(username=username).is_superuser:
            state = 'root'
        else:
            state = 'normal'
    else:
        state = 'logout'
    return render(request, 'index.html', {'state': state, 'error': error})


# stream video actively
def video(request):
    return StreamingHttpResponse(generateVideo(), content_type="multipart/x-mixed-replace; boundary=frame")


# generator used by aforementioned streaming method
def generateVideo():
    global video_capture
    global cameraChange
    while True:
        time.sleep(0.03)
        img = video_capture.read()[1]
        while img is None or cameraChange:
            video_capture = cv2.VideoCapture(camera)
            img = video_capture.read()[1]
            cameraChange = False
        img = bytes(cv2.imencode('.jpeg', processImg(img))[1])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n\r\n')


# post history alert message
# add new alert message actively
def alertPage(request):
    username = request.session.get('username')
    if username is not None:
        user = User.objects.get(username=username)

        if request.method == 'POST':
            threshold = request.POST.get('threshold')
            if threshold is not None:
                user.threshold = float(threshold)
                user.save()
                return redirect('/monitor/alertPage/')
            else:
                index = int(request.POST['index'])
                pos = int(request.POST['pos'])
                alerts = json.loads(user.historyAlert)
                del alerts[index][pos]
                if len(alerts[index]) == 0:
                    del alerts[index]
                user.historyAlert = json.dumps(alerts)
                user.save()
                return redirect('/monitor/alertPage/')
        elif request.method == 'GET':
            if user.historyAlert != '':
                alerts = json.loads(user.historyAlert)
            else:
                alerts = []
            alerts = [dict(enumerate(a)) for a in alerts]
            alerts = dict(enumerate(alerts))
            return render(request, 'alert.html', {'alerts': alerts, 'username': username, 'threshold': user.threshold})
    else:
        return redirect('/monitor/login/')


# stream alert message actively
def alert(request):
    username = request.session.get('username')
    return StreamingHttpResponse(generateAlert(username))


# generator used by aforementioned streaming method
def generateAlert(username):
    while True:
        user = User.objects.get(username=username)
        if user.update:
            user.update = False
            alerts = json.loads(user.historyAlert)
            data = alerts[-1]
            data = {'data': data, 'index': len(alerts) - 1, 'username': username}
            user.save()
            yield (json.dumps(data).encode('utf-8'))
            time.sleep(1)
        else:
            yield (''.encode('utf-8'))
            time.sleep(1)


# response of accessing login page
def login(request):
    if request.method == 'GET':
        return render(request, 'login.html')
    elif request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)
        if user is None:
            return render(request, 'login.html', {'error': 'Username or password incorrect'})
        else:
            request.session.clear_expired()
            request.session.set_expiry(0)
            request.session['username'] = username
            return redirect('/monitor/')


# response of accessing signup page
def signup(request):
    if request.method == 'GET':
        return render(request, 'signup.html')
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password_1 = request.POST.get('password_1')
        try:
            User.objects.get(username=username)
            return render(request, 'signup.html', {'error': 'User already existed'})
        except User.DoesNotExist:
            User.objects.create_user(username=username, password=password_1, email=email)
            if request.POST.get('type') is not None:
                return redirect('/monitor/manage/')
            request.session.clear_expired()
            request.session.set_expiry(0)
            request.session['username'] = username
            return redirect('/monitor/registerFace/')


# response of accessing account page
def account(request):
    username = request.session.get('username')
    if username is not None:
        user = User.objects.get(username=username)
        return render(request, 'account.html', {'username': username, 'email': user.email})
    else:
        return redirect('/monitor/login/')


# response of accessing change password page
def password(request):
    username = request.session.get('username')
    if request.method == 'GET':
        if username is not None:
            return render(request, 'password.html', {'username': username})
        else:
            return redirect('/monitor/login/')
    elif request.method == 'POST':
        if username is not None:
            password = request.POST['password_1']
            user = User.objects.get(username=username)
            user.password = password
            user.save()
            request.session.flush()
        return redirect('/monitor/login')


# response of accessing manage page
# root authority required
def manage(request):
    username = request.session.get('username')
    if username is not None:
        if User.objects.get(username=username).is_superuser:
            users = User.objects.all()
            return render(request, 'manage.html', {'users': users})
        else:
            return redirect('/monitor/')
    else:
        return redirect('/monitor/login/')


# response of adding normal user by root user
def addUser(request):
    return render(request, 'adduser.html')


# response of accessing every account info by root user
def accountInfo(request, id):
    try:
        user = User.objects.get(id=id)
    except User.DoesNotExist:
        return render(request, '404.html')
    img = ''
    if user.emotionData != '':
        data = eval(user.emotionData)
        sizes = [round(float(x), 2) * 100 for x in data.values()]
        labels = []
        for size, key in zip(sizes, data.keys()):
            if size < 0.5:
                labels.append('')
            else:
                labels.append(key)
        explode = [0, 0, 0, 0, 0, 0, 0]
        plt.pie(sizes, explode, labels, shadow=False, startangle=90, labeldistance=1.1, rotatelabels=True)
        plt.axis('equal')
        plt.legend(loc='upper right', fontsize='xx-small')
        buf = BytesIO()
        plt.savefig(buf, format='jpg')
        img = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('utf8')
        plt.close('all')
    return render(request, 'accountInfo.html', {'user': user, 'img': img})


# register user's face when finishing registration of normal info
def registerFace(request):
    if request.method == 'GET':
        username = request.session.get('username')
        if username is not None:
            return render(request, 'face.html', {'username': username})
        else:
            return redirect('/monitor/signup')
    elif request.method == 'POST':
        global video_capture
        img = video_capture.read()[1]
        while img is None:
            video_capture = cv2.VideoCapture(camera)
            img = video_capture.read()[1]
        imgStr = str(base64.b64encode(img))[2:-1]
        emb, imgBox = calEmbedding(img)
        progress = '%.2f' % (len(embeddings) / 500)
        if emb is not None:
            embeddings.append(emb)
            imgBox = cv2.imencode('.jpeg', imgBox)[1]
            imgBox = str(base64.b64encode(imgBox))[2:-1]
            if len(embeddings) < 500:
                print(len(embeddings))
                data = json.dumps({'img': imgBox, 'progress': progress, 'complete': 'false', 'error': ''})
            else:
                faceData = np.mean(embeddings, axis=0)
                embeddings.clear()
                username = request.session.get('username')
                if username is not None:
                    user = User.objects.get(username=username)
                    user.faceData = json.dumps(faceData.tolist())
                    user.save()
                    data = json.dumps(
                        {'img': imgBox, 'progress': progress, 'complete': 'true', 'error': ''})
                else:
                    data = json.dumps(
                        {'img': imgBox, 'progress': progress, 'complete': 'false', 'error': 'No Username'})
        else:
            data = json.dumps({'img': imgStr, 'progress': progress, 'complete': 'false', 'error': ''})

        return HttpResponse(data)


# face recognition and expression recognition
def processImg(bgr_image):
    # hyper-parameters for bounding boxes shape
    frame_window = 10
    emotion_offsets = (20, 40)

    global face_detection
    global emotion_classifier
    global emotion_target_size
    global emotion_labels
    global graph

    with graph.as_default():
        # starting lists for calculating modes
        emotion_window = []

        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        faces = detect_faces(face_detection, gray_image)

        for face_coordinates in faces:

            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            color_face = bgr_image[y1:y2, x1:x2]
            emb, _ = calEmbedding(color_face)  # feature vector of face
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]  # prediction of expression
            emotion_window.append(emotion_text)

            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except:
                continue

            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((255, 255, 0))
            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
            else:
                color = emotion_probability * np.asarray((0, 255, 0))

            color = color.astype(int)
            color = color.tolist()

            draw_bounding_box(face_coordinates, rgb_image, color)
            draw_text(face_coordinates, rgb_image, emotion_mode,
                      color, 0, -45, 1, 1)

            if emb is not None:
                count = 0
                for user in User.objects.all():
                    if user.faceData != '':
                        data = eval(user.faceData)
                        if calDist(data, emb) < 0.6:
                            draw_text(face_coordinates, rgb_image, user.username,
                                      color, 0, -15, 1, 1)
                            processData(user, emotion_text)
                            break
                    count += 1
                if count == len(User.objects.all()):
                    draw_text(face_coordinates, rgb_image, 'other',
                              color, 0, -15, 1, 1)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        return bgr_image


# statistics analyze of emotion data
# calculate the ratio of each expression and store them in database
# compare new data with old data. if new data exceeds threshold, generate alert
def processData(user, emotion_text):
    global emotion_count
    global emotion_data
    emotion_count += 1
    if emotion_count <= 100:
        if emotion_text in emotion_data:
            emotion_data[emotion_text] += 1
        else:
            emotion_data[emotion_text] = 1
    else:
        overThreshold = []
        underThreshold = []
        storedData = json.loads(user.emotionData)
        for key, value in storedData.items():
            if key not in emotion_data:
                emotion_data[key] = 0
            emotion_data[key] = (emotion_data[key] / 100 + value * user.emotionCount) / (
                    user.emotionCount + 1)
            if emotion_data[key] > user.threshold:
                overThreshold.append(key)
            else:
                underThreshold.append(key)

        if user.historyAlert != '':
            under = json.loads(user.underThreshold)
            diff = set(overThreshold).intersection(set(under))
            if len(diff) != 0:
                user.update = True
                history = json.loads(user.historyAlert)
                history.append(list(diff))
                user.historyAlert = json.dumps(history)
            else:
                user.update = False
        else:
            history = [overThreshold]
            user.historyAlert = json.dumps(history)
            user.update = True

        user.underThreshold = json.dumps(underThreshold)
        user.emotionData = json.dumps(emotion_data)
        user.emotionCount += 1
        user.save()

        emotion_count = 0
        emotion_data.clear()


# process face image
def preWhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0 / np.sqrt(size))
    y = (x - mean) / std_adj
    return y


# process face image
def l2Normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


# process face image
def alignImage(img, margin):
    faces = face_detection.detectMultiScale(img,
                                            scaleFactor=1.1,
                                            minNeighbors=3)
    if len(faces) == 1:
        (x, y, w, h) = faces[0]
        imgBox = img.copy()
        draw_bounding_box(faces[0], imgBox, np.asarray((255, 0, 0)).astype(int).tolist())
        cropped = img[y - margin // 2:y + h + margin // 2,
                  x - margin // 2:x + w + margin // 2, :]
        aligned = resize(cropped, (image_size, image_size), mode='reflect')

        return aligned, imgBox
    else:
        return None, None


# calculate feature vector of human face
def calEmbedding(img, margin=10):
    with graph.as_default():
        img, imgBox = alignImage(img, margin)
        if img is not None:
            alignedImage = preWhiten(img)
            emb = facenet_model.predict(np.expand_dims(np.array(alignedImage), axis=0))
            emb = l2Normalize(np.concatenate(emb))

            return emb, imgBox
        else:
            return None, None


# calculate the Euclidean distance of two faces' feature vector
def calDist(source, target):
    return distance.euclidean(source, target)
