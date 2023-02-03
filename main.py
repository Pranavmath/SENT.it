import requests
from flask import *
import numpy as np
from ml import predict, onehot_encode_inputs, predict_array
from PIL import Image
from flask_login import login_user, login_required, logout_user, current_user
from flask_login import LoginManager
from werkzeug.security import generate_password_hash, check_password_hash
from models import User, Messages
from app import *
import math
from flask_socketio import *

"""
Note: Since the model files are really big git lfs was used. 
Since an error was originally thrown after using git lfs this stack overflow article was used to fix the error:
https://stackoverflow.com/questions/33330771/git-lfs-this-exceeds-githubs-file-size-limit-of-100-00-mb

If other files with big sizes are added to the project:
Add them to git lfs in reference to the git lfs page and the stack overflow link above
"""


# This gives the distance in km between 2 points given their longitude and latitudes
def degtokm(lat1, long1, lat2, long2):
    R = 6371;  # km
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(long2 - long1)
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)) * math.sin(dLon / 2) * math.sin(dLon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a));
    d = R * c;
    return d


# Configuring flask_socketio and flask app
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")
create_database(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)
app.app_context()


# This adds test medical practitioners
def add_practitioners():
    # Medical practitioners:
    user1 = User(email="a@email.com",
                 firstname="ADoctor",
                 password=generate_password_hash("1234567", method='sha256'),
                 lastname="Bruh",
                 job="Medical Practitioner",
                 location=[25.980, -80.277])
    user2 = User(email="b@email.com",
                 firstname="BDoctor",
                 password=generate_password_hash("1234567", method='sha256'),
                 lastname="Bruh",
                 job="Medical Practitioner",
                 location=[26.626, -81.736])
    user3 = User(email="c@email.com",
                 firstname="CDoctor",
                 password=generate_password_hash("1234567", method='sha256'),
                 lastname="Bruh",
                 job="Medical Practitioner",
                 location=[28.481, -81.339])
    user4 = User(email="d@email.com",
                 firstname="DDoctor",
                 lastname="Bruh",
                 password=generate_password_hash("1234567", method='sha256'),
                 job="Medical Practitioner",
                 location=[40.398, -3.608])

    db.session.query(User).delete()

    db.session.add(user1)
    db.session.add(user2)
    db.session.add(user3)
    db.session.add(user4)

    db.session.commit()


@app.route("/npi", methods=["POST"])
def check_npi():
    number = request.form['number']

    URL = "https://npiregistry.cms.hhs.gov/api/?number=" + number + "&version=2.1"

    r = requests.get(url=URL)

    data = r.json()

    return data


# This just gets our current user
@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))


# Comment the code below once the people are added.
# with app.app_context(): add_practitioners()
# print("Test people added")

@app.route('/', methods=['GET', 'POST'])
def home_page():
    return render_template("home.html", current_user=current_user)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    return render_template("contact.html", current_user=current_user)

@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template("about.html", current_user=current_user)


@app.route('/image', methods=['GET', 'POST'])
@login_required
def image_page():
    # If they submit an image then given them the prediction
    if request.method == 'POST':
        if request.form.get("form_type") == "image":
            top_3 = get_top_3()

            image = request.files['image']
            img = Image.open(image)

            prediction = predict(img)  # EX: prediction = ["Melanoma, 0.89"]

            return render_template('image.html', predict=prediction, user=current_user, top=top_3)
        if request.form.get("form_type") == "followup":
            sendto_firstname = request.form['redirect']
            current_firstname = current_user.firstname

            # This is the alphabetically first user from the current and send_to firstnames
            user_1_firstname = sorted([current_firstname, sendto_firstname])[0]

            # This is the alphabetically second user from the current and send_to firstnames
            user_2_firstname = sorted([current_firstname, sendto_firstname])[1]

            # This is the messages object between the users
            m = Messages.query.filter_by(user_1_firstname=user_1_firstname, user_2_firstname=user_2_firstname).first()

            # Automatic prompt added to current messages
            prediction = [request.form.get("disease"), request.form.get("probability")]
            prompt = str(
                f'AUTOGENERATED MESSAGE || {current_user.firstname}: Hello Doctor, the website predicted that I have the following disease {prediction[0]} with the probability of {float(prediction[1]) * 100}%')

            # If they sent messages then get the list of messages
            if m:
                m.messages.append(prompt)
                db.session.commit()
            # If they have not sent any messages between each other than make the messages an empty list
            else:
                new_messages = Messages(user_1_firstname=user_1_firstname, user_2_firstname=user_2_firstname,
                                        messages=[prompt])
                db.session.add(new_messages)
                db.session.commit()

            # Messages object after commiting
            m = Messages.query.filter_by(user_1_firstname=user_1_firstname, user_2_firstname=user_2_firstname).first()
            messages = m.messages

            # The user will be added to the contacts of the medical practitioner
            # The contacts of a medical practitioners are all the patients that have talked to them
            send_to_user = User.query.filter_by(firstname=sendto_firstname).first()

            # This makes sure they aren't already in the contact
            if not (current_user.firstname in list(send_to_user.contacts)):
                send_to_user.contacts.append(current_user.firstname)
                db.session.commit()

            return render_template("chat.html", user=current_user, sendto_firstname=sendto_firstname, flash=flash,
                                   messages=messages)

    # If not then just return the prediction as [None, None] so nothing is displayed
    elif request.method == "GET":
        return render_template('image.html', predict=[None, None])


@app.route("/sign-up", methods=["GET", "POST"])
def signup():
    # If they submit the form then get all the data and add to database then redirect to home
    if request.method == "POST":
        firstname = request.form.get("firstname")
        lastname = request.form.get("lastname")
        email = request.form.get("email")
        password = request.form.get("password")
        confirmpassword = request.form.get("confirmpassword")
        job = request.form.get("job")

        try:
            location = [float(i) for i in request.form.get("loc").split(" ")]

            print(location)
        except:
            flash("Please enter a valid location", category="error")

        user = User.query.filter_by(email=email).first()

        if user:
            flash("Email already exists.", category="error")
        else:
            if len(email) < 4:
                flash("Email must be greater than 3 characters!", category="error")
            elif len(firstname) < 2:
                flash("Your first name must be greater than 1 characters!", category="error")
            elif len(password) != len(confirmpassword):
                flash("Your passwords must match!", category="error")
            elif len(password) < 7:
                flash("Your password must be greater than 6 characters!", category="error")
            else:
                new_user = User(email=email, firstname=firstname, lastname=lastname,
                                password=generate_password_hash(password, method='sha256'), job=job, location=location)
                db.session.add(new_user)
                db.session.commit()
                login_user(new_user, remember=True)
                flash("Account created!", category="success")
                return redirect(url_for('home_page'))

    return render_template("sign-up.html", user=current_user, check_npi=check_npi)


@app.route("/login", methods=["GET", "POST"])
def login():
    # If they are trying to log in, then check if the login credentials are correct and then redirect to home
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = User.query.filter_by(email=email).first()
        if user:
            if check_password_hash(user.password, password):
                flash("Logged in successfully!", category="success")
                login_user(user, remember=True)
                return redirect(url_for("home_page"))
            else:
                flash("Incorrect password, try again.", category="error")
        else:
            flash("Email does not exist.", category="error")
    return render_template("login.html", user=current_user)


@app.route("/symptoms", methods=["GET", "POST"])
@login_required
def symptoms():
    # When they send their symptoms then return the prediction
    if request.method == "POST":
        if request.form.get("form_type") == "symptoms":
            top_3 = get_top_3()
            symptoms_list = request.form.getlist("multiselect")

            # Right now we just print the list of symptoms, but later we will get predictions from it

            symptoms_array = np.asarray(onehot_encode_inputs(symptoms_list)).astype(np.float32)
            prediction = predict_array(symptoms_array)

            return render_template("symptoms.html", predict=prediction, top=top_3)
        if request.form.get("form_type") == "followup":
            sendto_firstname = request.form['redirect']
            current_firstname = current_user.firstname

            # This is the alphabetically first user from the current and send_to firstnames
            user_1_firstname = sorted([current_firstname, sendto_firstname])[0]

            # This is the alphabetically second user from the current and send_to firstnames
            user_2_firstname = sorted([current_firstname, sendto_firstname])[1]

            # This is the messages object between the users
            m = Messages.query.filter_by(user_1_firstname=user_1_firstname, user_2_firstname=user_2_firstname).first()

            # Automatic prompt added to current messages
            prediction = request.form.get("disease")
            prompt = str(
                f'AUTOGENERATED MESSAGE || {current_user.firstname}: Hello Doctor, the website predicted that I have the following disease {prediction} ')

            # If they sent messages then get the list of messages
            if m:
                m.messages.append(prompt)
                db.session.commit()
            # If they have not sent any messages between each other than make the messages an empty list
            else:
                new_messages = Messages(user_1_firstname=user_1_firstname, user_2_firstname=user_2_firstname,
                                        messages=[prompt])
                db.session.add(new_messages)
                db.session.commit()

            # Messages object after committing the prompt
            m = Messages.query.filter_by(user_1_firstname=user_1_firstname, user_2_firstname=user_2_firstname).first()
            messages = m.messages

            # The user will be added to the contacts of the medical practitioner
            # The contacts of a medical practitioners are all the patients that have talked to them
            send_to_user = User.query.filter_by(firstname=sendto_firstname).first()

            # This makes sure they aren't already in the contact
            if not (current_user.firstname in list(send_to_user.contacts)):
                send_to_user.contacts.append(current_user.firstname)
                db.session.commit()

            return render_template("chat.html", user=current_user, sendto_firstname=sendto_firstname, flash=flash,
                                   messages=messages)
    else:
        return render_template("symptoms.html", predict=None)


# This just logs out the user
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out successfully!", category="success")
    return redirect(url_for('login'))


def get_top_3():
    distances = {}

    # Adds all info to distances dict
    for med in User.query.filter_by(job="Medical Practitioner").all():
        patient_location = current_user.location
        med_location = med.location
        distance = round(degtokm(patient_location[0], patient_location[1], med_location[0], med_location[1]), 2)
        distances[med] = distance

    print(distances)

    # This gets the top_3 closet medical practitioners
    top_3 = dict(sorted(distances.items(), key=lambda x: x[1])[:3])

    return top_3


@app.route("/communicate-patient", methods=["GET", "POST"])
@login_required
def communicate_patient():
    # If they are a patient then allow them to access
    if current_user.job == "patient":
        # This is the dict with all the medical practitioners - key and their distances from the patient - value

        top_3 = get_top_3()

        # This gets run they click on the button to communicate to a specific medical practitioner
        if request.method == "POST":
            sendto_firstname = request.form['redirect']
            current_firstname = current_user.firstname

            # This is the alphabetically first user from the current and send_to firstnames
            user_1_firstname = sorted([current_firstname, sendto_firstname])[0]

            # This is the alphabetically second user from the current and send_to firstnames
            user_2_firstname = sorted([current_firstname, sendto_firstname])[1]

            # This is the messages object between the users
            m = Messages.query.filter_by(user_1_firstname=user_1_firstname, user_2_firstname=user_2_firstname).first()

            # If they sent messages then get the list of messages
            if m:
                messages = m.messages
            # If they have not sent any messages between each other than make the messages an empty list
            else:
                messages = []

            # The user will be added to the contacts of the medical practitioner
            # The contacts of a medical practitioners are all the patients that have talked to them
            send_to_user = User.query.filter_by(firstname=sendto_firstname).first()

            # This makes sure they aren't already in the contact
            if not (current_user.firstname in list(send_to_user.contacts)):
                send_to_user.contacts.append(current_user.firstname)
                db.session.commit()

            return render_template("chat.html", user=current_user, sendto_firstname=sendto_firstname, flash=flash,
                                   messages=messages)

        # If they haven't clicked on the button then just render the current page
        elif request.method == "GET":
            return render_template("communicate-patient.html", user=current_user, top=top_3)

    # Don't allow them to access since they are not a patient
    else:
        flash("Sorry but since you are not a patient you can't access that page.", category="error")
        return redirect(url_for("communicate_medical"))


@app.route("/communicate-medical", methods=["GET", "POST"])
@login_required
def communicate_medical():
    # If they are a medical practitioner allow them to access
    if current_user.job != "patient":
        # Get the list of patients' firstnames that they communicated to
        contacts = list(current_user.contacts)

        # If they click on the button to communicate with a patient
        if request.method == "POST":
            sendto_firstname = request.form['redirect']
            current_firstname = current_user.firstname

            # This is the alphabetically first user from the current and send_to firstnames
            user_1_firstname = sorted([current_firstname, sendto_firstname])[0]
            # This is the alphabetically second user from the current and send_to firstnames
            user_2_firstname = sorted([current_firstname, sendto_firstname])[1]

            # This is the messages object between the users
            m = Messages.query.filter_by(user_1_firstname=user_1_firstname, user_2_firstname=user_2_firstname).first()

            # If they sent messages then get the list of messages
            if m:
                messages = m.messages
            # If they have not sent any messages between each other than make the messages an empty list
            else:
                messages = []

            return render_template("chat.html", user=current_user, sendto_firstname=sendto_firstname, flash=flash,
                                   messages=messages)
        # If they haven't clicked on the button then just render the current page
        elif request.method == "GET":
            return render_template("communicate-medical.html", user=current_user, contacts=contacts)

    # Don't allow them to access since they are not a medical practioner
    else:
        flash("Sorry but since you are not a medical practitioner you can't access that page.", category="error")
        return redirect(url_for("communicate_patient"))


# Dict of all currently connected users and their socket id
users = {}


# Whenever a user connects to the socket this is ran
@socketio.on("connect")
def on_connect():
    # Get the user's current socketid and firstname and then append to the users dict
    current_socketid = request.sid
    current_firstname = current_user.firstname
    users[current_firstname] = current_socketid

    print("User connected: " + current_firstname)


@socketio.on("disconnect")
def on_disconnect():
    # Delete the user from the user dict when they disconnect
    current_firstname = current_user.firstname
    del users[current_firstname]

    print("User disconnected: " + current_firstname)


# This function is only ran when a user sends a message to the server when they are on chat.html
@socketio.on("message")
def handle_message(message, sendto_firstname):
    print(message)

    # Get the user's current socketid and firstname
    current_firstname = current_user.firstname
    current_socketid = request.sid

    # This is the alphabetically first user from the current and send_to firstnames
    user_1_firstname = sorted([current_firstname, sendto_firstname])[0]
    # This is the alphabetically second user from the current and send_to firstnames
    user_2_firstname = sorted([current_firstname, sendto_firstname])[1]

    # This is the messages object between the users
    m = Messages.query.filter_by(user_1_firstname=user_1_firstname, user_2_firstname=user_2_firstname).first()

    # If they sent messages then append this messages to the list of messages
    if m:
        m.messages.append(message)
        db.session.commit()

    # If they haven't sent any messages between each other than create the new messages object and append this message
    else:
        new_messages = Messages(user_1_firstname=user_1_firstname, user_2_firstname=user_2_firstname,
                                messages=[message])
        db.session.add(new_messages)
        db.session.commit()

    # If the user they want to send a message to is online then send the message to them
    if sendto_firstname in users.keys():
        # Gets the socket id of the user they want to send a message to via the users dict
        sendto_socketid = users[sendto_firstname]

        emit("message", message, to=[sendto_socketid, current_socketid])

    # If the user they want to send a message to is online then tell the current user that they are offline
    else:
        emit("message", [message, "offline"], to=current_socketid)


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True, use_reloader=False)
