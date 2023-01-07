from flask import *
from PIL import Image
from ml import predict
from flask_login import UserMixin, login_user, login_required, logout_user, current_user
from flask_login import LoginManager
from sqlalchemy.sql import func
from werkzeug.security import generate_password_hash, check_password_hash
from models import *
from app import *
import math
import flask_socketio
from flask_socketio import *

# Example pull request
# I love cheese

super(cheese)

def degtokm(lat1, long1, lat2, long2):
    R = 6371;  # km
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(long2 - long1)
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)) * math.sin(dLon / 2) * math.sin(dLon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a));
    d = R * c;
    return d


app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")
create_database(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)
app.app_context()


def run():
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


@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))


# Comment the code below once the people are added.
#with app.app_context(): run()
#print("Test people added")


@app.route('/', methods=['GET', 'POST'])
@login_required
def home():
    if request.method == 'POST':
        sendto_firstname = request.form['redirect']
        current_firstname = current_user.firstname
        prediction = predict()  # EX: prediction = ["Melanoma, 0.89"]

        user_1_firstname = sorted([current_firstname, sendto_firstname])[0]
        user_2_firstname = sorted([current_firstname, sendto_firstname])[1]

        m = Messages.query.filter_by(user_1_firstname=user_1_firstname, user_2_firstname=user_2_firstname).first()

        if m:
            messages = m.messages
        else:
            messages = []

        return render_template("chat.html", user=current_user, sendto_firstname=sendto_firstname, flash=flash,
                               messages=messages, predict=prediction)
        image = request.files['image']
        img = Image.open(image)

    elif request.method == "GET":
        return render_template('home.html', predict=[None, None])


@app.route("/sign-up", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        firstname = request.form.get("firstname")
        lastname = request.form.get("lastname")
        email = request.form.get("email")
        password = request.form.get("password")
        confirmpassword = request.form.get("confirmpassword")
        job = request.form.get("job")
        location = [float(i) for i in request.form.get("loc").split(" ")]

        print(location)

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
                return redirect(url_for('home'))

    return render_template("sign-up.html", user=current_user)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = User.query.filter_by(email=email).first()
        if user:
            if check_password_hash(user.password, password):
                flash("Logged in successfully!", category="success")
                login_user(user, remember=True)
                return redirect(url_for("home"))
            else:
                flash("Incorrect password, try again.", category="error")
        else:
            flash("Email does not exist.", category="error")
    return render_template("login.html", user=current_user)


@app.route("/symptoms", methods=["GET", "POST"])
@login_required
def symptoms():
    if request.method == "POST":
        check = request.form.getlist("multiselect")
        print(check)

    return render_template("symptoms.html")


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out successfully!", category="success")
    return redirect(url_for('login'))


@app.route("/communicate-patient", methods=["GET", "POST"])
@login_required
def communicate_patient():
    if current_user.job == "patient":
        distances = {}

        for med in User.query.filter_by(job="Medical Practitioner").all():
            patient_location = current_user.location
            med_location = med.location
            distance = round(degtokm(patient_location[0], patient_location[1], med_location[0], med_location[1]), 2)
            distances[med] = distance

        top_3 = dict(sorted(distances.items(), key=lambda x: x[1])[:3])

        if request.method == "POST":
            sendto_firstname = request.form['redirect']
            current_firstname = current_user.firstname

            user_1_firstname = sorted([current_firstname, sendto_firstname])[0]
            user_2_firstname = sorted([current_firstname, sendto_firstname])[1]

            m = Messages.query.filter_by(user_1_firstname=user_1_firstname, user_2_firstname=user_2_firstname).first()

            if m:
                messages = m.messages
            else:
                messages = []

            # The user will be added to the contacts of the medical practitioner that the user wants to send a message

            send_to_user = User.query.filter_by(firstname=sendto_firstname).first()

            if not (current_user.firstname in list(send_to_user.contacts)):
                send_to_user.contacts.append(current_user.firstname)
                db.session.commit()

            return render_template("chat.html", user=current_user, sendto_firstname=sendto_firstname, flash=flash,
                                   messages=messages)
        elif request.method == "GET":
            return render_template("communicate-patient.html", user=current_user, top=top_3)
    else:
        flash("Sorry but since you are not a patient you can't access that page.", category="error")
        return redirect(url_for("communicate_medical"))


@app.route("/communicate-medical", methods=["GET", "POST"])
@login_required
def communicate_medical():
    if current_user.job != "patient":
        contacts = list(current_user.contacts)

        if request.method == "POST":
            sendto_firstname = request.form['redirect']
            current_firstname = current_user.firstname

            user_1_firstname = sorted([current_firstname, sendto_firstname])[0]
            user_2_firstname = sorted([current_firstname, sendto_firstname])[1]

            m = Messages.query.filter_by(user_1_firstname=user_1_firstname, user_2_firstname=user_2_firstname).first()

            if m:
                messages = m.messages
            else:
                messages = []

            return render_template("chat.html", user=current_user, sendto_firstname=sendto_firstname, flash=flash, messages=messages)
        elif request.method == "GET":
            return render_template("communicate-medical.html", user=current_user, contacts=contacts)
    else:
        flash("Sorry but since you are not a medical practitioner you can't access that page.", category="error")
        return redirect(url_for("communicate_patient"))


users = {}


@socketio.on("connect")
def on_connect():
    current_socketid = request.sid
    current_firstname = current_user.firstname
    users[current_firstname] = current_socketid

    print("User connected: " + current_firstname)


@socketio.on("disconnect")
def on_disconnect():
    current_firstname = current_user.firstname
    del users[current_firstname]

    print("User disconnected: " + current_firstname)


@socketio.on("message")
def handle_message(message, sendto_firstname):
    print(message)

    if message != "User Connected!":
        current_firstname = current_user.firstname
        current_socketid = request.sid

        user_1_firstname = sorted([current_firstname, sendto_firstname])[0]
        user_2_firstname = sorted([current_firstname, sendto_firstname])[1]

        m = Messages.query.filter_by(user_1_firstname=user_1_firstname, user_2_firstname=user_2_firstname).first()

        if m:
            m.messages.append(message)
            db.session.commit()
        else:
            new_messages = Messages(user_1_firstname=user_1_firstname, user_2_firstname=user_2_firstname,
                                    messages=[message])
            db.session.add(new_messages)
            db.session.commit()

        if sendto_firstname in users.keys():
            sendto_socketid = users[sendto_firstname]

            emit("message", message, to=[sendto_socketid, current_socketid])
        else:
            emit("message", "offline", to=current_socketid)


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)

app.run()
