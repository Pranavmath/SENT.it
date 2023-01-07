from app import db
from flask_login import UserMixin
from sqlalchemy import PickleType
from sqlalchemy.ext.mutable import MutableList

# User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    firstname = db.Column(db.String(150))
    lastname = db.Column(db.String(150))
    job = db.Column(db.String(150))
    location = db.Column(db.PickleType())
    # Contacts are all the patients that have sent a message to the medical practioner
    contacts = db.Column(MutableList.as_mutable(PickleType), default=[])

# Messages model
class Messages(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    user_1_firstname = db.Column(db.String(300))  # This will be the firstname that is first alphabetically
    user_2_firstname = db.Column(db.String(300))  # This will be the firstname that is second alphabetically
    messages = db.Column(MutableList.as_mutable(PickleType), default=[])
