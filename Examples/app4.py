__author__ = "kai"
from flask import Flask, session
import os

app = Flask(__name__)
app.secret_key = os.urandom(25)


@app.route("/")
def index():
    if "user" in session:
        return session["user"]
    return 'No session <a href="/login">Please Log in </a>'

    return 'Please <a href="/login">Log in </a>'


@app.route("/login")
def login():
    session["user"] = "Kai"

    return "Logged in as: " + session["user"] + '<a href="/logout"> Log out</a>'


@app.route("/logout")
def logout():
    session.pop("user", None)
    return "Logged out. " + '<a href="/">Home</a>'


if __name__ == "__main__":
    app.run(debug=True)

