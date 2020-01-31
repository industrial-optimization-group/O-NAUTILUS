from flask_session import Session
from flask import session
import random, string

import dash

external_stylesheets = ['https://cdn.jsdelivr.net/gh/kognise/water.css@latest/dist/light.min.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# random key. make this available to the user?
# If two keys match, the older session will clash with the new session
app.server.secret_key = "".join(
    random.choices(string.ascii_letters + string.digits, k=16)
)
app.server.config.from_object("config.Config")
app.server.config["SESSION_TYPE"] = "filesystem"
app.server.config["SESSION_FILE_DIR"] = "flask_sessions/session" + "".join(
    random.choices(string.ascii_letters + string.digits, k=10)
)
app.config.suppress_callback_exceptions = True

sess = Session()
sess.init_app(app.server)
