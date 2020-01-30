from flask_session import Session

import dash


app = dash.Dash(__name__)

app.server.secret_key = '_5#y2L"F4Q8z]/'
app.server.config.from_object("config.Config")
app.server.config["SESSION_TYPE"] = "filesystem"
app.config.suppress_callback_exceptions = True

sess = Session()
sess.init_app(app.server)