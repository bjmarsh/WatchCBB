from flask import Flask
app = Flask(__name__)
from watchcbb.flask import views
