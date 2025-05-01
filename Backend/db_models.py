from flask_sqlalchemy import SQLAlchemy

# Initialize SQLAlchemy object
db = SQLAlchemy()

# Define User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(50), nullable=True)
    marital_status = db.Column(db.String(50), nullable=True)

# Define EmotionEntry model
class EmotionEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    emotion = db.Column(db.String(150), nullable=False)
    recommendation = db.Column(db.String(255), nullable=False)

# Debugging output
print("DB initialized in db_models.py:", db)
print("User model defined in db_models.py:", User)
print("EmotionEntry model defined in db_models.py:", EmotionEntry)
