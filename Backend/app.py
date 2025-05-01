# Import required libraries
from dotenv import load_dotenv  # For loading environment variables from a .env file
load_dotenv()
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, g  # Flask web framework components
from flask_sqlalchemy import SQLAlchemy  # ORM for database management
from flask_migrate import Migrate  # For database migrations
from werkzeug.security import generate_password_hash, check_password_hash  # For password security
import tensorflow as tf  # For machine learning
import numpy as np  # For numerical operations
import os  # For operating system related functions
import json  # For JSON handling
import requests  # For HTTP requests
import spotipy  # Python library for the Spotify API
from spotipy.oauth2 import SpotifyOAuth  # Authentication for Spotify API
import openai  # For OpenAI API integration
import re  # For regular expressions
from textblob import TextBlob  # For sentiment analysis
from datetime import datetime, timedelta  # For date and time operations
import random  # For random selections
from collections import Counter  # For counting occurrences
from recommendation_engine import get_api_recommendations  # Custom module for recommendations
from functools import lru_cache  # For function result caching
import logging  # For logging
from logging.handlers import RotatingFileHandler  # For log file rotation
from threading import Thread  # For multi-threading
from flask_limiter import Limiter  # For rate limiting
from flask_limiter.util import get_remote_address  # To get client IP for rate limiting

# Configure Flask application
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', os.urandom(24))  # Set secret key for session security
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///users.db')  # Database connection
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable modification tracking
app.config['SESSION_PERMANENT'] = True  # Make sessions permanent
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # Session lasts 7 days
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 10,  # Maximum number of database connections
    'max_overflow': 20,  # Maximum number of connections that can be created beyond pool_size
    'pool_recycle': 1800,  # Recycle connections after 30 minutes
}

# Set up logging
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)  # Log rotation setup
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# Initialize extensions
db = SQLAlchemy(app)  # Database ORM
migrate = Migrate(app, db)  # Database migrations
limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])  # Rate limiting

# Load TensorFlow model for mental health predictions
model_path = 'mental_health_model.h5'
if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        app.logger.info("TensorFlow model loaded successfully")
    except Exception as e:
        model = None
        app.logger.error(f"Error loading TensorFlow model: {e}")
else:
    model = None
    app.logger.warning("TensorFlow model not found")

# Load academic resources from JSON file
try:
    with open("resources.json", "r") as f:
        course_resources = json.load(f)
    app.logger.info("Academic resources loaded successfully")
except Exception as e:
    course_resources = {}
    app.logger.error(f"Error loading resources.json: {e}")

# Database Models
class User(db.Model):
    """User model for storing user account information"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True, index=True)
    password = db.Column(db.String(150), nullable=False)
    age = db.Column(db.Integer, nullable=False, default=18)
    gender = db.Column(db.String(50), nullable=False, default="Not Specified")
    marital_status = db.Column(db.String(50), nullable=False, default="Single")
    
    # Relationships with other models for easier access
    mood_logs = db.relationship('MoodLog', backref='user', lazy=True)
    journal_entries = db.relationship('JournalEntry', backref='user', lazy=True)
    progress_items = db.relationship('UserProgress', backref='user', lazy=True)
    music_history = db.relationship('UserMusicHistory', backref='user', lazy=True)

class UserProgress(db.Model):
    """Model for tracking user progress on recommendations"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), index=True)
    title = db.Column(db.String(255))
    completed = db.Column(db.Boolean, default=False)

class MoodLog(db.Model):
    """Model for tracking user moods over time"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), index=True)
    mood = db.Column(db.String(50))
    sentiment = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)

class UserMusicHistory(db.Model):
    """Model for tracking music preferences and liked tracks"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), index=True)
    track_title = db.Column(db.String(255))
    artist_name = db.Column(db.String(255))
    liked = db.Column(db.Boolean, default=False, index=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)

class JournalEntry(db.Model):
    """Model for storing user journal entries"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), index=True)
    entry_text = db.Column(db.Text, nullable=False)
    mood = db.Column(db.String(50))
    sentiment = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)

# Ensure database tables exist and create additional indexes
with app.app_context():
    db.create_all()  # Create tables if they don't exist
    try:
        # Create additional indexes for performance
        db.session.execute('CREATE INDEX IF NOT EXISTS idx_mood_sentiment ON mood_log(sentiment)')
        db.session.execute('CREATE INDEX IF NOT EXISTS idx_journal_mood ON journal_entry(mood)')
        db.session.commit()
        app.logger.info("Database initialized with indexes")
    except Exception as e:
        app.logger.error(f"Error creating database indexes: {e}")

# Constants and configuration data
ITEMS_PER_PAGE = 10  # For pagination
CACHE_TIMEOUT = 3600  # 1 hour in seconds for cache expiration

# Categorical options for user inputs
courses = ['Computer Science', 'Engineering', 'Mathematics', 'Business Administration', 'Psychology']
years_of_study = ['year 1', 'year 2', 'year 3', 'year 4']
# Mapping courses and years to numerical values for the ML model
course_encoding = {course: idx for idx, course in enumerate(courses)}
year_encoding = {year: idx for idx, year in enumerate(years_of_study)}

# Learning paths by course and year - structured educational recommendation paths
learning_paths = {
    'Computer Science': {
        'year 1': ['Intro to Programming', 'Basic Data Structures', 'Python Basics'],
        'year 2': ['OOP', 'Algorithms', 'Web Development'],
        'year 3': ['Machine Learning', 'Databases', 'Software Engineering'],
        'year 4': ['Deep Learning', 'Big Data Analytics', 'Cloud Computing']
    },
    'Engineering': {
        'year 1': ['Engineering Fundamentals', 'Basic Thermodynamics'],
        'year 2': ['Fluid Mechanics', 'Materials Science'],
        'year 3': ['Control Systems', 'Robotics'],
        'year 4': ['Sustainable Engineering', 'Manufacturing Processes']
    },
    'Mathematics': {
        'year 1': ['Algebra Fundamentals', 'Pre-Calculus', 'Mathematical Thinking'],
        'year 2': ['Linear Algebra', 'Probability & Statistics', 'Mathematical Modelling'],
        'year 3': ['Calculus II', 'Numerical Methods', 'Applied Mathematics'],
        'year 4': ['Advanced Statistics', 'Game Theory', 'Computational Mathematics']
    },
    'Business Administration': {
        'year 1': ['Introduction to Business', 'Marketing Fundamentals', 'Business Communication'],
        'year 2': ['Accounting Basics', 'Entrepreneurship', 'Financial Management'],
        'year 3': ['Strategic Management', 'Human Resource Management', 'Operations Management'],
        'year 4': ['Corporate Finance', 'International Business', 'Business Analytics']
    },
    'Psychology': {
        'year 1': ['Introduction to Psychology', 'Cognitive Psychology', 'Research Methods'],
        'year 2': ['Social Psychology', 'Developmental Psychology', 'Personality Theories'],
        'year 3': ['Clinical Psychology', 'Neuroscience', 'Abnormal Psychology'],
        'year 4': ['Psychotherapy Techniques', 'Forensic Psychology', 'Positive Psychology']
    }
}

# Artist bank for music recommendations, organized by mood and gender
artist_bank = {
    "The Weeknd": {"id": "1Xyo4u8uXC1ZmMpatF05PJ", "moods": ["sad", "relaxing"], "gender": "male"},
    "Drake": {"id": "3TVXtAsR1Inumwj472S9r4", "moods": ["happy", "motivational"], "gender": "male"},
    "SZA": {"id": "6LuN9FCkKOj5PcnpouEgny", "moods": ["relaxing", "sad"], "gender": "female"},
    "Ed Sheeran": {"id": "6eUKZXaKkcviH0Ku9w2n3V", "moods": ["happy", "sad"], "gender": "male"},
    "Adele": {"id": "4dpARuHxo51G3z768sgnrY", "moods": ["sad"], "gender": "female"},
    "Kanye West": {"id": "5K4W6rqBFWDnAN6FQUkS6x", "moods": ["motivational"], "gender": "male"},
    "Future": {"id": "1RyvyyTE3xzB2ZywiAwp0i", "moods": ["motivational"], "gender": "male"},
    "Selena Gomez": {"id": "0C8ZW7ezQVs4URX5aX7Kqx", "moods": ["happy", "relaxing"], "gender": "female"},
    "Justin Bieber": {"id": "1uNFoZAHBGtllmzznpCI3s", "moods": ["happy", "relaxing"], "gender": "male"},
    "Ariana Grande": {"id": "66CXWjxzNUsdJxJ2JdwvnR", "moods": ["relaxing", "happy"], "gender": "female"},
    "Rihanna": {"id": "5pKCCKE2ajJHZ9KAiaK11H", "moods": ["motivational", "relaxing"], "gender": "female"},
    "ASAP Rocky": {"id": "13ubrt8QOOCPljQ2FL1Kca", "moods": ["motivational", "sad"], "gender": "male"},
    "Katy Perry": {"id": "6jJ0s89eD6GaHleKKya26X", "moods": ["happy"], "gender": "female"},
    "Beyoncé": {"id": "6vWDO969PvNqNYHIOW5v0m", "moods": ["motivational", "relaxing"], "gender": "female"},
    "Shakira": {"id": "0EmeFodog0BfCgMzAIvKQp", "moods": ["happy", "motivational"], "gender": "female"},
    "Lady Gaga": {"id": "1HY2Jd0NmPuamShAr6KMms", "moods": ["happy", "sad"], "gender": "female"},
    "Nicki Minaj": {"id": "0hCNtLu0JehylgoiP8L4Gh", "moods": ["motivational"], "gender": "female"},
    "Bruno Mars": {"id": "0du5cEVh5yTK9QJze8zA0C", "moods": ["happy", "relaxing"], "gender": "male"},
    "Pitbull": {"id": "0TnOYISbd1XYRBk9myaseg", "moods": ["happy", "motivational"], "gender": "male"},
    "Billie Eilish": {"id": "6qqNVTkY8uBg9cP3Jd7DAH", "moods": ["sad", "relaxing"], "gender": "female"},
    "Doja Cat": {"id": "5cj0lLjcoR7YOSnhnX0Po5", "moods": ["happy", "motivational"], "gender": "female"},
    "Post Malone": {"id": "246dkjvS1zLTtiykXe5h60", "moods": ["sad", "motivational"], "gender": "male"},
    "Halsey": {"id": "26VFTg2z8YR0cCuwLzESi2", "moods": ["sad", "relaxing"], "gender": "female"},
    "Shawn Mendes": {"id": "7n2Ycct7Beij7Dj7meI4X0", "moods": ["happy", "relaxing"], "gender": "male"},
    "Dua Lipa": {"id": "6M2wZ9GZgrQXHCFfjv46we", "moods": ["motivational", "happy"], "gender": "female"},
    "Sam Smith": {"id": "2wY79sveU1sp5g7SokKOiI", "moods": ["sad", "relaxing"], "gender": "male"},
    "Olivia Rodrigo": {"id": "1McMsnEElThX1knmY4oliG", "moods": ["sad", "happy"], "gender": "female"},
    "Harry Styles": {"id": "6KImCVD70vtIoJWnq6nGn3", "moods": ["happy", "relaxing"], "gender": "male"},
    "Tyler, The Creator": {"id": "4V8LLVI7PbaPR0K2TGSxFF", "moods": ["motivational", "relaxing"], "gender": "male"},
    "Lizzo": {"id": "56oDRnqbIiwx4mymNEv7dS", "moods": ["happy", "motivational"], "gender": "female"},
    "Jorja Smith": {"id": "1CoZyIx7UvdxT5c8UkMzHd", "moods": ["relaxing", "sad"], "gender": "female"},
    "Tame Impala": {"id": "5INjqkS1o8h1imAzPqGZBb", "moods": ["relaxing"], "gender": "male"},
    "Miley Cyrus": {"id": "5YGY8feqx7naU7z4HrwZM6", "moods": ["motivational", "happy"], "gender": "female"},
    "Burna Boy": {"id": "3wcj11K77LjEY1PkEazffa", "moods": ["happy", "motivational"], "gender": "male"},
    "Jhene Aiko": {"id": "5ZS223C6JyBfXasXxrRqOk", "moods": ["relaxing", "sad"], "gender": "female"},
    "Lewis Capaldi": {"id": "4GNC7GD6oZMSxPGyXy4MNB", "moods": ["sad"], "gender": "male"},
    "Jessie Reyez": {"id": "2wIVse2owClT7go1WT98tk", "moods": ["relaxing", "sad"], "gender": "female"},
    "Tyla": {"id": "3M83EDbtgZuKcUQ6cJFkY3", "moods": ["happy", "relaxing"], "gender": "female"},
    "Frank Ocean": {"id": "2h93pZq0e7k5yf4dywlkpM", "moods": ["sad", "relaxing"], "gender": "male"},
    "Pink Floyd": {"id": "0k17h0D3J5VfsdmQ1iZtE9", "moods": ["relaxing", "sad"], "gender": "male"},
    "Taylor Swift": {"id": "06HL4z0CvFAxyc27GXpf02", "moods": ["happy", "sad"], "gender": "female"},
    "Imagine Dragons": {"id": "53XhwfbYqKCa1cC15pYq2q", "moods": ["motivational", "happy"], "gender": "male"},
    "Maroon 5": {"id": "04gDigrS5kc9YWfZHwBETP", "moods": ["happy", "sad"], "gender": "male"},
    "Coldplay": {"id": "4gzpq5DPGxSnKTe4SA8HAU", "moods": ["relaxing", "sad"], "gender": "male"},
    "Demi Lovato": {"id": "6S2OmqARrzebs0tKUEyXyp", "moods": ["motivational", "sad"], "gender": "female"},
    "BLACKPINK": {"id": "41MozSoPIsD1dJM0CLPjZF", "moods": ["motivational", "happy"], "gender": "female"},
    "BTS": {"id": "3Nrfpe0tUJi4K4DXYWgMUX", "moods": ["happy", "motivational"], "gender": "male"},
    "Charlie Puth": {"id": "6vuM5FYQyQfZHZh8b6xK6Y", "moods": ["sad", "happy"], "gender": "male"},
    "Tate McRae": {"id": "2oZ8rYZWQwU1bzoRnlWdT1", "moods": ["sad", "relaxing"], "gender": "female"},
    "ZAYN": {"id": "5ZsFI1h6hIdQRw2ti0hz81", "moods": ["sad", "relaxing"], "gender": "male"},
    "Troye Sivan": {"id": "3WGpXCj9YhhfX11TToZcXP", "moods": ["relaxing", "happy"], "gender": "male"},
    "Ellie Goulding": {"id": "0X2BH1fck6amBIoJhDVmmJ", "moods": ["happy", "relaxing"], "gender": "female"},
    "Lana Del Rey": {"id": "00FQb4jTyendYWaN8pK0wa", "moods": ["sad", "relaxing"], "gender": "female"},
    "OneRepublic": {"id": "5Pwc4xIPtQLFEnJriah9YJ", "moods": ["motivational", "happy"], "gender": "male"},
    "Zara Larsson": {"id": "1Xylc3o4UrD53lo9CvFvVg", "moods": ["happy", "motivational"], "gender": "female"},
    "Ne-Yo": {"id": "21E3waRsmPlU7jZsS13rcj", "moods": ["relaxing", "sad"], "gender": "male"},
    "Avicii": {"id": "1vCWHaC5f2uS3yhpwWbIA6", "moods": ["motivational", "happy"], "gender": "male"},
    "David Guetta": {"id": "1Cs0zKBU1kc0i8ypK3B9ai", "moods": ["motivational", "happy"], "gender": "male"},
    "Major Lazer": {"id": "738wLrAtLtCtFOLvQBXOXp", "moods": ["happy", "motivational"], "gender": "male"}
}

# Preprocess artist data for faster lookups
def preprocess_artist_data():
    """Create indexes of artists by mood and gender for faster lookups"""
    mood_to_artists = {}
    gender_to_artists = {}
    
    for name, data in artist_bank.items():
        # Index by mood
        for mood in data["moods"]:
            if mood not in mood_to_artists:
                mood_to_artists[mood] = []
            mood_to_artists[mood].append((name, data))
        
        # Index by gender
        gender = data["gender"]
        if gender not in gender_to_artists:
            gender_to_artists[gender] = []
        gender_to_artists[gender].append((name, data))
    
    return {"mood_to_artists": mood_to_artists, "gender_to_artists": gender_to_artists}

# Initialize preprocessed artist data
artist_indices = preprocess_artist_data()

# Spotify API setup
try:
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=os.getenv('SPOTIPY_CLIENT_ID', 'dbe7fa4db4f84b0280c94e5bb5fc1c93'),
        client_secret=os.getenv('SPOTIPY_CLIENT_SECRET', '2bf80d85b4034f31ade1767bf6faac61'),
        redirect_uri=os.getenv('SPOTIPY_REDIRECT_URI', 'http://localhost:5001/callback'),
        scope="user-read-playback-state,user-modify-playback-state"
    ))
    app.logger.info("Spotify API configured successfully")
except Exception as e:
    sp = None
    app.logger.error(f"Error configuring Spotify: {e}")

# OpenAI API setup for chatbot
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    app.logger.error("OpenAI API Key is missing. Set OPENAI_API_KEY in the environment.")

# Before request handler - loads user data if logged in
@app.before_request
def load_user_if_logged_in():
    """Load user data before each request if the user is logged in"""
    g.user = None
    if 'user_id' in session:
        try:
            g.user = db.session.get(User, session['user_id'])
        except:
            pass  # User not found or DB error

# Utility Functions
@lru_cache(maxsize=32)
def fetch_courses_from_api(platform, subject):
    """Fetch course data from external APIs with caching for performance"""
    api_endpoints = {
        'coursera': f'https://api.coursera.org/api/courses.v1?q=search&query={subject}',
        'udemy': f'https://www.udemy.com/api-2.0/courses/?search={subject}',
        'edx': f'https://www.edx.org/api/v2/catalog/search?subject={subject}'
    }
    try:
        response = requests.get(api_endpoints.get(platform), timeout=10)
        response.raise_for_status()
        return response.json().get("courses", [])[:3]  # Return up to 3 courses
    except requests.exceptions.RequestException as e:
        app.logger.error(f"API Fetch Error: {e}")
        return []

@lru_cache(maxsize=32)
def get_recommendations(course, year_of_study):
    """Get course recommendations with caching based on course and year"""
    # Get course recommendations from learning paths
    path_courses = learning_paths.get(course, {}).get(year_of_study, [])
    recommendations = [{
        "text": topic,
        "link": "#",
        "description": f"Learn about {topic} in this structured path."
    } for topic in path_courses]
    
    # Get external courses from Coursera API
    external_courses = fetch_courses_from_api('coursera', course)
    recommendations.extend([
        {"text": c.get("name", "Course"), "link": c.get("url", "#"), "description": c.get("headline", "Course Description")} 
        for c in external_courses
    ])
    return recommendations

def detect_mood(user_message):
    """Detect mood from user message text using keyword matching"""
    mood_words = {
        "happy": ["happy", "excited", "joyful", "glad", "good"],
        "sad": ["sad", "down", "depressed", "unhappy"],
        "anxious": ["anxious", "nervous", "worried"],
        "stressed": ["stressed", "overwhelmed", "pressure"]
    }

    # Check if any mood keywords appear in the message
    for mood, words in mood_words.items():
        if any(re.search(rf"\b{word}\b", user_message, re.IGNORECASE) for word in words):
            return mood
    return "neutral"  # Default mood

def analyze_sentiment(user_message):
    """Analyze sentiment of text using TextBlob library"""
    sentiment_score = TextBlob(user_message).sentiment.polarity
    if sentiment_score > 0.2:
        return "positive"
    elif sentiment_score < -0.2:
        return "negative"
    else:
        return "neutral"

def process_chatbot_response(user_message):
    """Process chatbot responses with OpenAI API"""
    # Initialize chat history if it doesn't exist
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    # Add user message to history
    session['chat_history'].append({"role": "user", "content": user_message})
    session['chat_history'] = session['chat_history'][-5:]  # Keep only last 5 messages

    try:
        # Get response from OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a mental health chatbot named MindBalance. You help users manage stress, anxiety, and emotions with a kind and supportive tone."}
            ] + session['chat_history'],
            temperature=0.7
        )
        bot_reply = response["choices"][0]["message"]["content"].strip()
        
        # Save bot response to history
        session['chat_history'].append({"role": "assistant", "content": bot_reply})
        session.modified = True
        return bot_reply
    except Exception as e:
        app.logger.error(f"OpenAI API Error: {e}")
        return "I'm having trouble processing your request right now. Please try again later."

def save_track_history_async(user_id, tracks):
    """Save music track history asynchronously to avoid blocking the request"""
    def background_task():
        try:
            with app.app_context():
                for track in tracks:
                    if isinstance(track, dict):
                        history = UserMusicHistory(
                            user_id=user_id,
                            track_title=track.get("title", "Unknown"),
                            artist_name=track.get("artist", "Unknown")
                        )
                        db.session.add(history)
                db.session.commit()
        except Exception as e:
            app.logger.error(f"Error saving track history: {e}")
    
    # Start background thread for the database operation
    Thread(target=background_task).start()

def fetch_music(emotion, genre=None, gender="all"):
    """Fetch music recommendations based on emotion, genre, and gender preference"""
    try:
        # Map genre names to Spotify genre IDs
        genre_map = {
            "hiphop": "hip-hop",
            "rnb": "r-n-b",
            "relaxing": "chill",
            "sad": "sad",
            "happy": "happy",
            "motivational": "pop",
            "pop": "pop",
            "rock": "rock",
            "jazz": "jazz",
            "classical": "classical"
        }

        genre_final = genre_map.get(genre.lower() if genre else "pop", "pop")

        # Modify genre slightly based on emotion to avoid repetition
        if emotion == "sad" and genre_final == "pop":
            genre_final = "acoustic"
        elif emotion == "motivational" and genre_final == "pop":
            genre_final = "dance"
        elif emotion == "relaxing" and genre_final == "pop":
            genre_final = "chill"

        # Filter artist IDs by mood and gender using preprocessed indices
        matching_artists = []
        
        if emotion in artist_indices["mood_to_artists"]:
            for name, data in artist_indices["mood_to_artists"][emotion]:
                if gender == "all" or data["gender"] == gender:
                    matching_artists.append(data["id"])
        
        if not matching_artists:
            # Fallback to just mood match if gender filter returns nothing
            matching_artists = [data["id"] for name, data in artist_indices["mood_to_artists"].get(emotion, [])]
        
        if not matching_artists:
            matching_artists = ["6eUKZXaKkcviH0Ku9w2n3V"]  # Fallback to Ed Sheeran

        # Randomize and pick up to 3 artists
        selected = random.sample(matching_artists, min(3, len(matching_artists)))

        app.logger.info(f"Spotify API → Emotion: {emotion}, Genre: {genre_final}, Gender: {gender}, Artists: {selected}")

        # Use YouTube fallback if Spotify is not available
        if not sp:
            return get_youtube_fallback(emotion, genre, gender)

        # Get recommendations from Spotify API
        results = sp.recommendations(
            seed_artists=selected,
            seed_genres=[genre_final],
            limit=10
        )

        tracks = results.get("tracks", [])
        music_list = []

        # Process tracks from Spotify API
        for t in tracks:
            if t.get("preview_url"):  # Only include tracks with preview URLs
                music_list.append({
                    "title": t["name"],
                    "artist": t["artists"][0]["name"],
                    "preview_url": t["preview_url"],
                    "spotify_url": t["external_urls"]["spotify"],
                    "image_url": t["album"]["images"][0]["url"] if t["album"]["images"] else None
                })

        # Filter out previously seen tracks to avoid repetition
        if 'recent_tracks' not in session:
            session['recent_tracks'] = []

        seen_ids = set(session['recent_tracks'])
        unique_tracks = [t for t in music_list if t["spotify_url"] not in seen_ids]

        # Update session memory of recently seen tracks
        session['recent_tracks'] += [t["spotify_url"] for t in unique_tracks]
        session['recent_tracks'] = session['recent_tracks'][-30:]  # Keep only recent 30
        session.modified = True

        # Get artist names for UI display
        matching_names = [
            name for name, data in artist_bank.items()
            if emotion in data["moods"] and (gender == "all" or data["gender"] == gender)
        ]
        selected_artists = matching_names[:2] if matching_names else ["Ed Sheeran"]

        # Return the final recommendations
        return {
            "artists": selected_artists,
            "tracks": unique_tracks[:6]  # Limit to 6 tracks
        }

    except Exception as e:
        app.logger.error(f"Error in fetch_music(): {e}")
        return get_youtube_fallback(emotion, genre, gender)

def get_youtube_fallback(emotion, genre=None, gender="all"):
    """Fallback to YouTube videos when Spotify is unavailable"""
    # YouTube track data organized by emotion and genre
    youtube_fallback = {
        "sad": {
            "classical": [
                {"title": "Clair de Lune", "artist": "Debussy", "youtube": "https://www.youtube.com/embed/CvFH_6DNRCY", "gender": "male", "thumbnail": "https://img.youtube.com/vi/CvFH_6DNRCY/hqdefault.jpg"},
                {"title": "Moonlight Sonata", "artist": "Beethoven", "youtube": "https://www.youtube.com/embed/4Tr0otuiQuU", "gender": "male", "thumbnail": "https://img.youtube.com/vi/4Tr0otuiQuU/hqdefault.jpg"}
            ],
            "pop": [
                {"title": "Someone Like You", "artist": "Adele", "youtube": "https://www.youtube.com/embed/hLQl3WQQoQ0", "gender": "female", "thumbnail": "https://img.youtube.com/vi/hLQl3WQQoQ0/hqdefault.jpg"},
                {"title": "Hello", "artist": "Adele", "youtube": "https://www.youtube.com/embed/YQHsXMglC9A", "gender": "female", "thumbnail": "https://img.youtube.com/vi/YQHsXMglC9A/hqdefault.jpg"},
                {"title": "Say Something", "artist": "A Great Big World", "youtube": "https://www.youtube.com/embed/-2U0Ivkn2Ds", "gender": "male", "thumbnail": "https://img.youtube.com/vi/-2U0Ivkn2Ds/hqdefault.jpg"}
            ],
            "rock": [
                {"title": "Nothing Else Matters", "artist": "Metallica", "youtube": "https://www.youtube.com/embed/tAGnKpE4NCI", "gender": "male", "thumbnail": "https://img.youtube.com/vi/tAGnKpE4NCI/hqdefault.jpg"},
                {"title": "Zombie", "artist": "The Cranberries", "youtube": "https://www.youtube.com/embed/6Ejga4kJUts", "gender": "female", "thumbnail": "https://img.youtube.com/vi/6Ejga4kJUts/hqdefault.jpg"}
            ],
            "jazz": [
                {"title": "Blue In Green", "artist": "Miles Davis", "youtube": "https://www.youtube.com/embed/TLDflhhdPCg", "gender": "male", "thumbnail": "https://img.youtube.com/vi/TLDflhhdPCg/hqdefault.jpg"},
                {"title": "I Put A Spell On You", "artist": "Nina Simone", "youtube": "https://www.youtube.com/embed/ua2k52n_Bvw", "gender": "female", "thumbnail": "https://img.youtube.com/vi/ua2k52n_Bvw/hqdefault.jpg"}
            ],
            "hiphop": [
                {"title": "Changes", "artist": "2Pac", "youtube": "https://www.youtube.com/embed/eXvBjCO19QY", "gender": "male", "thumbnail": "https://img.youtube.com/vi/eXvBjCO19QY/hqdefault.jpg"},
                {"title": "Ex-Factor", "artist": "Lauryn Hill", "youtube": "https://www.youtube.com/embed/cE-bnWqLqxE", "gender": "female", "thumbnail": "https://img.youtube.com/vi/cE-bnWqLqxE/hqdefault.jpg"}
            ]
        },
        "happy": {
            "pop": [
                {"title": "Happy", "artist": "Pharrell Williams", "youtube": "https://www.youtube.com/embed/ZbZSe6N_BXs", "gender": "male", "thumbnail": "https://img.youtube.com/vi/ZbZSe6N_BXs/hqdefault.jpg"},
                {"title": "Uptown Funk", "artist": "Mark Ronson ft. Bruno Mars", "youtube": "https://www.youtube.com/embed/OPf0YbXqDm0", "gender": "male", "thumbnail": "https://img.youtube.com/vi/OPf0YbXqDm0/hqdefault.jpg"},
                {"title": "Roar", "artist": "Katy Perry", "youtube": "https://www.youtube.com/embed/CevxZvSJLk8", "gender": "female", "thumbnail": "https://img.youtube.com/vi/CevxZvSJLk8/hqdefault.jpg"},
                {"title": "Run the World (Girls)", "artist": "Beyoncé", "youtube": "https://www.youtube.com/embed/VBmMU_iwe6U", "gender": "female", "thumbnail": "https://img.youtube.com/vi/VBmMU_iwe6U/hqdefault.jpg"}
            ],
            # More genres...
        },
        "relaxing": {
            # Various genres for relaxing music
        },
        "motivational": {
            # Various genres for motivational music
        }
    }

    # Get genre-specific tracks, fallback to pop tracks if genre not found
    genre_lower = genre.lower() if genre else "pop"
    mood_tracks = youtube_fallback.get(emotion, youtube_fallback["happy"]) # Default to happy if emotion not found
    all_tracks = mood_tracks.get(genre_lower, mood_tracks.get("pop", [])) # Default to pop if genre not found
    
    # Filter by gender if specified
    if gender != "all":
        filtered_tracks = [track for track in all_tracks if track.get("gender") == gender]
        tracks = filtered_tracks if filtered_tracks else all_tracks # Use unfiltered if no matches
    else:
        tracks = all_tracks
    
    # Ensure we have tracks - provide a default if none found
    if not tracks:
        tracks = [{"title": "Happy", "artist": "Pharrell Williams", "youtube": "https://www.youtube.com/embed/ZbZSe6N_BXs", "gender": "male", "thumbnail": "https://img.youtube.com/vi/ZbZSe6N_BXs/hqdefault.jpg"}]
    
    # Get artist names for display
    matching_artists = [name for name, data in artist_bank.items() 
                      if emotion in data["moods"] and (gender == "all" or data["gender"] == gender)]
    
    if not matching_artists:
        matching_artists = [name for name, data in artist_bank.items() if emotion in data["moods"]]
    
    selected_artists = matching_artists[:2] if matching_artists else ["Ed Sheeran"]
    
    # Randomize tracks for variety
    random.shuffle(tracks)
    
    # Add image_url field to match Spotify format for consistency
    for track in tracks:
        if "thumbnail" in track and "image_url" not in track:
            track["image_url"] = track["thumbnail"]
    
    # Return formatted response
    return {
        "artists": selected_artists,
        "tracks": tracks[:6]  # Limit to 6 tracks
    }

# Routes - URL endpoints for the application

@app.route('/')
def welcome():
    """Welcome page route - app entry point"""
    return render_template('welcome.html')

@app.route('/about-us')
def about_us():
    """About us page route"""
    return render_template('about_us.html')

@app.route('/self-referral')
def self_referral():
    """Self-referral resources page - links to professional help"""
    referral_links = [
        {"name": "Mind UK", "url": "https://www.mind.org.uk/"},
        {"name": "Samaritans", "url": "https://www.samaritans.org/"},
        {"name": "Young Minds", "url": "https://www.youngminds.org.uk/"},
    ]
    return render_template('self_referral.html', referral_links=referral_links)

@app.route('/music', methods=['GET'])
def music_library():
    """Music library page - music therapy feature"""
    return render_template("music.html")

@app.route('/music_api', methods=['GET'])
@limiter.limit("30 per minute")  # Rate limiting for API
def music_api():
    """API endpoint for music recommendations"""
    # Get query parameters
    emotion = request.args.get('emotion', 'happy')  # Default: happy
    gender = request.args.get('gender', 'all')      # Default: all genders
    genre = request.args.get('genre', 'pop').lower()  # Default: pop
    
    app.logger.info(f"Music API request - Emotion: {emotion}, Gender: {gender}, Genre: {genre}")

    try:
        # Get music recommendations
        result = fetch_music(emotion, genre, gender)

        # Process result
        if isinstance(result, dict) and 'tracks' in result:
            tracks = result['tracks']
            artists = result.get('artists', [])
        else:
            tracks = result
            # Filter artists by gender
            matching_artists = [
                name for name, data in artist_bank.items() 
                if emotion in data["moods"] and (gender == "all" or data["gender"] == gender)
            ]
            artists = matching_artists[:2] if matching_artists else ["Ed Sheeran"]

        # Get user's liked tracks if logged in
        liked_titles = []
        if g.user:
            liked_history = UserMusicHistory.query.filter_by(
                user_id=g.user.id, 
                liked=True
            ).all()
            liked_titles = [history.track_title for history in liked_history]
            
            # Save track view history asynchronously
            save_track_history_async(g.user.id, tracks)
        
        app.logger.info(f"Music API response - Found {len(tracks)} tracks, {len(artists)} artists")
        
        return jsonify({
            "artists": artists,
            "tracks": tracks,
            "liked_titles": liked_titles
        })
    except Exception as e:
        app.logger.error(f"Error in music_api: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

@app.route('/like_track', methods=['POST'])
@limiter.limit("20 per minute")
def like_track():
    """Endpoint to like a music track - saves user preferences"""
    # Check if user is logged in
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Please log in to like tracks"}), 401

    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "message": "Invalid request data"}), 400
            
        track_title = data.get("title")
        artist_name = data.get("artist")

        if not track_title or not artist_name:
            return jsonify({"success": False, "message": "Missing track information"}), 400

        # Check if already liked
        existing = UserMusicHistory.query.filter_by(
            user_id=session['user_id'],
            track_title=track_title,
            artist_name=artist_name,
            liked=True
        ).first()
        
        if existing:
            return jsonify({"success": True, "message": "Track already liked!"})

        # Add new liked track
        liked_track = UserMusicHistory(
            user_id=session['user_id'],
            track_title=track_title,
            artist_name=artist_name,
            liked=True
        )
        db.session.add(liked_track)
        db.session.commit()

        return jsonify({"success": True, "message": "❤️ Track liked successfully!"})
        
    except Exception as e:
        app.logger.error(f"Error in like_track: {e}")
        db.session.rollback()
        return jsonify({"success": False, "message": "Server error occurred"}), 500

@app.route('/chatbot', methods=['GET', 'POST'])
@limiter.limit("30 per minute")
def chatbot():
    """Chatbot page and API endpoint - mental health conversation feature"""
    if request.method == 'POST':
        user_message = request.form.get('message', '').strip()

        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400

        # Detect mood and sentiment from user message
        mood = detect_mood(user_message)
        sentiment = analyze_sentiment(user_message)

        # Save mood log if logged in
        if g.user:
            new_log = MoodLog(
                user_id=g.user.id,
                mood=mood,
                sentiment=sentiment
            )
            db.session.add(new_log)
            db.session.commit()

        # Get chatbot response from OpenAI
        bot_response = process_chatbot_response(user_message)

        return jsonify({
            "response": bot_response,
            "mood": mood,
            "sentiment": sentiment
        })

    # GET request - render chatbot page
    return render_template('chatbot.html')

@app.route('/mood-graph')
def mood_graph():
    """Mood tracking visualization page"""
    if not g.user:
        flash("⚠️ Please log in to view your mood trends.", "warning")
        return redirect(url_for('login'))

    return render_template('mood_graph.html')

@app.route('/api/mood-logs')
def api_mood_logs():
    """API endpoint for mood logging data - provides data for mood graphs"""
    if 'user_id' not in session:
        return jsonify({"error": "Not logged in"}), 401

    try:
        # Pagination parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)  # Default 50 logs
        
        # Query user's mood logs ordered by timestamp
        logs_query = MoodLog.query.filter_by(user_id=session['user_id']).order_by(MoodLog.timestamp)
        paginated_logs = logs_query.paginate(page=page, per_page=per_page)
        
        # Format data for frontend chart visualization
        data = {
            "timestamps": [log.timestamp.strftime("%Y-%m-%d %H:%M") for log in paginated_logs.items],
            "moods": [log.mood for log in paginated_logs.items],
            "sentiments": [log.sentiment for log in paginated_logs.items],
            "pagination": {
                "total": paginated_logs.total,
                "pages": paginated_logs.pages,
                "current": page,
                "per_page": per_page,
                "has_next": paginated_logs.has_next,
                "has_prev": paginated_logs.has_prev
            }
        }
        return jsonify(data)
    except Exception as e:
        app.logger.error(f"Error fetching mood logs: {e}")
        return jsonify({"error": "Failed to fetch mood data"}), 500

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    """Main dashboard page - central hub for personalized recommendations"""
    app.logger.info(f"Session before dashboard: {session}")
    
    # Check if user is logged in
    if not g.user:
        flash("⚠️ You must be logged in to access the dashboard.", "warning")
        return redirect(url_for('login'))

    # Handle form submission (POST request)
    if request.method == 'POST':
        try:
            # Get form inputs
            course = request.form.get('course')
            year_of_study = request.form.get('year_of_study')
            cgpa_range = request.form.get('cgpa')

            # Get mental health status
            depression = request.form.get('depression') == 'yes'
            anxiety = request.form.get('anxiety') == 'yes'
            panic_attack = request.form.get('panic_attack') == 'yes'

            # Validate inputs
            if not course or not year_of_study or not cgpa_range:
                flash("⚠️ All fields are required.", "danger")
                return redirect(url_for('dashboard'))

            # Parse CGPA range
            try:
                cgpa_values = list(map(float, cgpa_range.split(' - ')))
                cgpa_midpoint = sum(cgpa_values) / 2
            except ValueError:
                flash("⚠️ Invalid CGPA input.", "danger")
                return redirect(url_for('dashboard'))

            # Encode categorical variables for the model
            encoded_course = course_encoding.get(course, -1)
            encoded_year = year_encoding.get(year_of_study, -1)

            if encoded_course == -1 or encoded_year == -1:
                flash("⚠️ Invalid course or year selection.", "danger")
                return redirect(url_for('dashboard'))

            # Prepare data for model prediction
            user_data = [
                encoded_course,
                encoded_year,
                cgpa_midpoint,
                depression,
                anxiety,
                panic_attack
            ]

            # Make prediction if model is available
            prediction = model.predict(np.array([user_data], dtype=float))[0][0] if model else 0.5
            has_mental_health_issues = bool(depression or anxiety or panic_attack)

            # Get recommendations
            # API-based academic recommendations
            academic_recommendations = get_api_recommendations(
                course=course, 
                year_of_study=year_of_study, 
                cgpa_range=cgpa_range
            )
            
            # Mental health recommendations
            mental_health_resources = []
            if depression:
                mental_health_resources.extend([
                    {"text": "Managing Depression - NHS", "link": "https://www.nhs.uk/conditions/stress-anxiety-depression/", "description": "The UK's NHS provides guidance on recognizing and managing depression.", "category": "mental"},
                    {"text": "Mind UK - Depression Resources", "link": "https://www.mind.org.uk/information-support/types-of-mental-health-problems/depression/", "description": "A mental health charity offering advice and information for those dealing with depression.", "category": "mental"}
                ])
            if anxiety:
                mental_health_resources.extend([
                    {"text": "Anxiety UK Support", "link": "https://www.anxietyuk.org.uk/", "description": "A UK-based charity providing resources, support groups, and practical advice for managing anxiety.", "category": "mental"},
                    {"text": "Calm Breathing Exercises", "link": "https://www.calm.com/breathe", "description": "Guided breathing exercises from Calm to help reduce stress and anxiety.", "category": "mental"}
                ])
            if panic_attack:
                mental_health_resources.extend([
                    {"text": "How to Stop a Panic Attack - Healthline", "link": "https://www.healthline.com/health/how-to-stop-a-panic-attack", "description": "Steps to manage and stop panic attacks effectively.", "category": "mental"},
                    {"text": "Panic Disorder - NHS", "link": "https://www.nhs.uk/conditions/panic-disorder/", "description": "NHS overview of symptoms, causes, and treatment options.", "category": "mental"}
                ])
                
            # Combine recommendations
            recommendations = academic_recommendations + mental_health_resources

            # Format mental conditions for display
            mental_conditions = []
            if depression: mental_conditions.append("Depression")
            if anxiety: mental_conditions.append("Anxiety")
            if panic_attack: mental_conditions.append("Panic Attacks")

            # Render recommendations page with personalized content
            return render_template(
                'recommendations.html',
                recommendations=recommendations,
                has_mental_health_issues=has_mental_health_issues,
                cgpa_range=cgpa_range,
                mental_conditions=", ".join(mental_conditions) if mental_conditions else None
            )

        except Exception as e:
            app.logger.error(f"Error processing dashboard form: {e}")
            flash(f"⚠️ Error processing form: {str(e)}", "danger")
            return redirect(url_for('dashboard'))

    # GET request - render dashboard form
    return render_template(
        'dashboard.html',
        user=g.user,
        courses=courses,
        years_of_study=years_of_study
    )
@app.route('/journal', methods=['GET', 'POST'])
def journal():
    """Journal page for emotion tracking and reflection"""
    # Check if user is logged in
    if not g.user:
        flash("Please log in.", "warning")
        return redirect(url_for('login'))

    # Handle form submission (POST request)
    if request.method == 'POST':
        entry_text = request.form.get('entry_text')
        mood = request.form.get('mood')
        sentiment = analyze_sentiment(entry_text)  # Analyze sentiment from text
        
        # Create new journal entry
        new_entry = JournalEntry(
            user_id=g.user.id, 
            entry_text=entry_text, 
            mood=mood,
            sentiment=sentiment
        )
        db.session.add(new_entry)
        db.session.commit()
        flash("Entry saved!", "success")
        return redirect(url_for('journal'))

    # Pagination setup for displaying entries
    page = request.args.get('page', 1, type=int)
    per_page = 10  # Entries per page
    
    # Filtering by mood (optional)
    filter_mood = request.args.get('filter_mood')
    if filter_mood:
        entries_query = JournalEntry.query.filter_by(user_id=g.user.id, mood=filter_mood)
    else:
        entries_query = JournalEntry.query.filter_by(user_id=g.user.id)
    
    # Get paginated entries
    entries_paginated = entries_query.order_by(JournalEntry.timestamp.desc()).paginate(page=page, per_page=per_page)
    
    # Count moods for the chart (using all entries for accurate statistics)
    mood_counts = db.session.query(
        JournalEntry.mood, 
        db.func.count(JournalEntry.mood)
    ).filter(
        JournalEntry.user_id == g.user.id,
        JournalEntry.mood.isnot(None)  # Skip null moods
    ).group_by(JournalEntry.mood).all()
    
    mood_data = dict(mood_counts)

    # Render journal page with entries and mood data
    return render_template(
        "journal.html", 
        entries=entries_paginated, 
        mood_data=mood_data,
        filter_mood=filter_mood
    )

@app.route('/journal/edit/<int:entry_id>', methods=['GET', 'POST'])
def edit_journal(entry_id):
    """Edit a journal entry"""
    # Check if user is logged in
    if not g.user:
        flash("Please log in.", "warning")
        return redirect(url_for('login'))
        
    # Get the journal entry
    entry = JournalEntry.query.get_or_404(entry_id)
    
    # Check if the entry belongs to the logged-in user
    if entry.user_id != g.user.id:
        flash("❌ Unauthorized access", "danger")
        return redirect(url_for('journal'))

    # Handle form submission (POST request)
    if request.method == 'POST':
        entry.entry_text = request.form['entry_text']
        entry.mood = detect_mood(entry.entry_text)  # Re-detect mood
        entry.sentiment = analyze_sentiment(entry.entry_text)  # Re-analyze sentiment
        db.session.commit()
        flash("✏️ Entry updated!", "success")
        return redirect(url_for('journal'))

    # GET request - render edit form
    return render_template('edit_journal.html', entry=entry)

@app.route('/journal/delete/<int:entry_id>')
def delete_journal(entry_id):
    """Delete a journal entry"""
    # Check if user is logged in
    if not g.user:
        flash("Please log in.", "warning")
        return redirect(url_for('login'))
        
    # Get the journal entry
    entry = JournalEntry.query.get_or_404(entry_id)
    
    # Check if the entry belongs to the logged-in user
    if entry.user_id != g.user.id:
        flash("❌ Unauthorized access", "danger")
        return redirect(url_for('journal'))

    # Delete the entry
    db.session.delete(entry)
    db.session.commit()
    flash("🗑️ Entry deleted.", "info")
    return redirect(url_for('journal'))

@app.route('/profile')
def profile():
    """User profile page - personal information and progress tracking"""
    if not g.user:
        flash("⚠️ Please log in to view your profile.", "warning")
        return redirect(url_for('login'))

    # Get user-related data with query optimization
    mood_logs = MoodLog.query.filter_by(user_id=g.user.id).order_by(MoodLog.timestamp).limit(100).all()
    progress = UserProgress.query.filter_by(user_id=g.user.id).all()
    music_history = UserMusicHistory.query.filter_by(
        user_id=g.user.id, 
        liked=True
    ).order_by(UserMusicHistory.timestamp.desc()).limit(10).all()

    # Prepare mood data for chart visualization
    moods = [log.mood for log in mood_logs]
    sentiments = [log.sentiment for log in mood_logs]
    timestamps = [log.timestamp.strftime("%Y-%m-%d %H:%M") for log in mood_logs]

    # Render profile page with user data
    return render_template(
        'profile.html',
        user=g.user,
        moods=moods,
        sentiments=sentiments,
        timestamps=timestamps,
        progress=progress,
        music_history=music_history
    )

@app.route('/toggle_progress', methods=['POST'])
def toggle_progress():
    """Toggle progress item completion status"""
    if not g.user:
        flash("⚠️ Please log in.", "warning")
        return redirect(url_for('login'))

    item_id = request.form.get('item_id')
    progress_item = UserProgress.query.get(item_id)

    # Check if item exists and belongs to user
    if progress_item and progress_item.user_id == g.user.id:
        # Toggle completion status
        progress_item.completed = not progress_item.completed
        db.session.commit()

    return redirect(url_for('profile'))

@app.route('/toggle_progress_ajax', methods=['POST'])
def toggle_progress_ajax():
    """AJAX endpoint to toggle progress item completion - for smooth UI updates"""
    if 'user_id' not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401

    try:
        data = request.get_json()
        item_id = data.get('item_id')
        
        # Get progress item
        progress_item = UserProgress.query.get(item_id)
        
        # Check if item exists and belongs to user
        if not progress_item or progress_item.user_id != session['user_id']:
            return jsonify({"success": False, "message": "Invalid item ID"}), 400
            
        # Toggle completion status
        progress_item.completed = not progress_item.completed
        db.session.commit()
        
        return jsonify({
            "success": True,
            "completed": progress_item.completed,
            "message": f"Marked as {'completed' if progress_item.completed else 'incomplete'}."
        })
    except Exception as e:
        app.logger.error(f"Error in toggle_progress_ajax: {e}")
        return jsonify({"success": False, "message": "Server error occurred"}), 500

@app.route('/update_profile', methods=['POST'])
def update_profile():
    """Update user profile information"""
    if not g.user:
        flash("⚠️ Please log in.", "warning")
        return redirect(url_for('login'))

    try:
        # Update user data
        g.user.age = request.form['age']
        g.user.gender = request.form['gender']
        g.user.marital_status = request.form['marital_status']
        db.session.commit()
        flash("✅ Profile updated.", "success")
    except Exception as e:
        app.logger.error(f"Error updating profile: {e}")
        db.session.rollback()
        flash("⚠️ Error updating profile.", "danger")
        
    return redirect(url_for('profile'))

# Authentication routes
@app.route('/invalid_credentials')
def invalid_credentials():
    """Invalid credentials page - shown after failed login"""
    return render_template('invalid_credentials.html')

@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute")  # Rate limiting to prevent brute force attacks
def login():
    """User login page"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Look up user in database
        user = User.query.filter_by(username=username).first()
        
        # Check if user exists and password is correct
        if user and check_password_hash(user.password, password):
            # Set session variables
            session['user_id'] = user.id
            session.permanent = True
            flash("✅ Login successful!", "success")
            
            # Redirect to intended page if available, otherwise to dashboard
            next_page = request.args.get('next')
            if next_page and next_page.startswith('/'):  # Prevent open redirect vulnerability
                return redirect(next_page)
            return redirect(url_for('dashboard'))

        # Log failed login attempt
        app.logger.warning(f"Failed login attempt for username: {username}")
        return redirect(url_for('invalid_credentials'))

    # GET request - render login form
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
@limiter.limit("5 per minute")  # Rate limiting for account creation
def register():
    """User registration page"""
    if request.method == 'POST':
        try:
            # Get form data
            username = request.form['username']
            password = request.form['password']
            age = request.form['age']
            gender = request.form['gender']
            marital_status = request.form['marital_status']

            # Basic validation
            if not username or not password:
                flash("⚠️ Username and password are required.", "danger")
                return redirect(url_for('register'))
                
            # Check if username already exists
            if User.query.filter_by(username=username).first():
                flash("⚠️ Username already exists.", "danger")
                return redirect(url_for('register'))

            # Create user with secure password
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            new_user = User(
                username=username, 
                password=hashed_password, 
                age=age, 
                gender=gender, 
                marital_status=marital_status
            )
            
            # Save to database
            db.session.add(new_user)
            db.session.commit()
            
            app.logger.info(f"New user registered: {username}")
            flash("✅ Registration successful! Please login.", "success")
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Error in registration: {e}")
            flash("⚠️ An error occurred during registration.", "danger")
            return redirect(url_for('register'))

    # GET request - render registration form
    return render_template('register.html')

@app.route('/logout')
def logout():
    """User logout - clear session data"""
    # Remove session data
    session.pop('user_id', None)
    session.pop('chat_history', None)
    session.pop('recent_tracks', None)
    flash("ℹ️ Logged out successfully.", "info")
    return redirect(url_for('welcome'))

@app.route('/admin')
def admin_dashboard():
    """Admin dashboard for monitoring the application"""
    # Check if user is admin
    if not g.user or g.user.username != "admin":
        flash("❌ Unauthorized access", "danger")
        return redirect(url_for('dashboard'))

    # Pull data for admin view
    users = User.query.all()
    journals = JournalEntry.query.order_by(JournalEntry.timestamp.desc()).limit(30).all()
    moods = MoodLog.query.order_by(MoodLog.timestamp.desc()).limit(30).all()

    return render_template('admin_dashboard.html', users=users, journals=journals, moods=moods)

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    """404 error handler - page not found"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """500 error handler - server error"""
    app.logger.error(f"Server error: {e}")
    return render_template('500.html'), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    """Rate limit exceeded handler"""
    return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429

# Run the app
if __name__ == '__main__':
    # Create custom error pages if they don't exist
    try:
        with open('templates/404.html', 'a+') as f:
            if f.tell() == 0:
                f.write("<!DOCTYPE html>\n<html>\n<head>\n<title>Page Not Found</title>\n</head>\n")
                f.write("<body>\n<h1>404 - Page Not Found</h1>\n<p>The page you are looking for does not exist.</p>\n")
                f.write("<a href='/'>Return to home</a>\n</body>\n</html>")
                
        with open('templates/500.html', 'a+') as f:
            if f.tell() == 0:
                f.write("<!DOCTYPE html>\n<html>\n<head>\n<title>Server Error</title>\n</head>\n")
                f.write("<body>\n<h1>500 - Server Error</h1>\n<p>Something went wrong on our end. Please try again later.</p>\n")
                f.write("<a href='/'>Return to home</a>\n</body>\n</html>")
    except:
        pass
        
    app.run(debug=False, port=5001)  # Run the Flask application
