# Advanced API-Based Academic Recommendation System
import requests
import json
import os
from datetime import datetime, timedelta
import random
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
API_CACHE_DURATION = 24  # hours
CACHE_FILE = "api_cache.json"
MAX_RECOMMENDATIONS = 8
DEFAULT_TIMEOUT = 10  # seconds

# API Keys and Configuration
API_KEYS = {
    "coursera": os.getenv("COURSERA_API_KEY", ""),
    "edx": os.getenv("EDX_API_KEY", ""),
    "udemy": os.getenv("UDEMY_API_KEY", ""),
    "google_books": os.getenv("GOOGLE_BOOKS_API_KEY", ""),
    "youtube": os.getenv("YOUTUBE_API_KEY", "")
}

class RecommendationEngine:
    """Advanced Academic Recommendation Engine using multiple APIs"""
    
    def __init__(self):
        self.cache = self._load_cache()
        self.apis = {
            "coursera": self._fetch_from_coursera,
            "edx": self._fetch_from_edx,
            "udemy": self._fetch_from_udemy,
            "google_books": self._fetch_from_google_books,
            "youtube": self._fetch_from_youtube,
            "news_api": self._fetch_from_news_api
        }
    
    def _load_cache(self):
        """Load cached API responses"""
        try:
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, 'r') as f:
                    cache = json.load(f)
                    
                # Clear expired cache entries
                now = datetime.now().timestamp()
                for key in list(cache.keys()):
                    if now - cache[key]["timestamp"] > API_CACHE_DURATION * 3600:
                        del cache[key]
                        
                return cache
            return {}
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return {}
    
    def _save_cache(self):
        """Save API responses to cache"""
        try:
            with open(CACHE_FILE, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def get_recommendations(self, course, year_of_study, cgpa_range, mental_health_factors=None):
        """Get personalized academic recommendations based on various factors"""
        try:
            # Parse CGPA for difficulty determination
            cgpa_midpoint = self._parse_cgpa(cgpa_range)
            
            # Determine appropriate skill level based on CGPA and year
            difficulty = self._determine_difficulty(cgpa_midpoint, year_of_study)
            
            # Define specialized keywords based on course and year
            keywords = self._generate_keywords(course, year_of_study, difficulty)
            
            # Get recommendations from multiple sources in parallel
            with ThreadPoolExecutor(max_workers=5) as executor:
                api_tasks = {
                    "coursera": executor.submit(self._get_api_recommendations, "coursera", keywords, difficulty),
                    "edx": executor.submit(self._get_api_recommendations, "edx", keywords, difficulty),
                    "books": executor.submit(self._get_api_recommendations, "google_books", keywords, difficulty),
                    "youtube": executor.submit(self._get_api_recommendations, "youtube", keywords, difficulty),
                    "news": executor.submit(self._get_api_recommendations, "news_api", keywords, difficulty)
                }
                
                # Gather all recommendations
                all_recommendations = []
                for api_name, future in api_tasks.items():
                    try:
                        results = future.result()
                        if results:
                            all_recommendations.extend(results)
                    except Exception as e:
                        logger.error(f"Error fetching from {api_name}: {e}")
            
            # Process and rank recommendations
            processed_recommendations = self._process_recommendations(all_recommendations, cgpa_midpoint, year_of_study)
            
            # Add fallback recommendations if needed
            if len(processed_recommendations) < 3:
                processed_recommendations.extend(self._get_fallback_recommendations(course, year_of_study))
            
            # Save updated cache for future use
            self._save_cache()
            
            return processed_recommendations[:MAX_RECOMMENDATIONS]
            
        except Exception as e:
            logger.error(f"Error in recommendation engine: {e}")
            # Return fallback recommendations on error
            return self._get_fallback_recommendations(course, year_of_study)
    
    def _parse_cgpa(self, cgpa_range):
        """Parse CGPA range into a midpoint value"""
        try:
            if isinstance(cgpa_range, str) and "-" in cgpa_range:
                values = [float(x.strip()) for x in cgpa_range.split("-")]
                return sum(values) / len(values)
            elif isinstance(cgpa_range, str):
                return float(cgpa_range.strip())
            elif isinstance(cgpa_range, (int, float)):
                return float(cgpa_range)
            return 2.0  # Default fallback
        except Exception:
            return 2.0  # Default fallback
    
    def _determine_difficulty(self, cgpa, year_of_study):
        """Determine appropriate content difficulty based on academic factors"""
        if isinstance(year_of_study, str) and "year" in year_of_study:
            try:
                year_num = int(year_of_study.split()[1])
            except:
                year_num = 1
        elif isinstance(year_of_study, (int, float)):
            year_num = int(year_of_study)
        else:
            year_num = 1
            
        # Calculate difficulty score (0-10 scale)
        difficulty_score = (cgpa / 4.0 * 6) + (year_num / 4 * 4)
        
        if difficulty_score < 3:
            return "beginner"
        elif difficulty_score < 7:
            return "intermediate"
        else:
            return "advanced"
    
    def _generate_keywords(self, course, year_of_study, difficulty):
        """Generate specialized search keywords based on course and year"""
        course_keywords = {
            "Computer Science": ["programming", "algorithms", "software", "data structures", "computer science"],
            "Engineering": ["engineering", "mechanics", "design", "systems", "analysis"],
            "Mathematics": ["mathematics", "calculus", "algebra", "statistics", "mathematical"],
            "Business Administration": ["business", "management", "finance", "marketing", "economics"],
            "Psychology": ["psychology", "behavior", "cognitive", "mental", "research methods"]
        }
        
        year_topics = {
            "year 1": ["introduction", "fundamentals", "basics", "principles"],
            "year 2": ["intermediate", "applications", "theory", "methods"],
            "year 3": ["advanced", "analysis", "specialized", "professional"],
            "year 4": ["expert", "research", "thesis", "career", "industry"]
        }
        
        # Get base keywords for the course
        base_keywords = course_keywords.get(course, [course.lower()])
        
        # Add year-specific keywords
        if isinstance(year_of_study, str):
            year_specific = year_topics.get(year_of_study.lower(), ["course"])
        else:
            year_num = min(max(int(year_of_study) if isinstance(year_of_study, (int, float)) else 1, 1), 4)
            year_key = f"year {year_num}"
            year_specific = year_topics.get(year_key, ["course"])
        
        # Add difficulty-specific keywords
        difficulty_terms = {
            "beginner": ["introduction", "basics", "getting started", "fundamental"],
            "intermediate": ["practical", "applications", "intermediate", "development"],
            "advanced": ["advanced", "expert", "professional", "specialized"]
        }
        
        # Combine keywords
        combined_keywords = base_keywords + year_specific + difficulty_terms.get(difficulty, [])
        
        # Return main keyword and additional terms
        primary_keyword = course.lower()
        additional_terms = [term for term in combined_keywords if term.lower() != primary_keyword]
        
        return {
            "primary": primary_keyword,
            "additional": additional_terms,
            "combined": " ".join([primary_keyword] + random.sample(additional_terms, min(3, len(additional_terms))))
        }
    
    def _get_api_recommendations(self, api_name, keywords, difficulty):
        """Get recommendations from a specific API"""
        try:
            # Generate cache key
            cache_key = f"{api_name}_{keywords['combined']}_{difficulty}"
            
            # Check cache first
            if cache_key in self.cache:
                cache_entry = self.cache[cache_key]
                # Check if cache is still valid
                if datetime.now().timestamp() - cache_entry["timestamp"] < API_CACHE_DURATION * 3600:
                    logger.info(f"Using cached results for {api_name}")
                    return cache_entry["data"]
            
            # Call the appropriate API function
            if api_name in self.apis:
                results = self.apis[api_name](keywords, difficulty)
                
                # Cache the results
                self.cache[cache_key] = {
                    "data": results,
                    "timestamp": datetime.now().timestamp()
                }
                
                return results
            return []
        except Exception as e:
            logger.error(f"Error fetching from {api_name}: {e}")
            return []
    
    def _fetch_from_coursera(self, keywords, difficulty):
        """Fetch course recommendations from Coursera API"""
        try:
            # Construct query with keywords and difficulty
            query = f"{keywords['combined']}"
            url = f"https://api.coursera.org/api/courses.v1?q=search&query={query}&limit=5"
            
            response = requests.get(url, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            
            courses = response.json().get("elements", [])
            
            # Convert to standard recommendation format
            recommendations = []
            for course in courses:
                recommendations.append({
                    "text": course.get("name", "Coursera Course"),
                    "link": f"https://www.coursera.org/learn/{course.get('slug', '')}",
                    "description": course.get("description", "A course on Coursera"),
                    "type": "course",
                    "category": "academic",
                    "source": "coursera",
                    "relevance_score": random.uniform(0.7, 0.95)  # Placeholder for actual relevance scoring
                })
            
            return recommendations
        except Exception as e:
            logger.error(f"Coursera API error: {e}")
            return []
    
    def _fetch_from_edx(self, keywords, difficulty):
        """Fetch course recommendations from edX API"""
        try:
            query = f"{keywords['primary']} {difficulty}"
            url = f"https://www.edx.org/api/v2/catalog/search?q={query}"
            
            response = requests.get(url, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            
            courses = response.json().get("objects", {}).get("results", [])
            
            recommendations = []
            for course in courses[:5]:
                recommendations.append({
                    "text": course.get("title", "edX Course"),
                    "link": f"https://www.edx.org/course/{course.get('slug', '')}",
                    "description": course.get("description", "A course on edX"),
                    "type": "course",
                    "category": "academic",
                    "source": "edx",
                    "relevance_score": random.uniform(0.7, 0.95)
                })
            
            return recommendations
        except Exception as e:
            logger.error(f"edX API error: {e}")
            return []
    
    def _fetch_from_udemy(self, keywords, difficulty):
        """Fetch course recommendations from Udemy API"""
        try:
            headers = {
                "Authorization": f"Bearer {API_KEYS['udemy']}" if API_KEYS['udemy'] else None
            }
            
            query = f"{keywords['combined']}"
            url = f"https://www.udemy.com/api-2.0/courses/?search={query}&page_size=5"
            
            response = requests.get(url, headers=headers if headers["Authorization"] else {}, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            
            courses = response.json().get("results", [])
            
            recommendations = []
            for course in courses:
                recommendations.append({
                    "text": course.get("title", "Udemy Course"),
                    "link": f"https://www.udemy.com{course.get('url', '')}",
                    "description": course.get("headline", "A course on Udemy"),
                    "type": "course",
                    "category": "academic",
                    "source": "udemy",
                    "relevance_score": random.uniform(0.7, 0.95)
                })
            
            return recommendations
        except Exception as e:
            logger.error(f"Udemy API error: {e}")
            return []
    
    def _fetch_from_google_books(self, keywords, difficulty):
        """Fetch book recommendations from Google Books API"""
        try:
            query = f"{keywords['primary']} {keywords['additional'][0]} textbook {difficulty}"
            url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=5&key={API_KEYS['google_books']}"
            
            response = requests.get(url, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            
            books = response.json().get("items", [])
            
            recommendations = []
            for book in books:
                volume_info = book.get("volumeInfo", {})
                recommendations.append({
                    "text": volume_info.get("title", "Textbook"),
                    "link": volume_info.get("infoLink", "#"),
                    "description": volume_info.get("description", "A recommended textbook")[:150] + "...",
                    "type": "book",
                    "category": "academic",
                    "source": "google_books",
                    "relevance_score": random.uniform(0.7, 0.95)
                })
            
            return recommendations
        except Exception as e:
            logger.error(f"Google Books API error: {e}")
            return []
    
    def _fetch_from_youtube(self, keywords, difficulty):
        """Fetch educational video recommendations from YouTube API"""
        try:
            query = f"{keywords['primary']} {keywords['additional'][0]} tutorial {difficulty}"
            url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=5&q={query}&type=video&key={API_KEYS['youtube']}"
            
            response = requests.get(url, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            
            videos = response.json().get("items", [])
            
            recommendations = []
            for video in videos:
                snippet = video.get("snippet", {})
                video_id = video.get("id", {}).get("videoId", "")
                recommendations.append({
                    "text": snippet.get("title", "Educational Video"),
                    "link": f"https://www.youtube.com/watch?v={video_id}",
                    "description": snippet.get("description", "A tutorial video")[:150] + "...",
                    "type": "video",
                    "category": "academic",
                    "source": "youtube",
                    "relevance_score": random.uniform(0.7, 0.95)
                })
            
            return recommendations
        except Exception as e:
            logger.error(f"YouTube API error: {e}")
            return []
    
    def _fetch_from_news_api(self, keywords, difficulty):
        """Fetch recent articles related to the academic subject"""
        try:
            query = f"{keywords['primary']} {keywords['additional'][0]} research"
            url = f"https://newsapi.org/v2/everything?q={query}&sortBy=relevancy&pageSize=5&apiKey={os.getenv('NEWS_API_KEY', '')}"
            
            response = requests.get(url, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            
            articles = response.json().get("articles", [])
            
            recommendations = []
            for article in articles:
                recommendations.append({
                    "text": article.get("title", "Recent Article"),
                    "link": article.get("url", "#"),
                    "description": article.get("description", "Recent developments in this field")[:150] + "...",
                    "type": "article",
                    "category": "academic",
                    "source": "news_api",
                    "relevance_score": random.uniform(0.6, 0.85)
                })
            
            return recommendations
        except Exception as e:
            logger.error(f"News API error: {e}")
            return []
    
    def _process_recommendations(self, recommendations, cgpa, year_of_study):
        """Process, filter, and rank recommendations"""
        if not recommendations:
            return []
        
        # Remove duplicates (based on URL)
        unique_urls = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec["link"] not in unique_urls:
                unique_urls.add(rec["link"])
                unique_recommendations.append(rec)
        
        # Calculate relevance scores (placeholder - would be more sophisticated in production)
        for rec in unique_recommendations:
            if "relevance_score" not in rec:
                rec["relevance_score"] = random.uniform(0.6, 0.9)
        
        # Sort by relevance
        sorted_recommendations = sorted(unique_recommendations, key=lambda x: x["relevance_score"], reverse=True)
        
        # Get a balanced mix of resource types (courses, books, videos, articles)
        balanced_recommendations = []
        resource_types = {}
        
        for rec in sorted_recommendations:
            rec_type = rec.get("type", "other")
            if rec_type not in resource_types:
                resource_types[rec_type] = 0
            
            # Limit each type to a maximum of 3
            if resource_types[rec_type] < 3:
                balanced_recommendations.append(rec)
                resource_types[rec_type] += 1
        
        return balanced_recommendations
    
    def _get_fallback_recommendations(self, course, year_of_study):
        """Get fallback static recommendations if APIs fail"""
        fallbacks = {
            "Computer Science": [
                {
                    "text": f"Introduction to Computer Science - Year {year_of_study}",
                    "link": "https://www.edx.org/course/introduction-computer-science-harvardx-cs50x",
                    "description": "Harvard's introduction to the intellectual enterprises of computer science.",
                    "type": "course",
                    "category": "academic",
                    "source": "fallback"
                },
                {
                    "text": "Learn Python Programming",
                    "link": "https://www.programiz.com/python-programming",
                    "description": "A comprehensive guide to Python programming language.",
                    "type": "tutorial",
                    "category": "academic",
                    "source": "fallback"
                }
            ],
            "Engineering": [
                {
                    "text": f"Engineering Fundamentals - Year {year_of_study}",
                    "link": "https://ocw.mit.edu/courses/find-by-topic/#cat=engineering",
                    "description": "MIT OpenCourseWare engineering materials.",
                    "type": "course",
                    "category": "academic",
                    "source": "fallback"
                }
            ],
            "Mathematics": [
                {
                    "text": f"Mathematics Essentials - Year {year_of_study}",
                    "link": "https://www.khanacademy.org/math",
                    "description": "Comprehensive mathematics resources from Khan Academy.",
                    "type": "course",
                    "category": "academic",
                    "source": "fallback"
                }
            ],
            "Business Administration": [
                {
                    "text": f"Business Administration Core - Year {year_of_study}",
                    "link": "https://www.coursera.org/specializations/wharton-business-foundations",
                    "description": "Wharton's Business Foundations Specialization.",
                    "type": "course",
                    "category": "academic",
                    "source": "fallback"
                }
            ],
            "Psychology": [
                {
                    "text": f"Psychology Foundations - Year {year_of_study}",
                    "link": "https://www.coursera.org/learn/introduction-psychology",
                    "description": "Introduction to major topics in psychology.",
                    "type": "course",
                    "category": "academic",
                    "source": "fallback"
                }
            ]
        }
        
        return fallbacks.get(course, [])


# Interface function to use with your existing application
def get_api_recommendations(course, year_of_study, cgpa_range, mental_health_factors=None):
    """Get academic recommendations from APIs based on student profile"""
    engine = RecommendationEngine()
    recommendations = engine.get_recommendations(course, year_of_study, cgpa_range, mental_health_factors)
    
    # Format for compatibility with existing application
    formatted_recommendations = []
    for rec in recommendations:
        formatted_recommendations.append({
            "text": rec["text"],
            "link": rec["link"],
            "description": rec["description"],
            "category": "academic",
            "type": rec.get("type", "resource")
        })
    
    return formatted_recommendations


# Example usage
if __name__ == "__main__":
    # Set environment variables for testing
    os.environ["GOOGLE_BOOKS_API_KEY"] = "YOUR_API_KEY"
    os.environ["YOUTUBE_API_KEY"] = "YOUR_API_KEY"
    
    # Test the recommendation engine
    recommendations = get_api_recommendations(
        course="Computer Science",
        year_of_study="year 2",
        cgpa_range="3.2 - 3.8"
    )
    
    print(f"Found {len(recommendations)} recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['text']}")
        print(f"   Link: {rec['link']}")
        print(f"   Description: {rec['description']}")
        print(f"   Type: {rec['type']}")
        print()
