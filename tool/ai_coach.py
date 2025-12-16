"""
AI Coach Service
Provides conversational coaching based on user's progress pics, nutrition, workouts, and shot analysis
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import requests

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, that's okay

# You can use OpenAI, Anthropic Claude, or any other LLM API
# For this example, we'll use OpenAI's API
# Note: Also checks for OPEN_AI_API_KEY (with underscore) for backwards compatibility
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY') or os.environ.get('OPEN_AI_API_KEY', '')
OPENAI_API_URL = 'https://api.openai.com/v1/chat/completions'

# Alternative: Use Anthropic Claude
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages'

class AICoach:
    """
    AI Coach that provides personalized guidance based on:
    - Progress pictures
    - Nutrition data
    - Workout history
    - Shot analysis results
    - User goals and preferences
    """
    
    def __init__(self, user_id: str, db_service=None):
        """
        Initialize AI Coach for a specific user
        
        Args:
            user_id: Firebase user ID
            db_service: Service to interact with Firestore database
        """
        self.user_id = user_id
        self.db_service = db_service
        self.conversation_history = []
        self.user_context = {}
        
    def _get_system_prompt(self) -> str:
        """Generate the system prompt that defines the coach's personality and expertise"""
        return """You are an expert basketball and fitness coach with deep knowledge in:
- Basketball shooting mechanics and form analysis
- Strength and conditioning for athletes
- Sports nutrition and meal planning
- Progressive training program design
- Motivation and mental performance

Your coaching style:
- Be encouraging and supportive, like a real coach
- Ask thoughtful questions to understand the user's situation
- Provide specific, actionable advice
- Reference their actual data (progress pics, workouts, nutrition, shot analysis)
- Celebrate wins and help them learn from setbacks
- Keep them motivated with realistic, achievable goals

When analyzing data:
- Look for patterns and trends
- Identify areas for improvement
- Suggest specific exercises or nutrition adjustments
- Consider their goals and current fitness level
- Be honest but constructive about areas needing work

Formatting guidelines:
- Use bullet points (-) for lists of items, exercises, or recommendations
- Use numbered lists (1. 2. 3.) for step-by-step instructions or ordered plans
- Break up long paragraphs into shorter, digestible sections
- Use **bold** for important points or headings
- Structure workout plans with clear sections (e.g., "Monday:", "Tuesday:", etc.)
- Format meal plans with bullet points for each meal
- Use line breaks between different topics or sections

Always be conversational and natural. Ask follow-up questions to better understand their needs."""
    
    def _load_user_context(self) -> Dict[str, Any]:
        """Load all relevant user data to provide context to the AI"""
        if not self.db_service:
            return {}
        
        context = {
            'user_profile': {},
            'recent_shot_analyses': [],
            'nutrition_data': {
                'goals': {},
                'recent_entries': []
            },
            'workout_history': [],
            'progress_pics': [],
            'training_scores': []
        }
        
        try:
            # Load user profile
            user_ref = self.db_service.collection('users').document(self.user_id)
            user_doc = user_ref.get()
            if user_doc.exists:
                context['user_profile'] = user_doc.to_dict()
            
            # Load recent shot analyses (last 30 days)
            analyses_ref = self.db_service.collection('analyses')
            analyses_query = analyses_ref.where('userId', '==', self.user_id)\
                                        .order_by('createdAt', direction='DESCENDING')\
                                        .limit(10)
            context['recent_shot_analyses'] = [
                doc.to_dict() for doc in analyses_query.stream()
            ]
            
            # Load nutrition goals
            if 'nutritionGoals' in context['user_profile']:
                context['nutrition_data']['goals'] = context['user_profile']['nutritionGoals']
            
            # Load recent nutrition entries (last 7 days)
            nutrition_ref = self.db_service.collection('nutritionEntries')
            nutrition_query = nutrition_ref.where('userId', '==', self.user_id)\
                                          .order_by('createdAt', direction='DESCENDING')\
                                          .limit(20)
            context['nutrition_data']['recent_entries'] = [
                doc.to_dict() for doc in nutrition_query.stream()
            ]
            
            # Load workout history (last 30 days)
            workouts_ref = self.db_service.collection('workouts')
            workouts_query = workouts_ref.where('userId', '==', self.user_id)\
                                        .order_by('createdAt', direction='DESCENDING')\
                                        .limit(20)
            context['workout_history'] = [
                doc.to_dict() for doc in workouts_query.stream()
            ]
            
            # Load progress pictures (last 90 days)
            progress_ref = self.db_service.collection('progressPics')
            progress_query = progress_ref.where('userId', '==', self.user_id)\
                                        .order_by('createdAt', direction='DESCENDING')\
                                        .limit(10)
            context['progress_pics'] = [
                doc.to_dict() for doc in progress_query.stream()
            ]
            
            # Load training scores
            scores_ref = self.db_service.collection('trainingScores')
            scores_query = scores_ref.where('userId', '==', self.user_id)\
                                    .order_by('createdAt', direction='DESCENDING')\
                                    .limit(20)
            context['training_scores'] = [
                doc.to_dict() for doc in scores_query.stream()
            ]
            
        except Exception as e:
            print(f"Error loading user context: {e}")
        
        return context
    
    def _format_context_for_ai(self, context: Dict[str, Any]) -> str:
        """Format user context into a readable string for the AI"""
        formatted = "=== USER CONTEXT ===\n\n"
        
        # User profile
        if context.get('user_profile'):
            profile = context['user_profile']
            formatted += f"User: {profile.get('firstName', 'User')} {profile.get('lastName', '')}\n"
            if 'nutritionGoals' in profile:
                goals = profile['nutritionGoals']
                formatted += f"Nutrition Goals: {goals.get('calorieGoal', 'N/A')} cal/day, "
                formatted += f"{goals.get('proteinGoal', 'N/A')}g protein, "
                formatted += f"Target weight: {goals.get('idealWeight', 'N/A')} lbs\n"
        
        # Recent shot analyses
        if context.get('recent_shot_analyses'):
            formatted += f"\nRecent Shot Analyses ({len(context['recent_shot_analyses'])}):\n"
            for analysis in context['recent_shot_analyses'][:5]:
                score = analysis.get('overallScore', 'N/A')
                feedback = analysis.get('feedback', [])
                formatted += f"- Score: {score}%, Feedback: {', '.join(feedback[:2]) if feedback else 'N/A'}\n"
        
        # Nutrition data
        if context.get('nutrition_data', {}).get('recent_entries'):
            entries = context['nutrition_data']['recent_entries']
            formatted += f"\nRecent Nutrition Entries ({len(entries)}):\n"
            total_cals = sum(e.get('calories', 0) for e in entries[:7])
            total_protein = sum(e.get('protein', 0) for e in entries[:7])
            formatted += f"- Last 7 days avg: {total_cals/7:.0f} cal/day, {total_protein/7:.0f}g protein/day\n"
        
        # Workout history
        if context.get('workout_history'):
            workouts = context['workout_history']
            formatted += f"\nRecent Workouts ({len(workouts)}):\n"
            for workout in workouts[:5]:
                workout_type = workout.get('type', 'Unknown')
                duration = workout.get('duration', 'N/A')
                formatted += f"- {workout_type} ({duration} min)\n"
        
        # Training scores
        if context.get('training_scores'):
            scores = context['training_scores']
            avg_score = sum(s.get('similarityScore', 0) for s in scores) / len(scores) if scores else 0
            formatted += f"\nTraining Scores: Average similarity: {avg_score:.1f}% ({len(scores)} sessions)\n"
        
        formatted += "\n=== END CONTEXT ===\n"
        return formatted
    
    def _call_openai_api(self, messages: List[Dict[str, str]]) -> str:
        """Call OpenAI API for chat completion"""
        if not OPENAI_API_KEY:
            return "AI Coach is not configured. Please set OPENAI_API_KEY environment variable."
        
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': 'gpt-3.5-turbo',  # Cost-efficient: ~$0.002 per message vs GPT-4's ~$0.10
            'messages': messages,
            'temperature': 0.7,
            'max_tokens': 500  # Reduced to save costs - still plenty for good responses
        }
        
        try:
            response = requests.post(OPENAI_API_URL, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            return f"Error communicating with AI: {str(e)}"
    
    def _call_anthropic_api(self, messages: List[Dict[str, str]]) -> str:
        """Call Anthropic Claude API for chat completion"""
        if not ANTHROPIC_API_KEY:
            return "AI Coach is not configured. Please set ANTHROPIC_API_KEY environment variable."
        
        headers = {
            'x-api-key': ANTHROPIC_API_KEY,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json'
        }
        
        # Convert messages format for Anthropic
        system_message = next((m['content'] for m in messages if m['role'] == 'system'), '')
        user_messages = [m for m in messages if m['role'] != 'system']
        
        payload = {
            'model': 'claude-3-sonnet-20240229',  # or 'claude-3-haiku-20240307' for faster/cheaper
            'max_tokens': 1000,
            'system': system_message,
            'messages': user_messages
        }
        
        try:
            response = requests.post(ANTHROPIC_API_URL, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result['content'][0]['text']
        except Exception as e:
            return f"Error communicating with AI: {str(e)}"
    
    def chat(self, user_message: str, use_anthropic: bool = False) -> str:
        """
        Main chat method - handles conversation with the AI coach
        
        Args:
            user_message: User's message/question
            use_anthropic: If True, use Anthropic Claude; otherwise use OpenAI
        
        Returns:
            Coach's response
        """
        # Load fresh user context
        self.user_context = self._load_user_context()
        
        # Build conversation messages
        messages = []
        
        # System prompt with user context
        system_prompt = self._get_system_prompt()
        context_text = self._format_context_for_ai(self.user_context)
        full_system_prompt = f"{system_prompt}\n\n{context_text}"
        
        messages.append({
            'role': 'system',
            'content': full_system_prompt
        })
        
        # Add conversation history (last 10 messages to keep context manageable)
        for msg in self.conversation_history[-10:]:
            messages.append(msg)
        
        # Add current user message
        messages.append({
            'role': 'user',
            'content': user_message
        })
        
        # Call AI API
        if use_anthropic and ANTHROPIC_API_KEY:
            response = self._call_anthropic_api(messages)
        else:
            response = self._call_openai_api(messages)
        
        # Save to conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': user_message
        })
        self.conversation_history.append({
            'role': 'assistant',
            'content': response
        })
        
        return response
    
    def get_personalized_insights(self) -> Dict[str, Any]:
        """
        Generate personalized insights based on user's data
        Returns structured insights that can be displayed in the UI
        """
        self.user_context = self._load_user_context()
        
        insights = {
            'shot_analysis': None,
            'nutrition': None,
            'workouts': None,
            'progress': None,
            'recommendations': []
        }
        
        # Analyze shot performance
        if self.user_context.get('recent_shot_analyses'):
            analyses = self.user_context['recent_shot_analyses']
            if analyses:
                avg_score = sum(a.get('overallScore', 0) for a in analyses) / len(analyses)
                insights['shot_analysis'] = {
                    'average_score': avg_score,
                    'trend': 'improving' if len(analyses) > 1 and analyses[0].get('overallScore', 0) > analyses[-1].get('overallScore', 0) else 'stable',
                    'total_sessions': len(analyses)
                }
        
        # Analyze nutrition
        nutrition_data = self.user_context.get('nutrition_data', {})
        if nutrition_data.get('goals') and nutrition_data.get('recent_entries'):
            goals = nutrition_data['goals']
            entries = nutrition_data['recent_entries'][:7]  # Last 7 days
            if entries:
                avg_cals = sum(e.get('calories', 0) for e in entries) / len(entries)
                avg_protein = sum(e.get('protein', 0) for e in entries) / len(entries)
                insights['nutrition'] = {
                    'calorie_goal': goals.get('calorieGoal', 0),
                    'calorie_actual': avg_cals,
                    'protein_goal': goals.get('proteinGoal', 0),
                    'protein_actual': avg_protein,
                    'on_track': abs(avg_cals - goals.get('calorieGoal', 0)) < 200
                }
        
        # Analyze workouts
        if self.user_context.get('workout_history'):
            workouts = self.user_context['workout_history']
            if workouts:
                total_workouts = len(workouts)
                workout_types = {}
                for w in workouts:
                    w_type = w.get('type', 'Unknown')
                    workout_types[w_type] = workout_types.get(w_type, 0) + 1
                insights['workouts'] = {
                    'total_workouts': total_workouts,
                    'frequency': total_workouts / 30 if total_workouts > 0 else 0,  # workouts per day
                    'types': workout_types
                }
        
        # Generate recommendations
        recommendations = []
        if insights['shot_analysis'] and insights['shot_analysis']['average_score'] < 75:
            recommendations.append("Focus on improving your shooting form. Consider practicing with the benchmark comparison tool.")
        
        if insights['nutrition'] and not insights['nutrition']['on_track']:
            recommendations.append("Your nutrition is off track. Let's discuss your meal planning.")
        
        if insights['workouts'] and insights['workouts']['frequency'] < 0.3:
            recommendations.append("Try to increase your workout frequency to at least 3-4 times per week.")
        
        insights['recommendations'] = recommendations
        
        return insights
    
    def save_conversation(self, db_service):
        """Save conversation history to database"""
        if not db_service:
            return
        
        try:
            conversation_ref = db_service.collection('coachConversations').document(self.user_id)
            conversation_ref.set({
                'userId': self.user_id,
                'conversationHistory': self.conversation_history,
                'lastUpdated': datetime.now().isoformat()
            }, merge=True)
        except Exception as e:
            print(f"Error saving conversation: {e}")
    
    def load_conversation(self, db_service):
        """Load conversation history from database"""
        if not db_service:
            return
        
        try:
            conversation_ref = db_service.collection('coachConversations').document(self.user_id)
            conversation_doc = conversation_ref.get()
            if conversation_doc.exists:
                data = conversation_doc.to_dict()
                self.conversation_history = data.get('conversationHistory', [])
        except Exception as e:
            print(f"Error loading conversation: {e}")

