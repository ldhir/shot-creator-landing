# AI Coach System

A comprehensive AI-powered coaching system that provides personalized guidance based on user's progress pics, nutrition, workouts, and shot analysis data.

## Features

- **Conversational AI Coach**: Natural language interactions with personalized coaching advice
- **Multi-Modal Data Integration**: Analyzes progress pics, nutrition logs, workout history, and shot analysis
- **Personalized Insights**: Real-time insights based on user's actual data
- **Motivational Support**: Encouraging, coach-like personality that keeps users motivated
- **Training Plan Optimization**: Helps users fix and improve their current training plans

## Architecture

### Components

1. **AI Coach Service** (`ai_coach.py`)
   - Core coaching logic
   - Integrates with OpenAI or Anthropic Claude APIs
   - Loads and analyzes user data from Firestore
   - Generates personalized insights

2. **API Endpoints** (`app.py`)
   - `/api/coach/chat` - Chat with the AI coach
   - `/api/coach/insights` - Get personalized insights
   - `/api/coach/conversation` - Get conversation history
   - `/api/coach/clear` - Clear conversation history

3. **Frontend Interface** (`static/coach.html`)
   - Beautiful chat interface
   - Real-time insights panel
   - Quick question buttons
   - Conversation history

## Setup

### 1. Install Dependencies

The AI coach requires the `requests` library (already in requirements.txt):

```bash
pip install requests
```

### 2. Configure AI API

Choose one of the following AI providers:

#### Option A: OpenAI (Recommended)

1. Get an API key from [OpenAI](https://platform.openai.com/api-keys)
2. Set environment variable:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

#### Option B: Anthropic Claude

1. Get an API key from [Anthropic](https://console.anthropic.com/)
2. Set environment variable:
   ```bash
   export ANTHROPIC_API_KEY='your-api-key-here'
   ```

### 3. Firebase Configuration

The AI coach integrates with your existing Firebase/Firestore setup. Make sure you have:

- Firebase Admin SDK configured
- Collections: `users`, `analyses`, `nutritionEntries`, `workouts`, `progressPics`, `trainingScores`

### 4. Database Schema

The coach expects the following Firestore collections:

#### `users` Collection
```javascript
{
  userId: string,
  firstName: string,
  lastName: string,
  email: string,
  nutritionGoals: {
    calorieGoal: number,
    idealWeight: number,
    proteinGoal: number
  },
  createdAt: timestamp
}
```

#### `analyses` Collection
```javascript
{
  userId: string,
  overallScore: number,
  feedback: string[],
  shot_angles: array,
  landmark_frames: array,
  createdAt: timestamp
}
```

#### `nutritionEntries` Collection
```javascript
{
  userId: string,
  calories: number,
  protein: number,
  carbs: number,
  fats: number,
  mealType: string, // 'breakfast', 'lunch', 'dinner', 'snack'
  createdAt: timestamp
}
```

#### `workouts` Collection
```javascript
{
  userId: string,
  type: string, // 'strength', 'cardio', 'basketball', 'shooting'
  duration: number, // minutes
  exercises: array,
  notes: string,
  createdAt: timestamp
}
```

#### `progressPics` Collection
```javascript
{
  userId: string,
  imageUrl: string,
  weight: number,
  notes: string,
  createdAt: timestamp
}
```

#### `trainingScores` Collection
```javascript
{
  userId: string,
  player: string, // 'curry', 'lebron', etc.
  similarityScore: number,
  createdAt: timestamp
}
```

#### `coachConversations` Collection
```javascript
{
  userId: string,
  conversationHistory: array,
  lastUpdated: timestamp
}
```

## Usage

### Access the Coach Interface

1. Start your Flask server:
   ```bash
   python app.py
   ```

2. Navigate to: `http://localhost:5001/tool/coach`

3. Sign in with your Firebase account (or provide user ID)

### Using the Coach

1. **Ask Questions**: Type any question in the chat input
   - "How can I improve my shooting form?"
   - "What should I eat today?"
   - "Create a workout plan for me"
   - "How am I progressing?"

2. **Quick Questions**: Click the quick question buttons for common queries

3. **View Insights**: Check the insights panel at the top for real-time data summaries

4. **Conversation History**: Your conversations are automatically saved and restored

## How It Works

### 1. Context Loading

When you send a message, the coach:
- Loads your user profile
- Fetches recent shot analyses (last 10)
- Gets nutrition data (goals + last 20 entries)
- Retrieves workout history (last 20 workouts)
- Loads progress pictures (last 10)
- Gets training scores (last 20 sessions)

### 2. AI Processing

The coach:
- Formats all this data into a context string
- Sends it to the AI API with your message
- The AI analyzes patterns and provides personalized advice
- Response is conversational and actionable

### 3. Insights Generation

The coach automatically:
- Calculates average shot scores
- Compares nutrition to goals
- Analyzes workout frequency
- Identifies trends
- Generates recommendations

## Customization

### Modify Coach Personality

Edit the `_get_system_prompt()` method in `ai_coach.py` to change:
- Coaching style (encouraging, technical, etc.)
- Areas of expertise
- Response tone

### Add More Data Sources

To include additional data:

1. Add data loading in `_load_user_context()` method
2. Format it in `_format_context_for_ai()` method
3. Update insights in `get_personalized_insights()` method

### Change AI Model

In `ai_coach.py`:
- OpenAI: Change `model` in `_call_openai_api()` (e.g., 'gpt-4', 'gpt-3.5-turbo')
- Anthropic: Change `model` in `_call_anthropic_api()` (e.g., 'claude-3-sonnet', 'claude-3-haiku')

## Example Conversations

### Improving Shooting Form
```
User: "My shot analysis shows 65% similarity. How can I improve?"

Coach: "I see your recent shot analysis shows 65% similarity to the benchmark. 
Let's break this down:

1. **Elbow Position**: Your elbow angle is slightly off. Try keeping your 
   shooting elbow aligned with the basket.

2. **Follow Through**: Your follow-through could be more consistent. 
   Practice holding your follow-through position for 2 seconds after each shot.

3. **Practice Routine**: I recommend 100 shots per day focusing on form, 
   not speed. Quality over quantity.

Would you like me to create a specific shooting practice plan for you?"
```

### Nutrition Planning
```
User: "I'm not hitting my protein goal. What should I do?"

Coach: "I see you're averaging 90g protein per day but your goal is 150g. 
Here's a plan:

**Breakfast**: Add 2 eggs (12g) + Greek yogurt (20g) = +32g
**Lunch**: Include chicken breast (30g) or salmon (25g) = +25-30g
**Dinner**: Lean protein source (30g) = +30g
**Snack**: Protein shake (25g) = +25g

This gets you to ~150g easily! Want me to create a meal plan with specific 
recipes?"
```

### Workout Planning
```
User: "Create a workout plan for me"

Coach: "Based on your data, I see you've been doing 2 workouts per week. 
Let's get you to 4-5 workouts for better results!

**Monday**: Upper Body Strength
- Bench press: 3x8
- Rows: 3x8
- Shoulder press: 3x8
- Bicep curls: 3x10

**Wednesday**: Lower Body + Core
- Squats: 3x8
- Deadlifts: 3x8
- Planks: 3x60s
- Leg curls: 3x10

**Friday**: Basketball-Specific
- Shooting practice: 30 min
- Agility drills: 20 min
- Cardio: 20 min

**Saturday**: Full Body Circuit
- Circuit training: 45 min

Want me to adjust this based on your equipment or preferences?"
```

## Troubleshooting

### "AI Coach is not configured"
- Make sure you've set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` environment variable
- Restart your Flask server after setting the variable

### "Error communicating with AI"
- Check your API key is valid
- Verify you have API credits/quota
- Check internet connection
- Review API rate limits

### No insights showing
- Ensure user has data in Firestore collections
- Check Firebase connection
- Verify user ID is correct

### Conversation not saving
- Check Firebase Admin SDK is configured
- Verify `coachConversations` collection exists
- Check user permissions

## Cost Considerations

### OpenAI Pricing (as of 2024)
- GPT-4: ~$0.03 per 1K input tokens, ~$0.06 per 1K output tokens
- GPT-3.5-turbo: ~$0.0015 per 1K input tokens, ~$0.002 per 1K output tokens

### Anthropic Pricing (as of 2024)
- Claude 3 Sonnet: ~$0.003 per 1K input tokens, ~$0.015 per 1K output tokens
- Claude 3 Haiku: ~$0.00025 per 1K input tokens, ~$0.00125 per 1K output tokens

**Recommendation**: Start with GPT-3.5-turbo or Claude 3 Haiku for cost efficiency, upgrade to GPT-4 or Claude 3 Sonnet for better quality.

## Future Enhancements

- [ ] Image analysis for progress pics
- [ ] Voice input/output
- [ ] Scheduled check-ins
- [ ] Goal setting and tracking
- [ ] Integration with fitness wearables
- [ ] Multi-language support
- [ ] Coach personality customization
- [ ] Team/group coaching features

## Support

For issues or questions:
1. Check this README
2. Review the code comments in `ai_coach.py`
3. Check API provider documentation (OpenAI/Anthropic)
4. Review Firebase/Firestore setup

