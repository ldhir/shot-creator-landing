# AI Coach - Quick Start Guide

Get your AI coach up and running in 5 minutes!

## Step 1: Get an API Key

Choose one:

**OpenAI** (Recommended for beginners):
1. Go to https://platform.openai.com/api-keys
2. Create an account or sign in
3. Create a new API key
4. Copy the key

**Anthropic Claude** (Alternative):
1. Go to https://console.anthropic.com/
2. Create an account or sign in
3. Create an API key
4. Copy the key

## Step 2: Set Environment Variable

**On macOS/Linux:**
```bash
export OPENAI_API_KEY='sk-your-key-here'
```

**On Windows:**
```cmd
set OPENAI_API_KEY=sk-your-key-here
```

Or add to your `.env` file or shell profile (`~/.zshrc` or `~/.bashrc`).

## Step 3: Start the Server

```bash
cd tool
python app.py
```

## Step 4: Access the Coach

Open your browser and go to:
```
http://localhost:5001/tool/coach
```

## Step 5: Start Chatting!

1. If you're signed in with Firebase, you're ready to go!
2. If not, you'll be prompted for a user ID (you can use any string for testing)
3. Try asking:
   - "How can I improve my shooting form?"
   - "What should I eat today?"
   - "Create a workout plan for me"

## Testing Without Firebase

The coach will work even without Firebase, but it won't have access to your data. You can still:
- Have conversations
- Get general coaching advice
- Test the interface

To get full functionality, set up Firebase (see main README).

## Troubleshooting

**"AI Coach is not configured"**
- Make sure you set the environment variable
- Restart your Flask server after setting it

**"Error communicating with AI"**
- Check your API key is correct
- Verify you have credits/quota on your API account
- Check your internet connection

**Can't access the page**
- Make sure Flask server is running
- Check the URL is correct: `http://localhost:5001/tool/coach`
- Try `http://localhost:5001/tool/coach/` (with trailing slash)

## Next Steps

1. Set up Firebase to enable full data integration
2. Add nutrition/workout tracking to your app
3. Customize the coach's personality (edit `ai_coach.py`)
4. Add more data sources for richer insights

For detailed documentation, see `AI_COACH_README.md`.

