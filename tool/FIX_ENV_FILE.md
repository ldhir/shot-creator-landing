# Fix Your .env File

## Issue Found

You wrote `OPEN_AI_API_KEY` in your `.env` file, but it should be `OPENAI_API_KEY` (no underscore between OPEN and AI).

## Quick Fix

**Option 1: Fix the variable name (Recommended)**

In your `.env` file, change:
```
OPEN_AI_API_KEY="your-key-here"
```

To:
```
OPENAI_API_KEY="your-key-here"
```

**Option 2: Keep it as is (Works too!)**

I've updated the code to also check for `OPEN_AI_API_KEY`, so your current setup will work. But it's better to use the standard name `OPENAI_API_KEY`.

## Your .env File Should Look Like:

```
OPENAI_API_KEY="sk-your-actual-api-key-here"
```

**Note:** 
- No spaces around the `=`
- Quotes around the key value are optional but recommended
- No `export` keyword needed in .env files

## Install python-dotenv (If Not Already Installed)

The code now automatically loads `.env` files, but you need the package:

```bash
pip install python-dotenv
```

Or if you're using the requirements.txt:

```bash
pip install -r requirements.txt
```

## Verify It Works

1. Make sure your `.env` file is in the `tool` directory
2. Restart your Flask server:
   ```bash
   python app.py
   ```
3. Test the coach at `http://localhost:5001/tool/coach`
4. If you see "AI Coach is not configured", check:
   - The variable name in .env
   - The .env file is in the `tool` directory
   - You restarted the server after creating/editing .env

## Example .env File Location

```
shot-creator-landing-copy/
  tool/
    .env          ← Your .env file should be here
    app.py
    ai_coach.py
    ...
```

