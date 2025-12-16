# How to Set Your OpenAI API Key

## Quick Setup (Temporary - Only for This Session)

### On macOS/Linux:
```bash
export OPENAI_API_KEY='sk-your-actual-key-here'
```

### On Windows (Command Prompt):
```cmd
set OPENAI_API_KEY=sk-your-actual-key-here
```

### On Windows (PowerShell):
```powershell
$env:OPENAI_API_KEY='sk-your-actual-key-here'
```

**Then start your server:**
```bash
python app.py
```

⚠️ **Note**: This only works for the current terminal session. If you close the terminal, you'll need to set it again.

---

## Permanent Setup (Recommended)

### On macOS/Linux:

1. **Open your shell profile file:**
   ```bash
   # For zsh (default on newer macOS):
   nano ~/.zshrc
   
   # OR for bash:
   nano ~/.bashrc
   ```

2. **Add this line at the end:**
   ```bash
   export OPENAI_API_KEY='sk-your-actual-key-here'
   ```

3. **Save and exit:**
   - Press `Ctrl + X`
   - Press `Y` to confirm
   - Press `Enter`

4. **Reload your shell:**
   ```bash
   source ~/.zshrc
   # OR
   source ~/.bashrc
   ```

### On Windows:

1. **Open System Properties:**
   - Press `Win + R`
   - Type `sysdm.cpl` and press Enter
   - Click "Environment Variables"

2. **Add new variable:**
   - Under "User variables", click "New"
   - Variable name: `OPENAI_API_KEY`
   - Variable value: `sk-your-actual-key-here`
   - Click OK

3. **Restart your terminal/command prompt**

---

## Using a .env File (Alternative - Most Flexible)

1. **Create a `.env` file in the `tool` directory:**
   ```bash
   cd tool
   nano .env
   ```

2. **Add your key:**
   ```
   OPENAI_API_KEY=sk-your-actual-key-here
   ```

3. **Install python-dotenv:**
   ```bash
   pip install python-dotenv
   ```

4. **Update `app.py` to load .env file** (add at the top):
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   ```

---

## Verify It's Working

1. **Check if the variable is set:**
   ```bash
   # macOS/Linux:
   echo $OPENAI_API_KEY
   
   # Windows:
   echo %OPENAI_API_KEY%
   ```

2. **Start your server and look for:**
   - No error messages about missing API key
   - The coach should work when you test it

3. **Test the coach:**
   - Go to `http://localhost:5001/tool/coach`
   - Try asking a question
   - If you see "AI Coach is not configured", the key isn't being read

---

## Security Notes

⚠️ **IMPORTANT:**
- Never commit your API key to Git
- Add `.env` to `.gitignore` if you use it
- Don't share your API key publicly
- If your key is exposed, regenerate it on OpenAI's website

---

## Troubleshooting

**"AI Coach is not configured" error:**
- Make sure you set the environment variable
- Restart your Flask server after setting it
- Check the variable name is exactly `OPENAI_API_KEY` (case-sensitive)
- Verify the key starts with `sk-`

**Key not persisting:**
- Make sure you added it to `.zshrc` or `.bashrc` (not just exported in terminal)
- Restart your terminal after adding to profile
- On Windows, restart your computer after adding to Environment Variables

