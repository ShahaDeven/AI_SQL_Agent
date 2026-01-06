import os
from dotenv import load_dotenv
import google.genai as genai

# Load your API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Error: GOOGLE_API_KEY not found in .env")
else:
    print(f"🔑 Authenticating with key: {api_key[:5]}...")
    genai.configure(api_key=api_key)

    print("\n📡 Fetching available models...")
    try:
        count = 0
        for m in genai.list_models():
            # We only care about models that can generate text (content)
            if 'generateContent' in m.supported_generation_methods:
                print(f"  - {m.name}")
                count += 1
        
        if count == 0:
            print("⚠️ No text generation models found. Check your API key permissions.")
            
    except Exception as e:
        print(f"❌ Error listing models: {e}")