"""
Check Available Models
======================
Dynamically lists all models available on your Anthropic and Google Gemini accounts.

Usage:
    python scripts/check_models.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 50)
print("API KEY STATUS")
print("=" * 50)

anthropic_key = os.getenv("ANTHROPIC_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")

print(f"ANTHROPIC_API_KEY: {'✅ Set (' + anthropic_key[:8] + '...)' if anthropic_key else '❌ Not set'}")
print(f"GOOGLE_API_KEY:    {'✅ Set (' + google_key[:8] + '...)' if google_key else '❌ Not set'}")

# --- Anthropic ---
if anthropic_key:
    print("\n" + "=" * 50)
    print("ANTHROPIC — AVAILABLE MODELS")
    print("=" * 50)
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_key)
        models = client.models.list()
        for model in models.data:
            print(f"  • {model.id}  ({model.display_name})")
        if not models.data:
            print("  No models returned.")
    except ImportError:
        print("  ❌ 'anthropic' package not installed. Run: pip install anthropic")
    except Exception as e:
        print(f"  ❌ Error: {e}")

# --- Google Gemini ---
if google_key:
    print("\n" + "=" * 50)
    print("GOOGLE GEMINI — AVAILABLE MODELS")
    print("=" * 50)
    try:
        import google.generativeai as genai
        genai.configure(api_key=google_key)
        for model in genai.list_models():
            if "generateContent" in model.supported_generation_methods:
                print(f"  • {model.name}")
    except ImportError:
        print("  ❌ 'google-generativeai' package not installed. Run: pip install google-generativeai")
    except Exception as e:
        print(f"  ❌ Error: {e}")

if not anthropic_key and not google_key:
    print("\nNo API keys found. Add them to your .env file.")

print("\n" + "=" * 50)
print("DONE")
print("=" * 50)