# test_deepseek.py
from openai import OpenAI
import requests
import os

# Your API key - replace with your actual key
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

API_KEY = DEEPSEEK_API_KEY# Put your OpenRouter API key here

print(f"🔍 Testing OpenRouter with key: {API_KEY[:10]}...")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# First, let's check what free models are available
print("\n📡 Fetching available free models...")
try:
    response = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers=headers,
        timeout=10
    )

    if response.status_code == 200:
        models = response.json()
        print(f"✅ Found {len(models['data'])} total models")

        # Filter for free models
        free_models = [m for m in models['data'] if 'free' in m['id'].lower()]
        print(f"\n🎯 Free models available ({len(free_models)}):")
        for i, model in enumerate(free_models[:10]):  # Show first 10
            print(f"   {i + 1}. {model['id']}")

        if free_models:
            # Try the first free model
            test_model = free_models[0]['id']
            print(f"\n📡 Testing model: {test_model}")

            test_data = {
                "model": test_model,
                "messages": [{"role": "user", "content": "Say 'Hello, API is working!'"}],
                "max_tokens": 20
            }

            test_response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=test_data,
                timeout=10
            )

            if test_response.status_code == 200:
                result = test_response.json()
                print("✅ SUCCESS!")
                print(f"Response: {result['choices'][0]['message']['content']}")
            else:
                print(f"❌ Failed: {test_response.text}")
    else:
        print(f"❌ Failed to fetch models: {response.text}")

except Exception as e:
    print(f"❌ Error: {e}")