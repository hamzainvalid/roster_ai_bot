import os
from openai import OpenAI

# Initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def test_openai_call(prompt, system_message=None):
    """Simple test function to call OpenAI API"""

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-4"
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )

        print(f"✓ API call successful")
        print(f"Response: {response.choices[0].message.content}")
        print(f"Tokens used: {response.usage.total_tokens}")
        return response

    except Exception as e:
        print(f"✗ API call failed: {e}")
        return None


if __name__ == "__main__":
    # Test 1: Simple chat
    print("\n--- Test 1: Simple chat ---")
    test_openai_call("Say 'Hello, world!'")

    # Test 2: With system message
    print("\n--- Test 2: With system message ---")
    test_openai_call(
        "What's the capital of France?",
        system_message="You are a helpful geography expert. Keep answers brief."
    )

    # Test 3: Your specific use case
    print("\n--- Test 3: Follow-up generation test ---")
    test_openai_call(
        "### Task:\nSuggest 3-5 relevant follow-up questions or prompts that the user might naturally ask next\n### Context:\nUser asked about their April roster",
        system_message="You generate concise follow-up questions."
    )