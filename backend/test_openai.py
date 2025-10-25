import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pathlib import Path
import aiohttp
import json

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

async def test_emergent_direct():
    """Test Emergent LLM API with direct HTTP calls"""
    EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')
    
    if not EMERGENT_LLM_KEY:
        print("\n❌ EMERGENT_LLM_KEY not found in environment")
        return False
    
    print(f"\n✓ Found EMERGENT_LLM_KEY: {EMERGENT_LLM_KEY[:20]}...")
    
    # Try different endpoints
    endpoints = [
        "https://demobackend.emergentagent.com/llm/v1/chat/completions",
        "https://demobackend.emergentagent.com/v1/chat/completions",
        "https://demobackend.emergentagent.com/llm/chat",
        "https://demobackend.emergentagent.com/chat/completions",
    ]
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello, I work!'"}
        ],
        "max_tokens": 50
    }
    
    headers = {
        "Authorization": f"Bearer {EMERGENT_LLM_KEY}",
        "Content-Type": "application/json"
    }
    
    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints:
            print(f"\nTrying endpoint: {endpoint}")
            try:
                async with session.post(endpoint, json=payload, headers=headers) as resp:
                    print(f"  Status: {resp.status}")
                    text = await resp.text()
                    print(f"  Response preview: {text[:200]}")
                    
                    if resp.status == 200:
                        data = json.loads(text)
                        print(f"  ✓ Success!")
                        if 'choices' in data and len(data['choices']) > 0:
                            print(f"  Message: {data['choices'][0].get('message', {}).get('content', 'N/A')}")
                        return True
            except Exception as e:
                print(f"  ❌ Error: {str(e)[:100]}")
    
    return False

async def test_openai():
    """Test OpenAI API connection and available models"""
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not found in environment")
        return False
    
    print(f"✓ Found OPENAI_API_KEY: {OPENAI_API_KEY[:20]}...")
    
    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        print("✓ OpenAI client initialized")
        
        # Try to make a simple API call with different models
        models_to_test = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo", "gpt-4"]
        
        for model in models_to_test:
            try:
                print(f"\nTesting model: {model}")
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Say 'Hello, I work!'"}
                    ],
                    max_tokens=50
                )
                print(f"✓ {model} works!")
                print(f"  Response: {response.choices[0].message.content}")
                return True  # If we get here, API key is valid
            except Exception as e:
                error_msg = str(e)
                if len(error_msg) > 150:
                    error_msg = error_msg[:150] + "..."
                print(f"❌ {model} failed: {error_msg}")
        
        return False
    except Exception as e:
        print(f"❌ Failed to initialize OpenAI client: {str(e)}")
        return False

async def test_emergent():
    """Test Emergent LLM API connection"""
    EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')
    
    if not EMERGENT_LLM_KEY:
        print("\n❌ EMERGENT_LLM_KEY not found in environment")
        return False
    
    print(f"\n✓ Found EMERGENT_LLM_KEY: {EMERGENT_LLM_KEY[:20]}...")
    
    # Try different base URLs and model names
    configs = [
        {
            "base_url": "https://demobackend.emergentagent.com/llm/v1",
            "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo"]
        },
        {
            "base_url": "https://demobackend.emergentagent.com/v1",
            "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
        },
        {
            "base_url": "https://api.openai.com/v1",
            "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
        }
    ]
    
    for config in configs:
        print(f"\nTrying base_url: {config['base_url']}")
        try:
            client = AsyncOpenAI(
                api_key=EMERGENT_LLM_KEY,
                base_url=config['base_url']
            )
            print("✓ Emergent LLM client initialized")
            
            for model in config['models']:
                try:
                    print(f"  Testing model: {model}")
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "Say 'Hello, I work!'"}
                        ],
                        max_tokens=50
                    )
                    print(f"  ✓ {model} works with {config['base_url']}!")
                    print(f"    Response: {response.choices[0].message.content}")
                    return True  # If we get here, API key is valid
                except Exception as e:
                    error_msg = str(e)
                    if len(error_msg) > 100:
                        error_msg = error_msg[:100] + "..."
                    print(f"  ❌ {model} failed: {error_msg}")
        except Exception as e:
            print(f"❌ Failed to initialize client: {str(e)}")
    
    return False

async def main():
    print("=" * 60)
    print("Testing AI API Connections")
    print("=" * 60)
    
    openai_works = await test_openai()
    emergent_works = await test_emergent()
    emergent_direct_works = await test_emergent_direct()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"OpenAI API: {'✓ Working' if openai_works else '❌ Not Working'}")
    print(f"Emergent API (OpenAI wrapper): {'✓ Working' if emergent_works else '❌ Not Working'}")
    print(f"Emergent API (Direct HTTP): {'✓ Working' if emergent_direct_works else '❌ Not Working'}")
    
    if not openai_works and not emergent_works and not emergent_direct_works:
        print("\n⚠️  No API is working. Please check your API keys.")
        print("\n💡 Recommendations:")
        print("   1. Add credits to your OpenAI account or get a valid API key")
        print("   2. Contact Emergent Agent support for correct API endpoint")
        print("   3. Use a different AI service (e.g., Anthropic Claude, Google Gemini)")
    elif openai_works:
        print("\n✓ Using OpenAI API for the application")
    elif emergent_works or emergent_direct_works:
        print("\n✓ Using Emergent API for the application")

if __name__ == "__main__":
    asyncio.run(main())
