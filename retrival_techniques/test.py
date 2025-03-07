# import os
# from huggingface_hub import login
# from transformers import pipeline

# # Set token directly (recommended for scripts in servers)
# HF_TOKEN = os.getenv("HF_TOKEN")
# login(HF_TOKEN)

# # Load model
# qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
# print("Model loaded successfully!")
######################################################################################
# import openai
# import os
# # openai.api_key = "your_api_key"
# openai.api_key= os.getenv("OPENAI_API_KEY")
# print(openai.version)
# response = openai.embeddings.create(
#     model="text-embedding-3-small",
#     input="Test embedding"
# )
# print(response)


# response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[{"role": "user", "content": "Hello, how are you?"}]
# )

# print(response["choices"][0]["message"]["content"])

import openai
import os
import requests
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
openai.api_key= os.getenv("OPENAI_API_KEY")

print(openai.api_key)  # Just to verify if it's loaded correctly


# from openai import openAIError
# openai.api_key = "your_api_key"

try:
    
    response = openai.models.list()  # Fetch available models
    print("✅ OpenAI API is active!")
    
    # billing_info = openai.billings.usage()
    # print(f"💰 Remaining quota: ${billing_info['total_available']}")

except openai.error.AuthenticationError:
    print("❌ Invalid API key. Check your OpenAI API key.")
except openai.error.RateLimitError:
    print("⚠️ Rate limit exceeded or insufficient quota. Check billing page.")
except openai.error.OpenAIError as e:
    print(f"🚨 OpenAI API error: {e}")




# api_key = "your_api_key"
headers = {"Authorization": f"Bearer {openai.api_key}"}

response = requests.get("https://api.openai.com/v1/dashboard/billing/credit_grants", headers=headers)

if response.status_code == 200:
    data = response.json()
    total_quota = data["total_granted"]
    used_quota = data["total_used"]
    remaining_quota = total_quota - used_quota

    print(f"💰 Total Quota: ${total_quota}")
    print(f"⚡ Used Quota: ${used_quota}")
    print(f"✅ Remaining Quota: ${remaining_quota}")

else:
    print("❌ Failed to retrieve quota. Check your API key and billing status.")
