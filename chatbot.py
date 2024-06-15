from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variables
api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)

if api_key is None:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

# Set the OpenAI API key

def generate_response(query):
    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query}
    ])
    return response.choices[0].message.content