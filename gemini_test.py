
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

try:
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents="Hello! Introduce yourself in 3 sentences."
    )

    print("\n===== GEMINI AI RESPONSE =====\n")
    print(response.text)

    client.close()

except Exception as e:
    print("Error:", e)



