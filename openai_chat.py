import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
# === Chat Completion ===
chat_response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "who is the prime minister and president  in india ?"}
    ]
)

print("\n💬 Chat Response:")
print(chat_response.choices[0].message.content)