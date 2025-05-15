from openai import OpenAI

client = OpenAI(
    api_key="sk-**", # 换成你在 AiHubMix 生成的密钥
    base_url="https://aihubmix.com/v1"
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-4o-mini",
)

print(chat_completion)