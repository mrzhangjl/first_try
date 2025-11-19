from ollama import generate
model = "deepseek-r1:8b"

response = generate(model, '你都有哪些能力？')
print("deepseek:", response["response"])