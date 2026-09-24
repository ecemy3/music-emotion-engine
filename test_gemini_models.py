import google.genai as genai

# API Key'inizi buraya yazın
api_key = input("Gemini API Key girin: ")

client = genai.Client(api_key=api_key)

print("\n" + "="*60)
print("MEVCUT Gemini Modelleri:")
print("="*60)

for m in client.models.list():
    print(f"✓ {m.name}")
    if hasattr(m, 'display_name'):
        print(f"  Display Name: {m.display_name}")
    print()

print("="*60)
print("Not: Yukarıdaki model isimlerinden birini kullanın")
print("="*60)
