import openai

try:
    with open("apikey.txt", "r") as f:
        api_key = f.read()
except:
    api_key = ''

def call_gpt35(prompt):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-35-turbo-0125", 
                messages=[{"role": "user", "content": prompt}], 
                api_key=api_key, 
                request_timeout=5)
            break
        except:
            print("Timeout, retrying...")
            time.sleep(5)

    output_text = response['choices'][0]['message']['content']
    return output_text