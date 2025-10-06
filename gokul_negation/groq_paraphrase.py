import os
from groq import Groq
import random
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# set environ
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

def paraphrase_captions(captions):
    paraphrased_captions = []
    for caption in captions:
        temp = random.uniform(0.7, 1)

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "user", "content": f"Reword the following medical image caption in a different way, keeping the meaning intact. Maintain ANY and ALL negation used within the caption: {caption}"}
            ],
            temperature=temp
        )

        paraphrased_captions.append(response.choices[0].message.content)
    return paraphrased_captions

# test case
print(paraphrase_captions([
    "This image depicts a normal fundus."
]))

