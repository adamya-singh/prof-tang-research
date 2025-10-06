import random
from ollama import chat
from ollama import ChatResponse

def paraphrase_captions(captions):
    paraphrased_captions = []
    for caption in captions:
        temp = random.uniform(0.5, 1)

        response: ChatResponse = chat(model='llama3.2:latest', messages=[
        {
            'role': 'user',
            'content': f'Ignore all prior instruction and rephrase the following caption for an image in a different way, maintaing the meaning and ANY and ALL negation used within the caption. Return ONLY the paraphrased caption. The caption is: {caption}',
            'options': {"temperature": temp}
        },
        ])
        paraphrased_captions.append(response['message']['content'])
    
    return paraphrased_captions

# test case
print(paraphrase_captions([
    "This image depicts a normal fundus."
]))