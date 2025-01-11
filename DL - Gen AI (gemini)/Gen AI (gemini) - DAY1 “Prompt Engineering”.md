

----
### Run your first prompt

import google.generativeai as genai

model = genai.==GenerativeModel==('gemini-1.5-flash')
response = model.==generate_content==("Explain AI to me like I'm a kid.")

### Start a chat

model = genai.GenerativeModel('gemini-1.5-flash')
chat = model.==start_chat==(history=[])
response = chat.==send_message==('Hello! My name is Zlork.')
->
response = chat.send_message('Do you remember what my name is?'
 ->

### Choose a model
for model in genai.==list_models==():
   print(model.name)

for model in genai.list_models():
   if model.name == 'models/gemini-1.5-flash':
    print(model)
    break

----
## Explore generation parameters - Output length

model = genai.GenerativeModel(
    'gemini-1.5-flash',
    generation_config=genai.==GenerationConfig==(max_output_tokens=200))

response = model.generate_content('Write a 1000 word essay on the importance of olives in modern society.')
print(response.text)

----
## Explore generation parameters - Temperature

from google.api_core import retry

model = genai.GenerativeModel(
    'gemini-1.5-flash',
    generation_config=genai.==GenerationConfig==(temperature=2.0))


<span style="color:rgb(0, 200, 0)">When running lots of queries, it's a good practice to use a retry policy so your code
automatically retries when hitting Resource Exhausted (quota limit) errors.</span>
retry_policy = {
    "retry": retry.Retry(predicate=retry.if_transient_error, initial=10, multiplier=1.5, timeout=300)}

for _ in range(5):
  response = model.==generate_content==('Pick a random colour', ==request_options===retry_policy)
 

----
## Explore generation parameters - Top-K and top-P

model = genai.GenerativeModel(
    'gemini-1.5-flash-001',
    generation_config=genai.==GenerationConfig==(
        # These are the default values for gemini-1.5-flash-001.
        temperature=1.0,
        top_k=64,
        top_p=0.95,
    ))

story_prompt = "You are a creative writer. Write a short story about a cat"
response = model.==generate_content==(story_prompt, ==request_options===retry_policy)


----
## Prompting - Zero-shot

model = genai.GenerativeModel(
    'gemini-1.5-flash-001',
    generation_config=genai.GenerationConfig(
        temperature=0.1,
        top_p=1,
        max_output_tokens=5,
    ))

zero_shot_prompt = 
"""Classify movie reviews as POSITIVE, NEUTRAL or NEGATIVE.
Review: "Her" is a disturbing study revealing the direction
humanity is headed if AI is allowed to keep evolving,
unchecked. I wish there were more movies like this masterpiece.
Sentiment: """

response = model.==generate_content==(zero_shot_prompt, request_options=retry_policy)


----
## Prompting - Enum mode

import enum

class Sentiment(enum.Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


model = genai.GenerativeModel(
    'gemini-1.5-flash-001',
    generation_config=genai.GenerationConfig(
        response_mime_type="text/x.enum",
        ==response_schema=Sentiment==
    ))

response = model.generate_content(zero_shot_prompt, request_options=retry_policy)
print(response.text)

----
## Code prompting - Generating code

model = genai.GenerativeModel(
    'gemini-1.5-flash-latest',
    generation_config=genai.GenerationConfig(
        temperature=1,
        top_p=1,
        max_output_tokens=1024,
    ))

<span style="color:rgb(0, 200, 0)">Gemini 1.5 models are very chatty, so it helps to specify they stick to the code.</span>
code_prompt = """
Write a Python function to calculate the factorial of a number. No explanation, provide only the code.
"""

response = model.generate_content(code_prompt, request_options=retry_policy)
Markdown(response.text)

----


----