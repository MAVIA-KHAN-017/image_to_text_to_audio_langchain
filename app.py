from dotenv import find_dotenv,load_dotenv
from transformers import pipeline
from langchain import PromptTemplate,LLMChain,OpenAI
import requests
import os
import streamlit as st


load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN=os.getenv("HUGGINGFACEHUB_API_TOKEN")

def imageText2(url):

  image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
  text=image_to_text(url)[0]["generated_text"]
  
  print(text)
  return text

imageText2("grp.jpg")

def generate_story(scenario):
 template="""
 you are a story teller;
 you can generate  a story  based on a simple narrative,the story should be no more than  20 words;

 CONTEXT : {scenario}
 STORY:
 """
 prompt=PromptTemplate(template=template,input_variables=["scenario"])
 story_llm=LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo",temperature=1),prompt=prompt,verbose=True)
 story=story_llm.predict(scenario=scenario)
 print(story)
 return story



def text2image(message):
  API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
  headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
  payloads={
      "inputs":message
  }
  response = requests.post(API_URL, headers=headers, json=payloads)
  with open('audio.flac','wb') as file:
      file.write(response.content)

  
scenario=imageText2("grp.jpg")
story=generate_story(scenario)
text2image(story)

def main():
  st.set_page_config(page_title="img 2 audio story",page_icon="")
  st.header("Turn img into audio story")
  uploaded_file= st.file_uploader("Choose an img ...",type="jpg")
  if uploaded_file is not None:
    bytes_data=uploaded_file.getvalue()
    with open(uploaded_file.name,"wb") as file:
      file.write(bytes_data)
    st.image(uploaded_file,caption="upload image.",use_column_width=True)
    scenario=imageText2(uploaded_file.name)
    story=generate_story(scenario)
    text2image(story)

    with st.expander("Scenario"):
      st.write(scenario)
    with st.expander("Story"):
      st.write(story)

    st.audio("audio.flac")

if __name__=='__main__':
  main()

