import torch
import os
import gradio as gr

from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

#######------------- IBM WatsonX Setup -------------####

my_credentials = {
    "url": "https://eu-de.ml.cloud.ibm.com",
    "apikey": "a4Kk-fPi_ay3_gM-AgcQUvIn5Whv5xqrjc8rXdEf-LSi"
}

project_id = "d357908c-dcb7-4b0a-b104-53c8159d71ce"

params = {
    GenParams.MAX_NEW_TOKENS: 800,
    GenParams.TEMPERATURE: 0.1,
}

LLAMA2_model = Model(
    model_id="ibm/granite-13b-instruct-v2",  # ✅ Fixed model_id string
    credentials=my_credentials,
    params=params,
    project_id=project_id
)

llm = WatsonxLLM(LLAMA2_model)

#######------------- Prompt Template -------------####

temp = """
<s><<SYS>>
List the key points with details from the context: 
[INST] The context : {context} [/INST] 
<</SYS>>
"""

pt = PromptTemplate(
    input_variables=["context"],
    template=temp
)

prompt_to_LLAMA2 = LLMChain(llm=llm, prompt=pt)

#######------------- Speech2Text Function -------------####

def transcript_audio(audio_file):
    try:
        print(f"📁 Audio file received: {audio_file}")
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny.en",
            chunk_length_s=30,
        )
        transcript_txt = pipe(audio_file, batch_size=8)["text"]
        print(f"📝 Transcript: {transcript_txt[:100]}...")

        result = prompt_to_LLAMA2.run(transcript_txt)
        print(f"✅ Result: {result[:100]}...")
        return result

    except Exception as e:
        print(f"❌ ERROR in transcript_audio: {e}")
        return f"An error occurred: {str(e)}"


#######------------- Gradio Interface -------------####

audio_input = gr.Audio(sources="upload", type="filepath")
output_text = gr.Textbox()

iface = gr.Interface(
    fn=transcript_audio,
    inputs=audio_input,
    outputs=output_text,
    title="Audio Transcription App",
    description="Upload the audio file to get summarized key points."
)

iface.launch(server_name="0.0.0.0", server_port=7860)
