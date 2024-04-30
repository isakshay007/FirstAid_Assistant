import streamlit as st
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent, Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from PIL import Image
from lyzr_automata.tasks.task_literals import InputType, OutputType
import os

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["apikey"]

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("./logo/lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("First Aid Assistant🏥")
st.markdown("Built using Lyzr SDK🚀")

input = st.text_input("Please briefly describe the nature of the injury, its location, and its severity so we can provide the appropriate assistance.",placeholder=f"""Type here""")

open_ai_text_completion_model = OpenAIModel(
    api_key=st.secrets["apikey"],
    parameters={
        "model": "gpt-4-turbo-preview",
        "temperature": 0.2,
        "max_tokens": 1500,
    },
)


def first_aid_generation(input):
    generator_agent = Agent(
        role=" FIRST AID CONSULTANT expert",
        prompt_persona=f"your task is to PROVIDE step-by-step INSTRUCTIONS in response to user-provided details such as the NATURE OF EMERGENCY, LOCATION OF INJURY, and SEVERITY. You MUST also offer FOLLOW-UP ADVICE."
    )

    prompt = f"""
You are an Expert FIRST AID CONSULTANT. Your task is to PROVIDE step-by-step INSTRUCTIONS in response to user-provided details such as the NATURE OF EMERGENCY, LOCATION OF INJURY, and SEVERITY. You MUST also offer FOLLOW-UP ADVICE. Here's how you should approach it:

1. ANALYZE the user provided DETAILS LIKE NATURE OF EMERGENCY, LOCATION OF INJURY, and SEVERITY.

2. If any INFORMATION is MISSING, use your EXPERTISE to ASSESS the situation based on what is known.

3. GIVE CLEAR and CONCISE first aid instructions tailored to the specifics of the emergency, location of injury, and its severity in a structured format.

4. EMPHASIZE SAFETY for both the injured person and those providing aid.

5. PROVIDE FOLLOW-UP ADVICE on what to do after initial first aid has been administered, such as seeking medical attention or monitoring symptoms.

6. ENSURE that your instructions are easy to understand and follow in a high-stress situation.

 """

    generator_agent_task = Task(
        name="First Aid Generation",
        model=open_ai_text_completion_model,
        agent=generator_agent,
        instructions=prompt,
        default_input=input,
        output_type=OutputType.TEXT,
        input_type=InputType.TEXT,
    ).execute()

    return generator_agent_task 
   
if st.button("Get Help!"):
    solution = first_aid_generation(input)
    st.markdown(solution)

with st.expander("ℹ️ - About this App"):
    st.markdown("""
    This app uses Lyzr Automata Agent Optimize your code. For any inquiries or issues, please contact Lyzr.

    """)
    st.link_button("Lyzr", url='https://www.lyzr.ai/', use_container_width=True)
    st.link_button("Book a Demo", url='https://www.lyzr.ai/book-demo/', use_container_width=True)
    st.link_button("Discord", url='https://discord.gg/nm7zSyEFA2', use_container_width=True)
    st.link_button("Slack",
                   url='https://join.slack.com/t/genaiforenterprise/shared_invite/zt-2a7fr38f7-_QDOY1W1WSlSiYNAEncLGw',
                   use_container_width=True)