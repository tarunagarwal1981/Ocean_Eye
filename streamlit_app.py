import streamlit as st
import openai
from utils.nlp_utils import process_user_input
from modules.hull_performance import analyze_hull_performance

def get_api_key():
    """Retrieve the API key from Streamlit secrets or environment variables."""
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Set OPENAI_API_KEY as an environment variable.")
    return api_key

openai.api_key = get_api_key()

def get_gpt_response(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.5,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        st.error(f"Error in GPT response: {str(e)}")
        return "I'm sorry, I encountered an error while processing your request."

def handle_user_query(user_input: str) -> str:
    intent, vessel_name = process_user_input(user_input)
    
    if intent == "hull_performance" and vessel_name:
        chart, power_loss, hull_condition = analyze_hull_performance(vessel_name)
        if chart:
            st.pyplot(chart)
            return f"Here's the hull performance analysis for {vessel_name}. The excess power is {power_loss:.2f}% and the hull condition is {hull_condition}."
        else:
            return f"Sorry, I couldn't find hull performance data for {vessel_name}."
    elif intent in ["fuel_efficiency", "speed_performance", "general_info"]:
        # For now, we'll use GPT to handle these intents
        gpt_prompt = f"User asked about {intent} for vessel {vessel_name}. Provide a brief, informative response about {intent} in the context of maritime vessels."
        return get_gpt_response(gpt_prompt)
    else:
        # For unknown intents, we'll use GPT to generate a response
        gpt_prompt = f"User: {user_input}\nAssistant: As an AI specialized in vessel performance, "
        return get_gpt_response(gpt_prompt)

def main():
    st.title("Vessel Performance Chatbot")

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to know about vessel performance?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get chatbot response
        response = handle_user_query(prompt)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()
