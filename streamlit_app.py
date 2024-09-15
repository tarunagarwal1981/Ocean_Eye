import streamlit as st
import openai
from utils.nlp_utils import process_user_input, extract_vessel_name
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
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in vessel performance."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.5,
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        st.error(f"Error in GPT response: {str(e)}")
        return "I'm sorry, I encountered an error while processing your request."

def handle_user_query(user_input: str) -> str:
    intent, vessel_present = process_user_input(user_input)

    if intent == "hull_performance":
        if vessel_present:
            vessel_name = extract_vessel_name(user_input)
            return process_hull_performance(vessel_name)
        else:
            # Provide general hull performance information and ask for vessel name
            generic_response = get_gpt_response("Provide general information about hull performance in maritime vessels.")
            return f"{generic_response}\n\nTo provide specific hull performance data, I need a vessel name. Could you please provide one?"
    elif intent in ["fuel_efficiency", "speed_performance", "general_info"]:
        return get_gpt_response(f"Provide information about {intent} in the context of maritime vessels.")
    else:
        return get_gpt_response(f"The user asked: '{user_input}'. Provide a helpful response related to vessel performance.")

def process_hull_performance(vessel_name: str) -> str:
    chart, power_loss, hull_condition = analyze_hull_performance(vessel_name)
    if chart and power_loss is not None and hull_condition:
        st.pyplot(chart)
        return f"Here's the hull performance analysis for {vessel_name}. The excess power is {power_loss:.2f}% and the hull condition is {hull_condition}."
    else:
        return f"Sorry, I couldn't find specific hull performance data for {vessel_name}."

def main():
    st.title("Vessel Performance Chatbot")

    # Initialize chat history and conversation state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'awaiting_vessel_name' not in st.session_state:
        st.session_state.awaiting_vessel_name = False

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

        # Process user input
        if st.session_state.awaiting_vessel_name:
            # If we're waiting for the vessel name, process it directly
            response = process_hull_performance(prompt)
            st.session_state.awaiting_vessel_name = False
        else:
            response = handle_user_query(prompt)
            if "Could you please provide one?" in response:
                # Set the state to awaiting vessel name
                st.session_state.awaiting_vessel_name = True

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


if __name__ == "__main__":
    main()
