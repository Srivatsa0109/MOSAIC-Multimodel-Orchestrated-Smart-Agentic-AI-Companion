# pylint: disable = invalid-name
import os
import uuid
import logging
import json
from typing import Dict, Any

import streamlit as st
from langchain_core.messages import HumanMessage

from agents.agent import Agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def populate_envs(sender_email, receiver_email, subject):
    os.environ['FROM_EMAIL'] = sender_email
    os.environ['TO_EMAIL'] = receiver_email
    os.environ['EMAIL_SUBJECT'] = subject


def send_email(sender_email, receiver_email, subject, thread_id):
    try:
        populate_envs(sender_email, receiver_email, subject)
        config = {'configurable': {'thread_id': thread_id}}
        st.session_state.agent.graph.invoke(None, config=config)
        st.success('Email sent successfully!')
        # Clear session state
        for key in ['travel_info', 'thread_id']:
            st.session_state.pop(key, None)
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        st.error(f'Error sending email: {str(e)}')


def initialize_agent():
    if 'agent' not in st.session_state:
        st.session_state.agent = Agent()


def render_custom_css():
    st.markdown(
        '''
        <style>
        .main-title {
            font-size: 2.5em;
            color: #333;
            text-align: center;
            margin-bottom: 0.5em;
            font-weight: bold;
        }
        .sub-title {
            font-size: 1.2em;
            color: #333;
            text-align: left;
            margin-bottom: 0.5em;
        }
        .center-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 100%;
        }
        .query-box {
            width: 80%;
            max-width: 600px;
            margin-top: 0.5em;
            margin-bottom: 1em;
        }
        .query-container {
            width: 80%;
            max-width: 600px;
            margin: 0 auto;
        }
        .error-message {
            color: #dc3545;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 4px;
            padding: 10px;
            margin: 10px 0;
        }
        .success-message {
            color: #28a745;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 4px;
            padding: 10px;
            margin: 10px 0;
        }
        </style>
        ''', unsafe_allow_html=True)


def render_ui():
    st.markdown('<div class="center-container">', unsafe_allow_html=True)
    st.markdown('<div class="main-title">✈️🌍 AI Travel Agent 🏨🗺️</div>', unsafe_allow_html=True)
    st.markdown('<div class="query-container">', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter your travel query and get flight and hotel information:</div>', unsafe_allow_html=True)
    user_input = st.text_area(
        'Travel Query',
        height=200,
        key='query',
        placeholder='Type your travel query here...',
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.sidebar.image('images/ai-image.jpeg', caption='AI Travel Assistant')

    return user_input


def display_error(error_message: str):
    """Display error message in a styled container"""
    st.markdown(f'<div class="error-message">{error_message}</div>', unsafe_allow_html=True)


def display_success(message: str):
    """Display success message in a styled container"""
    st.markdown(f'<div class="success-message">{message}</div>', unsafe_allow_html=True)


def process_query(user_input: str):
    if not user_input:
        display_error('Please enter a travel query.')
        return

    try:
        thread_id = str(uuid.uuid4())
        st.session_state.thread_id = thread_id

        messages = [HumanMessage(content=user_input)]
        config = {'configurable': {'thread_id': thread_id}}

        result = st.session_state.agent.graph.invoke({'messages': messages}, config=config)
        
        # Check if the result contains an error
        if isinstance(result['messages'][-1].content, str):
            try:
                content = json.loads(result['messages'][-1].content)
                if isinstance(content, dict) and 'error' in content:
                    display_error(content['error'])
                    return
            except json.JSONDecodeError:
                pass

        st.subheader('Travel Information')
        st.write(result['messages'][-1].content)
        st.session_state.travel_info = result['messages'][-1].content
        display_success('Travel information retrieved successfully!')

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        display_error(f'Error: {str(e)}')


def render_email_form():
    send_email_option = st.radio('Do you want to send this information via email?', ('No', 'Yes'))
    if send_email_option == 'Yes':
        with st.form(key='email_form'):
            sender_email = st.text_input('Sender Email')
            receiver_email = st.text_input('Receiver Email')
            subject = st.text_input('Email Subject', 'Travel Information')
            submit_button = st.form_submit_button(label='Send Email')

        if submit_button:
            if not all([sender_email, receiver_email, subject]):
                display_error('Please fill out all email fields.')
            else:
                send_email(sender_email, receiver_email, subject, st.session_state.thread_id)


def main():
    initialize_agent()
    render_custom_css()
    user_input = render_ui()

    if st.button('Get Travel Information'):
        process_query(user_input)

    if 'travel_info' in st.session_state:
        render_email_form()


if __name__ == '__main__':
    main()
