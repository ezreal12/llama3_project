# streamlit run streamlit.py
import streamlit as st
import ChatModule as cm
# 채팅 메시지를 저장할 리스트 초기화
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

chat_module = cm.ChatModule()
st.title("Multi-agent function call")
st.subheader("박희수 작성")

# 사용자 입력 처리 함수
def handle_user_input():
    user_input = st.session_state.user_input
    if user_input:
        msg = chat_module.run_agent(user_input)
        st.session_state.messages.append(msg['generation'])
        # 입력창 초기화
        st.session_state.user_input = ""

# 사용자 입력 받기
st.text_input("메시지를 입력하세요:", key="user_input", on_change=handle_user_input)

# 저장된 메시지 출력
for message in st.session_state['messages']:
    st.write(message)
