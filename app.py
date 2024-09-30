from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# 전역 변수 초기화
chat_history = []  # 채팅 기록 저장
bot_name = "ChatBot"  # 봇의 이름
welcome_message = "안녕하세요! 무엇을 도와드릴까요?"  # 초기 메시지

# 서버가 최초 실행될 때 기본 메시지 추가
chat_history.append({"sender": bot_name, "message": welcome_message})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message")  # 사용자가 보낸 메시지
    if user_message:
        # 사용자 메시지를 채팅 기록에 추가
        chat_history.append({"sender": "User", "message": user_message})

        # 간단한 응답 로직 (예시: '안녕'을 입력하면 '안녕하세요!'로 응답)
        if "안녕" in user_message:
            bot_reply = "안녕하세요!"
        else:
            bot_reply = f"{user_message}에 대해 잘 모르겠어요."

        # 봇 응답을 채팅 기록에 추가
        chat_history.append({"sender": bot_name, "message": bot_reply})

        # 봇 응답을 반환
        return jsonify({"sender": bot_name, "message": bot_reply})
    return jsonify({"error": "메시지를 입력해주세요."})

if __name__ == '__main__':
    app.run(debug=True)
