import os
import requests

# NOTE: API key hardcoded per user request (NOT safe for production).
GEMINI_API_KEY = "AIzaSyAqy2Vn8kzqcuXrHrRKe0r8ZllIUEcYi8I"


# ---------------------------------------------------------
#  AI SAFETY CHATBOX FALLBACK (Gemini)
# ---------------------------------------------------------
def ai_chatbox_safe_response(query: str) -> str:
    """
    Safety-focused AI fallback using Gemini.
    Generates ONLY safety guidance and precautions.
    """

    # Ask the model to produce a concise, numbered emergency response (max 5 short lines)
    safety_prompt = f"""
You are McQueen — a focused Safety Assistant. Respond VERY CONCISELY (max 5 short lines).
User message: "{query}"

Return exactly:
1) One-line short status (e.g. "FIRE — evacuate now")
2) 2-4 short numbered actions (each one line), immediate priorities first
3) One-line when to call emergency services

Do NOT add long paragraphs, emojis, or unrelated commentary. Use plain short sentences.
"""

    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        payload = {
            "contents": [{"parts": [{"text": safety_prompt}]}]
        }

        response = requests.post(
            url,
            json=payload,
            params={"key": GEMINI_API_KEY},
            timeout=10
        )

        data = response.json()
        # attempt to read the model output in common fields
        try:
            out = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            out = response.text
        # sanitize and trim to reasonable length
        out = out.strip()
        if len(out) > 1500:
            out = out[:1500].rsplit('\n',1)[0]
        return out

    except Exception:
        # Fallback concise reply if external API is unavailable
        return "UNAVAILABLE — Unable to reach AI service. 1) Stay calm and move to a safe area. 2) Call local emergency services if immediate danger. 3) Secure exits and help others to safety."


# ---------------------------------------------------------
#  MAIN MESSAGE HANDLER
# ---------------------------------------------------------
def handle_message(message: str) -> dict:
    """
    Safety AI with:
    - Emergency detection
    - Safety guidance
    - AI fallback
    """

    message = (message or "").strip()
    if not message:
        return {"error": "no message provided"}

    msg = message.lower()

    # ---------------------------------------------------------
    #  1. INTRODUCTION LOGIC
    # ---------------------------------------------------------
    if msg in ["hi", "hello", "hey", "who are you", "introduce yourself"]:
        return {"reply": "McQueen — Safety assistant. Describe the situation (fire, robbery, injury)."}

    # ---------------------------------------------------------
    #  2. FIRE RELATED SAFETY
    # ---------------------------------------------------------
    if any(word in msg for word in ["fire", "burning", "smoke detected", "flames"]):
        return {"reply": "FIRE — evacuate now. 1) Stay low and exit immediately. 2) Do not open hot doors. 3) Call emergency services from a safe location."}

    # ---------------------------------------------------------
    #  3. ROBBERY / CRIME SAFETY
    # ---------------------------------------------------------
    if any(word in msg for word in ["robbery", "thief", "theft", "stolen", "break in"]):
        return {"reply": "INTRUSION — get to safety. 1) Do not confront the intruder. 2) Move to a locked room and call police. 3) Note details from a safe distance."}

    # ---------------------------------------------------------
    #  4. ACCIDENT / INJURY SAFETY
    # ---------------------------------------------------------
    if any(word in msg for word in ["accident", "collision", "injured", "crash"]):
        return {"reply": "ACCIDENT — check safety. 1) Do not move seriously injured persons. 2) Call ambulance if needed. 3) Provide basic first aid if trained."}

    # ---------------------------------------------------------
    #  5. HEALTH EMERGENCY SAFETY
    # ---------------------------------------------------------
    if any(word in msg for word in ["faint", "unconscious", "not breathing", "chest pain"]):
        return {"reply": "MEDICAL — call emergency services now. 1) Check responsiveness and breathing. 2) Start CPR if unresponsive and not breathing. 3) Send someone to meet responders."}

    # ---------------------------------------------------------
    #  6. GENERAL SAFETY QUESTIONS
    # ---------------------------------------------------------
    if any(word in msg for word in ["safe", "precautions", "what to do", "guide"]):
        return {"reply": "I can help. Describe the situation (fire, injury, intrusion). I will give short, actionable steps."}

    # ---------------------------------------------------------
    #  7. AI SAFETY CHATBOX FALLBACK
    # ---------------------------------------------------------
    safe_reply = ai_chatbox_safe_response(message)
    # ensure concise single-string reply
    if isinstance(safe_reply, str):
        return {"reply": safe_reply}
    return {"reply": str(safe_reply)}


# ---------------------------------------------------------
#  FLASK API
# ---------------------------------------------------------
if __name__ == "__main__":
    from flask import Flask, request, jsonify
    from flask_cors import CORS

    app = Flask("mcqueen_service")
    CORS(app)

    @app.route('/', methods=['GET'])
    def root_redirect():
        # Redirect users who open the mcqueen service root to the main webapp
        from flask import redirect
        return redirect('http://127.0.0.1:8000/main.html')

    @app.route("/api/mcqueen", methods=["POST"])
    def api_mcqueen():
        data = request.get_json(silent=True) or {}
        message = data.get("message", "")
        if not message:
            return jsonify({"error": "no message provided"}), 400

        result = handle_message(message)
        return jsonify(result)

    app.run(host="0.0.0.0", port=8600, debug=False)
