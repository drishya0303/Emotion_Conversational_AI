
# --- Import Required Packages ---
from transformers import pipeline
import random
import streamlit as st
import pandas as pd
import altair as alt

# --- Load Pre-trained Emotion Detection Model ---
emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

# --- Enhanced Function to Detect Emotion ---
def detect_emotion(text):
    """
    Detect emotion in the text using a pre-trained model.
    Returns the detected emotion, confidence score, and all scores.
    """
    scores = emotion_model(text)
    dominant_emotion = max(scores[0], key=lambda x: x["score"])
    confidence_threshold = 0.5  # Minimum confidence threshold
    if dominant_emotion["score"] >= confidence_threshold:
        return dominant_emotion["label"], dominant_emotion["score"], scores[0]
    else:
        return "neutral", 0, scores[0]

# --- Function to Generate Tone-Adaptive Responses ---
def generate_response(emotion):
    """
    Generate a response based on the detected emotion.
    Includes dynamic phrasing for varied user interactions.
    """
    responses = {
        "joy": [
            "Your joy lights up the moment! 🌟 Keep spreading positivity!",
            "Feeling great, huh? The world shines brighter when you're happy!",
            "Your happiness is inspiring—share the good vibes!",
            "Feeling great? You're the energy boost we all need today! 💛",
            "Ah, I see you're feeling great! Let's keep the positivity alive.",
             
            
        ],
        "sadness": [
            "It's okay to feel down—better days are always ahead. 💙",
            "Even in sadness, there’s hope for brighter moments. Hang in there.",
            "I'm here to support you through the tough times. Things will improve. 💙",
            "It's okay to feel a little blue; moments like these make the brighter ones shine even more. 🌈",
            "Remember, tough times are temporary. Let hope guide you toward brighter moments"

        ],
        "anger": [
            "Feeling frustrated? Let's pause and take a deep breath together. 🧘",
            "Anger can drive change—channel it into something meaningful!",
            "Let your mind cool and reflect—growth often stems from challenges. 🔥",
            "I sense you're upset. Let's take a deep breath together. 🧘",
            "I can tell you're feeling upset. Let's pause for a moment and breathe deeply together. 🧘",
        ],
        "neutral": [
            "I sense you're feeling calm—let's keep the peace alive.",
            "Not sure how you're feeling, but I'm here for a meaningful conversation.",
            "You seem neutral—ready for some inspiring dialogue?",
            "It seems like a good day for some thoughtful exploration.",
            "Feeling neutral? Perfect space to spark some curiosity and ideas."
        ]
    }
    fallback_response = "I'm not sure how you're feeling, but I'm here to listen!"

    return random.choice(responses.get(emotion.lower(), [fallback_response]))

# --- Streamlit Frontend ---
def main():
    # Set up Streamlit UI configuration
    st.set_page_config(
        page_title="Wisdom AI",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # UI Title and Description
    st.title("HerInsight AI: Conversations Crafted with Wisdom")
    st.write("Enter your message below and let the AI respond with wisdom tailored to your emotions.")
    st.write("**Personalized interaction with engaging visuals and emotional insights!**")

    # Input box for user message
    user_message = st.text_input("Ask Me:", "")

    if user_message.strip():
        # Detect emotion
        emotion, confidence, scores = detect_emotion(user_message)
        response = generate_response(emotion)

        # Display results in Streamlit
        st.markdown(f"### Detected Emotion: **{emotion.capitalize()}**")
        st.markdown(f"Confidence Level: **{confidence:.2f}**")
        st.markdown(f"HerInsight AI: **{response}**")

        # Add Emotion Visualization (Bar Chart)
        df_scores = pd.DataFrame(scores)
        chart = alt.Chart(df_scores).mark_bar().encode(
            x=alt.X("label:N", title="Emotion"),
            y=alt.Y("score:Q", title="Confidence"),
            color="label:N"
        ).properties(
            title="Emotion Confidence Scores"
        )
        st.altair_chart(chart, use_container_width=True)

        # Change background colors based on emotion
        color_mapping = {
            "joy": "#99EDC3",      # Mint Green for joy
            "sadness": "#FFE135",  # Yellow for sadness 
            "anger": "#ed9999",    # Soft Red for anger 
            "neutral": "#99eded"   # Light Blue for neutral 
}
        emotion_color = color_mapping.get(emotion.lower(), "#ffffff")  # Default white
        st.markdown(f"<style>div.stApp {{ background-color: {emotion_color}; }}</style>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
