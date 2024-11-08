import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load the model and vectorizer
loaded_model = joblib.load("sentiment_analysis_model.joblib")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.joblib")  # Ensure to load your TF-IDF vectorizer

# Streamlit app layout with larger title and font sizes
st.markdown("<h1 style='text-align: center; font-weight: bold; color: red; font-size: 40px;'>🌟 Dynamic Product Recommendations: Analyzing Flipkart Reviews with Sentiment Analysis🌟</h1>", unsafe_allow_html=True)

# File uploader for smiley background
smiley_image = st.file_uploader("Upload a smiley background image (PNG)", type=["png", "jpg", "jpeg"])

if smiley_image is not None:
    st.image(smiley_image, use_column_width=True)  # Display the uploaded smiley image

# Initialize session state for user input and review history
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'history' not in st.session_state:
    st.session_state.history = []

# Create two columns for layout
col1, col2 = st.columns([1, 3])  # Adjust column ratios as needed

# Column for input and analysis
with col1:
    # Text input for user review with increased font size
    user_input = st.text_input("Enter a product review:", value=st.session_state.user_input, 
                                label_visibility="visible", 
                                help="Type your review here...")

    # Button to analyze sentiment
    if st.button("🔍 Analyze Sentiment"):
        if user_input:
            try:
                # Transform user input using TF-IDF vectorizer
                review_tfidf = tfidf_vectorizer.transform([user_input])
                # Make prediction
                prediction = loaded_model.predict(review_tfidf)
                prediction_proba = loaded_model.predict_proba(review_tfidf)

                # Determine sentiment label
                sentiment = 'Positive' if prediction[0] == 1 else 'Negative' if prediction[0] == 0 else 'Neutral'

                # Display results with larger text
                st.success(f"Predicted Sentiment: **{sentiment}**", unsafe_allow_html=True)
                st.write(f"Sentiment Score: Positive: {prediction_proba[0][1]:.2f}, Negative: {prediction_proba[0][0]:.2f}")

                # Visual representation of sentiment scores
                sentiment_data = pd.Series({'Positive': prediction_proba[0][1], 'Negative': prediction_proba[0][0]})
                sentiment_data.plot(kind='bar', title='Sentiment Score Distribution', color=['green', 'red'])
                plt.ylabel('Score')
                plt.xticks(rotation=0)  # Rotate x-axis labels
                st.pyplot(plt)

                # Add to review history
                st.session_state.history.append(f"- \"{user_input}\": **{sentiment}**")

                # Save the current input for future resets
                st.session_state.user_input = user_input
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
        else:
            st.write("Please enter a review.")

    # Button to reset input
    if st.button("🔄 Reset Input"):
        st.session_state.user_input = ""  # Clear the input text
        st.session_state.history = []       # Clear the review history
        st.write("Input has been reset! Please enter a new review.")

# Column for review history
with col2:
    # Button to show review history
    if st.button("📜 Show Review History"):
        st.write("**Review History:**")
        if st.session_state.history:
            for review in st.session_state.history:
                st.write(review)
        else:
            st.write("No reviews analyzed yet.")
