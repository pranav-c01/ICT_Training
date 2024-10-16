import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
import io


def main():
    st.title("Azure Text Analytics App")


    # Load environment variables
    load_dotenv()
    ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
    ai_key = os.getenv('AI_SERVICE_KEY')


    # Create client using endpoint and key
    credential = AzureKeyCredential(ai_key)
    ai_client = TextAnalyticsClient(endpoint=ai_endpoint, credential=credential)


    # Text input options
    input_option = st.radio("Choose input method:", ("Enter Text", "Upload File"))


    if input_option == "Enter Text":
        text = st.text_area("Enter the text you want to analyze:", height=200)
    else:
        uploaded_file = st.file_uploader("Choose a text file", type=["txt"])
        if uploaded_file is not None:
            text = io.TextIOWrapper(uploaded_file).read()
        else:
            text = ""


    if st.button("Analyze"):
        if text:
            try:
                # Get language
                detectedLanguage = ai_client.detect_language(documents=[text])[0]
                st.subheader("Language Detection")
                st.write(f"Detected Language: {detectedLanguage.primary_language.name}")


                # Get sentiment
                sentimentAnalysis = ai_client.analyze_sentiment(documents=[text])[0]
                st.subheader("Sentiment Analysis")
                st.write(f"Overall Sentiment: {sentimentAnalysis.sentiment}")
               
                # Display sentiment scores
                sentiment_scores = {
                    "Positive": sentimentAnalysis.confidence_scores.positive,
                    "Neutral": sentimentAnalysis.confidence_scores.neutral,
                    "Negative": sentimentAnalysis.confidence_scores.negative
                }
                sentiment_df = pd.DataFrame(sentiment_scores.items(), columns=["Sentiment", "Score"])

                st.bar_chart(sentiment_df.set_index("Sentiment"))


                # Get key phrases
                phrases = ai_client.extract_key_phrases(documents=[text])[0].key_phrases
                if len(phrases) > 0:
                    st.subheader("Key Phrases")
                    st.write(", ".join(phrases))
                else:
                    st.info("No key phrases found in the text.")


                # Get entities
                entities = ai_client.recognize_entities(documents=[text])[0].entities
                if len(entities) > 0:
                    st.subheader("Entities")
                    entity_data = [(entity.text, entity.category, entity.confidence_score) for entity in entities]
                    st.table(entity_data)
                else:
                    st.info("No entities found in the text.")


                # Get linked entities
                linked_entities = ai_client.recognize_linked_entities(documents=[text])[0].entities
                if len(linked_entities) > 0:
                    st.subheader("Linked Entities")
                    for linked_entity in linked_entities:
                        with st.expander(f"{linked_entity.name} ({linked_entity.confidence_score:.2f})"):
                            st.markdown(f"**Name:** {linked_entity.name}")
                            st.markdown(f"**Data Source:** {linked_entity.data_source}")
                            st.markdown(f"**URL:** [{linked_entity.url}]({linked_entity.url})")
                            st.markdown(f"**Data Source Entity ID:** {linked_entity.data_source_entity_id}")
                            st.markdown(f"**Language:** {linked_entity.language}")
                            st.markdown("**Matches:**")
                            for match in linked_entity.matches:
                                st.markdown(f"- Text: '{match.text}' (Confidence: {match.confidence_score:.2f})")
                                st.markdown(f"  Offset: {match.offset}, Length: {match.length}")
                else:
                    st.info("No linked entities found in the text.")


            except Exception as ex:
                st.error(f"An error occurred: {str(ex)}")
        else:
            st.warning("Please enter some text or upload a file to analyze.")


if __name__ == "__main__":
    main()
