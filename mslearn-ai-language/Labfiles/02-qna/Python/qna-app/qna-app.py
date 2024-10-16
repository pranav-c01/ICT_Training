from dotenv import load_dotenv
import os

# Import namespaces
import streamlit as st
from azure.ai.language.questionanswering import QuestionAnsweringClient
from azure.core.credentials import AzureKeyCredential

def main():
    try:
        # Get Configuration Settings
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')
        ai_project_name = os.getenv('QA_PROJECT_NAME')
        ai_deployment_name = os.getenv('QA_DEPLOYMENT_NAME')

        # Create client using endpoint and key
        client = QuestionAnsweringClient(endpoint=ai_endpoint, credential=AzureKeyCredential(ai_key))

        # Streamlit app layout
        st.title("Azure QnA App")
        st.write("Ask a question related to the provided knowledge base.")

        # Input text box for the question
        user_question = st.text_input("Enter your question:")

        if st.button("Submit"):
            if user_question:
                # Submit a question and display the answer
                try:
                    response = client.get_answers(
                        project_name=ai_project_name,
                        deployment_name=ai_deployment_name,
                        question=user_question
                    )

                    if response.answers:
                        # Display the first answer
                        answer = response.answers[0]
                        st.write("**Answer:**", answer.answer)
                        st.write("**Confidence Score:**", answer.confidence_score)
                    else:
                        st.write("No answer found.")

                except Exception as ex:
                    st.error(f"Error: {ex}")
            else:
                st.warning("Please enter a question.")

        # Submit a question and display the answer



    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    main()
