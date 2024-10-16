from dotenv import load_dotenv
import os

# Import namespaces
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

def main():
    try:
        # Get Configuration Settings
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')

        # Create client using endpoint and key
        text_analytics_client = TextAnalyticsClient(endpoint=ai_endpoint, credential=AzureKeyCredential(ai_key))

        # Analyze each text file in the reviews folder
        reviews_folder = 'reviews'
        for file_name in os.listdir(reviews_folder):
            # Read the file contents
            print('\n-------------\n' + file_name)
            text = open(os.path.join(reviews_folder, file_name), encoding='utf8').read()
            print('\n' + text)

            # Get language
            response = text_analytics_client.detect_language([text])
            language = response[0].primary_language.name if response else "Unknown"
            print(f'Language: {language}')

            # Get sentiment
            sentiment_response = text_analytics_client.analyze_sentiment([text])
            sentiment = sentiment_response[0].sentiment if sentiment_response else "Unknown"
            print(f'Sentiment: {sentiment}')

            # Get key phrases
            key_phrases_response = text_analytics_client.extract_key_phrases([text])
            key_phrases = key_phrases_response[0].key_phrases if key_phrases_response else []
            print(f'Key Phrases: {key_phrases}')

            # Get entities
            entities_response = text_analytics_client.recognize_entities([text])
            entities = entities_response[0].entities if entities_response else []
            print(f'Entities: {[entity.text for entity in entities]}')

            # Get linked entities
            linked_entities_response = text_analytics_client.recognize_linked_entities([text])
            linked_entities = linked_entities_response[0].entities if linked_entities_response else []
            print(f'Linked Entities: {[entity.name for entity in linked_entities]}')


    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    main()