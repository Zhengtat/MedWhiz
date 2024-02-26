#Model building
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from summarizer import Summarizer
from transformers import pipeline
import joblib
import re
#Abstractive summary of the keywords
import openai
#Converting PDF to text
import fitz
#Changing Language
from googletrans import Translator
import json
import requests

#Model Building
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('mtsamples.csv')

# Fill missing values with an empty string
df = df.fillna('')

# Preprocess the data
def preprocess_text(text):
    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Apply stemming
    stemmer = nltk.stem.PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

df['processed_text'] = df['transcription'].apply(preprocess_text)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'], df[['medical_specialty', 'sample_name', 'keywords']],
    test_size=0.2, random_state=42
)

# Encode labels using separate LabelEncoders for each output
le_specialty = LabelEncoder()
le_sample_name = LabelEncoder()
le_keywords = LabelEncoder()

y_train_encoded = {
    'medical_specialty': le_specialty.fit_transform(y_train['medical_specialty']),
    'sample_name': le_sample_name.fit_transform(y_train['sample_name'].astype(str)),
    'keywords': le_keywords.fit_transform(y_train['keywords'].astype(str))
}

# Load the pre-trained model
best_model = joblib.load('best_model.joblib')


def pdf_to_text(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf_document:
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            text += page.get_text()

    return text

#uncomment this 2 ltr
#pdf_file_path = 'path/to/your/pdf/file.pdf' #input goes here
#new_text = pdf_to_text(pdf_file_path)

new_text = "A 45-year-old female patient presents with acute-onset severe chest pain, dyspnea, and tachypnea. " \
            "She reports a two-day history of worsening cough, greenish sputum production, and subjective fever. "\
            "The patient has a past medical history significant for asthma diagnosed in childhood, with intermittent use of inhaled corticosteroids and bronchodilators. "\
            "Furthermore, she is currently prescribed multiple medications for the management of hypertension, including an angiotensin-converting enzyme (ACE) inhibitor and a calcium channel blocker."\
            "Upon physical examination, the patient appears distressed, with increased work of breathing and accessory muscle use. "\
            "Auscultation reveals decreased breath sounds in the left lower lung field, and there are crackles upon inspiration. "\
            "Arterial blood gas analysis indicates hypoxemia and respiratory alkalosis."\
            "Given the clinical suspicion of pneumonia, a chest X-ray is promptly performed, revealing a consolidation in the left lower lobe. "\
            "Laboratory investigations show an elevated white blood cell count, with a predominance of neutrophils. "\
            "The patient is started on empiric broad-spectrum antibiotics, including a third-generation cephalosporin and a macrolide."\
            "Further diagnostic tests, such as sputum culture and blood cultures, are ordered to identify the causative organism. "\
            "The patient is closely monitored for any signs of respiratory distress, and oxygen supplementation is initiated to maintain adequate oxygenation."\
            "In summary, this complex case involves a middle-aged female with a history of asthma and hypertension, "\
            "presenting with acute respiratory distress and clinical findings suggestive of pneumonia. "\
            "Timely intervention and appropriate management are crucial to ensure a favorable outcome for the patient."



# Extractive summarization of the new text
headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiZTZkYjdmMTctNjg2NS00M2I3LTgxNjAtMTVjZDVjOTY3ZDkxIiwidHlwZSI6ImFwaV90b2tlbiJ9.WA1l2WFha-l7_6-ZjFdr3hwUuFQebH12LWbFv8nBn88"}

url = "https://api.edenai.run/v2/text/summarize"
payload = {
    "providers": "microsoft,connexun",
    "language": "en",
    "text": new_text,
    "fallback_providers": ""
}

response = requests.post(url, json=payload, headers=headers)

result = json.loads(response.text)

processed_new_text = preprocess_text(new_text)
predicted_outputs_encoded = best_model.predict([processed_new_text])[0]

# Decode the predicted outputs
predicted_outputs = {
    'medical_specialty': le_specialty.inverse_transform([predicted_outputs_encoded])[0],
    'sample_name': le_sample_name.inverse_transform([predicted_outputs_encoded])[0],
    'keywords': le_keywords.inverse_transform([predicted_outputs_encoded])[0]
}


'''# Prompt ChatGPT for a brief summary
summary_prompt = f"Provide a brief summary of the sample name '{predicted_outputs['sample_name']}' and its severity."

# You can use your preferred method to interact with ChatGPT here
openai.api_key = 'insert ur api key here'
def interact_with_chatgpt(prompt):
    # Use the OpenAI GPT-3 API to get a response
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",  # You may choose a different engine
        prompt=prompt,
        max_tokens=150,  # Adjust max tokens as needed
        temperature=0.7,  # Adjust temperature for creativity
        stop=None  # You can specify stop conditions if needed
    )

    return response.choices[0].text.strip()

summary_response = interact_with_chatgpt(summary_prompt)

# Print the generated summary from ChatGPT'''

def translate_outputs(outputs, target_language='english'):
    translator = Translator()

    translated_outputs = {
        'medical_specialty': translator.translate(outputs['medical_specialty'], dest=target_language).text,
        'sample_name': translator.translate(outputs['sample_name'], dest=target_language).text,
        'keywords': translator.translate(outputs['keywords'], dest=target_language).text,
    }

    return translated_outputs

def get_language_code(language_name):
    language_mapping = {
        'english': 'en',
        'simplified chinese': 'zh-cn',
        'bahasa melayu': 'ms',
        'tamil': 'ta'
    }

    return language_mapping.get(language_name.lower(), 'en') 

input_language = 'english'  # Replace this with the input from the frontend
target_language_code = get_language_code(input_language)


translator = Translator()
translated_outputs = translate_outputs(predicted_outputs, target_language_code)
translated_executive_summary = translator.translate(result['microsoft']['result'],dest=target_language_code).text
#translated_elaboration = translator.translate(summary_response,dest=target_language_code).text
translated_medical_domain = translator.translate("Medical Domain",dest=target_language_code).text
translated_noc = translator.translate("Name of condition",dest=target_language_code).text
translated_short_summary = translator.translate("Short summary:",dest=target_language_code).text
translated_elab_on_ur_condition =  translator.translate("Elaboration on your condition:",dest=target_language_code).text
# Print the translated outputs
print(f"{translated_medical_domain}: {translated_outputs['medical_specialty']}")
print(f"{translated_noc}: {translated_outputs['sample_name']}")
print(translated_short_summary + ": ")
print(translated_executive_summary)
print(translated_elab_on_ur_condition + ":\n")
#print(translated_elaboration)