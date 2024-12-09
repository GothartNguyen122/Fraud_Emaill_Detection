import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import joblib

df = pd.read_csv('../input/fraud-email-dataset/fraud_email_.csv')
# Data cleaning
df.dropna(inplace=True)
df.isnull().sum()
df['Text'] = df['Text'].apply(lambda s: BeautifulSoup(s).text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['Text'], df['Class'], test_size=0.2, random_state=42
)

# Fit model and save both model and vectorizer
def fit_and_save_model(model):
    clf_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', model)
    ])
    
    # Train the pipeline
    clf_pipeline.fit(X_train, y_train)

    # Evaluate the model
    score = clf_pipeline.score(X_test, y_test)
    print(f'Score: {score * 100:.2f}%')

    # Save the trained model
    model_path = 'fraud_email_model.pkl'
    vectorizer_path = 'tfidf_vectorizer.pkl'
    
    # Save both the model pipeline and vectorizer
    joblib.dump(clf_pipeline, model_path)
    joblib.dump(clf_pipeline.named_steps['tfidf'], vectorizer_path)

    print(f"Model saved at: {model_path}")
    print(f"Vectorizer saved at: {vectorizer_path}")
    
    return clf_pipeline

# Train the model
svc_model = fit_and_save_model(LinearSVC())
