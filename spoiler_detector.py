import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import unidecode
import re
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from joblib import dump, load
import seaborn as sns

# Define a class to handle movie review and script data processing
class SpoilerDetector:
    def __init__(self, reviews_path, scripts_path):
        # Load review and script data from JSON files
        self.df_reviews = pd.read_json(reviews_path, lines=True)
        print(len(self.df_reviews["user_id"].values))
        self.df_scripts = pd.read_json(scripts_path, lines=True)
        self.stop_words = set(stopwords.words('english'))  # Set of English stop words
        self.port_stemmer = PorterStemmer()  # Porter Stemmer for stemming words

    def sample_data(self):
        # Sample 1000 unique movies for analysis
        sampled_movies = self.df_reviews['movie_id'].drop_duplicates().sample(n=1000, random_state=42)
        # Filter the reviews and scripts to include only the sampled movies
        self.df_reviews = self.df_reviews[self.df_reviews['movie_id'].isin(sampled_movies)]
        self.df_scripts = self.df_scripts[self.df_scripts['movie_id'].isin(sampled_movies)]
        sampled_reviews = []
        # Sample specific numbers of spoiler and non-spoiler reviews for each movie
        for movie_id in sampled_movies:
            movie_reviews = self.df_reviews[self.df_reviews['movie_id'] == movie_id]
            spoilers = movie_reviews[movie_reviews['is_spoiler'] == True]
            non_spoilers = movie_reviews[movie_reviews['is_spoiler'] == False]
            if len(spoilers) >= 28 and len(non_spoilers) >= 10:
                sampled_spoilers = spoilers.sample(n=28, random_state=42)
                sampled_non_spoilers = non_spoilers.sample(n=10, random_state=42)
                sampled_reviews.append(pd.concat([sampled_spoilers, sampled_non_spoilers]))
        self.df_reviews = pd.concat(sampled_reviews)  # Combine sampled reviews

    def preprocess_text(self, text):
        # Fix contractions (e.g., "don't" -> "do not")
        text = contractions.fix(text)
        # Remove URLs from the text
        text = re.sub(r"http\S+", "", text)
        # Remove accents and convert to ASCII
        text = unidecode.unidecode(text)
        # Remove non-alphabetic characters
        text = re.sub(r"[^a-zA-Z]", " ", text)
        # Tokenize, stem, and remove stop words
        tokens = [self.port_stemmer.stem(token) for token in word_tokenize(text.lower()) if token not in self.stop_words]
        # Create bigrams (pairs of consecutive words)
        bigram_tokens = list(nltk.bigrams(tokens))
        bigram_tokens = ['_'.join(bigram) for bigram in bigram_tokens]
        # Return a combined string of tokens and bigrams
        return ' '.join(tokens + bigram_tokens)

    def preprocess_reviews(self):
        # Apply text preprocessing to each review
        self.df_reviews['cleaned_review'] = self.df_reviews['review_text'].apply(self.preprocess_text)

    def preprocess_scripts(self):
        # Apply text preprocessing to each movie script
        self.df_scripts['cleaned_script'] = self.df_scripts['plot_summary'].apply(self.preprocess_text)

    def merge_data(self):
        # Merge review and script data on the 'movie_id' column
        self.df = pd.merge(self.df_reviews, self.df_scripts, on='movie_id')
        # Save the merged data to a CSV file
        self.df.to_csv('merged_reviews_scripts.csv', index=False)

    def create_custom_features(self, reviews, scripts):
        custom_features = []
        # Calculate overlap between words in the review and script for each pair
        for review, script in zip(reviews, scripts):
            review_words = set(review.split())
            script_words = set(script.split())
            overlap = len(review_words.intersection(script_words)) / len(review_words)
            custom_features.append([overlap])
        return np.array(custom_features)  # Return array of custom features

    def create_features(self):
        X_reviews = self.df['cleaned_review'].values  # Extract cleaned reviews
        X_scripts = self.df['cleaned_script'].values  # Extract cleaned scripts
        # Create TF-IDF features for reviews
        tfidf_vectorizer_reviews = TfidfVectorizer(max_features=7500)
        X_reviews_tfidf = tfidf_vectorizer_reviews.fit_transform(X_reviews).toarray()
        dump(tfidf_vectorizer_reviews, 'tfidf_vectorizer_reviews.joblib')  # Save the vectorizer
        # Create TF-IDF features for scripts
        tfidf_vectorizer_scripts = TfidfVectorizer(max_features=3500)
        X_scripts_tfidf = tfidf_vectorizer_scripts.fit_transform(X_scripts).toarray()
        dump(tfidf_vectorizer_scripts, 'tfidf_vectorizer_scripts.joblib')  # Save the vectorizer
        # Create custom features
        custom_features = self.create_custom_features(X_reviews, X_scripts)
        # Combine all features
        X = np.hstack((X_reviews_tfidf, X_scripts_tfidf, custom_features))
        y = self.df['is_spoiler'].values  # Target labels
        # Select the top 5000 features
        selector = SelectKBest(f_classif, k=5000)
        X_selected = selector.fit_transform(X, y)
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
        # Apply SMOTE to handle class imbalance
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        return X_train_smote, X_test, y_train_smote, y_test

    def train_model(self, X_train, y_train):
        # Define hyperparameter search space for Logistic Regression
        param_dist = {'C': np.logspace(-3, 3, 7), 'penalty': ['l1', 'l2']}
        lr_model = LogisticRegression(max_iter=1000, solver='liblinear')  # Initialize the model
        # Perform Randomized Search for hyperparameter tuning
        random_search = RandomizedSearchCV(lr_model, param_distributions=param_dist, n_iter=10, cv=5, random_state=42)
        random_search.fit(X_train, y_train)
        # Print the best parameters found
        print(f"Best Logistic Regression parameters: {random_search.best_params_}")
        lr_best = random_search.best_estimator_  # Best model
        # Train a Random Forest classifier
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        models = [lr_best, rf_model]  # Ensemble of models
        dump(models, 'ensemble_model.joblib')  # Save the trained models
        return models

    def predict_ensemble(self, models, X):
        # Predict probabilities with each model and average them
        predictions = np.array([model.predict_proba(X)[:, 1] for model in models])
        return np.mean(predictions, axis=0) > 0.5  # Average prediction and threshold

    def evaluate_model(self, models, X_test, y_test):
        # Get predictions from the ensemble model
        y_pred = self.predict_ensemble(models, X_test)
        # Print evaluation metrics
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", conf_matrix)
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
        # Plot Precision-Recall curve
        def plot_precision_recall_curve(y_test, y_pred_prob):
            precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve for Spoilers')
            plt.show()

        # Display metrics for the positive class
        def display_positive_class_metrics(y_test, y_pred):
            report = classification_report(y_test, y_pred, target_names=['Non-Spoiler', 'Spoiler'], output_dict=True)
            print('Positive Class Metrics (Spoiler):\n')
            print(f"Precision: {report['Spoiler']['precision']:.2f}")
            print(f"Recall: {report['Spoiler']['recall']:.2f}")
            print(f"F1-Score: {report['Spoiler']['f1-score']:.2f}")

        # Plot the confusion matrix
        def plot_confusion_matrix(y_test, y_pred):
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, annot_kws={"size": 17},
                        xticklabels=['Non-Spoiler', 'Spoiler'], yticklabels=['Non-Spoiler', 'Spoiler'])
            plt.xlabel('Predicted', fontsize=16)
            plt.ylabel('Actual', fontsize=16)
            plt.title('Confusion Matrix', fontsize=18)
            plt.show()

        # Showcase the model performance
        def showcase_model_performance(models, X_test, y_test):
            y_pred_prob = np.mean([model.predict_proba(X_test)[:, 1] for model in models], axis=0)
            y_pred = y_pred_prob > 0.5
            # Plot Precision-Recall Curve
            plot_precision_recall_curve(y_test, y_pred_prob)
            # Display metrics for the positive class
            display_positive_class_metrics(y_test, y_pred)
            # Plot the confusion matrix
            plot_confusion_matrix(y_test, y_pred)
        showcase_model_performance(models, X_test, y_test)

    def check_model(self, personal_script, personal_review, models):
        # Preprocess the personal script and review
        personal_script_cleaned = self.preprocess_text(personal_script)
        personal_review_cleaned = self.preprocess_text(personal_review)
        # Load the TF-IDF vectorizers
        tfidf_vectorizer_reviews = load('tfidf_vectorizer_reviews.joblib')
        tfidf_vectorizer_scripts = load('tfidf_vectorizer_scripts.joblib')
        # Transform the personal script and review
        X_personal_review = tfidf_vectorizer_reviews.transform([personal_review_cleaned]).toarray()
        X_personal_script = tfidf_vectorizer_scripts.transform([personal_script_cleaned]).toarray()
        # Create custom features for the personal review and script
        custom_features = self.create_custom_features([personal_review_cleaned], [personal_script_cleaned])
        # Combine features
        X_personal = np.hstack((X_personal_review, X_personal_script, custom_features))
        # Make a prediction with the ensemble model
        personal_prediction = self.predict_ensemble(models, X_personal)
        print(f"Review: {personal_review}\nIs Spoiler: {personal_prediction[0]}")

    # Main function to run the entire process
    def run_model(self):
        # Initialize the data parser with file paths to reviews and scripts
        self.sample_data()  # Sample data
        print("Data sampling done!")
        self.preprocess_reviews()  # Preprocess reviews
        print("Reviews processing done!")
        self.preprocess_scripts()  # Preprocess scripts
        print("Scripts processing done!")
        self.merge_data()  # Merge reviews and scripts
        print("Data merging done!")
        # Create features and split the data
        X_train, X_test, y_train, y_test = self.create_features()
        print("Creating features done!")
        # Train the model
        models = self.train_model(X_train, y_train)
        print("Training done!")
        # Evaluate the model
        self.evaluate_model(models, X_test, y_test)
        print("Evaluating done!")



# Execute the main function when the script is run directly
if __name__ == "__main__":
    spoiler_detctor_model = SpoilerDetector('project_data/IMDB_reviews.json', 'project_data/IMDB_movie_details.json')
    spoiler_detctor_model.run_model()

    # # Test the model with a personal script and reviews
    # personal_script = "Mosh eat icecream in the desert and then died from heartatack."
    # personal_reviews = [
    #     "Mosh",
    #     "desert",
    #     "heartatackk in the desert",
    #     "Mosh eat icecream la la la its funnnnnn",
    #     "Mosh walked in the desert",
    #     "Mosh got sick",
    #     "Mosh died in the desert"
    # ]
    #
    # for review in personal_reviews:
    #     spoiler_detctor_model.check_model(personal_script, review, models)