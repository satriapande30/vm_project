import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from datetime import datetime

class SVMClassifier:
    def __init__(self, input_file, output_dir, batch_size=1000):
        self.input_file = input_file
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.logger = self._setup_logger()

    def _setup_logger(self):
        log_dir = os.path.join(self.output_dir, 'log')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"svm_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(log_file)
        stream_handler = logging.StreamHandler()
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        
        return logger

    def load_data(self):
        self.logger.info("Loading data...")
        data = pd.read_csv(self.input_file)
        X = data.iloc[:, 1:].values
        y = data.iloc[:, 0].values
        y = self.label_encoder.fit_transform(y)
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def preprocess_data(self, X_train, X_test):
        self.logger.info("Preprocessing data...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def train_model(self, X_train, y_train):
        self.logger.info("Training SVM model...")
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 1],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        
        svm = SVC()
        grid_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1, verbose=1)
        
        n_samples = X_train.shape[0]
        n_batches = (n_samples - 1) // self.batch_size + 1
        
        for i in tqdm(range(n_batches), desc="Training batches"):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, n_samples)
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]
            grid_search.fit(X_batch, y_batch)
        
        self.model = grid_search.best_estimator_
        self.logger.info(f"Best parameters: {grid_search.best_params_}")

        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        self.logger.info(f"Cross-validation scores: {cv_scores}")
        self.logger.info(f"Mean CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
        
        return grid_search.best_params_, cv_scores

    def evaluate_model(self, X_test, y_test):
        self.logger.info("Evaluating model...")
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f"Precision: {precision:.4f}")
        self.logger.info(f"Recall: {recall:.4f}")
        self.logger.info(f"F1 Score: {f1:.4f}")
        
        return accuracy, precision, recall, f1

    def plot_confusion_matrix(self, X_test, y_test):
        self.logger.info("Plotting confusion matrix...")
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        vis_dir = os.path.join(self.output_dir, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)
        plt.savefig(os.path.join(vis_dir, 'confusion_matrix.png'))
        plt.close()
        self.logger.info(f"Confusion matrix saved to {os.path.join(vis_dir, 'confusion_matrix.png')}")

    def plot_best_parameters(self, best_params):
        self.logger.info("Plotting best parameters...")
        plt.figure(figsize=(10, 6))
        for i, (key, value) in enumerate(best_params.items()):
            plt.bar(i, 1, label=f"{key}: {value}")
        plt.title('Best SVM Parameters')
        plt.xlabel('Parameter')
        plt.ylabel('Value')
        plt.legend()
        plt.xticks([])
        
        vis_dir = os.path.join(self.output_dir, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)
        plt.savefig(os.path.join(vis_dir, 'best_parameters.png'))
        plt.close()
        self.logger.info(f"Best parameters plot saved to {os.path.join(vis_dir, 'best_parameters.png')}")

    def plot_cv_scores(self, cv_scores):
        self.logger.info("Plotting cross-validation scores...")
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o')
        plt.title('Cross-Validation Scores')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        mean_cv = np.mean(cv_scores)
        plt.axhline(y=mean_cv, color='r', linestyle='--', label=f'Mean CV accuracy: {mean_cv:.4f}')
        plt.legend()
        
        vis_dir = os.path.join(self.output_dir, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)
        plt.savefig(os.path.join(vis_dir, 'cv_scores.png'))
        plt.close()
        self.logger.info(f"Cross-validation scores plot saved to {os.path.join(vis_dir, 'cv_scores.png')}")

    def plot_evaluation_metrics(self, accuracy, precision, recall, f1):
        self.logger.info("Plotting evaluation metrics...")
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        values = [accuracy, precision, recall, f1]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values)
        plt.title('Model Evaluation Metrics')
        plt.ylim(0, 1)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.4f}',
                     ha='center', va='bottom')
        
        vis_dir = os.path.join(self.output_dir, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)
        plt.savefig(os.path.join(vis_dir, 'evaluation_metrics.png'))
        plt.close()
        self.logger.info(f"Evaluation metrics plot saved to {os.path.join(vis_dir, 'evaluation_metrics.png')}")

    def plot_hand_prediction(self, X_test, y_test):
        self.logger.info("Plotting hand prediction examples...")
        y_pred = self.model.predict(X_test)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Hand Gesture Predictions (Sample)')
        
        for i, ax in enumerate(axes.flat):
            if i < len(y_test):
                ax.scatter(X_test[i, ::3], X_test[i, 1::3], c='b', label='Actual')
                ax.scatter(X_test[i, ::3], X_test[i, 1::3], c='r', marker='x', label='Predicted')
                ax.set_title(f'True: {self.label_encoder.inverse_transform([y_test[i]])[0]}\nPred: {self.label_encoder.inverse_transform([y_pred[i]])[0]}')
                ax.legend()
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        vis_dir = os.path.join(self.output_dir, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)
        plt.savefig(os.path.join(vis_dir, 'hand_predictions.png'))
        plt.close()
        self.logger.info(f"Hand prediction examples saved to {os.path.join(vis_dir, 'hand_predictions.png')}")

    def plot_final_results(self, accuracy, precision, recall, f1):
        self.logger.info("Plotting final results...")
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        values = [accuracy, precision, recall, f1]
        
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        ax = sns.barplot(x=metrics, y=values)
        plt.title('Final Model Performance', fontsize=16)
        plt.ylim(0, 1)
        
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=12)
        
        plt.ylabel('Score', fontsize=14)
        plt.xlabel('Metric', fontsize=14)
        
        vis_dir = os.path.join(self.output_dir, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)
        plt.savefig(os.path.join(vis_dir, 'final_results.png'))
        plt.close()
        self.logger.info(f"Final results plot saved to {os.path.join(vis_dir, 'final_results.png')}")

    def save_model(self):
        self.logger.info("Saving model...")
        model_dir = os.path.join(self.output_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, 'svm_model.pkl')
        joblib.dump((self.model, self.scaler, self.label_encoder), model_file)
        self.logger.info(f"Model saved to {model_file}")

    def run(self):
        X_train, X_test, y_train, y_test = self.load_data()
        X_train_scaled, X_test_scaled = self.preprocess_data(X_train, X_test)
        best_params, cv_scores = self.train_model(X_train_scaled, y_train)
        
        self.plot_best_parameters(best_params)
        self.plot_cv_scores(cv_scores)
        
        accuracy, precision, recall, f1 = self.evaluate_model(X_test_scaled, y_test)
        
        self.plot_evaluation_metrics(accuracy, precision, recall, f1)
        self.plot_confusion_matrix(X_test_scaled, y_test)
        self.plot_hand_prediction(X_test_scaled, y_test)
        self.plot_final_results(accuracy, precision, recall, f1)
        
        self.save_model()
        
        return accuracy, precision, recall, f1

if __name__ == "__main__":
    input_file = "D:/KULIAH/SKRIPSI/vm_project/output/features/hand_landmark_features.csv"
    output_dir = "D:/KULIAH/SKRIPSI/vm_project/output"
    
    classifier = SVMClassifier(input_file, output_dir)
    accuracy, precision, recall, f1 = classifier.run()
    
    classifier.logger.info("Training and evaluation completed.")
    classifier.logger.info(f"Final Accuracy: {accuracy:.4f}")
    classifier.logger.info(f"Final Precision: {precision:.4f}")
    classifier.logger.info(f"Final Recall: {recall:.4f}")
    classifier.logger.info(f"Final F1 Score: {f1:.4f}")