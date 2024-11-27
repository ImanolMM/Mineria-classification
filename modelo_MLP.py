import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

nltk.download('stopwords')
# Parámetros del modelo
HIDDEN_LAYER_SIZES = (128,128)  # Tamaño de las capas ocultas
MAX_ITER = 25 # Número máximo de iteraciones
RANDOM_STATE = 42  # Semilla aleatoria

# Crear carpeta única basada en los parámetros
try:
    layer_sizes_str = "_".join(map(str, HIDDEN_LAYER_SIZES))
except:
    layer_sizes_str = str(HIDDEN_LAYER_SIZES)
folder_name = f"results/MLP/HIDDEN_LAYERS_{layer_sizes_str}_MAX_ITER_{MAX_ITER}"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Cargar los datos
train_df = pd.read_csv('data/train.csv')
dev_df = pd.read_csv('data/dev.csv')
test_df = pd.read_csv('data/test.csv')

# Preprocesar los textos
def preprocess_text(text):
    # Convertir a minúsculas
    text = text.lower()
    
    # Eliminar números
    text = re.sub(r'\d+', '', text)
    
    # Eliminar puntuación
    text = re.sub(r'[^\w\s]', '', text)
    
    # Eliminar stopwords (opcional)
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    
    # Stemming (opcional)
    stemmer = SnowballStemmer('english')
    text = " ".join([stemmer.stem(word) for word in text.split()])
    
    # Eliminar espacios extra
    text = text.strip()
    
    return text
train_df['open_response'] = train_df['open_response'].apply(preprocess_text)
dev_df['open_response'] = dev_df['open_response'].apply(preprocess_text)
test_df['open_response'] = test_df['open_response'].apply(preprocess_text)

# Vectorización del texto usando TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df['open_response'])
X_dev = vectorizer.transform(dev_df['open_response'])
X_test = vectorizer.transform(test_df['open_response'])

# Codificar la variable de salida
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['gs_text34'])
y_dev = label_encoder.transform(dev_df['gs_text34'])
y_test = label_encoder.transform(test_df['gs_text34'])

# Crear y entrenar el modelo MLP
mlp = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYER_SIZES, max_iter=MAX_ITER, random_state=RANDOM_STATE)
mlp.fit(X_train, y_train)

# Predecir las etiquetas
y_pred_dev = mlp.predict(X_dev)
y_pred_test = mlp.predict(X_test)

# Métricas y evaluación
metrics = {}

def evaluate_model(y_true, y_pred, set_name):
    report = classification_report(y_true, y_pred, target_names=label_encoder.classes_, output_dict=True)
    accuracy = accuracy_score(y_true, y_pred)
    
    metrics[set_name] = {
        "accuracy": accuracy,
        "macro avg": {
            "precision": report['macro avg']['precision'],
            "recall": report['macro avg']['recall'],
            "f1-score": report['macro avg']['f1-score']
        },
        "weighted avg": {
            "precision": report['weighted avg']['precision'],
            "recall": report['weighted avg']['recall'],
            "f1-score": report['weighted avg']['f1-score']
        },
        "classification_report": report
    }

# Evaluar Dev y Test
evaluate_model(y_dev, y_pred_dev, "Dev")
evaluate_model(y_test, y_pred_test, "Test")

# Guardar las métricas en un archivo JSON
with open(f"{folder_name}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Graficar la matriz de confusión y guardarla
def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{folder_name}/{filename}')
    plt.close()

plot_confusion_matrix(y_dev, y_pred_dev, label_encoder.classes_, "Confusion Matrix (Dev)", "confusion_matrix_dev.png")
plot_confusion_matrix(y_test, y_pred_test, label_encoder.classes_, "Confusion Matrix (Test)", "confusion_matrix_test.png")

# Graficar la precisión durante el entrenamiento y guardarla
plt.figure(figsize=(10, 6))
plt.plot(mlp.loss_curve_)
plt.title("Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.savefig(f'{folder_name}/loss_curve.png')
plt.close()

# Guardar el modelo, vectorizador y codificador
joblib.dump(mlp, f'{folder_name}/mlp_model.pkl')
joblib.dump(vectorizer, f'{folder_name}/tfidf_vectorizer.pkl')
joblib.dump(label_encoder, f'{folder_name}/label_encoder.pkl')

print(f"Model, metrics, and plots saved in: {folder_name}")
