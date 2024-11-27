import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import spacy
import json

# Spacy Tokenizer
nlp = spacy.load("en_core_web_sm")
def spacy_tokenizer(text):
    return [token.text.lower() for token in nlp(text)]

# Configuración global
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
EPOCHS = 25
EMBEDDING_DIM = 100
LEARNING_RATE = 0.0001
NUM_CLASSES = None

# Crear directorio de resultados
params_str = f"BATCH{BATCH_SIZE}_EPOCHS{EPOCHS}_EMB{EMBEDDING_DIM}_LR{LEARNING_RATE}"
output_dir = f"results/CNN/{params_str}"
os.makedirs(output_dir, exist_ok=True)

# 1. Cargar los datos
train_data = pd.read_csv('data/train.csv')
dev_data = pd.read_csv('data/dev.csv')
test_data = pd.read_csv('data/test.csv')

# Preprocesamiento
def preprocess_text(df):
    df['text'] = df['module'] + ' ' + df['site'] + ' ' + df['open_response']
    df['label'] = df['gs_text34']
    return df[['text', 'label']]

train_data = preprocess_text(train_data)
dev_data = preprocess_text(dev_data)
test_data = preprocess_text(test_data)

# Crear vocabulario
counter = Counter()
for text in train_data['text']:
    counter.update(spacy_tokenizer(text))

# Mapeo del vocabulario
vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common())}
vocab['<unk>'] = 0
vocab['<pad>'] = 1
vocab_size = len(vocab)

# Función para tokenizar y codificar texto
def encode_text(text):
    return [vocab.get(token, vocab['<unk>']) for token in spacy_tokenizer(text)]

# Codificar datos
def encode_data(df):
    texts = [torch.tensor(encode_text(text)) for text in df['text']]
    labels = [label for label in df['label'].astype('category').cat.codes]
    return texts, torch.tensor(labels), list(df['label'].astype('category').cat.categories)

train_texts, train_labels, classes = encode_data(train_data)
dev_texts, dev_labels, _ = encode_data(dev_data)
test_texts, test_labels, _ = encode_data(test_data)

NUM_CLASSES = len(classes)

# Dataset personalizado
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def collate_batch(batch):
    texts, labels = zip(*batch)
    lengths = [len(text) for text in texts]
    texts = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=vocab['<pad>'])
    return texts, torch.tensor(labels), lengths

train_dataset = TextDataset(train_texts, train_labels)
dev_dataset = TextDataset(dev_texts, dev_labels)
test_dataset = TextDataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# 2. Modelo CNN
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab['<pad>'])
        self.conv1 = nn.Conv2d(1, 100, (3, embed_dim))
        self.conv2 = nn.Conv2d(1, 100, (4, embed_dim))
        self.conv3 = nn.Conv2d(1, 100, (5, embed_dim))
        self.fc = nn.Linear(300, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, text):
        embedded = self.embedding(text).unsqueeze(1)
        conv1_out = torch.relu(self.conv1(embedded)).squeeze(3)
        conv2_out = torch.relu(self.conv2(embedded)).squeeze(3)
        conv3_out = torch.relu(self.conv3(embedded)).squeeze(3)
        pooled1 = torch.max(conv1_out, dim=2)[0]
        pooled2 = torch.max(conv2_out, dim=2)[0]
        pooled3 = torch.max(conv3_out, dim=2)[0]
        concat = torch.cat((pooled1, pooled2, pooled3), dim=1)
        return self.fc(self.dropout(concat))

model = TextCNN(vocab_size, EMBEDDING_DIM, NUM_CLASSES).to(device)

# 3. Entrenamiento
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train(model, loader):
    model.train()
    total_loss = 0
    correct = 0
    for texts, labels, _ in loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

# Función para evaluación con métricas
def evaluate_with_metrics(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for texts, labels, _ in loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    # Obtener el reporte de métricas
    metrics_report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    return total_loss / len(loader), correct / len(loader.dataset), y_true, y_pred, metrics_report

train_losses = []
train_accuracies, dev_accuracies = [], []

for epoch in range(EPOCHS):
    train_loss, train_acc = train(model, train_loader)
    dev_loss, dev_acc, _, _, _ = evaluate_with_metrics(model, dev_loader)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    dev_accuracies.append(dev_acc)
    print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Dev Loss={dev_loss:.4f}, Dev Acc={dev_acc:.4f}")

# Guardar el modelo
model_path = os.path.join(output_dir, 'model.pth')
torch.save(model.state_dict(), model_path)

# Graficar pérdidas
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')
plt.savefig(os.path.join(output_dir, 'loss.png'))
plt.close()

# Evaluar en el conjunto de prueba y guardar las métricas
test_loss, test_acc, y_true, y_pred, metrics_report = evaluate_with_metrics(model, test_loader)

# Guardar las métricas en un archivo JSON
metrics_path = os.path.join(output_dir, 'metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics_report, f, indent=4)

# Imprimir y graficar el reporte de métricas
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=classes))

# Graficar matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()

# Resumen final de métricas
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Metrics saved to {metrics_path}")
