# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
Named Entity Recognition (NER) is a task in Natural Language Processing (NLP) that identifies entities such as names of persons, locations, organizations, etc., in text.

## DESIGN STEPS

### STEP 1:
Collect and preprocess the NER dataset by tokenizing sentences and mapping words and tags to indices.

### STEP 2:
Split the dataset into training and testing sets and create data loaders for batch processing.

### STEP 3:
Initialize the BiLSTM model with embedding, LSTM, and fully connected layers.

### STEP 4:
Define the loss function and optimizer for training the model.

### STEP 5:
Train the model over multiple epochs and update weights using backpropagation.

### STEP 6:
Evaluate the model on test data and analyze predictions for named entity recognition.

## PROGRAM

### Name: Yamuna M

### Register Number: 212223230248

```python
# Model definition
class BiLSTMTagger(nn.Module):
  def __init__(self,vocab_size, target_size, embedding_dim=50, hidden_dim=100):
    super(BiLSTMTagger,self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.dropout = nn.Dropout(0.1)
    self.lstm=nn.LSTM(embedding_dim,hidden_dim,batch_first=True,bidirectional=True)
    self.fc=nn.Linear(hidden_dim*2,target_size)

  def forward(self, x):
    x=self.embedding(x)
    x=self.dropout(x)
    x,_=self.lstm(x)
    return self.fc(x)
model =BiLSTMTagger(len(word2idx)+1,len(tag2idx)).to(device)
loss_fn =nn.CrossEntropyLoss()
optimizer =torch.optim.Adam(model.parameters(),lr=0.001)

# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase after each epoch
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids)
                loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
                val_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(test_loader))
        print(f"Epoch {epoch+1}: Train loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}")

    return train_losses, val_losses
```

### OUTPUT

## Loss Vs Epoch Plot
<img width="791" height="655" alt="Screenshot 2026-03-14 173916" src="https://github.com/user-attachments/assets/3e0062cc-e735-4fb0-8c0f-e7ad2508be69" />

### Sample Text Prediction
<img width="436" height="541" alt="image" src="https://github.com/user-attachments/assets/ac833bba-bfa3-4a83-b6fb-522e4fa0b23e" />

## RESULT
The BiLSTM-based NER model was successfully trained and evaluated, showing a steady decrease in training and validation loss.
The model is able to correctly identify named entities such as persons, locations, and organizations from input text.
