# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
**Problem Statement:**  
Predicting stock prices is a complex task due to market volatility. Using historical closing prices, we aim to develop a Recurrent Neural Network (RNN) model that can analyze time-series data and generate future price predictions.

**Dataset:**  
The dataset consists of historical stock closing prices from `trainset.csv` and `testset.csv`. The data is normalized using MinMax scaling, and sequences of 60 past values are used as input features. The model learns patterns from training data to predict upcoming prices, helping traders and investors make informed decisions.

## Design Steps
### Step 1:
- Data Collection & Preprocessing: Load historical stock prices, normalize using MinMaxScaler, and create sequences for time-series input.

### Step 2:
- Model Design: Build an RNN with two layers, define input/output sizes, and set activation functions.

### Step 3:
- Training Process: Train the model using MSE loss and Adam optimizer for 20 epochs with batch-size optimization.

### Step 4:
- Evaluation & Prediction: Test on unseen data, inverse transform predictions, and compare with actual prices.

### Step 5:
- Visualization & Interpretation: Plot training loss and predictions to analyze performance and potential improvements.



## Program
#### Name:CHANDRU P
#### Register Number:212223110007
```Python 
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layer
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)             # out: (batch_size, seq_len, hidden_size)
        out = out[:, -1, :]                  # take last time step
        out = self.fc(out)
        return out
```

```
model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

```
## Step 3: Train the Model
num_epochs = 50
train_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        # Forward pass
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')

```

## Output

### True Stock Price, Predicted Stock Price vs time
<img width="1079" height="728" alt="image" src="https://github.com/user-attachments/assets/d7ddd202-11fc-4623-b2a5-6e48301cf5ba" />



### Predictions 
<img width="293" height="47" alt="image" src="https://github.com/user-attachments/assets/96ae437b-60bb-48e0-88bb-69dbff292f4b" />



## Result
Thus, a Recurrent Neural Network model for stock price prediction has successfully been devoloped.

