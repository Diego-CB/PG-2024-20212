import torch
import torch.nn.functional as F
from torch import nn

class ConvNet(nn.Module):
    def __init__(self, num_features=6):
        self.num_features = num_features
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(128 * 6 * 6, 1024)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, self.num_features)
    
    def forward(self, x):
        # Forward pass through the network
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout2(x)
        
        # Flatten the tensor before fully connected layers
        x = x.view(-1, 128 * 6 * 6)  # Reshaping for the fully connected layer
        
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)  # Final output
        
        # Apply log_softmax to the output
        return F.log_softmax(x, dim=1)

class ConvNetLarge(nn.Module):
    def __init__(self, num_features=6):  # Set num_features to match number of output classes
        super(ConvNetLarge, self).__init__()
        self.num_features = num_features
        
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces size by half
        self.dropout1 = nn.Dropout(0.25)

        # Second convolutional block
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Further reduces size
        self.dropout2 = nn.Dropout(0.25)

        # Fully connected layers
        # After 2 max-pool layers, image size reduces to 12x12
        self.fc1 = nn.Linear(256 * 12 * 12, 2048)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, 1024)
        self.dropout4 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1024, self.num_features)  # Output layer

    def forward(self, x):
        # First block
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout1(x)

        # Second block
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Flatten the output for fully connected layers
        x = x.view(-1, 256 * 12 * 12)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = F.relu(self.fc2(x))
        x = self.dropout4(x)
        x = self.fc3(x)  # Final output, no activation since we use softmax later

        return F.log_softmax(x, dim=1)  # Apply log_softmax to the output
