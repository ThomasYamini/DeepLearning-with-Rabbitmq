import pika
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random

epochs = 5
learning_rate = 0.001
batch_size = 64
num_test_examples = 1000

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_dataset = torch.utils.data.Subset(train_dataset, range(num_test_examples))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def __add__(self, other):
        # Implement the addition of two CNNModel instances
        if not isinstance(other, CNNModel):
            raise ValueError("Unsupported operand type for +: 'CNNModel' and {}".format(type(other)))

        # Create a new CNNModel instance
        result_model = CNNModel()

        # Combine the parameters of the two models
        result_model_dict = result_model.state_dict()
        for key, value in self.state_dict().items():
            result_model_dict[key] = value
        for key, value in other.state_dict().items():
            result_model_dict[key] = value

        # Load the combined state_dict into the result_model
        result_model.load_state_dict(result_model_dict)

        return result_model
    
def get_data_loader(train_dataset, start, end, batch_size):
    subset = torch.utils.data.Subset(train_dataset, range(start, end))
    return DataLoader(subset, batch_size=batch_size, shuffle=True)

def load_combined_model(channel):
    _, _, body = channel.basic_get(queue='combined_model_queue', auto_ack=True)
    combined_model_state_dict = json.loads(body.decode('utf-8'))
    return combined_model_state_dict


def retrieve_and_update_model(channel):
    # Wait for the training to be complete
    while True:
        _, _, body = channel.basic_get(queue='task_queue', auto_ack=True)
        if body and body.decode('utf-8') == 'TRAINING_COMPLETE':
            break

    param_accumulator = {}

    while True:
        _, _, body = channel.basic_get(queue='merge_queue', auto_ack=True)
        if body is None:
            break

        message = json.loads(body.decode('utf-8'))
        start = message['start']
        end = message['end']
        param_name = message['param_name']
        param_value = message['param_value']

        # Accumulate the parameters
        if (start, end) not in param_accumulator:
            param_accumulator[(start, end)] = {}

        param_accumulator[(start, end)][param_name] = torch.tensor(param_value)

    # Create a temporary model to hold the received parameter
    temporary_model = CNNModel()

    # Dictionary to accumulate parameter values
    param_accumulator_mean = {}

    for (start, end), param_dict in param_accumulator.items():
        for param_name, param_value in param_dict.items():
            if param_name not in param_accumulator_mean:
                param_accumulator_mean[param_name] = torch.zeros_like(param_value)

            param_accumulator_mean[param_name] += param_value

    # Calculate the mean for each parameter
    num_chunks = len(param_accumulator)
    for param_name, param_value_sum in param_accumulator_mean.items():
        param_mean = param_value_sum / num_chunks
        # Directly assign the mean values to the model parameters
        temporary_model.state_dict()[param_name].copy_(param_mean)

    return temporary_model



def train_model(local_model, start, end, process_id):
    print(f"Process {process_id} is training on data from index {start} to {end}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(local_model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = local_model(data)
            loss = criterion(output[start:end], target[start:end])
            loss.backward()
            optimizer.step()

    # Send only the local model parameters to the merge queue
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='merge_queue', durable=True)

    model_state_dict = local_model.state_dict()
    for key, value in model_state_dict.items():
        channel.basic_publish(
            exchange='',
            routing_key='merge_queue',
            body=json.dumps({
                'start': start,
                'end': end,
                'param_name': key,
                'param_value': value.numpy().tolist()
            })
        )
    # Send the model completion message
    channel.basic_publish(exchange='', routing_key='task_queue', body="TRAINING_COMPLETE")

    connection.close()

    print(f"Process {process_id} finished training.")


def test(final_model):
    print("Tests on the MNIST Dataset")
    # Testing section
    # Load and preprocess the MNIST test dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

    # Get a batch of images and labels
    images, labels = next(iter(test_loader))

    # Randomly select 10 indices
    selected_indices = random.sample(range(len(images)), 10)

    # Test the combined model on the selected images
    with torch.no_grad():
        for i in selected_indices:
            image = images[i].unsqueeze(0)
            label = labels[i].item()

            output = final_model(image)
            _, prediction = torch.max(output, 1)
            prediction = prediction.item()

            # Print the prediction along with the image
            print(f"Prediction: {prediction}, Actual Label: {label}")

            # Display the image
            plt.imshow(image.squeeze(), cmap='gray')
            plt.title(f"Prediction: {prediction}, Actual Label: {label}")
            plt.show()
