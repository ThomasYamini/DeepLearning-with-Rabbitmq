import pika
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from multiprocessing import Process
import multiprocessing
from utils import CNNModel, retrieve_and_update_model, train_model, test
import json

# Hyperparameters
learning_rate = 0.001
batch_size = 64
epochs = 5
num_processes = multiprocessing.cpu_count() - 1
num_test_examples = 1000

# Load and preprocess MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_dataset = torch.utils.data.Subset(train_dataset, range(num_test_examples))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# RabbitMQ connection setup
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Declare the queue
channel.queue_declare(queue='task_queue', durable=True)
channel.queue_declare(queue='merge_queue', durable=True)

# Define the callback function to receive messages
def callback(ch, method, properties, body):
    try:
        # Decode the message
        message = body.decode('utf-8')
        print("Received message:", message)
        
        if message == 'TRAINING_COMPLETE':
            print("Training complete message received.")
        else: 
            start, end = map(int, message.split('_'))
            print(f"Received start: {start}, end: {end}")

            # Split the training data indices for each process
            step = (end - start) // num_processes
            ranges = [(i * step, (i + 1) * step) for i in range(num_processes)]

            for i in range(epochs):
                print(f"Epoch {i} started")

                if i==0: # for the first iteration, the model must be initialized with empty values
                    # Create and start a process for each training subset
                    processes = []
                    for i, (s, e) in enumerate(ranges):
                        process = Process(target=train_model, args=(CNNModel(), s, e, i ))
                        process.start()
                        processes.append(process)

                    # Wait for all processes to finish
                    for process in processes:
                        process.join()

                else: # afterwards, the model will be updated with the sent parameters
                    updated_model = retrieve_and_update_model(channel)

                    processes = []
                    for i, (s, e) in enumerate(ranges):
                        process = Process(target=train_model, args=(updated_model, s, e, i))
                        process.start()
                        processes.append(process)

                    # Wait for all processes to finish
                    for process in processes:
                        process.join()

            # Once the model is enough trained, we can test it :
            test(updated_model)

    except Exception as e:
        print(f"Error in callback: {e}")


# Set up the consumer
channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='task_queue', on_message_callback=callback)


# Start consuming messages
if __name__ == '__main__':
    print('Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()