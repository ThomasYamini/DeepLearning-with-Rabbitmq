import pika

# RabbitMQ connection setup
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Declare the queue
channel.queue_declare(queue='task_queue', durable=True)

# Send a message to train a segment of the model
message = "0_500"  # Example: Train the model on data from index 0 to 500


channel.basic_publish(exchange='',
                      routing_key='task_queue',
                      body=message,
                      properties=pika.BasicProperties(
                          delivery_mode=2,  # Make the message persistent
                      ))

print(f" [x] Sent '{message}'")

# Close the connection
connection.close()
