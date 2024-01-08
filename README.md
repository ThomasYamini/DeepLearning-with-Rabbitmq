## Project Summary: Leveraging RabbitMQ for Distributed Classification

In this project, the goal was to implement a distributed classification algorithm using PyTorch for image classification on the MNIST dataset. The approach involved parallelizing the training process across multiple processes, each handling a specific subset of the data. RabbitMQ was employed as a message broker to enable communication and coordination between these processes.

To make this work : 
Make sure rabbitmq is installed and started and  install torch with `pip install torch`.  

* To begin, launch `launch.py` -> `python launch.py`  
This code will send a message to the `task_queue` to launch the algorithm.  
The function was made to indicate on which portion of the dataset the training should be on : you can manually modify the value in the file.  

* In another terminal, launch `training_with_rounds.py` -> `python training_with_rounds.py`  
This function splits the MNIST dataset into n, where n is the number of processors on your machine.
Each processors will train on its chunk of the data then send the weights and biaises to a channel :`merge_queue`  
Once the training is complete, each worker will also send a message to announce that : `TRAINING_COMPLETE`  
The code will wait for the message `TRAINING_COMPLETE`. Then, merge each information sent by the workers to update the model with the new parameters.  
It will do this for the number of epochs given as hyperparameter at the beginning of the code.  
Finally, the testing section will wait for the last update on the model (last epoch).   
Then, it will load a test sample from the MNIST Dataset and print the predicted label with the correct one.  


### Key Advantages of Using RabbitMQ:  

1. **Decoupling Training Processes:**  
   - RabbitMQ decouples training processes, allowing them to operate independently. Each process handles a specific subset of the dataset without direct dependencies.  

2. **Dynamic Model Updates:**  
   - Trained model parameters are efficiently communicated between processes through RabbitMQ queues. This dynamic update mechanism enhances the overall efficiency of the distributed training.  

3. **Fault Tolerance:**  
   - RabbitMQ provides fault tolerance by decoupling producers and consumers. If one component fails, the others can continue processing without interruption. This is essential for building robust and resilient distributed systems.  

4. **Scalability and Flexibility:**  
   - RabbitMQ enables easy scalability. Additional processes can be added without modifying the existing code, making it flexible and adaptable to varying computational resources.  

5. **Reduced Communication Overheads:**  
   - RabbitMQ minimizes communication overheads by efficiently managing message queues. Processes only exchange essential information, reducing latency and improving training speed.  

6. **Enhanced Fault Tolerance:**  
   - RabbitMQ enhances fault tolerance. In the event of a process failure, others can continue training, and the system gracefully handles the recovery.  

7. **Simplified Coordination:**  
   - RabbitMQ simplifies coordination between training processes. The code focuses on model training logic, while RabbitMQ efficiently manages inter-process communication.  

### Project Outcomes:  

- The distributed classification algorithm successfully leveraged RabbitMQ for task distribution and communication between processes.
- The parallelized training process demonstrated improved efficiency in model training, especially when utilizing multiple cores or processors.


### Conclusion:

In conclusion, RabbitMQ proved to be a valuable tool for implementing a distributed classification algorithm. Its features, such as fault tolerance, and scalability, were instrumental in achieving efficient parallelization of the training process. The project not only demonstrated the practical application of RabbitMQ in distributed computing but also highlighted its potential for handling more complex and resource-intensive tasks in machine learning workflows.  