import queue
import threading
import time

# Create a queue to hold the tasks
task_queue = queue.Queue()

# Define a functn to process the tasks
def process_task(task):
    print(f"Processing task: {task}")
    time.sleep(task)
    print(f"Task completed: {task}")
    print("*"*50)

# Define a function for the daemon to run
def daemon():
    while task:=task_queue.get(block=True):
        print("_"*50)
        print(f"Tasks todo: {task_queue.qsize()} | {task_queue.queue}")
        # Process the task
        process_task(task)

        # Mark the task as done
        task_queue.task_done()
        print(f"Queue size: {task_queue.qsize()}")
        print("Queue contents: ", list(task_queue.queue))
        time.sleep(2)

# Start the daemon in a separate thread
threading.Thread(target=daemon, daemon=True).start()

# Add tasks to the queue
task_queue.put(4)
task_queue.put(5)
task_queue.put(7)
time.sleep(30)
task_queue.put(4)

# Wait for all tasks to be processed
task_queue.join()