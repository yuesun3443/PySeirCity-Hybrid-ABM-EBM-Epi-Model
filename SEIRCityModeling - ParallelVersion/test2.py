import multiprocessing as mp

def worker(shared_obj, index):
    """
    Worker function that modifies the shared object.
    """
    # Access and modify the shared object
    shared_obj['data'][index] += 1
    #shared_obj['nested']['counter'] += 1

if __name__ == '__main__':
    # Create a Manager object
    manager = mp.Manager()

    # Initialize a complex shared object
    shared_obj = manager.dict()
    shared_obj['data'] = manager.list([i for i in range(1000)])
    shared_obj['nested'] = manager.dict({'counter': 0, 'info': 'Shared data'})
    shared_obj['settings'] = {'option1': True, 'option2': False}

    # Create a list to hold the process objects
    processes = []

    # Start multiple processes
    for i in range(len(shared_obj['data'])):
        p = mp.Process(target=worker, args=(shared_obj, i))
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Output the modified shared object
    print("Modified shared object:")
    print(shared_obj['data'])