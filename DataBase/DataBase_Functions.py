from datasets import load_dataset, load_from_disk
import os
import numpy as np
import torch
from queue import Queue
import threading




class Custom_DataSet_Manager():
    
    #Checks if there is dataset folder present, if not it creates it
    def __init__(self, DataSet_path, train_split, val_split, test_split, random_state):
        self.dataset_path = DataSet_path
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state
        
        self.flag_file = os.path.join(self.dataset_path, "download_complete.flag")
        
        
        
    def download_database(self, dataset_name):
        #Create folder if not present 
        os.makedirs(self.dataset_path, exist_ok=True)

        print("Downloading dataset...")
        dataset = load_dataset(dataset_name)
        dataset.save_to_disk(self.dataset_path)
        
        #Add flag but only if the dataset is completed. 
        #If downloading above is interrupted then its not present
        with open(self.flag_file, "w") as f:
            f.write("downloaded")

        print("Dataset downloaded and flagged!")
        
    def is_downloaded(self):
        # Check if the flag file exists
        return os.path.exists(self.flag_file)     
    
    def load_dataset_from_disk(self):
        #Check for flag
        if not self.is_downloaded():
            raise RuntimeError("Dataset not downloaded or incomplete. Download it first")
            
        #Load it to split it on run
        Dataset = load_from_disk(self.dataset_path)
        
        train, val, test = self.split_dataset(Dataset)
        return train, val, test
    
    def split_dataset(self,dataset):
        #Split dataset into train, val and test. Ready for work :). 
        #OFc with given random state or diseaster 
        
        #Train bc this dataset (at least the one for reconstruction - upscaling one may 
        #be different and require changes/addons) has only train. 
        #We need to split it on our own
        
        #Just load the data and shuffle it (so we mix the classes and hopefully mix them uniformly for training)
        #Cant use stratifying as we do not know the classes a priori (unsupervised learning)
        Data =  dataset["train"].shuffle(seed=self.random_state)
        
        #Split it into train and subset
        split_dataset = Data.train_test_split(test_size= (1 -self.train_split) , seed=self.random_state)
        
        train_subset = split_dataset['train']
        subset = split_dataset['test']
        
        #Split the subset into the val and test 
        test_fraction = self.val_split / ((self.val_split + self.test_split))
        
        split_dataset_1 = subset.train_test_split(test_size= test_fraction , seed=self.random_state)
        
        val_subset = split_dataset_1['train']
        test_subset = split_dataset_1['test']

        return train_subset, val_subset, test_subset
        
        
def Reconstruction_data_tests(train_subset, val_subset, test_subset):
        #############################################
        #Add some tests for given random_state = 111 (with prints) - again, valid only for reconstruction dataset as for now:
    
        print("Running datasaet tests...")
        #Train
        #From split we want to always replicate 
        n1 = "26633.jpg"
        n2 = "31329.jpg" 
        
        #From split
        name_1 = train_subset[0]['filename']
        name_2 = train_subset[1]['filename']
        
        #print(name_1)
        #print(name_2)
        
        assert n1 == name_1, f"Names do not match! Dataset is different than it should be by Config constants '{n1}' != '{name_1}'"
        assert n2 == name_2, f"Names do not match! Dataset is different than it should be by Config constants '{n2}' != '{name_2}'"
        
        print("Train passed")
        ###########
        #Val
        #From split we want to always replicate 
        n1 = "17396.jpg"
        n2 = "83545.jpg" 
        
        #From split
        name_1 = val_subset[0]['filename']
        name_2 = val_subset[1]['filename']
        
        #print(name_1)
        #print(name_2)
        
        assert n1 == name_1, f"Names do not match! Dataset is different than it should be by Config constants '{n1}' != '{name_1}'"
        assert n2 == name_2, f"Names do not match! Dataset is different than it should be by Config constants '{n2}' != '{name_2}'"
        
        print("Val passed")
        ###########
        #Test
        #From split we want to always replicate 
        n1 = "37436.jpg"
        n2 = "100478.jpg" 
        
        #From split
        name_1 = test_subset[0]['filename']
        name_2 = test_subset[1]['filename']
        
        #print(name_1)
        #print(name_2)
        
        assert n1 == name_1, f"Names do not match! Dataset is different than it should be by Config constants '{n1}' != '{name_1}'"
        assert n2 == name_2, f"Names do not match! Dataset is different than it should be by Config constants '{n2}' != '{name_2}'"
        
        print("Test passed")
    
    
    
    
    
class Async_DataLoader():
    def __init__(self, dataset, batch_size=32, num_workers=2, device='cuda', max_queue=10):
        self.dataset = dataset
        #Taking sample of from dataset to initialize the shape of images
        sample_img = np.array(dataset[0]["image"], dtype=np.uint8)
        self.C, self.H, self.W = sample_img.shape[2], sample_img.shape[0], sample_img.shape[1]
        
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.queue = Queue(maxsize=max_queue)
        self.num_workers = num_workers

        #Epoch control
        self.next_idx = 0               #Next step (batch) idx
        self.idx_lock = threading.Lock()
        self.active_workers = 0 
        self.threads = []
        self.epoch_event = threading.Event()
        self.indices = np.arange(len(self.dataset))  #images order in current epoch

        #Preallocate pinned buffers
        self.pinned_bufs = [torch.empty((self.batch_size, self.C, self.H, self.W), 
                                        dtype=torch.float32).pin_memory() 
                            for _ in range(num_workers)]
        
        #activate function for loading batches into the queue
        self._start_prefetch()

    def _start_prefetch(self):
        
        def get_chunk():
            
            """
            Functions for getting start idx and end idx of batch 
            (for indexes list as we shuffle)
            """
        
            #Lock function so we only can acces it from one thread (worker)
            #It assures that we cannot have the same batch operated twice
            with self.idx_lock:
                start = self.next_idx
                
                if start >= len(self.dataset):
                    return None, None
                
                end = min(start + self.batch_size, len(self.dataset))
                self.next_idx = end
                
                return start, end


        def worker(worker_id):
            """
            Function for taking and processing batch. Single worker operation
            """
            pinned_buf = self.pinned_bufs[worker_id]
            while True:
                #Wait for epoch to start
                self.epoch_event.wait()
                while True:
                    start, end = get_chunk()
                    if start is None:
                        break
                    actual_bs = end - start
                    for i in range(actual_bs):
                        idx = self.indices[start + i]
                        img = np.array(self.dataset[idx]["image"], dtype=np.float32) / 255.0
                        pinned_buf[i] = torch.from_numpy(img).permute(2,0,1)
                        
                        
                    #Put given batch of imgs in the queue
                    ###################
                    # Place to put the operations on batches ( Augmentation / damage etc.)
                    ##################
                    self.queue.put(pinned_buf[:actual_bs].clone())
                    
                    
                #One worker done, check if was last worker (so last batch)
                #If it was last one then put None. It will end the epoch when reached
                with self.idx_lock:
                    self.active_workers -= 1
                    
                    if self.active_workers == 0:
                        self.queue.put(None)  ##None at the end ends epoch when reached
                        self.epoch_event.clear()  #Wait for next epoch with prefetching

        # start worker threads
        for wid in range(self.num_workers):
            t = threading.Thread(target=worker, args=(wid,))
            t.daemon = True
            t.start()
            self.threads.append(t)

    def start_epoch(self, shuffle=True):
        
        """Start a new epoch. It resets queue and shuffle data."""
        
        self.queue.queue.clear()
        self.next_idx = 0
        self.active_workers = self.num_workers
        
        #Shuffle indexes if specified so they have other order in next epoch
        if shuffle:
            np.random.shuffle(self.indices)
            
        self.epoch_event.set() #It allows workers to start

    def get_batch(self):
        """Returns batch next in queue"""
        
        batch = self.queue.get()
        if batch is None:
            return None
        
        return batch.to(self.device, non_blocking=True)

    def get_num_batches(self):
        
        """Get number of batches (steps) for given dataset length and batch size"""
        steps = (len(self.dataset) + self.batch_size - 1) // self.batch_size
        return steps    
    
    
    
    
    
    
    