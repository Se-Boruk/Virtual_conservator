from datasets import load_dataset, load_from_disk
import os


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
    
    
    
    
    
    
    
    
    
    
    
    
    