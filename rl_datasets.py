from PIL import Image
import pickle

from torch.utils.data import Dataset, DataLoader


class MiniGridDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # open a file, where you stored the pickled data
        file = open(self.root_dir, 'rb')

        # dump information to that file
        self.data = pickle.load(file)

        # close the file
        file.close()

        
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        image = self.data[idx][0]

        image = Image.fromarray(image)

        #image = cv2.resize(image, (224,224) , interpolation = cv2.INTER_AREA)

        if self.transform:
            image = self.transform(image)
        
        #.to_tensorv2(image)
        #image = torch.from_numpy(image).reshape(3, 224, 224)
        #plt.imshow(image)
        #plt.show()
        
        #print(image.shape)
        #image = image.reshape(3, 224, 224)
        return image


class StateActionReturnDataset(Dataset):
    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):        
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        states = states / 255.
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)

        return states #, actions, rtgs, timesteps
