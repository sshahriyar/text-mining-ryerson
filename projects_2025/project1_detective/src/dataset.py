# ================================
# Utilities to process the dataset
# ================================

# Import required packages
from torch.utils.data import Dataset

class PassagesDataset(Dataset):
    """
    A PyTorch Dataset class for handling the OUTFOX dataset.
    It takes a dataset of passages and their corresponding labels,
    and provides methods to access the data.
    """

    def __init__(self, dataset, mode='OUTFOX', need_ids=False):
        """
        Initialize the PassagesDataset with the dataset and mode.
        Args:
            dataset (list): A list of tuples containing passages and labels.
            mode (str): The mode of the dataset ('OUTFOX' or 'other').
            need_ids (bool): Whether to include IDs in the output.
        """
        self.mode = mode
        self.dataset = dataset
        self.need_ids = need_ids
        self.classes = []
        self.model_name_set = {}
        LLM_name = set()
        for item in self.dataset:
            LLM_name.add(item[2])
        for i,name in enumerate(LLM_name):
            self.model_name_set[name] = (i,i)
            self.classes.append(name)

        print(f'there are {len(self.classes)} classes in {mode} dataset')
        print(f'the classes are {self.classes}')

    def get_class(self):
        """
        Get the list of classes in the dataset.
        Returns:
            list: A list of classes in the dataset.
        """
        return self.classes

    def __len__(self):
        """
        Get the length of the dataset.
        Returns:
            int: The length of the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get an item from the dataset by index.
        Args:
            idx (int): The index of the item.
        Returns:
            tuple: A tuple containing the text, label, and source of the item.
        """
        text,label,src,id=self.dataset[idx]
        write_model,write_model_set=1000,1000
        for name in self.model_name_set.keys():
            if name in src:
                write_model,write_model_set=self.model_name_set[name]
                break
        assert write_model!=1000,f'write_model is empty,src is {src}'

        if self.need_ids:
            return text,int(label),int(write_model),int(write_model_set),int(id)
        else:
            return text,int(label),int(write_model),int(write_model_set)