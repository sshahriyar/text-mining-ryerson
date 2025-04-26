# ====================================
# Utilities to read the OUTFOX dataset
# ====================================

# Import the required packages
import pickle
import os

def load_pkl(path):
    """
    Load a pickle file from the given path.
    Args:
        path (str): The path to the pickle file.
    Returns:
        object: The loaded object from the pickle file.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pkl(obj, path):
    """
    Save an object to a pickle file at the given path.
    Args:
        obj (object): The object to be saved.
        path (str): The path to save the pickle file.
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def make_mix_data(human_path, lm_path, ps_path):
    """
    Create a mixed dataset of humans and language models.
    Args:
        human_path (str): The path to the pickle file containing humans.
        lm_path (str): The path to the pickle file containing language models.
        ps_path (str): The path to the pickle file containing person similarities.
    Returns:
        list: A list of tuples containing mixed data.
    """
    humans = load_pkl(human_path)
    lms = load_pkl(lm_path)
    pss = load_pkl(ps_path)
    humans_with_label_ps, lms_with_label_ps = [(human, '1', ps) for human, ps in zip(humans, pss)], [(lm, '0', ps) for lm, ps in zip(lms, pss)]
    all_with_label_ps = humans_with_label_ps + lms_with_label_ps
    for i in range(len(all_with_label_ps)):
        all_with_label_ps[i] = (all_with_label_ps[i][0], all_with_label_ps[i][1], all_with_label_ps[i][1], i)
    return all_with_label_ps

def mix_train_data(path):
    """
    Create a mixed dataset of humans and language models.
    Args:
        path (str): The path to the directory containing the data.
    Returns:
        list: A list of tuples containing mixed data.
    """
    humans = load_pkl(os.path.join(path,'common/train/train_humans.pkl'))
    lms_chatgpt = load_pkl(os.path.join(path,'chatgpt/train/train_lms.pkl'))
    lms_davinci = load_pkl(os.path.join(path,'text_davinci_003/train/train_lms.pkl'))
    lms_flan_t5 = load_pkl(os.path.join(path,'flan_t5_xxl/train/train_lms.pkl'))
    humans_with_label = [(human, '1', 'human') for human in humans]
    lms_chatgpt_with_label = [(lm, '0', 'chatgpt') for lm in lms_chatgpt]
    lms_davinci_with_label = [(lm, '0', 'davinci') for lm in lms_davinci]
    lms_flan_t5_with_label = [(lm, '0', 'flan_t5') for lm in lms_flan_t5]
    lms_with_label = lms_chatgpt_with_label + lms_davinci_with_label + lms_flan_t5_with_label
    all_with_label = humans_with_label + lms_with_label
    for i in range(len(all_with_label)):
        all_with_label[i] = (all_with_label[i][0], all_with_label[i][1], all_with_label[i][2], i)
    return all_with_label

def mix_test_data(path,attack='none'):
    """
    Create a mixed dataset of humans and language models.
    Args:
        path (str): The path to the directory containing the data.
        attack (str): The type of attack to be used.
        Default is 'none'.
    Returns:
        list: A list of tuples containing mixed data.
    """
    humans = load_pkl(os.path.join(path,'common/test/test_humans.pkl'))
    lms_outfox = load_pkl(os.path.join(path,'chatgpt/test/test_outfox_attacks.pkl'))
    lms_dipper = load_pkl(os.path.join(path,'dipper/chatgpt/test_attacks.pkl'))
    lms_chatgpt = load_pkl(os.path.join(path,'chatgpt/test/test_lms.pkl'))
    humans_with_label = [(human, '1', 'human') for human in humans]
    lms_outfox_with_label = [(lm, '0', 'outfox') for lm in lms_outfox]
    lms_dipper_with_label = [(lm, '0', 'dipper') for lm in lms_dipper]
    lms_chatgpt_with_label = [(lm, '0', 'chatgpt') for lm in lms_chatgpt]
    if attack=='none':
        lms_with_label = lms_chatgpt_with_label
    elif attack=='outfox':
        lms_with_label = lms_outfox_with_label
    elif attack=='dipper':
        lms_with_label = lms_dipper_with_label
    all_with_label = humans_with_label + lms_with_label
    for i in range(len(all_with_label)):
        all_with_label[i] = (all_with_label[i][0], all_with_label[i][1], all_with_label[i][2], i)
    return all_with_label

def load_OUTFOX(datapath,attack='none'):
    """
    Load the OUTFOX dataset.
    Args:
        datapath (str): The path to the directory containing the data.
        attack (str): The type of attack to be used.
        Default is 'none'.
    Returns:
        dict: A dictionary containing the train and test datasets.
    """
    train_passages = mix_train_data(datapath)
    test_passages = mix_test_data(datapath,attack)
    data_dict = {'train': train_passages, 'test': test_passages}
    return data_dict