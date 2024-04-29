from google.cloud import storage
from google.cloud.storage import Client, transfer_manager
import math

def calculate_training_steps(pref_dataset_size,prompt_dataset_size,reward_batch_size,rl_batch_size,reward_no_of_epochs,rl_no_of_epochs):

    if reward_no_of_epochs!='':
        reward_steps_per_epoch = math.ceil(pref_dataset_size / reward_batch_size)
        # Calculate number of steps in the reward model training
        reward_model_train_steps = reward_steps_per_epoch * reward_no_of_epochs
    else:
        reward_model_train_steps=''

    if rl_no_of_epochs!='':
        rl_steps_per_epoch = math.ceil(prompt_dataset_size / rl_batch_size)
        # Calculate the number of steps in the RL training
        reinforcement_learning_train_steps = rl_steps_per_epoch * rl_no_of_epochs
    else:
        reinforcement_learning_train_steps=''

    return reward_model_train_steps,reinforcement_learning_train_steps
    


def create_bucket_class_location(bucket_name,region):
    """Create a new bucket in the US region with the coldline storage
    class
    """

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    bucket.storage_class = "Standard"
    new_bucket = storage_client.create_bucket(bucket, location=region)

    print(
        "Created bucket {} in {} with storage class {}".format(
            new_bucket.name, new_bucket.location, new_bucket.storage_class
        )
    )
    return new_bucket



def upload_datasets_to_gcs(bucket_name, filenames, source_directory="", workers=8):

    workers=8

    storage_client = Client()
    bucket = storage_client.bucket(bucket_name)

    results = transfer_manager.upload_many_from_filenames(
        bucket, filenames, source_directory=source_directory, max_workers=workers
    )

    for name, result in zip(filenames, results):
        # The results list is either `None` or an exception for each filename in
        # the input list, in order.

        if isinstance(result, Exception):
            print("Failed to upload {} due to exception: {}".format(name, result))
        else:
            print("Uploaded {} to {}.".format(name, bucket.name))
  

# Function to print the information in the prompt dataset with a better visualization
def print_d(d):
    for key, val in d.items():        
        print(f"key:{key}\nval:{val}\n")


