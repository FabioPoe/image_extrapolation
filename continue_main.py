#this file aims to further train the model which i already trained and saved
from architectures import *
from datasets import *
from utils import *
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import json

#getting the configurations from the config file
with open('working_config.json', 'r') as f:
    config = json.load(f)


# reading in 1 pic
import os
full_filenames = []
for folder in os.listdir(config["dataset_path"]):
    filenames = os.listdir(os.path.join(config["dataset_path"],folder))
    for file in filenames:
        full_filenames.append(os.path.join(config["dataset_path"],folder,file))



dataset = RandomSeqDataset(full_filenames)
n_samples = len(dataset)

shuffled_indices = np.random.permutation(n_samples)
testset_inds = shuffled_indices[:int(n_samples / 10)]
validationset_inds = shuffled_indices[int(n_samples / 10):int(n_samples / 10) * 2]
trainingset_inds = shuffled_indices[int(n_samples / 10) * 2:]

testset = Subset(dataset, indices=testset_inds)
validationset = Subset(dataset, indices=validationset_inds)
trainingset = Subset(dataset, indices=trainingset_inds)

# Create dataloaders from each subset
test_loader = DataLoader(testset,  # we want to load our dataset
                         shuffle=False,  # shuffle for training
                         batch_size=1,  # 1 sample at a time
                         num_workers=0,
                         collate_fn=stack_if_possible_collate_fn  # no background workers
                         )
validation_loader = DataLoader(validationset,  # we want to load our dataset
                               shuffle=False,  # shuffle for training
                               batch_size=4,  # stack 4 samples to a minibatch
                               num_workers=0,
                               collate_fn=stack_if_possible_collate_fn  # 2 background workers
                               )
training_loader = DataLoader(trainingset,  # we want to load our dataset
                             shuffle=False,  # shuffle for training
                             batch_size=4,  # stack 4 samples to a minibatch
                             num_workers=0,
                             collate_fn=stack_if_possible_collate_fn  # 2 background workers
                             )

#claring some folders
for key in config["to_clear"]:
    clear_folder(config["to_clear"][key])


# Create an instance of our CNN
cnn = torch.load(os.path.join("results", "my_model_A.pt"))

# GPU will be much faster here
device = torch.device(config["device"])
cnn.to(device=device)

# also try rmse and absolute error
loss_function = torch.nn.MSELoss(reduction="mean")

# Use a SGD optimizer
# optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01)
optimizer = torch.optim.Adam(cnn.parameters(), lr=config["learningrate"])

# Define a tensorboard summary writer that writes to directory "results_path/tensorboard"
writer = SummaryWriter(log_dir=os.path.join(config["results_path"], 'tensorboard'), flush_secs=1)

best_loss = np.float("inf")
# Optimize our dsnn model using SGD:
print("Continuing the training:")

# for update in tqdm(range(200)):
with tqdm(total=config["n_updates"]) as pbar:
    for update in range(config["n_updates"]):
        x = 0
        for data in training_loader:
            # Compute the output
            input_array, known_array, target_array, sample_id, original = data
            # ic(input_array.unsqueeze(1).type(torch.float32).shape)
            output = cnn(input_array.unsqueeze(1).type(torch.float32))

            # Compute the loss
            loss = loss_function(output, original.type(torch.float32).unsqueeze(1))
            # Compute the gradients
            loss.backward()
            # Preform the update
            optimizer.step()
            # Reset the accumulated gradients
            optimizer.zero_grad()
            x += 1
            if x == 10:
                break

        # automatic saving of the best model
        if update % 5 == 0:
            y = 0
            sum_loss = 0
            for tdata in test_loader:
                input_array, known_array, target_array, sample_id, original = data
                output = cnn(input_array.unsqueeze(1).type(torch.float32))
                loss = loss_function(output, original.type(torch.float32).unsqueeze(1))
                sum_loss += loss
                y += 1
                if y == 50:
                    break

            if sum_loss < best_loss:
                torch.save(cnn, os.path.join(config["results_path"], "autosave", f"loss_{sum_loss}.pt"))
                best_loss = loss

        # Tensorboard
        if update % 1 == 0:
            # Add losse as scalars to tensorboard
            writer.add_scalar(tag="training/loss", scalar_value=sum_loss,
                              global_step=update)

            # Add images to Tensorboard
            writer.add_image(tag="training/output", img_tensor=output[0], global_step=update)

            writer.add_image(tag="training/input", img_tensor=(original.unsqueeze(1)[0]), global_step=update)

        torch.save(cnn, os.path.join(config["results_path"], "my_model_A.pt"))
        pbar.update(1)

print("training finished")
#torch.save(cnn, os.path.join(config["results_path"], "my_model_A.pt"))
print("model saved")

r"""note to myself:
activate tensorboard like this:

activate programing_in_python_2
tensorboard --logdir=C:\Users\fabio\Google_Drive\AI_Studium\programing_python\ass2\ex5\results\tensorboard

http://localhost:6006/"""
