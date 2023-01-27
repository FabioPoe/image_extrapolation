"""
Author: Fabio Pöschko
Matr.Nr.: K11905017
Exercise 5
"""

from architectures import *
from dataset_numpy import *
from utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import json

#!!!!Disclaimer!!!!
# Before executing this i manually executed the file: "data_preprocessing.py", i decided to preprocess the data
#to make the training faster


#getting the configurations from the config file
with open('working_config.json', 'r') as f:
    config = json.load(f)

#creating the dataset
testset = RandomSeqDataset(r"D:\studium_tests\small_original_dataset2.npz",config["testset_key"],cuda=False)
trainset = RandomSeqDataset(r"D:\studium_tests\small_original_dataset2.npz",config["trainset_key"],cuda = False)


# Create dataloaders
test_loader = DataLoader(testset,  # we want to load our dataset
                         shuffle=False,  # shuffle for training
                         batch_size=1,  # 1 sample at a time
                         num_workers=0,
                         collate_fn=stack_if_possible_collate_fn  # no background workers
                         )

training_loader = DataLoader(trainset,  # we want to load our dataset
                             shuffle=False,  # shuffle for training
                             batch_size=4,  # stack 4 samples to a minibatch
                             num_workers=0,
                             collate_fn=stack_if_possible_collate_fn  # 2 background workers
                             )

#claring some folders, this is important if i execute the code more than once
#this clears f.e. the tensorboard folder
for key in config["to_clear"]:
    clear_folder(config["to_clear"][key])

#creating the net, this is only executed the first time i train, when i just want to continue training,
# because i had to interrupt it, then i comment it out
net = CNN(n_input_channels=config["network_config"]["n_input_channels"], n_hidden_layers=config["network_config"]["n_hidden_layers"], n_hidden_kernels=config["network_config"]["n_hidden_kernels"],
          n_output_channels=config["network_config"]["n_output_channels"])
torch.save(net, os.path.join(config["results_path"], "my_pc_cnn.pt"))

# loading the net
net = torch.load(os.path.join(config["results_path"], "my_pc_cnn.pt"))
#taking a look at the net
print(net)

# gpu is probably much faster, but i didnt manage to make mine work
device = torch.device(config["device"])
net.to(device=device)

# defining the loss
loss_function = torch.nn.MSELoss(reduction="mean")

# defining an optimizer, i tried SGD and Adam
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

# Define a tensorboard summary writer that writes to directory "results_path/tensorboard"
writer = SummaryWriter(log_dir=os.path.join(config["results_path"], 'tensorboard'), flush_secs=1)

#this best loss is used for autosaving
best_loss = np.float("inf")
with tqdm(total=config["n_updates"]) as pbar:
    for update in range(config["n_updates"]):
        x = 0
        for data in training_loader:
            # Compute the output
            input_array, original = data

            # ic(input_array.unsqueeze(1).type(torch.float32).shape)
            output = net(input_array.unsqueeze(1).type(torch.float32))

            # Compute the loss
            loss = loss_function(output, original.type(torch.float32).unsqueeze(1))
            # Compute the gradients
            loss.backward()
            # Preform the update
            optimizer.step()
            # Reset the accumulated gradients
            optimizer.zero_grad()

            # taking a look at the first iterations
            x += 1
            if x == 10:
                break

        # automatic saving of the best model
        if update % 1 == 0:
            with torch.no_grad():
                y = 0
                sum_loss = 0
                for tdata in test_loader:
                    input_array, original = data

                    output = net(input_array.unsqueeze(1).type(torch.float32))
                    sum_loss += loss_function(output, original.type(torch.float32).unsqueeze(1))

                    #taking a look at the first iterations
                    y += 1
                    if y == 1:
                        print(sum_loss)
                        break

            if sum_loss < best_loss:
                torch.save(net, os.path.join(config["results_path"], "autosave", f"loss_{sum_loss}.pt"))
                best_loss = loss

        # Tensorboard
        if update % 1 == 0:
            # Add losse as scalars to tensorboard
            writer.add_scalar(tag="training/loss", scalar_value=sum_loss,
                              global_step=update)

            # Add images to Tensorboard
            writer.add_image(tag="training/output", img_tensor=output[0], global_step=update)

            writer.add_image(tag="training/input", img_tensor=(original.unsqueeze(1)[0]), global_step=update)

        torch.save(net, os.path.join(config["results_path"], "my_pc_cnn.pt"))
        pbar.update(1)

print("training finished")

#!!!!Disclaimer!!!!
# after finishing the training, i executed: "creating_predictions.py" to create the predictions i submitted



r"""note to myself:
activate tensorboard like this:

activate programing_in_python_2
tensorboard --logdir=C:\Users\fabio\Google_Drive\AI_Studium\programing_python\ass2\ex5\results\tensorboard

http://localhost:6006/"""
