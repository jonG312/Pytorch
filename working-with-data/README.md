PyTorch has two primitives to work with data: torch.utils.data.DataLoader and torch.utils.data.Dataset. Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset.

PyTorch offers domain-specific libraries such as TorchText, TorchVision, and TorchAudio, all of which include datasets.

```
import torch

from torch import nn

from torch.utils.data import DataLoader

from torchvision import datasets

from torchvision.transforms import ToTensor
```

The torchvision.datasets module contains Dataset objects for many real-world vision data like CIFAR, COCO (full list here). in this example is implement the FGVCAircraft dataset. Every TorchVision Dataset includes two arguments: transform and target_transform to modify the samples and labels respectively.

# Download training data from open datasets.

```
`training_data = datasets.FashionMNIST(

    root="data",

    train=True,

    download=True,

    transform=ToTensor(),

)
```
​

# Download test data from open datasets.


```
test_data = datasets.FashionMNIST(

    root="data",

    train=False,

    download=True,

    transform=ToTensor(),

)
```

Output:

```
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data\FashionMNIST\raw\train-images-idx3-ubyte.gz

  0%|          | 0/26421880 [00:00<?, ?it/s]

Extracting data\FashionMNIST\raw\train-images-idx3-ubyte.gz to data\FashionMNIST\raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data\FashionMNIST\raw\train-labels-idx1-ubyte.gz

  0%|          | 0/29515 [00:00<?, ?it/s]

Extracting data\FashionMNIST\raw\train-labels-idx1-ubyte.gz to data\FashionMNIST\raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data\FashionMNIST\raw\t10k-images-idx3-ubyte.gz

  0%|          | 0/4422102 [00:00<?, ?it/s]

Extracting data\FashionMNIST\raw\t10k-images-idx3-ubyte.gz to data\FashionMNIST\raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data\FashionMNIST\raw\t10k-labels-idx1-ubyte.gz

  0%|          | 0/5148 [00:00<?, ?it/s]

Extracting data\FashionMNIST\raw\t10k-labels-idx1-ubyte.gz to data\FashionMNIST\raw
```

We pass the Dataset as an argument to DataLoader. This wraps an iterable over our dataset, and supports automatic batching, sampling, shuffling and multiprocess data loading. Here we define a batch size of 64, i.e. each element in the dataloader iterable will return a batch of 64 features and labels.

```
batch_size = 64

# Create data loaders.

train_dataloader = DataLoader(training_data, batch_size=batch_size)

test_dataloader = DataLoader(test_data, batch_size=batch_size)
```

```
for X, y in test_dataloader:

    print(f"Shape of X [N, C, H, W]: {X.shape}")

    print(f"Shape of y: {y.shape} {y.dtype}")

    break
```
Output:

```
Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
Shape of y: torch.Size([64]) torch.int64
```

#Creating Models

To define a neural network in PyTorch, we create a class that inherits from nn.Module. We define the layers of the network in the init function and specify how data will pass through the network in the forward function. To accelerate operations in the neural network, we move it to the GPU or MPS if available.

```
# Get cpu, gpu or mps device for training.


device = (

    "cuda"

    if torch.cuda.is_available()

    else "mps"

    if torch.backends.mps.is_available()

    else "cpu"

)

print(f"Using {device} device")

# Define model


class NeuralNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(

            nn.Linear(28*28, 512),

            nn.ReLU(),

            nn.Linear(512, 512),

            nn.ReLU(),

            nn.Linear(512, 10)

        )

    def forward(self, x):

        x = self.flatten(x)

        logits = self.linear_relu_stack(x)

        return logits​

model = NeuralNetwork().to(device)

print(model)
```
output:

```
Using cuda device
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
```

#Optimizing the Model Parameters

To train a model, we need a loss function and an optimizer.
```

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```

In a single training loop, the model makes predictions on the training dataset (fed to it in batches), and backpropagates the prediction error to adjust the model’s parameters.

def train(dataloader, model, loss_fn, optimizer):
```
    size = len(dataloader.dataset)

    model.train()

    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)

​

        # Compute prediction error

        pred = model(X)

        loss = loss_fn(pred, y)

​

        # Backpropagation

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

​

        if batch % 100 == 0:

            loss, current = loss.item(), (batch + 1) * len(X)

            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

We also check the model’s performance against the test dataset to ensure it is learning.
```

def test(dataloader, model, loss_fn):

    size = len(dataloader.dataset)

    num_batches = len(dataloader)

    model.eval()

    test_loss, correct = 0, 0

    with torch.no_grad():

        for X, y in dataloader:

            X, y = X.to(device), y.to(device)

            pred = model(X)

            test_loss += loss_fn(pred, y).item()

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches

    correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

The training process is conducted over several iterations (epochs). During each epoch, the model learns parameters to make better predictions. We print the model’s accuracy and loss at each epoch; we’d like to see the accuracy increase and the loss decrease with every epoch.

```

epochs = 5

for t in range(epochs):

    print(f"Epoch {t+1}\n-------------------------------")

    train(train_dataloader, model, loss_fn, optimizer)

    test(test_dataloader, model, loss_fn)

print("Done!")

```
output:

```
Epoch 1
-------------------------------
loss: 1.138490  [   64/60000]
loss: 1.135146  [ 6464/60000]
loss: 0.961674  [12864/60000]
loss: 1.098670  [19264/60000]
loss: 0.971702  [25664/60000]
loss: 0.998448  [32064/60000]
loss: 1.041859  [38464/60000]
loss: 0.991777  [44864/60000]
loss: 1.031904  [51264/60000]
loss: 0.952392  [57664/60000]
Test Error: 
 Accuracy: 66.2%, Avg loss: 0.968153 

Epoch 2
-------------------------------
loss: 1.018221  [   64/60000]
loss: 1.036232  [ 6464/60000]
loss: 0.847323  [12864/60000]
loss: 1.007006  [19264/60000]
loss: 0.884415  [25664/60000]
loss: 0.907204  [32064/60000]
loss: 0.965358  [38464/60000]
loss: 0.920192  [44864/60000]
loss: 0.956015  [51264/60000]
loss: 0.887610  [57664/60000]
Test Error: 
 Accuracy: 67.3%, Avg loss: 0.897317 

Epoch 3
-------------------------------
loss: 0.930661  [   64/60000]
loss: 0.968605  [ 6464/60000]
loss: 0.766641  [12864/60000]
loss: 0.943508  [19264/60000]
loss: 0.827224  [25664/60000]
loss: 0.842010  [32064/60000]
loss: 0.911448  [38464/60000]
loss: 0.872818  [44864/60000]
loss: 0.901815  [51264/60000]
loss: 0.841978  [57664/60000]
Test Error: 
 Accuracy: 68.4%, Avg loss: 0.846972 

Epoch 4
-------------------------------
loss: 0.864001  [   64/60000]
loss: 0.919024  [ 6464/60000]
loss: 0.707034  [12864/60000]
loss: 0.897042  [19264/60000]
loss: 0.787120  [25664/60000]
loss: 0.793601  [32064/60000]
loss: 0.870726  [38464/60000]
loss: 0.840002  [44864/60000]
loss: 0.861512  [51264/60000]
loss: 0.807451  [57664/60000]
Test Error: 
 Accuracy: 69.8%, Avg loss: 0.809172 

Epoch 5
-------------------------------
loss: 0.811093  [   64/60000]
loss: 0.879912  [ 6464/60000]
loss: 0.660973  [12864/60000]
loss: 0.861724  [19264/60000]
loss: 0.757106  [25664/60000]
loss: 0.756420  [32064/60000]
loss: 0.838164  [38464/60000]
loss: 0.815785  [44864/60000]
loss: 0.830085  [51264/60000]
loss: 0.779859  [57664/60000]
Test Error: 
 Accuracy: 71.2%, Avg loss: 0.779233 

Done!
```

#Saving Models


```
torch.save(model.state_dict(), "model.pth")

print("Saved PyTorch Model State to model.pth")

Saved PyTorch Model State to model.pth
```

#Loading Models

The process for loading a model includes re-creating the model structure and loading the state dictionary into it.

```
model = NeuralNetwork().to(device)

model.load_state_dict(torch.load("model.pth"))

<All keys matched successfully>

Using the models to make predictions.

classes = [

    "T-shirt/top",

    "Trouser",

    "Pullover",

    "Dress",

    "Coat",

    "Sandal",

    "Shirt",

    "Sneaker",

    "Bag",

    "Ankle boot",

]

​

model.eval()

x, y = test_data[0][0], test_data[0][1]

with torch.no_grad():

    x = x.to(device)

    pred = model(x)

    predicted, actual = classes[pred[0].argmax(0)], classes[y]

    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```
output:
```
Predicted: "Pullover", Actual: "Ankle boot"
```

