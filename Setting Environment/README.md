<h1 align = center>Setting Environment</h1>

## Prerequisites

  - Windows 7 and greater; Windows 10 or greater recommended.
  - Windows Server 2008 r2 and greater.

## Pytorch Installation


 - Installing pytorch in Windows + Cuda.<br>

 *run the following line of code at anaconda prompt*

 **Creating environment**
```
conda create --name pytorch
```
**Activating environment**
```
$ conda activate pytorch
```
**Installing pytorch  in Windows + CUDA 11.7**
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

- Cheking installation

**From the command line, type:**

```
python
```
**then enter the following code:**
```
import torch
x = torch.rand(5, 3)
print(x)
```
**The output should be something similar to:**
```
tensor([[0.6565, 0.7372, 0.2125],
        [0.8420, 0.2653, 0.0788],
        [0.6932, 0.0910, 0.4680],
        [0.9644, 0.2188, 0.5525],
        [0.4070, 0.0107, 0.3263]])
```
**to check if the GPU driver and CUDA is enabled and accessible by PyTorch, run the following commands to return whether or not the CUDA driver is enabled:**

```
import torch
torch.cuda.is_available()
```
**The output should be something similar to:**
```
<function is_available at 0x00000200CB877BE0>
```
**Install Jupyter notebook, type:**

```
install jupyter notebook
```
**install Kernel**

```
conda install ipykernel
```

**display pytorch in Jupyter notebook, type:**

```
python -m ipykernel install --user --name pytorch --display "pytorch"
```
**Open Jupyter Notebook
```
jupyter notebook
```


