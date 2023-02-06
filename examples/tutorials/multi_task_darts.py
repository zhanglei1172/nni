"""
Searching in DARTS search space
===============================

In this tutorial, we demonstrate how to search in the famous model space proposed in `DARTS`_.

Through this process, you will learn:

* How to use the built-in model spaces from NNI's model space hub.
* How to use one-shot exploration strategies to explore a model space.
* How to customize evaluators to achieve the best performance.

In the end, we get a strong-performing model on CIFAR-10 dataset, which achieves up to 97.28% accuracy.

.. attention::

   Running this tutorial requires a GPU.
   If you don't have one, you can set ``gpus`` in :class:`~nni.retiarii.evaluator.pytorch.Classification` to be 0,
   but do note that it will be much slower.

.. _DARTS: https://arxiv.org/abs/1806.09055

Use a pre-searched DARTS model
------------------------------

Similar to `the beginner tutorial of PyTorch <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`__,
we begin with CIFAR-10 dataset, which is a image classification dataset of 10 categories.
The images in CIFAR-10 are of size 3x32x32, i.e., RGB-colored images of 32x32 pixels in size.

We first load the CIFAR-10 dataset with torchvision.
"""

import io
import warnings
import graphviz
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from typing import Any, Dict, Iterable, Tuple, Union, Optional, List, Type, cast
from torchvision import transforms
import torch.nn
from torchvision.datasets import CIFAR10
from torch.utils.data import SubsetRandomSampler
import torch.nn.functional as nn_functional
import torch.optim as optim
import torchmetrics
import torchmetrics.classification
import torchmetrics.regression

import nni
import nni.nas.nn.pytorch as nn
from nni.nas.utils.serializer import model_wrapper
from nni.retiarii.evaluator.pytorch import Lightning, Trainer
from nni.nas.evaluator.pytorch.lightning import SupervisedLearningModule
from nni.retiarii.strategy import DARTS as DartsStrategy
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from nni.retiarii.nn.pytorch import CrossEntropyLoss, MSELoss
from nni.retiarii.evaluator.pytorch import DataLoader
from nni.retiarii import fixed_arch
from nni.retiarii.hub.pytorch import DARTS as DartsSpace
from nni.retiarii.hub.pytorch.nasnet import AuxiliaryHead, CellBuilder, NDSStage


CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]



# %% space define  

class MT_AuxiliaryHead(AuxiliaryHead):
    def __init__(self, C: int, num_labels: int, dataset: Literal['imagenet', 'cifar']):
        super().__init__(C, num_labels, dataset)
        self.reg = nn.Linear(768, 1)

    def forward(self, x):
        x = self.features(x)
        x_cls = self.classifier(x.view(x.size(0), -1))
        x_reg = self.reg(x.view(x.size(0), -1))
        return x_cls, x_reg

@model_wrapper
class MT_DartsSpace(DartsSpace):
    def __init__(self,
                 width: Union[Tuple[int, ...], int] = (16, 24, 32),
                 num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
                 dataset: Literal['cifar', 'imagenet'] = 'cifar',
                 auxiliary_loss: bool = False,
                 drop_path_prob: float = 0.):
        nn.Module.__init__(self)

        op_candidates = self.DARTS_OPS
        merge_op='all'
        num_nodes_per_cell=4
        self.dataset = dataset
        self.num_labels = 10 if dataset == 'cifar' else 1000
        self.auxiliary_loss = auxiliary_loss
        self.drop_path_prob = drop_path_prob

        # preprocess the specified width and depth
        if isinstance(width, Iterable):
            C = nn.ValueChoice(list(width), label='width')
        else:
            C = width

        self.num_cells: nn.MaybeChoice[int] = cast(int, num_cells)
        if isinstance(num_cells, Iterable):
            self.num_cells = nn.ValueChoice(list(num_cells), label='depth')
        num_cells_per_stage = [(i + 1) * self.num_cells // 3 - i * self.num_cells // 3 for i in range(3)]

        # auxiliary head is different for network targetted at different datasets
        if dataset == 'imagenet':
            self.stem0 = nn.Sequential(
                nn.Conv2d(3, cast(int, C // 2), kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(cast(int, C // 2)),
                nn.ReLU(inplace=True),
                nn.Conv2d(cast(int, C // 2), cast(int, C), 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C),
            )
            self.stem1 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(cast(int, C), cast(int, C), 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C),
            )
            C_pprev = C_prev = C_curr = C
            last_cell_reduce = True
        elif dataset == 'cifar':
            self.stem = nn.Sequential(
                nn.Conv2d(3, cast(int, 3 * C), 3, padding=1, bias=False),
                nn.BatchNorm2d(cast(int, 3 * C))
            )
            C_pprev = C_prev = 3 * C
            C_curr = C
            last_cell_reduce = False
        else:
            raise ValueError(f'Unsupported dataset: {dataset}')

        self.stages = nn.ModuleList()
        for stage_idx in range(3):
            if stage_idx > 0:
                C_curr *= 2
            # For a stage, we get C_in, C_curr, and C_out.
            # C_in is only used in the first cell.
            # C_curr is number of channels for each operator in current stage.
            # C_out is usually `C * num_nodes_per_cell` because of concat operator.
            cell_builder = CellBuilder(op_candidates, C_pprev, C_prev, C_curr, num_nodes_per_cell,
                                       merge_op, stage_idx > 0, last_cell_reduce, drop_path_prob)
            stage: Union[NDSStage, nn.Sequential] = NDSStage(cell_builder, num_cells_per_stage[stage_idx])

            if isinstance(stage, NDSStage):
                stage.estimated_out_channels_prev = cast(int, C_prev)
                stage.estimated_out_channels = cast(int, C_curr * num_nodes_per_cell)
                stage.downsampling = stage_idx > 0

            self.stages.append(stage)

            # NOTE: output_node_indices will be computed on-the-fly in trial code.
            # When constructing model space, it's just all the nodes in the cell,
            # which happens to be the case of one-shot supernet.

            # C_pprev is output channel number of last second cell among all the cells already built.
            if len(stage) > 1:
                # Contains more than one cell
                C_pprev = len(cast(nn.Cell, stage[-2]).output_node_indices) * C_curr
            else:
                # Look up in the out channels of last stage.
                C_pprev = C_prev

            # This was originally,
            # C_prev = num_nodes_per_cell * C_curr.
            # but due to loose end, it becomes,
            C_prev = len(cast(nn.Cell, stage[-1]).output_node_indices) * C_curr

            # Useful in aligning the pprev and prev cell.
            last_cell_reduce = cell_builder.last_cell_reduce

            if stage_idx == 2:
                C_to_auxiliary = C_prev

        if auxiliary_loss:
            assert isinstance(self.stages[2], nn.Sequential), 'Auxiliary loss can only be enabled in retrain mode.'
            self.auxiliary_head = MT_AuxiliaryHead(C_to_auxiliary, self.num_labels, dataset=self.dataset)  # type: ignore

        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(cast(int, C_prev), self.num_labels)
        # C_prev = self.classifier.in_features
        self.regressor = nn.Linear(cast(int, C_prev), 1)
    
    def forward(self, inputs):
        if self.dataset == 'imagenet':
            s0 = self.stem0(inputs)
            s1 = self.stem1(s0)
        else:
            s0 = s1 = self.stem(inputs)

        for stage_idx, stage in enumerate(self.stages):
            if stage_idx == 2 and self.auxiliary_loss and self.training:
                assert isinstance(stage, nn.Sequential), 'Auxiliary loss is only supported for fixed architecture.'
                for block_idx, block in enumerate(stage):
                    # auxiliary loss is attached to the first cell of the last stage.
                    s0, s1 = block([s0, s1])
                    if block_idx == 0:
                        logits_aux = self.auxiliary_head(s1)
            else:
                s0, s1 = stage([s0, s1])

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        reg_out = self.regressor(out.view(out.size(0), -1))
        if self.training and self.auxiliary_loss:
            return (logits, reg_out), logits_aux  # type: ignore
        else:
            return (logits, reg_out)



# search

model_space = MT_DartsSpace(
    width=16,           # the initial filters (channel number) for the model
    num_cells=8,        # the number of stacked cells in total
    dataset='cifar'     # to give a hint about input resolution, here is 32x32
)



fast_dev_run = True



transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

mt_transform = lambda x: (x, torch.tensor(np.random.randn(), dtype=torch.float32)) # generate multi task label

# %% define MT loss

class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.cls_loss = CrossEntropyLoss()
        self.reg_loss = MSELoss()

    def forward(self, predictions, targets):
        # Implement your custom loss calculation here
        # ...
        loss = self.cls_loss(predictions[0], targets[0]) + self.reg_loss(
            predictions[1], targets[1])
        return loss


# %% define MT metric

class _AccuracyWithLogits(torchmetrics.Accuracy):
    # Only for torchmetrics < 0.11
    def update(self, preds, targets):
        return super().update(nn_functional.softmax(preds[0], dim=-1), targets[0])  # type: ignore

class _R2WithInputs(torchmetrics.R2Score):
    # Only for torchmetrics < 0.11
    def update(self, preds, targets):
        return super().update(preds[1], targets[1].view(*preds[1].shape))  # type: ignore

# %%

train_data = nni.trace(CIFAR10)(root='./data', train=True, download=True, transform=transform, target_transform=mt_transform)

num_samples = len(train_data)
indices = np.random.permutation(num_samples)
split = num_samples // 2

search_train_loader = DataLoader(
    train_data, batch_size=64, num_workers=0,
    sampler=SubsetRandomSampler(indices[:split]),
)

search_valid_loader = DataLoader(
    train_data, batch_size=64, num_workers=0,
    sampler=SubsetRandomSampler(indices[split:]),
)


# %% define MT evaluator

@nni.trace
class MT_Module(SupervisedLearningModule):
    def __init__(self, criterion: Type[nn.Module] = nn.CrossEntropyLoss,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 export_onnx: bool = False,
                 num_classes: Optional[int] = None):

        from packaging.version import Version
        if Version(torchmetrics.__version__) < Version('0.11.0'):
            # Older version accepts num_classes = None
            metrics = {'acc': _AccuracyWithLogits(),'R2': _R2WithInputs()}  # type: ignore # pylint: disable=no-value-for-parameter
        else:
            if num_classes is None:
                raise ValueError('num_classes must be specified for torchmetrics >= 0.11. '
                                 'Please either specify it or use an older version of torchmetrics.')
            metrics = {'acc': torchmetrics.Accuracy('multiclass', num_classes=num_classes),'R2': torchmetrics.R2Score()}

        super().__init__(criterion, metrics,  # type: ignore
                         learning_rate=learning_rate, weight_decay=weight_decay, optimizer=optimizer,
                         export_onnx=export_onnx)
        
@nni.trace
class Multi_task(Lightning):
    """
    Evaluator that is used for classification.

    Available callback metrics in :class:`Classification` are:

    - train_loss
    - train_acc
    - val_loss
    - val_acc

    Parameters
    ----------
    criterion : nn.Module
        Class for criterion module (not an instance). default: ``nn.CrossEntropyLoss``
    learning_rate : float
        Learning rate. default: 0.001
    weight_decay : float
        L2 weight decay. default: 0
    optimizer : Optimizer
        Class for optimizer (not an instance). default: ``Adam``
    train_dataloaders : DataLoader
        Used in ``trainer.fit()``. A PyTorch DataLoader with training samples.
        If the ``lightning_module`` has a predefined train_dataloader method this will be skipped.
    val_dataloaders : DataLoader or List of DataLoader
        Used in ``trainer.fit()``. Either a single PyTorch Dataloader or a list of them, specifying validation samples.
        If the ``lightning_module`` has a predefined val_dataloaders method this will be skipped.
    export_onnx : bool
        If true, model will be exported to ``model.onnx`` before training starts. default true
    num_classes : int
        Number of classes for classification task.
        Required for torchmetrics >= 0.11.0. default: None
    trainer_kwargs : dict
        Optional keyword arguments passed to trainer. See
        `Lightning documentation <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`__ for details.

    Examples
    --------
    >>> evaluator = Classification()

    To use customized criterion and optimizer:

    >>> evaluator = Classification(nn.LabelSmoothingCrossEntropy, optimizer=torch.optim.SGD)

    Extra keyword arguments will be passed to trainer, some of which might be necessary to enable GPU acceleration:

    >>> evaluator = Classification(accelerator='gpu', devices=2, strategy='ddp')
    """

    def __init__(self, criterion: Type[nn.Module] = nn.CrossEntropyLoss,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 train_dataloaders: Optional[DataLoader] = None,
                 val_dataloaders: Union[DataLoader, List[DataLoader], None] = None,
                 export_onnx: bool = False,
                 train_dataloader: Optional[DataLoader] = None,
                 num_classes: Optional[int] = None,
                 **trainer_kwargs):
        if train_dataloader is not None:
            warnings.warn('`train_dataloader` is deprecated and replaced with `train_dataloaders`.', DeprecationWarning)
            train_dataloaders = train_dataloader
        module = MT_Module(criterion=criterion, learning_rate=learning_rate,
                                      weight_decay=weight_decay, optimizer=optimizer, export_onnx=export_onnx,
                                      num_classes=num_classes)
        super().__init__(module, Trainer(**trainer_kwargs),
                         train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)

evaluator = Multi_task(
    criterion=CustomLoss,
    learning_rate=1e-3,
    weight_decay=1e-4,
    train_dataloaders=search_train_loader,
    val_dataloaders=search_valid_loader,
    max_epochs=10,
    gpus=1,
    fast_dev_run=fast_dev_run,
)

# %% search


strategy = DartsStrategy()




config = RetiariiExeConfig(execution_engine='oneshot')
experiment = RetiariiExperiment(model_space, evaluator=evaluator, strategy=strategy)
experiment.run(config)


exported_arch = experiment.export_top_models()[0]

exported_arch



def plot_single_cell(arch_dict, cell_name):
    g = graphviz.Digraph(
        node_attr=dict(style='filled', shape='rect', align='center'),
        format='png'
    )
    g.body.extend(['rankdir=LR'])

    g.node('c_{k-2}', fillcolor='darkseagreen2')
    g.node('c_{k-1}', fillcolor='darkseagreen2')
    assert len(arch_dict) % 2 == 0

    for i in range(2, 6):
        g.node(str(i), fillcolor='lightblue')

    for i in range(2, 6):
        for j in range(2):
            op = arch_dict[f'{cell_name}/op_{i}_{j}']
            from_ = arch_dict[f'{cell_name}/input_{i}_{j}']
            if from_ == 0:
                u = 'c_{k-2}'
            elif from_ == 1:
                u = 'c_{k-1}'
            else:
                u = str(from_)
            v = str(i)
            g.edge(u, v, label=op, fillcolor='gray')

    g.node('c_{k}', fillcolor='palegoldenrod')
    for i in range(2, 6):
        g.edge(str(i), 'c_{k}', fillcolor='gray')

    g.attr(label=f'{cell_name.capitalize()} cell')

    image = Image.open(io.BytesIO(g.pipe()))
    return image

def plot_double_cells(arch_dict):
    image1 = plot_single_cell(arch_dict, 'normal')
    image2 = plot_single_cell(arch_dict, 'reduce')
    height_ratio = max(image1.size[1] / image1.size[0], image2.size[1] / image2.size[0]) 
    _, axs = plt.subplots(1, 2, figsize=(20, 10 * height_ratio))
    axs[0].imshow(image1)
    axs[1].imshow(image2)
    axs[0].axis('off')
    axs[1].axis('off')
    plt.show()

plot_double_cells(exported_arch)

#  retrain searched model


with fixed_arch(exported_arch):
    final_model = MT_DartsSpace(width=16, num_cells=8, dataset='cifar')

# %% retrain

train_loader = DataLoader(train_data, batch_size=96, num_workers=6)  # Use the original training data


transform_valid = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])
valid_data = nni.trace(CIFAR10)(root='./data', train=False, download=True, transform=transform_valid, target_transform=mt_transform)
valid_loader = DataLoader(valid_data, batch_size=256, num_workers=6)

max_epochs = 100

evaluator = Multi_task(
    criterion=CustomLoss,
    learning_rate=1e-3,
    weight_decay=1e-4,
    train_dataloaders=train_loader,
    val_dataloaders=valid_loader,
    max_epochs=max_epochs,
    gpus=1,
    export_onnx=False,          # Disable ONNX export for this experiment
    fast_dev_run=fast_dev_run   # Should be false for fully training
)

evaluator.fit(final_model)


# %% Reproduce results in DARTS paper


class DartsMTModule(MT_Module):
    def __init__(
        self,
        criterion: Type[nn.Module] = nn.CrossEntropyLoss,
        learning_rate: float = 0.001,
        weight_decay: float = 0.,
        auxiliary_loss_weight: float = 0.4,
        max_epochs: int = 600
    ):
        self.auxiliary_loss_weight = auxiliary_loss_weight
        # Training length will be used in LR scheduler
        self.max_epochs = max_epochs
        super().__init__(criterion=criterion, learning_rate=learning_rate, weight_decay=weight_decay, export_onnx=False)

    def configure_optimizers(self):
        """Customized optimizer with momentum, as well as a scheduler."""
        optimizer = torch.optim.SGD(
            self.parameters(),
            momentum=0.9,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epochs, eta_min=1e-3)
        }

    def training_step(self, batch, batch_idx):
        """Training step, customized with auxiliary loss."""
        x, y = batch
        if self.auxiliary_loss_weight:
            y_hat, y_aux = self(x)
            loss_main = self.criterion(y_hat, y)
            loss_aux = self.criterion(y_aux, y)
            self.log('train_loss_main', loss_main)
            self.log('train_loss_aux', loss_aux)
            loss = loss_main + self.auxiliary_loss_weight * loss_aux
        else:
            y_hat = self(x)
            loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        for name, metric in self.metrics.items():
            self.log('train_' + name, metric(y_hat, y), prog_bar=True)
        return loss

    def on_train_epoch_start(self):
        # Set drop path probability before every epoch. This has no effect if drop path is not enabled in model.
        self.model.set_drop_path_prob(self.model.drop_path_prob * self.current_epoch / self.max_epochs)

        # Logging learning rate at the beginning of every epoch
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])



max_epochs = 50

evaluator = Lightning(
    DartsMTModule(CustomLoss, 0.025, 3e-4, 0., max_epochs),
    Trainer(
        gpus=1,
        max_epochs=max_epochs,
        fast_dev_run=fast_dev_run,
    ),
    train_dataloaders=search_train_loader,
    val_dataloaders=search_valid_loader
)



strategy = DartsStrategy(gradient_clip_val=5.)



model_space = MT_DartsSpace(width=16, num_cells=8, dataset='cifar')

config = RetiariiExeConfig(execution_engine='oneshot')
experiment = RetiariiExperiment(model_space, evaluator=evaluator, strategy=strategy)
experiment.run(config)

exported_arch = experiment.export_top_models()[0]

exported_arch


def cutout_transform(img, length: int = 16):
    h, w = img.size(1), img.size(2)
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = np.clip(y - length // 2, 0, h)
    y2 = np.clip(y + length // 2, 0, h)
    x1 = np.clip(x - length // 2, 0, w)
    x2 = np.clip(x + length // 2, 0, w)

    mask[y1: y2, x1: x2] = 0.
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img *= mask
    return img

transform_with_cutout = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    cutout_transform,
])



train_data_cutout = nni.trace(CIFAR10)(root='./data', train=True, download=True, transform=transform_with_cutout, target_transform=mt_transform)
train_loader_cutout = DataLoader(train_data_cutout, batch_size=96)



with fixed_arch(exported_arch):
    final_model = MT_DartsSpace(width=36, num_cells=20, dataset='cifar', auxiliary_loss=True, drop_path_prob=0.2)



max_epochs = 600

evaluator = Lightning(
    DartsMTModule(CustomLoss, 0.025, 3e-4, 0.4, max_epochs),
    trainer=Trainer(
        gpus=1,
        gradient_clip_val=5.,
        max_epochs=max_epochs,
        fast_dev_run=fast_dev_run
    ),
    train_dataloaders=train_loader_cutout,
    val_dataloaders=valid_loader,
)

evaluator.fit(final_model)
