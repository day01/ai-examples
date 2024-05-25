import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import WeightedRandomSampler, DataLoader, TensorDataset

from config.polish_name import PolishNameConfig

EPOCH = 1_000
EPOCH_SEPARATOR = 100


class PolishNameModel:
    _xtrain_dataset: Tensor
    _ytrain_dataset: Tensor
    _train_sampler: WeightedRandomSampler

    _xdev_dataset: Tensor
    _ydev_dataset: Tensor
    _dev_sampler: WeightedRandomSampler

    _xtest_dataset: Tensor
    _ytest_dataset: Tensor
    _test_sampler: WeightedRandomSampler

    _vocab_size: int
    _dataset_initialized: bool = False
    _C: Tensor
    _W1: Tensor
    _b1: Tensor
    _W2: Tensor
    _b2: Tensor

    def __init__(self, config: PolishNameConfig):
        self.config = config

    @property
    def _parameters(self) -> list[Tensor]:
        return [self._C, self._W1, self._b1, self._W2, self._b2]

    def build_model(self, names: pd.DataFrame):
        self._prepare_dataset(names)
        self._prepare_parameters()

    def train_model(self):
        lossi = []
        stepi = []
        for i in range(EPOCH):
            train_loader = DataLoader(
                TensorDataset(self._xtrain_dataset, self._ytrain_dataset),
                batch_size=self.config.batch_size,
                sampler=self._train_sampler,
            )

            print(f"Epoch {i}/{EPOCH} training with {len(train_loader)} batches")
            for j, (x, y) in enumerate(train_loader):
                loss = self._calculate_loss(x, y)
                for p in self._parameters:
                    p.grad = None
                loss.backward()
                ix = (i * len(train_loader)) + j

                learning_rate = 10 ** -(int((i / EPOCH_SEPARATOR)) + 1)
                for p in self._parameters:
                    if p.grad is not None:
                        p.data += -learning_rate * p.grad
                    else:
                        print(f"What the fck {p}")
                stepi.append(ix)
                lossi.append(loss.log10().item())

        plt.plot(stepi, lossi)
        plt.show()
        self.train_loss()
        pass

    def _calculate_loss(self, x, y):
        emb = self._C[x]
        h = torch.tanh(
            emb.view(-1, self.config.block_size * self.config.embedding_vectors)
            @ self._W1
            + self._b1
        )
        logits = h @ self._W2 + self._b2
        return F.cross_entropy(logits, y)

    def test_loss(self):
        loss = self._calculate_loss(self._xtest_dataset, self._ytest_dataset)
        print(f"Test loss is {loss}")

    def train_loss(self):
        loss = self._calculate_loss(self._xtrain_dataset, self._ytrain_dataset)
        print(f"Train loss is {loss}")

    def dev_loss(self):
        loss = self._calculate_loss(self._xdev_dataset, self._ydev_dataset)
        print(f"Dev loss is {loss}")

    def _prepare_dataset(self, names: pd.DataFrame):
        additional_end_of_string = 1
        self.chars = sorted(list(set("".join(names.name))))
        self._vocab_size = len(self.chars) + additional_end_of_string
        self.stoi = {s: i + 1 for i, s in enumerate(self.chars)}
        self.stoi["."] = 0
        self.itos = {i: s for s, i in self.stoi.items()}

        names = names.sample(frac=1, random_state=self.config.seed).reset_index(
            drop=True
        )
        percentiles = names.occurrences.quantile([0.9, 0.7, 0.5, 0.2])
        names["weight"] = names.occurrences.apply(
            lambda x: self._assign_weight(x, percentiles)
        )

        n1 = int(0.8 * len(names))
        n2 = int(0.9 * len(names))

        self._xtrain_dataset, self._ytrain_dataset, self._train_sampler = (
            self._build_dataset(names[:n1])
        )
        self._xdev_dataset, self._ydev_dataset, dev_sampler = self._build_dataset(
            names[n1:n2]
        )
        self._xtest_dataset, self._ytest_dataset, self._test_sampler = (
            self._build_dataset(names[n2:])
        )

    def _prepare_parameters(self):
        g = torch.Generator().manual_seed(self.config.seed)
        self._C = torch.randn(
            (self._vocab_size, self.config.embedding_vectors), generator=g
        )

        self._W1 = torch.nn.init.normal_(
            torch.empty(
                (
                    self.config.embedding_vectors * self.config.block_size,
                    self.config.hidden_layers,
                )
            ),
            std=0.01,
        )
        self._b1 = torch.randn((self.config.hidden_layers,), generator=g)

        self._W2 = torch.nn.init.normal_(
            torch.empty((self.config.hidden_layers, self._vocab_size)), std=0.01
        )
        self._b2 = torch.randn((self._vocab_size,), generator=g)

        print(f"Number of parameters : {sum(p.nelement() for p in self._parameters)}")
        for p in self._parameters:
            p.requires_grad = True

    @staticmethod
    def _assign_weight(occurrences, percentiles):
        if occurrences >= percentiles[0.9]:
            return 1.0
        elif occurrences >= percentiles[0.7]:
            return 0.7
        elif occurrences >= percentiles[0.5]:
            return 0.5
        else:
            return 0.2

    def predict(self, i):
        g = torch.Generator().manual_seed(123409876 + i)
        out = []
        context = [0] * self.config.block_size
        while True:
            emb = self._C[torch.tensor([context])]
            h = torch.tanh(emb.view(1, -1) @ self._W1 + self._b1)
            logits = h @ self._W2 + self._b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break

        print("".join(self.itos[i] for i in out))

    def _build_dataset(self, names: pd.DataFrame):
        X, Y, weight_context = [], [], []

        for w in names.itertuples():
            context = [0] * self.config.block_size
            for ch in w.name + ".":
                ix = self.stoi[ch]
                X.append(context)
                Y.append(ix)
                weight_context.append(w.weight)
                context = context[1:] + [ix]

        X = torch.tensor(X)
        Y = torch.tensor(Y)
        weight_context = torch.tensor(weight_context)
        return X, Y, WeightedRandomSampler(weight_context, len(weight_context))

    def display_dimension(self):
        plt.figure(figsize=(8, 8))
        plt.scatter(self._C[:, 0].data, self._C[:, 1].data, s=200)
        for i in range(self._C.shape[0]):
            plt.text(
                self._C[i, 0].item(),
                self._C[i, 1].item(),
                self.itos[i],
                ha="center",
                va="center",
                color="white",
            )
        plt.grid(True)
        plt.show()
