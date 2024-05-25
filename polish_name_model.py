import argparse

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import WeightedRandomSampler, DataLoader, TensorDataset

from config.polish_name import PolishNameConfig
from polish_name_data_loader import PolishNameDataLoader


class PolishNameModel:
    xtrain_dataset: Tensor
    ytrain_dataset: Tensor
    train_sampler: WeightedRandomSampler

    xdev_dataset: Tensor
    ydev_dataset: Tensor
    dev_sampler: WeightedRandomSampler

    xtest_dataset: Tensor
    ytest_dataset: Tensor
    test_sampler: WeightedRandomSampler

    vocab_size: int
    dataset_initialized: bool = False
    C: Tensor
    W1: Tensor
    b1: Tensor
    W2: Tensor
    b2: Tensor

    def __init__(self, config: PolishNameConfig):
        self.config = config

    @property
    def _parameters(self) -> list[Tensor]:
        return [self.C, self.W1, self.b1, self.W2, self.b2]

    def build_model(self, names):
        self._prepare_dataset(names)
        self.prepare_parameters()

    def train_model(self):
        lossi = []
        stepi = []
        epoch = 1_000  # int(len(self.xtrain_dataset) // self.config.batch_size)
        for i in range(epoch):
            train_loader = DataLoader(TensorDataset(self.xtrain_dataset, self.ytrain_dataset),
                                      batch_size=self.config.batch_size,
                                      sampler=self.train_sampler)
            print(f'Epoch {i}/{epoch} training with {len(train_loader)} batches')
            for j, (x, y) in enumerate(train_loader):
                loss = self._calculate_loss(x, y)
                for p in self._parameters:
                    p.grad = None
                loss.backward()
                ix = (i * len(train_loader)) + j

                lerning_rate = 0.1 if i < 1000 else 0.001
                for p in self._parameters:
                    if p.grad is not None:
                        p.data += -lerning_rate * p.grad
                    else:
                        print(f'What the fck {p}')
                stepi.append(ix)
                lossi.append(loss.log10().item())
        plt.plot(stepi, lossi)
        plt.show()
        self.train_loss()
        pass

    def _calculate_loss(self, x, y):
        emb = self.C[x]
        h = torch.tanh(emb.view(-1, self.config.block_size * self.config.embedding_vectors) @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return F.cross_entropy(logits, y)

    def test_loss(self):
        loss = self._calculate_loss(self.xtest_dataset, self.ytest_dataset)
        print(f'Test loss is {loss}')

    def train_loss(self):
        loss = self._calculate_loss(self.xtrain_dataset, self.ytrain_dataset)
        print(f'Train loss is {loss}')

    def dev_loss(self):
        loss = self._calculate_loss(self.xdev_dataset, self.ydev_dataset)
        print(f'Dev loss is {loss}')

    def _prepare_dataset(self, names: pd.DataFrame):
        additional_end_of_string = 1
        self.chars = sorted(list(set(''.join(names.name))))
        self.vocab_size = len(self.chars) + additional_end_of_string
        self.stoi = {s: i + 1 for i, s in enumerate(self.chars)}
        self.stoi['.'] = 0
        self.itos = {i: s for s, i in self.stoi.items()}

        names = names.sample(frac=1, random_state=config.seed).reset_index(drop=True)
        # weights = torch.DoubleTensor(names.occurrences)
        percentiles = names.occurrences.quantile([0.9, 0.7, 0.5, 0.2])
        names['weight'] = names.occurrences.apply(lambda x: self._assign_weight(x, percentiles))

        n1 = int(0.8 * len(names))
        n2 = int(0.9 * len(names))

        self.xtrain_dataset, self.ytrain_dataset, self.train_sampler = self._build_dataset(names[:n1])
        self.xdev_dataset, self.ydev_dataset, dev_sampler = self._build_dataset(names[n1:n2])
        self.xtest_dataset, self.ytest_dataset, self.test_sampler = self._build_dataset(names[n2:])

    def prepare_parameters(self):
        g = torch.Generator().manual_seed(self.config.seed)
        self.C = torch.randn((self.vocab_size, self.config.embedding_vectors), generator=g)
        self.W1 = torch.nn.init.normal_(
            torch.empty((self.config.embedding_vectors * self.config.block_size, self.config.hidden_layers)), std=0.01)

        # self.W1 = torch.randn((self.config.embedding_vectors * self.config.block_size, self.config.hidden_layers),
        #                       generator=g)
        self.b1 = torch.randn((self.config.hidden_layers,), generator=g)
        self.W2 = torch.nn.init.normal_(torch.empty((self.config.hidden_layers, self.vocab_size)), std=0.01)

        # self.W2 = torch.randn((self.config.hidden_layers, self.vocab_size), generator=g)
        self.b2 = torch.randn((self.vocab_size,), generator=g)
        print(f'Number of parameters : {sum(p.nelement() for p in self._parameters)}')
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
            emb = self.C[torch.tensor([context])]
            h = torch.tanh(emb.view(1, -1) @ self.W1 + self.b1)
            logits = h @ self.W2 + self.b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break

        print(''.join(self.itos[i] for i in out))

    def _build_dataset(self, names: pd.DataFrame):
        X, Y, weight_context = [], [], []

        for w in names.itertuples():
            context = [0] * self.config.block_size
            for ch in w.name + '.':
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
        plt.scatter(self.C[:, 0].data, self.C[:, 1].data, s=200)
        for i in range(self.C.shape[0]):
            plt.text(self.C[i, 0].item(), self.C[i, 1].item(), self.itos[i], ha="center", va="center", color='white')
        plt.grid('minor')
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ignore", help="ignore loading data", action="store_true")
    args = parser.parse_args()

    config = PolishNameConfig.load_config('./config/polish_name_config.yaml')
    data_loader = PolishNameDataLoader(config)

    if not args.ignore:
        data_loader.refresh_data()

    male, female = data_loader.load_data()
    model = PolishNameModel(config)
    model.build_model(male)
    model.train_model()
    model.train_loss()
    model.dev_loss()
    model.test_loss()

    model.display_dimension()

    for index in range(10):
        model.predict(index)
    print("model initialized")
