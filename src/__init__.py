import argparse
import tensorflow_addons as tfa
from src.data.simple3 import Simple3
from src.data.imagenette import Imagenette
from src.data.cifar10 import Cifar10
from src.loss.offline_triplet import OfflineTripletLoss
from src.model.siamese import SiameseModel
from src.model.siamese_offlinetriplet import SiameseOfflineTripletModel

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-ds", help="Dataset", default="simple3",
                    choices=["simple3", "cifar10", "imagenette"], type=str)
parser.add_argument("--loss", "-l", help="Loss function", default="contrastive",
                    choices=["contrastive", "offline-triplet", "semi-hard-triplet", "hard-triplet"], type=str)
parser.add_argument("--margin", "-m", help="Margin for the loss function", default=1.0, type=float)
parser.add_argument("--dimensions", "-d", help="The dimension of Siamese output", default=512, type=int)
parser.add_argument("--epochs", "-e", help="Number of epochs, each epoch consists of 100 steps", default=5, type=int)


class SimpleParams:
    @classmethod
    def parse(cls):
        args = parser.parse_args()
        print('New params received: {}'.format(args))
        return cls(args.dataset, args.loss, args.margin, args.dimensions, args.epochs)

    def __init__(self, dataset, loss, margin, dimensions, epochs):
        self.dataset = dataset
        self.loss = loss
        self.margin = margin
        self.dimensions = dimensions
        self.epochs = epochs

    def get_dataset(self, **kwargs):
        cls = Simple3
        if self.dataset == "cifar10":
            cls = Cifar10
        elif self.dataset == "imagenette":
            cls = Imagenette

        return cls(**kwargs)

    def get_loss_class(self):
        cls = tfa.losses.ContrastiveLoss
        if self.dataset == "offline-triplet":
            cls = OfflineTripletLoss
        elif self.dataset == "semi-hard-triplet":
            cls = tfa.losses.TripletSemiHardLoss
        elif self.dataset == "hard-triplet":
            cls = tfa.losses.TripletHardLoss
        return cls

    def get_siamese_class(self):
        cls = SiameseModel
        if self.dataset == "offline-triplet":
            cls = SiameseOfflineTripletModel
        return cls
