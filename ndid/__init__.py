import time
import argparse
import tensorflow_addons as tfa
from ndid.data.simple3 import Simple3
from ndid.data.imagenette import Imagenette
from ndid.data.cifar10 import Cifar10
from ndid.loss.offline_triplet import OfflineTripletLoss
from ndid.model.siamese import SiameseModel
from ndid.model.siamese_offlinetriplet import SiameseOfflineTripletModel
from ndid.model.alexnet import AlexNetModel
from ndid.model.efficientnet import EfficientNetModel
from ndid.model.mobilenet import MobileNetModel
from ndid.model.resnet import ResNetModel
from ndid.model.vgg16 import VGG16Model
from ndid.model.vit import VitModel

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-D", help="Dataset", default="simple3",
                    choices=["simple3", "cifar10", "imagenette"], type=str)
parser.add_argument("--model", "-M", help="Model", default="alexnet",
                    choices=["alexnet", "efficientnet", "mobilenet", "resnet", "vgg16", "vit"], type=str)

# model params
parser.add_argument("--loss", "-l", help="Loss function", default="contrastive",
                    choices=["contrastive", "easy-triplet", "semi-hard-triplet", "hard-triplet"], type=str)
parser.add_argument("--margin", "-m", help="Margin for the loss function", default=1.0, type=float)
parser.add_argument("--dimensions", "-d", help="The dimension of Siamese output", default=512, type=int)
parser.add_argument("--epochs", "-e", help="Number of epochs, each epoch consists of 100 steps", default=5, type=int)

# other params
parser.add_argument("--seed", "-s", help="Skip setting seed value", default=False, type=bool)
parser.add_argument("--ukbench", help="Compute UKBench vectors", default=True, type=bool)
parser.add_argument("--save-vectors", help="Save embeddings", default=True, type=bool)
parser.add_argument("--project-vectors", help="Project embeddings", default=True, type=bool)


class SimpleParams:
    @classmethod
    def parse(cls):
        args = parser.parse_args()
        print('Params received: {}'.format(args))
        return cls(args.dataset, args.model, args.loss, args.margin, args.dimensions, args.epochs, args.seed, args.ukbench, args.save_vectors, args.project_vectors)

    def __init__(self, dataset, model, loss, margin, dimensions, epochs, seed, ukbench, save_vectors, project_vectors):
        self.dataset = dataset
        self.model = model

        self.loss = loss
        self.margin = margin
        self.dimensions = dimensions
        self.epochs = epochs

        self.ukbench = ukbench
        self.save_vectors = save_vectors
        self.project_vectors = project_vectors

        self.seed = ''
        self.basename = model + '_' + dataset + '_d' + str(dimensions) + '_m' + str(margin) + '_s' + str(epochs * 100) + '_' + loss
        if not seed:
            self.seed = str(int(time.time()))
            self.basename += '_' + self.seed

    def get_dataset(self, **kwargs):
        if self.dataset == "cifar10":
            cls = Cifar10
        elif self.dataset == "imagenette":
            cls = Imagenette
        elif self.dataset == "simple3":
            cls = Simple3
        else:
            raise ValueError("Dataset not found")
        return cls(**kwargs)

    def get_model(self, **kwargs):
        if self.model == "alexnet":
            cls = AlexNetModel
        elif self.model == "efficientnet":
            cls = EfficientNetModel
        elif self.model == "mobilenet":
            cls = MobileNetModel
        elif self.model == "resnet":
            cls = ResNetModel
        elif self.model == "vgg16":
            cls = VGG16Model
        elif self.model == "vit":
            cls = VitModel
        else:
            raise ValueError("Model not found")
        return cls(**kwargs)

    def get_loss_class(self):
        if self.loss == "contrastive":
            cls = tfa.losses.ContrastiveLoss
        elif self.loss == "easy-triplet":
            cls = OfflineTripletLoss
        elif self.loss == "semi-hard-triplet":
            cls = tfa.losses.TripletSemiHardLoss
        elif self.loss == "hard-triplet":
            cls = tfa.losses.TripletHardLoss
        else:
            raise ValueError("Loss not found")
        return cls

    def get_siamese_class(self):
        cls = SiameseModel
        if self.loss == "easy-triplet":
            cls = SiameseOfflineTripletModel
        return cls
