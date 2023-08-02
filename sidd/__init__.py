import time
import argparse
from sidd.data.simple3 import Simple3
from sidd.data.imagenette import Imagenette
from sidd.data.cifar10 import Cifar10
from sidd.loss import ContrastiveLoss, TripletEasyLoss, TripletSemiHardLoss, TripletHardLoss
from sidd.model.siamese import SiameseModel
from sidd.model.siamese_offlinetriplet import SiameseOfflineTripletModel
from sidd.model.alexnet import AlexNetModel
from sidd.model.efficientnet import EfficientNetModel
from sidd.model.mobilenet import MobileNetModel
from sidd.model.resnet import ResNetModel
from sidd.model.vgg16 import VGG16Model
from sidd.model.vit import VitModel

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-D", help="Dataset", default="simple3",
                    choices=["simple3", "cifar10", "imagenette"], type=str)
parser.add_argument("--model", "-M", help="Model", default="alexnet",
                    choices=["alexnet", "efficientnet", "mobilenet", "resnet", "vgg16", "vit"], type=str)
parser.add_argument("--weights", "-W", help="Weights", default="imagenet",
                    choices=["imagenet", "imagenetplus", "dataset"], type=str)

# model params
parser.add_argument("--loss", "-l", help="Loss function", default="contrastive",
                    choices=["contrastive", "easy-triplet", "semi-hard-triplet", "hard-triplet"], type=str)
parser.add_argument("--margin", "-m", help="Margin for the loss function", default=1.5, type=float)
parser.add_argument("--dimensions", "-d", help="The dimension of Siamese output", default=512, type=int)
parser.add_argument("--epochs", "-e", help="Number of epochs, each epoch consists of 100 steps", default=15, type=int)

# other params
parser.add_argument("--seed", "-s", help="Set seed value", default="", type=str)
parser.add_argument("--ukbench", help="Compute UKBench vectors", default=False, type=bool)

# what to save
parser.add_argument("--cnn-vectors", help="Save CNN's embeddings", default=False, type=bool)
parser.add_argument("--save-vectors", help="Save embeddings", default=False, type=bool)
parser.add_argument("--compute-stats", help="Compute FAISS statistical analysis", default=False, type=bool)
parser.add_argument("--project-vectors", help="Project embeddings", default=False, type=bool)


class SimpleParams:
    @classmethod
    def parse(cls):
        args = parser.parse_args()
        print('Params received: {}'.format(args))
        return cls(args.dataset, args.model, args.weights,
                   args.loss, args.margin, args.dimensions, args.epochs,
                   args.seed, args.ukbench,
                   args.cnn_vectors, args.save_vectors, args.project_vectors, args.compute_stats)

    def __init__(self, dataset, model, weights,
                 loss, margin, dimensions, epochs,
                 seed, ukbench,
                 cnn_vectors, save_vectors, project_vectors, compute_stats):
        self.dataset = dataset
        self.model = model
        self.weights = weights

        self.loss = loss
        self.margin = margin
        self.dimensions = dimensions
        self.epochs = epochs

        self.cnn_vectors = cnn_vectors
        self.save_vectors = save_vectors
        self.project_vectors = project_vectors
        self.compute_stats = compute_stats

        self.basename = model + '_' + dataset + '_' + weights + '_d' + str(dimensions) + '_m' + str(margin) + '_s' + str(epochs * 100) + '_' + loss

        self.ukbench = ukbench
        if len(seed) > 0:
            self.seed = seed
        else:
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

    def get_model_class(self):
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
        return cls

    def get_model(self, **kwargs):
        return self.get_model_class()(**kwargs)

    def get_loss_class(self):
        if self.loss == "contrastive":
            cls = ContrastiveLoss
        elif self.loss == "easy-triplet":
            cls = TripletEasyLoss
        elif self.loss == "semi-hard-triplet":
            cls = TripletSemiHardLoss
        elif self.loss == "hard-triplet":
            cls = TripletHardLoss
        else:
            raise ValueError("Loss not found")
        return cls

    def get_siamese_class(self):
        cls = SiameseModel
        if self.loss == "easy-triplet":
            cls = SiameseOfflineTripletModel
        return cls
