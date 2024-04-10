import time
import argparse
from sidd.data.californiand import CaliforniaND
from sidd.data.holidays import Holidays
from sidd.data.imagenet1k import ImageNet1k
from sidd.data.mirflickr import MFND
from sidd.data.mirflickr_full import MFNDFull
from sidd.data.simple3 import Simple3
from sidd.data.imagenette import Imagenette
from sidd.data.cifar10 import Cifar10
from sidd.data.ukbench import UKBench
from sidd.loss import ContrastiveLoss, TripletEasyLoss, TripletSemiHardLoss, TripletHardLoss
from sidd.model.siamese import SiameseModel
from sidd.model.siamese_offlinetriplet import SiameseOfflineTripletModel
from sidd.model.alexnet import AlexNetModel
from sidd.model.efficientnet import EfficientNetModel
from sidd.model.mobilenet import MobileNetModel
from sidd.model.resnet import ResNetModel
from sidd.model.simclr import SimclrModel
from sidd.model.vgg16 import VGG16Model
from sidd.model.vit import VitModel

parser = argparse.ArgumentParser()
# Model params
parser.add_argument("--dataset", "-D", help="Dataset", default="simple3",
                    choices=["simple3", "cifar10", "imagenette", "imagenet"], type=str)
parser.add_argument("--model", "-M", help="Model", default="alexnet",
                    choices=["alexnet", "efficientnet", "mobilenet", "resnet", "vgg16", "vit", "simclr"], type=str)
parser.add_argument("--weights", "-W", help="Weights", default="imagenet",
                    choices=["imagenet", "imagenetplus", "dataset"], type=str)
parser.add_argument("--loss", "-l", help="Loss function", default="contrastive",
                    choices=["contrastive", "easy-triplet", "semi-hard-triplet", "hard-triplet"], type=str)
parser.add_argument("--margin", "-m", help="Margin for the loss function", default=1.5, type=float)
parser.add_argument("--dimensions", "-d", help="The dimension of Siamese output", default=512, type=int)
parser.add_argument("--epochs", "-e", help="Number of epochs, each epoch consists of 100 steps", default=15, type=int)

# Evaluation params
parser.add_argument("--eval-dataset", "-ED", help="Evaluation datasets", default="ukbench",
                    choices=["ukbench", "holidays", "mirflickr", "mirflickr-full", "californiand"], type=str)

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
                   args.eval_dataset,
                   args.seed, args.ukbench,
                   args.cnn_vectors, args.save_vectors, args.project_vectors, args.compute_stats)

    def __init__(self, dataset, model, weights,
                 loss, margin, dimensions, epochs,
                 eval_dataset,
                 seed, ukbench,
                 cnn_vectors, save_vectors, project_vectors, compute_stats):
        self.dataset = dataset
        self.model = model
        self.weights = weights
        self.loss = loss
        self.margin = margin
        self.dimensions = dimensions
        self.epochs = epochs

        self.eval_dataset = eval_dataset

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
        elif self.dataset == "imagenet":
            cls = ImageNet1k
        else:
            raise ValueError("Dataset not found")
        return cls(**kwargs)

    def get_eval_dataset(self, **kwargs):
        if self.eval_dataset == "ukbench":
            cls = UKBench
        elif self.eval_dataset == "holidays":
            cls = Holidays
        elif self.eval_dataset == "mirflickr":
            cls = MFND
        elif self.eval_dataset == "mirflickr-full":
            cls = MFNDFull
        elif self.eval_dataset == "californiand":
            cls = CaliforniaND
        else:
            raise ValueError("Evaluation dataset not found")
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
        elif self.model == "simclr":
            cls = SimclrModel
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
