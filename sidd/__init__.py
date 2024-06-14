import time
import argparse
from sidd.data.californiand import CaliforniaND
from sidd.data.copydays import CopyDays
from sidd.data.holidays import Holidays
from sidd.data.imagenet1k import ImageNet1k
from sidd.data.mirflickr25k import Mirflickr25k
from sidd.data.mirflickr import Mirflickr
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
# CNN model params
parser.add_argument("--cnn-model", "-CM", help="CNN Model", default="alexnet",
                    choices=["alexnet", "efficientnet", "mobilenet", "resnet", "vgg16", "vit", "simclr"], type=str)
parser.add_argument("--cnn-weights", "-CW", help="CNN Weights", default="load",
                    choices=["load", "train", "finetune"], type=str)
parser.add_argument("--cnn-dataset", "-CD", help="CNN Dataset", default="imagenet",
                    choices=["simple3", "cifar10", "imagenette", "imagenet"], type=str)

# Siamese model params
parser.add_argument("--dataset", "-D", help="Siamese Dataset", default="ukbench",
                    choices=["ukbench", "holidays", "mirflickr", "mirflickr25k", "californiand", "copydays"], type=str)
parser.add_argument("--loss", "-l", help="Siamese Loss function", default="contrastive",
                    choices=["contrastive", "easy-triplet", "semi-hard-triplet", "hard-triplet"], type=str)
parser.add_argument("--margin", "-m", help="Siamese margin for the loss function", default=[1.5], type=float, nargs='*')
parser.add_argument("--dimensions", "-d", help="Siamese output dimension", default=[512], type=int, nargs='*')
parser.add_argument("--epochs", "-e", help="Siamese number of epochs (each epoch consists of 100 steps)", default=[15], type=int, nargs='*')

# Evaluation params
parser.add_argument("--eval-dataset", "-ED", help="Evaluation datasets",
                    choices=["ukbench", "holidays", "mirflickr", "mirflickr25k", "californiand", "copydays"], type=str)

# other params
parser.add_argument("--seed", "-s", help="Set seed value", default="", type=str)

# what to save
parser.add_argument("--save-vectors", help="Save embeddings", default=False, type=bool)
parser.add_argument("--compute-stats", help="Compute FAISS statistical analysis", default=False, type=bool)
parser.add_argument("--project-vectors", help="Project embeddings", default=False, type=bool)


class SiameseCliParams:
    @classmethod
    def parse(cls):
        args = parser.parse_args()
        print('Params received: {}'.format(args))
        return cls(args.cnn_model, args.cnn_weights, args.cnn_dataset,
                   args.dataset, args.loss,
                   args.margin, args.dimensions, args.epochs,
                   args.eval_dataset, args.seed,
                   args.save_vectors, args.project_vectors, args.compute_stats)

    def __init__(self, cnn_model, cnn_weights, cnn_dataset,
                 dataset, loss,
                 margin, dimensions, epochs,
                 eval_dataset, seed,
                 save_vectors, project_vectors, compute_stats):
        self.cnn_model = cnn_model
        self.cnn_weights = cnn_weights
        self.cnn_dataset = cnn_dataset

        self.dataset = dataset
        self.loss = loss
        self.margin = margin
        self.dimensions = dimensions
        self.epochs = epochs

        self.eval_dataset = eval_dataset

        self.save_vectors = save_vectors
        self.project_vectors = project_vectors
        self.compute_stats = compute_stats

        if len(seed) > 0:
            self.seed = seed
        else:
            self.seed = str(int(time.time()))

        # Construct model names
        self.core_name = cnn_model
        if cnn_weights == "finetune":
            self.core_name += '_imagenetF' + dataset
        else:
            self.core_name += '_' + dataset

        self.cnn_name = self.core_name + '_' + self.seed
        if cnn_weights == "load":
            self.cnn_name = self.core_name

    def get_cnn_dataset(self, **kwargs):
        return self.get_dataset_class(self.cnn_dataset)(**kwargs)

    def get_dataset(self, **kwargs):
        return self.get_dataset_class(self.dataset)(**kwargs)

    def get_eval_dataset(self, **kwargs):
        return self.get_dataset_class(self.eval_dataset)(**kwargs)

    def get_dataset_class(self, dsname):
        if dsname == "cifar10":
            cls = Cifar10
        elif dsname == "imagenette":
            cls = Imagenette
        elif dsname == "simple3":
            cls = Simple3
        elif dsname == "imagenet":
            cls = ImageNet1k
        elif dsname == "ukbench":
            cls = UKBench
        elif dsname == "holidays":
            cls = Holidays
        elif dsname == "mirflickr25k":
            cls = Mirflickr25k
        elif dsname == "mirflickr":
            cls = Mirflickr
        elif dsname == "copydays":
            cls = CopyDays
        elif dsname == "californiand":
            cls = CaliforniaND
        else:
            raise ValueError("Dataset not found")
        return cls

    def get_model_class(self):
        if self.cnn_model == "alexnet":
            cls = AlexNetModel
        elif self.cnn_model == "efficientnet":
            cls = EfficientNetModel
        elif self.cnn_model == "mobilenet":
            cls = MobileNetModel
        elif self.cnn_model == "resnet":
            cls = ResNetModel
        elif self.cnn_model == "vgg16":
            cls = VGG16Model
        elif self.cnn_model == "vit":
            cls = VitModel
        elif self.cnn_model == "simclr":
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
