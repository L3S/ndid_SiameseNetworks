from model.alexnet import AlexNetModel
from common import get_modeldir

model_name = 'alexnet_cifar10-new'

alexnet = AlexNetModel()

# train
train_ds, test_ds, validation_ds = alexnet.x_dataset()
alexnet.x_train(train_ds, test_ds)
alexnet.evaluate(validation_ds)

# save
alexnet.save_weights(get_modeldir(model_name + '.h5'))
alexnet.save(get_modeldir(model_name + '.tf'))

# print('evaluate')
# res = alexnet.predict(validation_ds)
