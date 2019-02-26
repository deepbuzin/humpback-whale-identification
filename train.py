from core.siamese import Siamese

model = Siamese('mobilenet_like', input_shape=(672, 896, 3), embedding_size=128)
model.load_weights('model/mobilenet_imagenet.h5')
model.train('data/split_train.csv', 'data/train', meta_dir='D:\\IdeaProjects\\whales\\data\\meta', epochs=20, batch_size=24, learning_rate=0.001)


