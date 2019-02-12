from core.siamese import Siamese

model = Siamese('mobilenet_like', input_shape=(672, 896, 3), embedding_size=128, strategy='batch_hard')
model.load_weights('model/mobilenet_imagenet.h5')
model.train('data/train.csv', 'data/train', epochs=20, batch_size=32, learning_rate=0.0001, margin=1.0)


