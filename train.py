from core.siamese import Siamese

model = Siamese('mobilenet_like', input_shape=(672, 896, 3), embedding_size=128, strategy='batch_all')
# model.load_weights('cache/cache-190208-093118/training/checkpoint-03.h5')
# TODO mark pretrained mobilenet
model.load_weights('cache/cache-190208-093118/training/checkpoint-03.h5')
model.train('data/train.csv', 'data/train', epochs=20, batch_size=32, learning_rate=0.0001, margin=2.0)


