from core.siamese import Siamese

model = Siamese('resnet_like_33', input_shape=(384, 512, 3), embedding_size=128, strategy='batch_all')
model.train('data/train.csv', 'data/train', epochs=20, batch_size=10, learning_rate=0.0001, margin=1.0)


