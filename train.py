from core.siamese import Siamese


model = Siamese('resnet_like_33', input_shape=(384, 512, 3), embedding_size=128)
model.train('data/train.csv', 'data/train', epochs=10, batch_size=10, learning_rate=0.001, margin=0.5, strategy='batch_all')


