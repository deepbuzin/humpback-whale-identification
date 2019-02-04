from core.siamese import Siamese

model = Siamese('resnet_like_33', input_shape=(384, 512, 3), embedding_size=128, strategy='batch_all')
model.load_weights('trained/30-Jan-19/weights/checkpoint-05')
model.load_embeddings('D:/IdeaProjects/whales/embeddings.npy')
# model.make_embeddings('data/train.csv', 'data/train')
model.predict('data/test')


# TODO automatically put weights, embeddings and predictions in a proper folder and write an info file on top  of it
# TODO tsne from sklearn


