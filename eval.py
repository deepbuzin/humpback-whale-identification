from core.siamese import Siamese


model = Siamese('mobilenet_like', input_shape=(672, 896, 3), embedding_size=128, strategy='batch_all')
model.load_weights('cache/cache-190208-081908/training/checkpoint-02.h5')
model.make_embeddings('data/train.csv', 'data/train', batch_size=24)
model.predict('data/test')
model.make_kaggle_csv('cache/cache-190208-081908/idx_to_whales_mapping.npy')





