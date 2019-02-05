from core.siamese import Siamese


model = Siamese('shallow_mnist', input_shape=(28, 28, 3), embedding_size=64, strategy='batch_all')
model.load_weights('cache/cache-190205-065005/final_weights.h5')
model.load_embeddings('cache/cache-190205-070856/embeddings.pkl')
model.load_predictions('cache/cache-190205-072026/predictions.pkl')
model.make_kaggle_csv('cache/cache-190205-065005/idx_to_whales_mapping.npy')
model.draw_tsne(model.predictions.values[:, 1:])





