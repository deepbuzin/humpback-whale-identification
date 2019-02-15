import numpy as np
import pandas as pd
import os
from core.siamese import Siamese


def test_embeddings():
    model = Siamese('dummy', input_shape=(6, 8, 3), embedding_size=3)

    model.make_embeddings('data_tiny/train.csv', 'data_tiny/train', batch_size=1)
    emb = pd.read_pickle(os.path.join(model.cache_dir, 'embeddings.pkl'))
    print(emb)

    model.predict('data_tiny/train')
    pred = pd.read_pickle(os.path.join(model.cache_dir, 'predictions.pkl'))
    print(pred)

    model.make_kaggle_csv(os.path.join(model.cache_dir, 'idx_to_whales_mapping.npy'))

    # model_2 = Siamese('dummy', input_shape=(6, 8, 3), embedding_size=3)
    # model_2.load_embeddings(os.path.join(model.cache_dir, 'embeddings.pkl'))
    # model_2.predict('data_tiny/test')
    #
    # mapping = np.load(os.path.join(model.cache_dir, 'idx_to_whales_mapping.npy'))
    # mapping_2 = np.load(os.path.join(model_2.cache_dir, 'idx_to_whales_mapping.npy'))
    #
    # print(mapping == mapping_2)


if __name__ == '__main__':
    test_embeddings()


