import argparse
import os
from sklearn.model_selection import train_test_split

from sasformer.data import raw_data_to_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/', type=str,
                        help='Directory containing data files.', metavar='data_dir')
    parser.add_argument('--name', default='sas-55m-20k', type=str,
                        help='Directory containing parquet files within data_dir.', metavar='sub_dir')
    parser.add_argument('--test_size', default=0.25, type=float,
                        help='Proportion of raw data to split for testing.', metavar='test_size')
    parser.add_argument('--seed', default=9, type=int,
                        help='Random seed. Use 9 to reproduce our results.', metavar='seed')
    namespace = parser.parse_args()

    # define paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, namespace.data_dir)

    df = raw_data_to_df(data_dir, step=2)
    df_train, df_test = train_test_split(
        df,
        test_size=namespace.test_size,
        stratify=df['model_label'],
        random_state=namespace.seed
    )
    # train
    train_fp = os.path.join(
        data_dir, namespace.name, 'train.parquet')
    df_train.to_parquet(train_fp, index=False)
    # test
    test_fp = os.path.join(
        data_dir, namespace.name, 'test.parquet')
    df_test.to_parquet(test_fp, index=False)
    print(f'Data written to: {os.path.join(data_dir, namespace.name)}.')
    print(f'Train: {df_train.shape}    Test: {df_test.shape}')
