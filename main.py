from data_modules import YahooS5DataModule


def main():
    data_module = YahooS5DataModule(dataset_path='datasets/datasets_files/yahoo_s5/A1Benchmark',
                                    dataset_name='A1Benchmark',
                                    dataset_configs_path='datasets/datasets_configs/yahoo_s5_config.json',
                                    train_size=0.6,
                                    valid_size=0.1,
                                    test_size=0.3,
                                    batch_size=1,
                                    seq_len=24,
                                    step=10,
                                    shuffle=False,
                                    normalize=True,
                                    normalize_range=(0, 1))
    data_module.prepare_data()
    data_module.setup()
    test_loader = data_module.test_dataloader()
    count = 0
    for batch in test_loader:
        if count == 10:
            pass
        print(batch[0])
        print('----------------')
        count += 1

    train_loader = data_module.train_dataloader()
    for batch in train_loader:
        print(batch)
        break


if __name__ == '__main__':
    main()
