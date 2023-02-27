import torch

def CreateDataLoader(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    print_args = not torch.distributed.is_initialized()
    if not print_args:
        if torch.distributed.get_rank() == 0:
            print_args = True
    if print_args:
        print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader