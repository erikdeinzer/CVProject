data_root = '/data'
dataset_type = 'CCPD'





model_cfg = [
    dict(type='YOLOv5', in_features=128,),
    dict(type='YOLOv5', in_features=256,),]



eval_cfg = dict(
    type='EntropyLoss',
)

train_pipeline = [
        dict(type='ToTensor'),
        dict(type='Augment1', param1=0.5, param2=0.5),
        dict(type='Augment2', param1=0.5, param2=0.5),
        dict(type='PackInputs')
    ]


ccpd_dataset_train = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='train.txt',
    pipeline=train_pipeline,
    test_mode=False,
)

dataset_train = dict(
    type='RepeatDataset',
    times=1, # repeat times
    dataset=ccpd_dataset_train,
)

train_dataloader = dict(
    batch_size=32, # batch size for each GPU
    num_workers=4, # number of workers for data loading
    persistent_workers=True, # keep workers alive
    shuffle=True, # shuffle the dataset
    drop_last=True, # drop last, incomplete batch
    dataset=dataset_train,
)

eval_pipeline = dict(
    [
        dict(type='ToTensor'),
        dict(type='PackInputs')
    ]
)

dataset_test = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='test.txt',
    pipeline=eval_pipeline,
    test_mode=True,
)

eval_dataloader = dict(
    batch_size=32, # batch size for each GPU
    num_workers=4, # number of workers for data loading
    persistent_workers=True, # keep workers alive
    shuffle=False, # shuffle the dataset
    drop_last=False, # drop last, incomplete batch
    dataset=dataset_test,
)



    