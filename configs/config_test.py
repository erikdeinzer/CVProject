data_root = '/data'
dataset_type = 'CCPD'





model_cfg = [
    dict(type='YOLOv5', in_features=128,),
    dict(type='YOLOv5', in_features=256,),]



eval_cfg = dict(
    type='EntropyLoss',
)

transforms = [
        dict(type='ToTensor'),
        dict(type='Augment1', param1=0.5, param2=0.5),
        dict(type='Augment2', param1=0.5, param2=0.5),
        dict(type='PackInputs')
    ]



train_dataloader = dict(
    type='RepeatDataset',
    times=1, # repeat times
    batch_size = 32, # batch size for each GPU
    num_workers=4, # number of workers for data loading
    persistent_workers=True, # keep workers alive
    shuffle=True, # shuffle the dataset
    drop_last=True, # drop last, incomplete batch
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.txt',
        pipeline=transforms,
        test_mode=False,
    )
)

eval_pipeline = dict(
    [
        dict(type='ToTensor'),
        dict(type='PackInputs')
    ]
)
eval_dataloader = dict(
    type='RepeatDataset',
    times=1, # repeat times
    batch_size = 32, # batch size for each GPU
    num_workers=4, # number of workers for data loading
    persistent_workers=True, # keep workers alive
    shuffle=True, # shuffle the dataset
    drop_last=True, # drop last, incomplete batch
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='eval.txt',
        pipeline=eval_pipeline,
        test_mode=False,
    )
)



    