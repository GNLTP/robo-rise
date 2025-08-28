from dataset.data_mujocoworld import MujocoWorldDataset, collate_fn
from dataset.data_BatchSampler import PerTableBatchSampler
from torch.utils.data import DataLoader

def data_provider(args, flag):
    Data = MujocoWorldDataset

    if flag == "train":
        drop_last = True
        batch_size = args.batch_size

    else:
        drop_last = True
        batch_size = args.batch_size

    data_set = Data(
        split=flag,
        path=args.data_path,
        path_time=args.path_time,
        target_obj=args.target_obj,
        num_obs=args.num_obs,
        num_action=args.num_action,
        voxel_size=args.voxel_size,
        cam_ids=args.cam_ids,
        aug=args.aug,
        aug_trans_min=args.aug_trans_min,
        aug_trans_max=args.aug_trans_max,
        aug_rot_min=args.aug_rot_min,
        aug_rot_max=args.aug_rot_max,
        aug_jitter=args.aug_jitter,
        aug_jitter_params=args.aug_jitter_params,
        aug_jitter_prob=args.aug_jitter_prob,
        with_cloud=args.with_cloud,
        depth_scale=1.0
    )

    print(flag, len(data_set))

    sampler = PerTableBatchSampler(
        data_set,
        batch_size=batch_size,
        drop_last=drop_last,
    )

    data_loader = DataLoader(
        data_set,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    return data_set, data_loader
