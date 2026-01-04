from tqdm import tqdm
from sstan.video_dataset import WLASLVideoDataset
from sstan.dataset import Sign_Dataset

def main():
    dataset = WLASLVideoDataset(
        split_file_path="/home/hirooka/transformer-based-sign-language-recognition/data/official_wlasl/splits/asl100.json",
        split="train",
        video_dir_path="/home/hirooka/transformer-based-sign-language-recognition/data/official_wlasl/video",
        seq_len=64,
        sampling_strategy="rnd_start",
        transforms=None
    )
    print(WLASLVideoDataset.gloss2index.keys())

    # subset = "asl100"
    # dataset = Sign_Dataset(
    #     index_file_path='./data/official_wlasl/splits/{}.json'.format(subset), 
    #     pose_root="./data/official_wlasl/pose_per_individual_videos",
    #     split="train",
    #     num_samples=64, num_copies=4, sample_strategy="rnd_start", skeleton_augmentation=None)
    
    print(len(dataset))
    for i, (data, label) in enumerate(tqdm(dataset)):
        print(f"{type(data)}: {data.shape}")
        print(f"{type(label)}: {label.shape}")

if __name__=="__main__":
    main()