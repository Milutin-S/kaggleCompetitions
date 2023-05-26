from pathlib import Path
import torchvision.transforms as transforms

# TODO: Change parameters in GENERAL_TRANSFORMATIONS ❓

# PATHS
PREDICTIONS_PATH = Path("./Digit_Recognizer\Predictions")
TRAIN_PATH = Path("./Digit_Recognizer/Dataset/train.csv")
TEST_PATH = Path("./Digit_Recognizer/Dataset/test.csv")
OUTPUT_DIR = Path("./Digit_Recognizer/Train_Output")
MODELS_DIR = Path("./Digit_Recognizer/Train_Output/models")
PARAMETERS_DIR = Path("./Digit_Recognizer/Train_Output/parameters")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PARAMETERS_DIR.mkdir(parents=True, exist_ok=True)

# CONSTS
BATCH_SIZE = 32  # 64
LEARNING_RATE = 1e-4  # 6e-5, 3e-5
EPOCH_NUM = 100  # 200
IMAGE_SHAPE = (28, 28)
RAND_TRESHOLD_69 = 0.8

# TRAIN_TRANSFORMS = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
# )

# transforms = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
#     # Actual mean and standard deviation of MNIST dataset
# )

REQUIRED_TRANSFORMS = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

TRAIN_GENERAL_TRANSFORMS = transforms.Compose(
    [
        # transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomChoice(
            [
                transforms.RandomApply(
                    [transforms.RandomResizedCrop(size=28, scale=(0.8, 1.2))], p=0.5  #
                ),
                transforms.RandomApply(
                    [
                        transforms.RandomAffine(
                            degrees=20, translate=(0.1, 0.1), fill=True
                        )  # 10,
                    ],
                    p=0.5,
                ),
            ]
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.6),  # 0.4
        transforms.RandomErasing(p=0.5, scale=(0.007, 0.007)),
    ]
)
TRAIN_0_TRANSFORM = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomRotation(degrees=180)], p=0.5),  # 45
    ]
)
TRAIN_3_TRANSFORM = transforms.RandomVerticalFlip(p=0.5)
TRAIN_8_TRANSFORM = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ]
)
TRAIN_69_SWITCH = transforms.Compose(
    [transforms.RandomHorizontalFlip(p=1), transforms.RandomVerticalFlip(p=1)]
)

if __name__ == "__main__":
    print(LEARNING_RATE)
