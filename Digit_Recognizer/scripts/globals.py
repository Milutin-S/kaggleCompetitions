from pathlib import Path
import torchvision.transforms as transforms

# PATHS
TRAIN_PATH = Path("./Digit_Recognizer/Dataset/train.csv")
TEST_PATH = Path("./Digit_Recognizer/Dataset/test.csv")
OUTPUT_DIR = Path("./Digit_Recognizer/Train_Output")
MODELS_DIR = Path("./Digit_Recognizer/Train_Output/models")
PARAMETERS_DIR = Path("./Digit_Recognizer/Train_Output/parameters")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PARAMETERS_DIR.mkdir(parents=True, exist_ok=True)

# CONSTS
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCH_NUM = 100

# TRAIN_TRANSFORMS = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
# )

# transforms = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
#     # Actual mean and standard deviation of MNIST dataset
# )

TRAIN_TRANSFORMS = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
