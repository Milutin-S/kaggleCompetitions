from pathlib import Path
import torchvision.transforms as transforms

TRAIN_PATH = Path('./Digit_Recognizer/Dataset/train.csv')
TEST_PATH = Path('./Digit_Recognizer/Dataset/test.csv')

transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    # Actual mean and standard deviation of MNIST dataset
)