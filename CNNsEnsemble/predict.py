import glob
from utils import predictions_to_csv
from nnmodels import *
from models import create_model
from configparser import ConfigParser
import ntpath
from datetime import datetime
from quickdrawdataset import *
from transforms import create_transforms
from tqdm import tqdm

def load_most_recent_best_model(model, folder):
    pths_folders = glob.glob(folder + "/pth_*")
    pths_folders = [os.path.basename(os.path.normpath(pth_folder)) for pth_folder in pths_folders]
    dates = [datetime.strptime(pth_folder[pth_folder.find("_") + 1:], "%Y-%m-%d_%H-%M-%S") for pth_folder in pths_folders]

    checkpoints_folder = os.path.join(folder, pths_folders[np.argmax(dates)])
    checkpoints = glob.glob(checkpoints_folder + "/*_cnn.pth")
    checkpoints_name = [ntpath.basename(checkpoint) for checkpoint in checkpoints]
    accuracies = [float(checkpoint_name[:checkpoint_name.find("_")]) for checkpoint_name in checkpoints_name]
    best_checkpoint = checkpoints[np.argmax(accuracies)]

    model.load_state_dict(torch.load(best_checkpoint))

def predict(model, model_config):
    # print ('Testing model: {}'.format(str(local_model)))
    model.cuda()
    model.eval()

    batch_size = model_config.getint("batch_size")
    transformations = create_transforms(model_config)
    test_set = QuickDrawDataset(model_config.get("data_path"), split='test', transform=transformations)
    test_loader = TestQuickDrawLoader(test_set, batch_size=batch_size, pin_memory=(torch.cuda.is_available()),
                                      num_workers=0)

    predictions = []
    with torch.no_grad():
        for i, (images, _) in tqdm(enumerate(test_loader), total=int(len(test_set)/batch_size) + (len(test_set)%batch_size > 0)):
            images = images.cuda()
            images = Variable(images)

            # compute y_pred
            y_pred = model(images)
            predictions.extend(y_pred.data.cpu().numpy())

    return np.array(predictions)


if __name__ == '__main__':
    predictions = np.zeros((10000, len(classes)))
    config = ConfigParser()
    config.read('models.config')

    submissions = glob.glob("./log/quickdraw/*_*.csv")
    accuracies = []
    for submission in submissions:
        try:
            accuracies.append(float(submission[submission.rfind("_") + 1:submission.rfind(".")]))
        except ValueError:
            continue

    best_submission_pth = submissions[np.argmax(accuracies)]
    best_submission = np.loadtxt(best_submission_pth, delimiter=',', skiprows=1, dtype=str)
    best_submission = best_submission[:,1]

    for i, model_name in enumerate(config.sections()):
        model_config = config[model_name]
        model = create_model(model_config)
        print("Predicting with", model_config.name, "({}/{})".format(i+1, len(config.sections())))
        checkpoints_folder = os.path.join(model_config.get("save_path"), "quickdraw", model_config.name)
        load_most_recent_best_model(model, checkpoints_folder)
        predictions += predict(model, model_config) / float(len(config.sections()))

    predictions = classes[predictions.argmax(axis=1)]
    print((predictions == best_submission).sum() / 10000)


    csv_name = "ensemble.csv"
    predictions_to_csv(predictions, os.path.join(config.get("DEFAULT", "save_path"), "quickdraw", csv_name))