import matplotlib.pyplot as plt
import os

def plot_history(val_acc_history, output_dir='.'):
    plt.figure()
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'val_accuracy.png'))
    plt.close()
