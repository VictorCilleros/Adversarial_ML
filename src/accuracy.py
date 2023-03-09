from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt


class Accurator():
    def __init__(self,y_test):
        self.y_test = y_test

    def print_report(self, predictions) -> None:
        print(classification_report(self.y_test, predictions))

    def print_confusionMatrix(self, predictions) -> None:

        cm = confusion_matrix(self.y_test, predictions)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(cm)
        ax.grid(False)
        ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0', 'Predicted 1'))
        ax.yaxis.set(ticks=(0, 1), ticklabels=('True 0', 'True 1'))
        ax.set_ylim(1.5, -0.5)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
        plt.show()

