from sklearn import datasets,svm, metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
_, axes = plt.subplots(2,4)

images_and_labels = list(zip(digits.images, digits.target))
for ax,(image, label) in zip(axes[0, :], images_and_labels[:4]):
    ax.set_axis_off()
    ax.imshow(image , cmap= plt.cm.gray_r , interpolation = 'nearest')
    ax.set_title('Training: %i' % label)

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

clf =svm.SVC(gamma = 0.001)

x_train,x_test , y_train,y_test= train_test_split(data,digits.target,test_size = 0.5 , shuffle = False)


clf.fit(x_train,y_train)
pred = clf.predict(x_test)

images_and_predictions = list(zip(digits.images[n_samples // 2:], pred))

for ax, (image, prediction) in zip(axes[1, :], images_and_predictions[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Prediction: %s' % pred)


print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(y_test, pred)))
disp = metrics.plot_confusion_matrix(clf, x_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)

plt.show()