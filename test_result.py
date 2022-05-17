from train_lte_nr import return_loader, train_sample, test
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score


lte_file = 'dataset/data_lte.mat'
nr_file = 'dataset/data_nr.mat'

_, _, test_dataloader, _, x_test_transformed, _, test_label = return_loader(lte_file, nr_file)
mse, latent_vector = test()

def _z_score(mse):
    median_val_mse = np.median(mse)
    diff = np.abs(mse - median_val_mse)
    median_of_diff = np.median(diff)
    
    return 0.6745 * diff/median_of_diff

z_score = _z_score(mse)
Threshold= 0.01
outliers = mse > Threshold

print("There are {} outliers in a total of {} signals.".format(np.sum(outliers), len(z_score)))


# Plot reconstruction loss

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(mse, label='MSE', color='b', linewidth=1)
plt.axhline(Threshold, label='Threshold', color='r', linestyle='-')
plt.legend(loc='upper left')
ax.set_title('Reconstruction loss graph in deep autoencoder', fontsize=16)
plt.xlabel('Test samples')
plt.ylabel('Loss')
plt.savefig('results/dense_autoencoder_singleMCS_multiprotocol.png')
plt.show

# lte label = 0, nr label = 1
lte_indices = np.where(test_label == 0)
nr_indices = np.where(test_label == 1)

z_lte = latent_vector[lte_indices[0], :].to('cpu')
z_nr = latent_vector[nr_indices[0], :].to('cpu')

plt.subplots(figsize=(8,8))
plt.scatter(z_lte[:,0],z_lte[:,1], s=5, c='g', alpha=0.5, label='LTE')
plt.scatter(z_nr[:,0],z_nr[:,1], s=5, c='r', alpha=0.5, label='5G NR')
plt.legend()
plt.title('Deep autoencoder\'s Latent Space Representation')

plt.savefig('results/Dense autoencoder\'s Latent Space Representation_singleMCS_multiprotocol.png')

# each latent space of lte and nr

plt.subplots(figsize=(8,8))
plt.scatter(z_lte[:,0],z_lte[:,1], s=5, c='g', alpha=0.5, label='LTE')
plt.legend()
plt.title('Deep autoencoder\'s LTE Latent Space Representation')
plt.savefig('results/DeepAE_latent_LTE.png')

plt.subplots(figsize=(8,8))
plt.scatter(z_nr[:,0],z_nr[:,1], s=5, c='r', alpha=0.5, label='5G NR')
plt.legend()
plt.title('Deep autoencoder\'s NR Latent Space Representation')
plt.savefig('results/DeepAE_latent_NR.png')

test_label = test_label.T
test_label = test_label.flatten()
outlier_label = np.logical_xor(test_label,outliers)
print(np.sum(test_label!=outlier_label))
cm = confusion_matrix(test_label,outlier_label)

ax=plt.subplot()
sns.set(font_scale=1.2)
sns.heatmap(cm,annot=True, fmt='g')

ax.set_xlabel('Predicted values')
ax.set_ylabel('True values')

ax.set_yticklabels(['LTE','NR'])
ax.set_xticklabels(['LTE','NR'])
ax.set_title('Confusion matrix for deep autoencoder')

plt.savefig('results/conf_matrix_dense_autoencoder_singleMCS_multiprotocol.png')


# Precision, Recall and F1-score

print("Precision of classification:", "%.4f" % (precision_score(test_label,outlier_label)*100),"%")
print("Recall of classification:","%.4f" % (recall_score(test_label,outlier_label)*100),"%")
print("F1-score of classification:","%.4f" % (f1_score(test_label,outlier_label)))