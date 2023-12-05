import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, concatenate
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from cgan import *
from cnn import *

def preprocess_data(x, y, digits_to_delete, percentage_to_delete):
    indices_to_delete = np.zeros_like(y, dtype=bool)
    for digit in digits_to_delete:
        digit_indices = np.where(y == digit)[0]
        np.random.shuffle(digit_indices)
        num_to_delete = int(len(digit_indices) * percentage_to_delete)
        indices_to_delete[digit_indices[:num_to_delete]] = True

    x_filtered = x[~indices_to_delete]
    y_filtered = y[~indices_to_delete]

    x_filtered = x_filtered.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    # y_filtered = to_categorical(y_filtered, 10)

    return x_filtered, y_filtered

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def calculate_deleted_digit_errors(digits_to_delete, y_pred, y_true):
    # Predict the classes for the test set

    # Identify indices corresponding to the deleted digits
    deleted_digit_indices = np.isin(y_true, digits_to_delete)

    # Create a confusion matrix for deleted digits
    cm_deleted_digits = confusion_matrix(y_true[deleted_digit_indices], y_pred[deleted_digit_indices])

    # Calculate the total number of errors made on deleted digits
    errors_on_deleted_digits = np.sum(np.sum(cm_deleted_digits)) - np.trace(cm_deleted_digits)

    # Calculate the overall number of errors
    total_errors = np.sum(np.sum(confusion_matrix(y_true, y_pred))) - np.trace(confusion_matrix(y_true, y_pred))

    # Calculate the ratio of errors on deleted digits to total errors
    error_ratio = errors_on_deleted_digits / total_errors

    return errors_on_deleted_digits, total_errors, error_ratio

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Digits and percentage to delete
    digits_to_delete = [3, 5, 7]
    deleted_digits_order = len(digits_to_delete)
    percentage_to_delete = 0.95

    # Preprocess the training data
    x_train_filtered, y_train_filtered = preprocess_data(x_train, y_train, digits_to_delete, percentage_to_delete)
    y_train_filtered_onehot = to_categorical(y_train_filtered, 10)
    num_deleted_digits = x_train.shape[0] - x_train_filtered.shape[0]
    print(f'Deleted {num_deleted_digits} digits from the training dataset')

    x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    y_test = to_categorical(y_test, 10)
    ##### Compile CNN FOR CLASSIFICATION

    errors_deleted_digits_filtered = 0
    total_errors_filtered = 0

    errors_deleted_digits_mod = 0
    total_errors_mod = 0

    n_trials = 5

    gen_model = load_model('c_gan 99.h5')
    for i in range(n_trials):

        cnn_model = create_model()
        trained_model, history = train_model(cnn_model, x_train_filtered, y_train_filtered_onehot)

        # Evaluate the model on the original test set

        test_loss, test_acc = trained_model.evaluate(x_test, y_test)
        print(f'Test accuracy: {test_acc * 100:.2f}%')

        # Predict the classes for the test set
        y_pred = np.argmax(trained_model.predict(x_test), axis=1)
        y_true = np.argmax(y_test, axis=1)

        errors_deleted_digits, total_errors, error_ratio = calculate_deleted_digit_errors(digits_to_delete, y_pred, y_true)
        errors_deleted_digits_filtered += errors_deleted_digits
        total_errors_filtered += total_errors

        print(f"Errors on deleted digits: {errors_deleted_digits}")
        print(f"Total errors: {total_errors}")
        print(f"Error ratio on deleted digits: {error_ratio * 100:.2f}%")

        # Plot the confusion matrix
        class_names = [str(i) for i in range(10)]
        plot_confusion_matrix(y_true, y_pred, class_names)

        ###### Compile CNN FOR CLASSIFICATION

        ###############################################################
        # cGAN digit generation
        latent_dim = 100
        # discriminator model
        # discriminator = define_discriminator()
        # # generator model
        # generator = define_generator(latent_dim)
        # # create the gan model
        # gan = define_gan(generator, discriminator)
        # # dataset
        # dataset = [x_train_filtered, y_train_filtered]
        # # train model
        # train_gan(generator, discriminator, gan, latent_dim, dataset)

        ### Existing cGAN pretrained

        [inputs, labels] = generate_latent_points(latent_dim, num_deleted_digits)
        labels = np.asarray([x for digit in digits_to_delete for x in [digit] * int(np.ceil(num_deleted_digits / deleted_digits_order))])
        labels = labels[:num_deleted_digits]
        X = gen_model.predict([inputs, labels])
        ###############################################################
        x_train_mod = np.concatenate((x_train_filtered, X), axis=0)
        y_train_mod = np.concatenate((y_train_filtered, labels), axis=0)
        y_train_mod_onehot = to_categorical(y_train_mod, 10)
        # Generate second model trained on synthetic data
        cnn_mod_model = create_model()
        trained_model, history = train_model(cnn_mod_model, x_train_mod, y_train_mod_onehot)

        test_loss, test_acc = trained_model.evaluate(x_test, y_test)
        print(f'Test accuracy: {test_acc * 100:.2f}%')

        # Predict the classes for the test set
        y_pred = np.argmax(trained_model.predict(x_test), axis=1)
        y_true = np.argmax(y_test, axis=1)

        errors_deleted_digits, total_errors, error_ratio = calculate_deleted_digit_errors(digits_to_delete, y_pred, y_true)
        errors_deleted_digits_mod += errors_deleted_digits
        total_errors_mod += total_errors

        print(f"Errors on deleted digits after cGAN : {errors_deleted_digits}")
        print(f"Total errors after cGAN: {total_errors}")
        print(f"Error ratio on deleted digits after cGAN: {error_ratio * 100:.2f}%")

    # Figure generation
    NDD_filtered = total_errors_filtered - errors_deleted_digits_filtered
    NDD_mod = total_errors_mod - errors_deleted_digits_mod
    # NDD_control = total_errors_control - errors_deleted_digits_control

    categories = ("w/ Unbalanced Data",
                  "w/ Synthetically Balanced Data",
                  )
    error_counts = {
        "Non-deleted digit errors": np.array([NDD_filtered/n_trials, NDD_mod/n_trials]),
        "Deleted digit errors": np.array([errors_deleted_digits_filtered/n_trials, errors_deleted_digits_mod/n_trials])
    }

    width = .5
    fig, ax = plt.subplots()
    bottom = np.zeros(2)
    for boolean, error_count in error_counts.items():
        p = ax.bar(categories, error_count, width, label = boolean, bottom=bottom)
        bottom += error_count

    ax.set_title("Error counts on deleted digits across different models")
    ax.legend(loc='upper right')
    ax.set_ylabel('# of Errors')
    ax.set_ylim(top=400)
    plt.show()






