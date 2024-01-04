from tensorflow import keras
import gc
import numpy as np
import pandas as pd

import misc
from data_loader import data_loader, NoLabelDataLoader


def build_finetune_network(base_builder, model_extension, cut_layer, optimizer, metrics, loss,
                           non_trainable_cut_layer=None):
    base_model = base_builder()
    extended_layers = []
    if non_trainable_cut_layer is not None:
        non_trainable_part = keras.Model(inputs=base_model.input,
                                         outputs=base_model.get_layer(non_trainable_cut_layer).output)
        for layer in non_trainable_part.layers:
            layer.trainable = False
        extended_layers.append(non_trainable_part)
        trainable_part = keras.Model(inputs=non_trainable_part.output, outputs=base_model.get_layer(cut_layer).output)
        extended_layers.append(trainable_part)

    else:
        backbone = keras.Model(inputs=base_model.input, outputs=base_model.get_layer(cut_layer).output)
        for layer in backbone.layers:
            layer.trainable = False
        extended_layers.append(backbone)

    extended_layers.append(model_extension)
    extended_model = keras.Sequential(extended_layers)

    extended_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return extended_model


def custom_preprocess(x, augment, strong_augment, preprocess_input):
    if augment:
        x = misc.apply_standard_data_augmentation(x)
    if strong_augment:
        x = misc.apply_data_augmentation(x)
    if preprocess_input is not None:
        x = preprocess_input(x)
    return x


def save_history(loss, acc, val_loss, val_acc, path):
    history_df = pd.DataFrame({
        "loss": loss,
        "acc": acc,
        "val_loss": val_loss,
        "val_acc": val_acc
    })

    history_df.to_csv(path)


def load_history(path):
    history_df = pd.read_csv(path)
    loss_history = history_df['loss']
    val_loss_history = history_df['val_loss']
    acc_history = history_df['acc']
    val_acc_history = history_df['val_acc']

    return loss_history, acc_history, val_loss_history, val_acc_history


def reload_model(model, path):
    model.save(path)
    del model
    gc.collect()
    keras.backend.clear_session()
    return keras.models.load_model(path)


def train_network(model, save_path, train_info, val_info, train_dir, batch_size=64, epochs=10,
                  augment=True, strong_augment=False, preprocess_input=None, reload_rate=1, prev_history=None):
    train_loader = data_loader(train_info, directory=train_dir, batch_size=batch_size,
                               resize_shape=(224, 224))
    val_loader = data_loader(val_info, directory=train_dir, batch_size=batch_size, resize_shape=(224, 224))

    loss_history = []
    acc_history = []
    val_loss_history = []
    val_acc_history = []

    if prev_history is not None:
        loss_history = prev_history['loss']
        acc_history = prev_history['acc']
        val_loss_history = prev_history['val_loss']
        val_acc_history = prev_history['val_acc']

    history_suffix = "_history.csv"

    for epoch in range(epochs):
        if epoch > 0 and epoch % reload_rate == 0:
            save_history(loss_history, acc_history, val_loss_history, val_acc_history, save_path + history_suffix)
            model = reload_model(model, save_path)

        epoch_loss = 0
        epoch_acc = 0
        train_loader.shuffle_dataframe()
        for i in range(train_loader.number_of_batch()):
            next_batch, next_labels, _ = (train_loader
                                          .get_batch(i, lambda x: custom_preprocess(x, augment,
                                                                                    strong_augment,
                                                                                    preprocess_input)))
            labels = keras.utils.to_categorical(next_labels, num_classes=251)
            loss, acc = model.train_on_batch(next_batch, labels)
            epoch_loss += loss * len(next_batch)
            epoch_acc += acc * len(next_batch)
            print('\rEpoch {}/{} iteration {}/{} Loss: {:5.4f} Accuracy: {:5.4f}'
                  .format(epoch + 1, epochs, i + 1,
                          train_loader.number_of_batch(),
                          epoch_loss / min(batch_size * (i + 1), len(train_info)),
                          epoch_acc / min(batch_size * (i + 1), len(train_info))),
                  end='')

        loss_history.append(epoch_loss / float(len(train_info)))
        acc_history.append(epoch_acc / float(len(train_info)))

        val_loss = 0
        val_acc = 0

        for i in range(val_loader.number_of_batch()):
            batch, labels, _ = (val_loader
                                .get_batch(i, preprocessing=lambda x: custom_preprocess(x, False,
                                                                                        False,
                                                                                        preprocess_input)))
            labels = keras.utils.to_categorical(labels, num_classes=251)
            loss, acc = model.evaluate(batch, labels, verbose=0)
            val_loss += loss * len(batch)
            val_acc += acc * len(batch)

        val_loss_history.append(val_loss / float(len(val_info)))
        val_acc_history.append(val_acc / float(len(val_info)))

        print(
            ' Validation Loss: {:5.4f}, Validation accuracy: {:5.4f}'.
            format(val_loss_history[-1], val_acc_history[-1]))

    save_history(loss_history, acc_history, val_loss_history, val_acc_history, save_path + history_suffix)
    model = reload_model(model, save_path)

    return model, loss_history, acc_history, val_loss_history, val_acc_history


def evaluate_model(model, info, dir, batch_size=64, preprocess_input=None):
    loader = data_loader(info, dir, batch_size=batch_size, resize_shape=(224, 224))

    total_loss = 0
    total_acc = 0
    for i in range(loader.number_of_batch()):
        batch, labels, _ = loader.get_batch(i, preprocessing=lambda x: custom_preprocess(x, False,
                                                                                         False,
                                                                                         preprocess_input))
        labels = keras.utils.to_categorical(labels, num_classes=251)
        loss, acc = model.evaluate(batch, labels, verbose=0)
        total_loss += loss
        total_acc += acc
        print(loss, acc)

    print(
        'Validation Loss: {:5.4f}, Validation accuracy: {:5.4f}'.
        format(total_loss / loader.number_of_batch(), total_acc / loader.number_of_batch()))


def predict(model, dir, batch_size, preprocess_input=None):
    loader = NoLabelDataLoader(dir, batch_size=batch_size, target_size=(224, 224))
    predictions = []
    image_names = []
    for i in range(loader.number_of_batches()):
        batch, names = loader.get_batch(i, preprocessing=lambda x: custom_preprocess(x, False,
                                                                                     False,
                                                                                     preprocess_input))
        pred = model.predict(batch)
        for p in pred:
            predictions.append(p)

        for n in names:
            image_names.append(n)

    return np.array(predictions), image_names
