from tensorflow import keras

import misc
from data_loader import data_loader


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


def custom_preprocess(x, training, preprocess_input):
    if training:
        x = misc.apply_standard_data_augmentation(x)
        x = misc.apply_data_augmentation(x)
    if preprocess_input is not None:
        x = preprocess_input(x)
    return x


def train_network(model, train_info, val_info, train_dir, batch_size=64, epochs=1,
                  augment=False, preprocess_input=None):
    train_loader = data_loader(train_info, directory=train_dir, batch_size=batch_size,
                               resize_shape=(224, 224))
    val_loader = data_loader(val_info, directory=train_dir, batch_size=batch_size, resize_shape=(224, 224))

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        train_loader.shuffle_dataframe()
        for i in range(len(train_info) // batch_size):
            next_batch, next_labels, _ = (train_loader
                                          .get_batch(i, lambda x: custom_preprocess(x,augment, preprocess_input)))
            labels = keras.utils.to_categorical(next_labels, num_classes=251)
            loss, acc = model.train_on_batch(next_batch, labels)
            epoch_loss += loss
            epoch_acc += acc
            print('\rEpoch {}/{} iteration {}/{} Loss: {:10.4f} Accuracy: {:10.4f}'
                  .format(epoch + 1, epochs, i + 1,
                          len(train_info) // batch_size,
                          epoch_loss / (i + 1),
                          epoch_acc / (i + 1)),
                  end='')

        val_loss = 0
        val_acc = 0

        for i in range(len(val_info) // batch_size):
            batch, labels, _ = (val_loader
                                .get_batch(i,preprocessing=lambda x: custom_preprocess(x, False, preprocess_input)))
            labels = keras.utils.to_categorical(labels, num_classes=251)
            loss, acc = model.evaluate(batch, labels, verbose=0)
            val_loss += loss
            val_acc += acc

        print(
            ' Validation Loss: {:10.4f}, Validation accuracy: {:10.4f}'.
            format(val_loss / (len(val_info) // batch_size), val_acc / (len(val_info) // batch_size)))


def evaluate_model(model, info, dir, batch_size=64, preprocess_input=None):
    loader = data_loader(info, dir, batch_size=batch_size, resize_shape=(224, 224))

    total_loss = 0
    total_acc = 0
    for i in range(len(info) // batch_size):
        batch, labels, _ = loader.get_batch(i, preprocessing=lambda x: custom_preprocess(x, False, preprocess_input))
        labels = keras.utils.to_categorical(labels, num_classes=251)
        loss, acc = model.evaluate(batch, labels, verbose=0)
        total_loss += loss
        total_acc += acc
        print(loss, acc)

    print(
        'Validation Loss: {:10.4f}, Validation accuracy: {:10.4f}'.
        format(total_loss / (len(info) // batch_size), total_acc / (len(info) // batch_size)))


def predict(model, info, dir, batch_size, preprocess_input=None):
    loader = data_loader(info, dir, batch_size=batch_size, resize_shape=(224, 224))
    predictions = []
    for i in range(len(info) // batch_size):
        batch, _, _ = loader.get_batch(i, preprocessing=lambda x: custom_preprocess(x, False, preprocess_input))
        pred = model.predict(batch)
        for p in pred:
            predictions.append(p)

    return predictions
