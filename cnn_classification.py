from tensorflow import keras

import misc
from data_loader import data_loader
from keras.applications.efficientnet import preprocess_input


def build_finetune_network(cut_layer='top_dropout', non_trainable_cut_layer='block6d_add', n_classes=251):
    base_model = keras.applications.EfficientNetB0()
    non_trainable_part = keras.Model(inputs=base_model.input,
                                     outputs=base_model.get_layer(non_trainable_cut_layer).output)
    trainable_part = keras.Model(inputs=non_trainable_part.output, outputs=base_model.get_layer(cut_layer).output)
    non_trainable_part.trainable = False
    model = keras.Sequential([non_trainable_part,
                              trainable_part,
                              keras.layers.Dense(n_classes, activation='softmax')])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def custom_preprocess(x):
    # x = misc.apply_standard_data_augmentation(x)
    return preprocess_input(x)


def train_network(model, train_info, train_dir, validation_split=0.2, batch_size=128, epochs=1):
    val_set = train_info.groupby(train_info['label'], group_keys=False).apply(
        lambda group: group.sample(frac=validation_split)
    )
    train_set = train_info[~train_info.iloc[:, 0].isin(val_set.iloc[:, 0])]

    train_loader = data_loader(train_set, directory=train_dir, batch_size=batch_size,
                               resize_shape=(224, 224))
    val_loader = data_loader(val_set, directory=train_dir, batch_size=batch_size, resize_shape=(224, 224))

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        train_loader.shuffle_dataframe()
        for i in range(len(train_set) // batch_size):
            batch, labels, _ = train_loader.get_batch(i, preprocessing=custom_preprocess)
            labels = keras.utils.to_categorical(labels, num_classes=251)
            loss, acc = model.train_on_batch(batch, labels)
            epoch_loss += loss
            epoch_acc += acc
            print('\rEpoch {}/{} iteration {}/{} Loss: {} Accuracy: {}'.format(epoch + 1, epochs, i + 1,
                                                                               len(train_set) // batch_size,
                                                                               epoch_loss / (i + 1),
                                                                               epoch_acc / (i + 1)),
                  end='')

        val_loss = 0
        val_acc = 0

        for i in range(len(val_set) // batch_size):
            batch, labels, _ = val_loader.get_batch(i, preprocessing=custom_preprocess)
            labels = keras.utils.to_categorical(labels, num_classes=251)
            loss, acc = model.evaluate(batch, labels)
            val_loss += loss
            val_acc += acc

        print(' Validation Loss: {}, Validation accuracy: {}'.format(val_loss / (len(val_set) // batch_size),
                                                                     val_acc / (len(val_set) // batch_size)))
