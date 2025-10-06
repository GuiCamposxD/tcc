import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Concatenate, Add, Activation, Multiply
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

# Hierarchical label mappings
c2_to_c1 = {0:0, 1:0, 2:1, 3:1, 4:2, 5:2}
fine_to_c2 = {0:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3, 12:4, 13:4, 14:4, 15:5, 16:5, 17:5}

def scheduler(epoch):
    learning_rate_init = 0.001
    if epoch > 55:
        learning_rate_init = 0.0002
    if epoch > 70:
        learning_rate_init = 0.00005
    return learning_rate_init

class LossWeightsModifier(tf.keras.callbacks.Callback):
    def __init__(self, alpha, beta, gamma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    def on_epoch_end(self, epoch, logs={}):
        if epoch == 15:
            K.set_value(self.alpha, 0.1)
            K.set_value(self.beta, 0.8)
            K.set_value(self.gamma, 0.1)
        if epoch == 25:
            K.set_value(self.alpha, 0.1)
            K.set_value(self.beta, 0.2)
            K.set_value(self.gamma, 0.7)
        if epoch == 35:
            K.set_value(self.alpha, 0)
            K.set_value(self.beta, 0)
            K.set_value(self.gamma, 1)

def create_model(input_shape, bt_strategy=True, branch_neurons=128, att_neurons=128):
    alpha = K.variable(0.98 if bt_strategy else 0.33, dtype="float32", name="alpha")
    beta = K.variable(0.01 if bt_strategy else 0.33, dtype="float32", name="beta")
    gamma = K.variable(0.01 if bt_strategy else 0.34, dtype="float32", name="gamma")

    img_input = Input(shape=input_shape, name='input')

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Coarse 1 branch
    c_1_bch = Flatten(name='c1_flatten')(x)
    c_1_bch = Dense(branch_neurons, activation='relu')(c_1_bch)
    c_1_bch = BatchNormalization()(c_1_bch)
    c_1_bch = Dropout(0.5)(c_1_bch)
    c_1_bch = Dense(branch_neurons, activation='relu')(c_1_bch)
    c_1_bch = BatchNormalization()(c_1_bch)
    c_1_bch = Dropout(0.5)(c_1_bch)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Coarse 2 branch
    c_2_bch = Flatten(name='c2_flatten')(x)
    c_2_bch = Dense(branch_neurons, activation='relu')(c_2_bch)
    c_2_bch = BatchNormalization()(c_2_bch)
    c_2_bch = Dropout(0.5)(c_2_bch)
    c_2_bch = Dense(branch_neurons, activation='relu')(c_2_bch)
    c_2_bch = BatchNormalization()(c_2_bch)
    c_2_bch = Dropout(0.5)(c_2_bch)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    # Fine branch
    x = Flatten(name='flatten')(x)
    x = Dense(branch_neurons, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(branch_neurons, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # Attention for coarse 1
    sfcn_1_1 = Dense(att_neurons, name='fc1_1')(c_1_bch)
    sfcn_1_1 = Dense(1, name='fc1_2')(sfcn_1_1)
    sfcn_1_2 = Dense(att_neurons, name='fc1_3')(c_2_bch)
    sfcn_1_2 = Dense(1, name='fc1_4')(sfcn_1_2)
    sfcn_1_3 = Dense(att_neurons, name='fc1_5')(x)
    sfcn_1_3 = Dense(1, name='fc1_6')(sfcn_1_3)
    score_vector_1 = Concatenate()([sfcn_1_1, sfcn_1_2, sfcn_1_3])
    att_weights_1 = Activation('softmax', name='attention_weights_1')(score_vector_1)
    weighted_c1_bch_1 = Multiply()([c_1_bch, att_weights_1[:, 0:1]])
    weighted_c2_bch_1 = Multiply()([c_2_bch, att_weights_1[:, 1:2]])
    weighted_x_1 = Multiply()([x, att_weights_1[:, 2:3]])
    weightned_sum_1 = Add()([weighted_c1_bch_1, weighted_c2_bch_1, weighted_x_1])
    coarse_1_concat = Concatenate()([c_1_bch, weightned_sum_1])
    c_1_pred = Dense(3, activation='softmax', name='c1_predictions_clothes')(coarse_1_concat)

    # Attention for coarse 2
    sfcn_2_1 = Dense(att_neurons, name='fc2_1')(c_1_bch)
    sfcn_2_1 = Dense(1, name='fc2_2')(sfcn_2_1)
    sfcn_2_2 = Dense(att_neurons, name='fc2_3')(c_2_bch)
    sfcn_2_2 = Dense(1, name='fc2_4')(sfcn_2_2)
    sfcn_2_3 = Dense(att_neurons, name='fc2_5')(x)
    sfcn_2_3 = Dense(1, name='fc2_6')(sfcn_2_3)
    score_vector_2 = Concatenate()([sfcn_2_1, sfcn_2_2, sfcn_2_3])
    att_weights_2 = Activation('softmax', name='attention_weights_2')(score_vector_2)
    weighted_c1_bch_2 = Multiply()([c_1_bch, att_weights_2[:, 0:1]])
    weighted_c2_bch_2 = Multiply()([c_2_bch, att_weights_2[:, 1:2]])
    weighted_x_2 = Multiply()([x, att_weights_2[:, 2:3]])
    weightned_sum_2 = Add()([weighted_c1_bch_2, weighted_c2_bch_2, weighted_x_2])
    coarse_2_concat = Concatenate()([c_2_bch, weightned_sum_2])
    c_2_pred = Dense(6, activation='softmax', name='c2_predictions_clothes')(coarse_2_concat)

    # Attention for fine
    sfcn_3_1 = Dense(att_neurons, name='fc3_1')(c_1_bch)
    sfcn_3_1 = Dense(1, name='fc3_2')(sfcn_3_1)
    sfcn_3_2 = Dense(att_neurons, name='fc3_3')(c_2_bch)
    sfcn_3_2 = Dense(1, name='fc3_4')(sfcn_3_2)
    sfcn_3_3 = Dense(att_neurons, name='fc3_5')(x)
    sfcn_3_3 = Dense(1, name='fc3_6')(sfcn_3_3)
    score_vector_3 = Concatenate()([sfcn_3_1, sfcn_3_2, sfcn_3_3])
    att_weights_3 = Activation('softmax', name='attention_weights_3')(score_vector_3)
    weighted_c1_bch_3 = Multiply()([c_1_bch, att_weights_3[:, 0:1]])
    weighted_c2_bch_3 = Multiply()([c_2_bch, att_weights_3[:, 1:2]])
    weighted_x_3 = Multiply()([x, att_weights_3[:, 2:3]])
    weightned_sum_3 = Add()([weighted_c1_bch_3, weighted_c2_bch_3, weighted_x_3])
    fine_concat = Concatenate()([x, weightned_sum_3])
    fine_pred = Dense(18, activation='softmax', name='predictions_clothes')(fine_concat)

    model = Model(img_input, [c_1_pred, c_2_pred, fine_pred], name='bacnn')

    opt = optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
    model.compile(
        loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
        optimizer=opt,
        loss_weights=[alpha, beta, gamma],
        metrics=['accuracy']
    )

    change_lr = LearningRateScheduler(scheduler)
    change_lw = LossWeightsModifier(alpha, beta, gamma) if bt_strategy else None
    
    return model, [change_lr, change_lw] if bt_strategy else [change_lr]

if __name__ == '__main__':
    # Load data
    data_path = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
    pkl_file = os.path.join(data_path, 'dados_32x32.pkl')
    print(f'Loading data from: {pkl_file}')
    print(f'Files in {data_path}: {os.listdir(data_path)}')
    
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
        x_train = data['x_train']
        y_train = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']

    # Prepare labels
    train_labels_fine = to_categorical(y_train)
    test_labels_fine = to_categorical(y_test)
    train_labels_c2 = to_categorical([fine_to_c2[i] for i in y_train])
    test_labels_c2 = to_categorical([fine_to_c2[i] for i in y_test])
    train_labels_c1 = to_categorical([c2_to_c1[fine_to_c2[i]] for i in y_train])
    test_labels_c1 = to_categorical([c2_to_c1[fine_to_c2[i]] for i in y_test])

    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    # Create model
    input_shape = (32, 32, 3)
    model, callbacks = create_model(input_shape, bt_strategy=True, branch_neurons=128, att_neurons=128)

    # Train
    batch_size = int(os.environ.get('SM_HP_BATCH_SIZE', '128'))
    epochs = int(os.environ.get('SM_HP_EPOCHS', '100'))
    print(f'Training with batch_size={batch_size}, epochs={epochs}')
    print(f'Train shape: {x_train.shape}, Test shape: {x_test.shape}')
    
    model.fit(
        x_train,
        [train_labels_c1, train_labels_c2, train_labels_fine],
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, [test_labels_c1, test_labels_c2, test_labels_fine]),
        callbacks=callbacks
    )

    # Save model
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    model.save(f'{model_dir}/bacnn_model.h5')
