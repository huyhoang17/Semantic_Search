from keras.callbacks import ModelCheckpoint

from src.models import custom_model
from src.loaders import TextSequenceGenerator

num_epochs = 10
batch_size = 16

data_generator = TextSequenceGenerator(
    mode="train", batch_size=batch_size, shuffle=True)
val_data_generator = TextSequenceGenerator(
    mode="val", batch_size=batch_size, shuffle=False)

checkpointer = ModelCheckpoint(
    filepath='path-to-model-folder/models/checkpoint.hdf5',
    verbose=1, save_best_only=True
)
custom_model.fit_generator(
    data_generator, steps_per_epoch=100000 // batch_size,
    epochs=num_epochs,
    validation_data=val_data_generator, validation_steps=10000 // batch_size,
    callbacks=[checkpointer]
)
model_json = custom_model.to_json()
with open('path-to-model-folder/models/config.json', 'w') as f:
    f.write(model_json)

custom_model.save_weights(
    'path-to-model-folder/models/best_model.hdf5')
