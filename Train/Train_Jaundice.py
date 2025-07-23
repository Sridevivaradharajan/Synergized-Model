# train_jaundice.py
# CNN Training Script for Jaundice Detection (Model only, without preprocessing)

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Base model setup
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# --- Training phase 1 (frozen base model) ---
# Placeholder: Replace 'train' and 'validation' with your actual datasets
model.fit(train, epochs=10, validation_data=validation, class_weight=class_weight_dict)

# --- Fine-tune last few layers ---
for layer in base_model.layers[:-5]:
    layer.trainable = True

# Re-compile with lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Continue training
model.fit(train, epochs=5, validation_data=validation, class_weight=class_weight_dict)

# Save trained model (optional)
model.save("models/jaundice_model.h5")
