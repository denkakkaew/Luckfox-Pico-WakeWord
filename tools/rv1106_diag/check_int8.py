#!/usr/bin/env python3
"""Local INT8 validation: train, post-training-quantize to a full INT8 TFLite,
and evaluate INT8 accuracy on the test set using TFLite's int8 interpreter.

This is a faithful local int8 signal (the rknn 'simulator' runs ~float and hides
int8 collapse). Tells us if the model survives int8 at all, before any board trip.
"""
import sys, pathlib
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(pathlib.Path("/workspaces/Luckfox-Pico-WakeWord/pc")))
import train_export as T

data_dir = T.resolve_data_dir(None)
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=str(data_dir), batch_size=T.BATCH_SIZE, validation_split=0.2,
    seed=T.SEED, output_sequence_length=T.CLIP_SAMPLES, subset="both")
labels = list(train_ds.class_names)
sq = lambda a, l: (tf.squeeze(a, -1), l)
train_ds = train_ds.map(sq); val_ds = val_ds.map(sq)
test_ds = val_ds.shard(2, 0)
train_spec = T.to_spectrogram_ds(train_ds).cache().prefetch(tf.data.AUTOTUNE)
test_spec = T.to_spectrogram_ds(test_ds)

model = T.build_model(len(labels))
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
model.fit(train_spec, epochs=T.EPOCHS, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])
_, fa = model.evaluate(test_spec, verbose=0)
print(f"FLOAT keras test acc: {fa:.3f}")

# Post-training full-INT8 quantization
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
def rep():
    for spec, _ in train_spec.unbatch().batch(1).take(200):
        yield [tf.cast(spec, tf.float32)]
conv.representative_dataset = rep
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
int8_tfl = conv.convert()

# Evaluate the INT8 tflite on the test set
interp = tf.lite.Interpreter(model_content=int8_tfl); interp.allocate_tensors()
ind = interp.get_input_details()[0]; outd = interp.get_output_details()[0]
in_scale, in_zp = ind["quantization"]
correct = 0; total = 0
pred_hist = np.zeros(len(labels), int); true_hist = np.zeros(len(labels), int)
for specs, labs in test_spec.unbatch().batch(1):
    x = specs.numpy().astype(np.float32)
    q = np.clip(np.round(x / in_scale) + in_zp, -128, 127).astype(np.int8)
    interp.set_tensor(ind["index"], q); interp.invoke()
    o = interp.get_tensor(outd["index"])[0].astype(np.int32)
    p = int(np.argmax(o)); t = int(labs.numpy()[0])
    correct += (p == t); total += 1
    pred_hist[p] += 1; true_hist[t] += 1
print(f"INT8 tflite test acc: {correct/total:.3f}  (n={total})")
print("INT8 prediction histogram:", dict(zip(labels, pred_hist.tolist())))
print("true distribution       :", dict(zip(labels, true_hist.tolist())))
