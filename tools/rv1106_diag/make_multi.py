import sys, pathlib
import numpy as np, tensorflow as tf
sys.path.insert(0, "/workspaces/Luckfox-Pico-WakeWord/pc")
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
model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
model.fit(train_spec, epochs=T.EPOCHS, verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])

# layers to tap (skip dropout/flatten): conv,conv,pool,conv,pool,dense,dense
taps = [l for l in model.layers if not isinstance(l, (tf.keras.layers.Dropout, tf.keras.layers.Flatten))]
multi = tf.keras.Model(inputs=model.input, outputs=[l.output for l in taps])
for i,l in enumerate(taps): print(f"tap{i}: {l.name} shape={l.output.shape}")

# one 'yes' test spectrogram
xin=None
for specs, labs in test_spec.unbatch().batch(1):
    if int(labs.numpy()[0])==labels.index("yes"): xin=specs.numpy(); break
xin.reshape(-1).astype(np.float32).tofile("/tmp/spec_in.f32")  # (124,128,1) frame-major
print("test input shape", xin.shape, "-> /tmp/spec_in.f32")

# golden float per-layer activations
acts = multi.predict(xin, verbose=0)
for i,a in enumerate(acts):
    np.save(f"/tmp/golden_{i}.npy", a.astype(np.float32))
    print(f"golden_{i}: shape={a.shape} n_elems={a.size} mean={a.mean():.3f} std={a.std():.3f}")
# export multi-output float tflite for rknn
tfl = tf.lite.TFLiteConverter.from_keras_model(multi).convert()
pathlib.Path("/tmp/kws_multi.tflite").write_bytes(tfl)
print("wrote /tmp/kws_multi.tflite")

# also save single-output tflites per tap (for hardware bisection)
for i,l in enumerate(taps):
    sm = tf.keras.Model(inputs=model.input, outputs=l.output)
    open(f"/tmp/tap_{i}.tflite","wb").write(tf.lite.TFLiteConverter.from_keras_model(sm).convert())
print("wrote per-tap single-output tflites tap_0..tap_%d" % (len(taps)-1))
