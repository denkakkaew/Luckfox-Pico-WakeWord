from rknn.api import RKNN
r = RKNN(verbose=False)
r.config(target_platform="rv1106")
assert r.load_tflite(model="/tmp/kws_int8.tflite") == 0, "load failed"
assert r.build(do_quantization=False) == 0, "build failed"   # already int8; don't requantize
assert r.export_rknn("/tmp/kws_i8.rknn") == 0, "export failed"
print("exported /tmp/kws_i8.rknn")
