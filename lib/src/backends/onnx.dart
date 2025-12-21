import 'dart:collection';

import 'package:ort/ort.dart' hide Tensor, DeviceType;
import 'package:ort/ort.dart' as ort;
import 'package:transformers/src/utils/devices.dart';
import 'package:transformers/src/utils/tensor.dart' show TensorDataType;

/// @type {Record<import("../utils/devices.js").DeviceType, ONNXExecutionProviders>}
final Map<DeviceType, ExecutionProvider?> DEVICE_TO_EXECUTION_PROVIDER_MAPPING = {
  DeviceType.auto: null, // Auto-detect based on device and environment
  DeviceType.gpu: null, // Auto-detect GPU
  DeviceType.cpu: CPUExecutionProvider(), // CPU
  DeviceType.wasm: null, // WebAssembly
  DeviceType.webgpu: null, // WebGPU TODO: Not yet implemented in ort
  DeviceType.cuda: CUDAExecutionProvider(), // CUDA
  DeviceType.dml: DirectMLExecutionProvider(), // DirectML

  DeviceType.webnn: null, // WebNN (default) TODO: Not yet implemented in ort
  DeviceType.webnnNpu: null, // WebNN NPU TODO
  DeviceType.webnnGpu: null, // WebNN GPU TODO
  DeviceType.webnnCpu: null, // WebNN CPU TODO
};

final List<ExecutionProvider> defaultDevices = [CPUExecutionProvider()];

/// DO NOT ACCESS DIRECTLY. Use [getSupportedDevices] instead.
List<ExecutionProvider>? _supportedDevices;

/// Get the list of supported devices, sorted by priority/performance.
/// TODO: we need to handle sorted by priority/performance
List<ExecutionProvider> getSupportedDevices() {
  if (_supportedDevices != null) return _supportedDevices!;

  _supportedDevices = getAvailableExecutionProviders();
  return _supportedDevices!;
}

/// This is a wrapper class around the ort library's [OrtValue] to make
/// things easier to work with.
class Tensor<T> with ListMixin<T> {
  /// The type of the data stored in the tensor
  final TensorDataType dataType;

  /// The data stored in the tensor
  List<T> get data => ortTensor.data.cast<T>();

  /// The shape of the data
  final List<int> shape;

  /// The underlying reference to the onnx tensor
  ort.Tensor ortTensor;

  Tensor._(this.dataType, this.shape, this.ortTensor);

  /// The shape/dimension of the tensor
  List<int> get dims => ortTensor.shape;

  /// Type of the tensor
  TensorDataType get type => dataType;

  /// The number of elements in the tensor
  int get size => data.length;

  /// The location of the tensor data
  AllocationDevice get location => ortTensor.memoryInfo().allocationDevice();

  /// Create an onnx tensor with the given type
  static Tensor<T> create<T>({
    required TensorDataType dataType,
    required List<T> data,
    List<int>? shape,
  }) {
    // Assume 1-D tensor if shape omitted
    shape ??= [1, data.length];

    final ort.TensorElementType ortDataType = _tensorDataTypeToOrtDataType(dataType);

    final ortValue = ort.Tensor.fromArray(
      dtype: ortDataType,
      data: data,
      shape: shape,
    );

    return Tensor._(dataType, shape, ortValue);
  }

  static Tensor<T> createFromOrtTensor<T>(ort.Tensor ortTensor) {
    final TensorDataType dataType = switch (ortTensor.dtype) {
      TensorElementType.float64 => TensorDataType.float64,
      TensorElementType.float32 => TensorDataType.float32,
      TensorElementType.float16 => TensorDataType.float16,
      TensorElementType.bfloat16 => TensorDataType.bfloat16,

      TensorElementType.int64 => TensorDataType.int64,
      TensorElementType.uint64 => TensorDataType.uint64,
      TensorElementType.int32 => TensorDataType.int32,
      TensorElementType.uint32 => TensorDataType.uint32,
      TensorElementType.int16 => TensorDataType.int16,
      TensorElementType.uint16 => TensorDataType.uint16,
      TensorElementType.int8 => TensorDataType.int8,
      TensorElementType.uint8 => TensorDataType.uint8,

      TensorElementType.bool => TensorDataType.bool,
      TensorElementType.string => TensorDataType.string,

      TensorElementType.complex64 => throw UnimplementedError(
          'OrtDataType.complex64 is not currently supported'),
      TensorElementType.complex128 => throw UnimplementedError(
          'OrtDataType.complex128 is not currently supported'),

      TensorElementType.float8E4M3Fn => throw UnimplementedError(),
      TensorElementType.float8E4M3Fnuz => throw UnimplementedError(),
      TensorElementType.float8E5M2 => throw UnimplementedError(),
      TensorElementType.float8E5M2Fnuz => throw UnimplementedError(),
      TensorElementType.uint4 => throw UnimplementedError(),
      TensorElementType.int4 => throw UnimplementedError(),
      TensorElementType.undefined => throw ArgumentError('Unsupported data type ${TensorElementType.undefined}'),
    };
    final List<int> shape = ortTensor.shape;

    return Tensor._(dataType, shape, ortTensor);
  }

  static TensorElementType _tensorDataTypeToOrtDataType(TensorDataType dataType) => switch (dataType) {
    TensorDataType.float64 => TensorElementType.float64,
    TensorDataType.float32 => TensorElementType.float32,
    TensorDataType.float16 => TensorElementType.float16,
    TensorDataType.bfloat16 => TensorElementType.bfloat16,

    TensorDataType.int64 => TensorElementType.int64,
    TensorDataType.uint64 => TensorElementType.uint64,
    TensorDataType.int32 => TensorElementType.int32,
    TensorDataType.uint32 => TensorElementType.uint32,
    TensorDataType.int16 => TensorElementType.int16,
    TensorDataType.uint16 => TensorElementType.uint16,
    TensorDataType.int8 => TensorElementType.int8,
    TensorDataType.uint8 => TensorElementType.uint8,
    TensorDataType.uint4 => TensorElementType.uint8,
    TensorDataType.int4 => TensorElementType.int8,

    TensorDataType.string => TensorElementType.string,
    TensorDataType.bool => TensorElementType.bool,
  };

  /// Casts the generic [T] of this class to a new type. Ideal for updating a
  /// [Tensor<dynamic>] to it's known type.
  ///
  /// Note: This does not convert the data to a new type - only updating dart's
  /// generic type for this class.
  ///
  /// Important: The returned [Tensor] points to the same underlying data and is
  /// not copied into a new Tensor.
  @override
  Tensor<U> cast<U>() => Tensor._(dataType, shape, ortTensor);

  Future<void> dispose() async {
    ortTensor.dispose();
  }

  @override
  set length(int newLength) {
    throw StateError('Tensor is not growable');
  }

  @override
  int get length => data.length;

  @override
  T operator [](int index) => data[index];

  @override
  void operator []=(int index, T value) {
    data[index] = value;
  }
}

/// Creates a view to another [Tensor] by subarray.
class TensorSubarray<T> extends Tensor<T> {
  final int begin;

  final int end;

  @override
  List<T> get data => this;

  final List<int> _dims;

  @override
  List<int> get dims => _dims;

  TensorSubarray._(super.dataType, super.shape, super.ortTensor, this._dims, int? begin, int? end)
      : begin = begin ?? 0,
        end = end ?? ortTensor.data.length,
        super._();

  static TensorSubarray<T> subarray<T>(Tensor<T> tensor, List<int> iterDims, [int? begin, int? end]) {
    return TensorSubarray<T>._(tensor.dataType, tensor.shape, tensor.ortTensor, iterDims, begin, end);
  }

  @override
  Tensor<U> cast<U>() => TensorSubarray._(dataType, shape, ortTensor, _dims, begin, end);

  @override
  set length(int newLength) {
    throw StateError('Tensor is not growable');
  }

  @override
  int get length => super.data.length - begin - (super.data.length - end);

  @override
  T operator [](int index) => super.data[index + begin];

  @override
  void operator []=(int index, T value) {
    super.data[index + begin] = value;
  }
}

/// Map a device to the execution providers to use for the given device.
/// @param {import("../utils/devices.js").DeviceType|"auto"|null} [device=null] (Optional) The device to run the inference on.
/// @returns {ONNXExecutionProviders[]} The execution providers to use for the given device.
List<ExecutionProvider> deviceToExecutionProviders([DeviceType? device]) {
  // Use the default execution providers if the user hasn't specified anything
  if (device == null) return defaultDevices;

  final List<ExecutionProvider> supportedDevices = getSupportedDevices();

  // Handle overloaded cases
  switch (device) {
    case DeviceType.auto:
      return supportedDevices;
    case DeviceType.gpu:
      return supportedDevices.where((x) => [
        // OrtProvider.WEB_GPU, // TODO: Not yet implemented in ort
        CUDAExecutionProvider(),
        ROCmExecutionProvider(),
        DirectMLExecutionProvider(),
        // OrtProvider.WEB_NN, // webnn-gpu TODO: Not yet implemented in ort
      ].contains(x)).toList();
    default:
      // Continue on :)
  }

  final ExecutionProvider? ortDevice = DEVICE_TO_EXECUTION_PROVIDER_MAPPING[device];
  if (ortDevice != null && supportedDevices.contains(ortDevice)) {
    return [ortDevice];
  }

  throw Exception('Unsupported device: "$device". Should be one of: ${supportedDevices.join(', ')}.');
}

/// Create an ONNX inference session
/// [buffer_or_path] - The path to the model [String] or a buffer of the model [List<int>]
///
/// @param {Uint8Array|string} buffer_or_path The ONNX model buffer or path.
/// @param {import('onnxruntime-common').InferenceSession.SessionOptions} session_options ONNX inference session options.
/// @param {Object} session_config ONNX inference session configuration.
/// @returns {Promise<import('onnxruntime-common').InferenceSession & { config: Object}>} The ONNX inference session.
Future<Session> createInferenceSession(
  dynamic buffer_or_path,
  Map<String, dynamic> session_options,
  Map<String, dynamic> session_config,
) async {
  // TODO: session_config is not currently used. Should it be? How would it be used?
  if (buffer_or_path is String) {
    SessionBuilder builder = Session.builder();
    final executionProviders = session_options['executionProviders'];
    if (executionProviders != null) {
      builder = builder.withExecutionProviders(executionProviders);
    }
    return await builder.commitFromFile(buffer_or_path);
  }

  return await createInferenceSessionBytes(buffer_or_path, session_options);
}

/// Create sn ONNX Runtime session with the given bytes as the model
Future<Session> createInferenceSessionBytes(
  List<int> buffer,
  Map<String, dynamic> sessionOptions,
  [Map<String, dynamic>? session_config]
) async {
  SessionBuilder builder = Session.builder();
  final executionProviders = sessionOptions['executionProviders'];
  if (executionProviders != null) {
    builder = builder.withExecutionProviders(executionProviders);
  }
  return await builder.commitFromMemory(buffer);
}

/// TODO: This is not implemented and will always be false
/// Check if ONNX's WASM backend is being proxied.
/// @returns {boolean} Whether ONNX's WASM backend is being proxied.
bool isONNXProxy() => false;
