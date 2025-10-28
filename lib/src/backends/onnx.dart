import 'dart:io';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';
import 'package:transformers/src/utils/devices.dart';
import 'package:transformers/src/utils/tensor.dart' show TensorDataType;

final OnnxRuntime ort = OnnxRuntime();

/// @type {Record<import("../utils/devices.js").DeviceType, ONNXExecutionProviders>}
const Map<DeviceType, OrtProvider?> DEVICE_TO_EXECUTION_PROVIDER_MAPPING = {
  DeviceType.auto: null, // Auto-detect based on device and environment
  DeviceType.gpu: null, // Auto-detect GPU
  DeviceType.cpu: OrtProvider.CPU, // CPU
  DeviceType.wasm: OrtProvider.WEB_ASSEMBLY, // WebAssembly
  DeviceType.webgpu: OrtProvider.WEB_GPU, // WebGPU
  DeviceType.cuda: OrtProvider.CUDA, // CUDA
  DeviceType.dml: OrtProvider.DIRECT_ML, // DirectML

  DeviceType.webnn: OrtProvider.WEB_NN, // WebNN (default)
  DeviceType.webnnNpu: OrtProvider.WEB_NN, // WebNN NPU TODO
  DeviceType.webnnGpu: OrtProvider.WEB_NN, // WebNN GPU TODO
  DeviceType.webnnCpu: OrtProvider.WEB_NN, // WebNN CPU TODO
};

const List<OrtProvider> defaultDevices = [OrtProvider.CPU];

/// DO NOT ACCESS DIRECTLY. Use [getSupportedDevices] instead.
List<OrtProvider>? _supportedDevices;

/// Get the list of supported devices, sorted by priority/performance.
/// TODO: Check if we need to handle sorted by priority/performance or if the flutter_onnxruntime does this already.
Future<List<OrtProvider>> getSupportedDevices() async {
  if (_supportedDevices != null) return _supportedDevices!;

  _supportedDevices = await ort.getAvailableProviders();
  return _supportedDevices!;
}

extension OrtSessionOptionsEx on OrtSessionOptions {
  OrtSessionOptions clone() => OrtSessionOptions(
    intraOpNumThreads: intraOpNumThreads,
    interOpNumThreads: interOpNumThreads,
    providers: providers,
    useArena: useArena,
    deviceId: deviceId,
  );

  OrtSessionOptions cloneWith({
    int? intraOpNumThreads,
    int? interOpNumThreads,
    List<OrtProvider>? providers,
    bool? useArena,
    int? deviceId,
  }) => OrtSessionOptions(
    intraOpNumThreads: intraOpNumThreads ?? this.intraOpNumThreads,
    interOpNumThreads: interOpNumThreads ?? this.interOpNumThreads,
    providers: providers ?? this.providers,
    useArena: useArena ?? this.useArena,
    deviceId: deviceId ?? this.deviceId,
  );

  static OrtSessionOptions fromJson(Map<String, dynamic> json) {
    throw UnimplementedError('`fromJson` is not implemented for OrtSessionOptions');
    return OrtSessionOptions(
      // intraOpNumThreads: intraOpNumThreads,
      // interOpNumThreads: interOpNumThreads,
      // providers: providers,
      // useArena: useArena,
      // deviceId: deviceId,
    );
  }

  Map<String, dynamic> toJson() => toMap();
}

/// This is a wrapper class around the flutter_onnxruntime [OrtValue] to make
/// things easier to work with.
class Tensor<T> {
  /// The type of the data stored in the tensor
  final TensorDataType dataType;

  /// The data stored in the tensor
  List<T> data;

  /// The shape of the data
  final List<int> shape;

  /// The underlying reference to the onnx tensor
  OrtValue ortValue;

  Tensor._(this.dataType, this.data, this.shape, this.ortValue);

  /// The shape/dimension of the tensor
  List<int> get dims => ortValue.shape;

  /// Type of the tensor
  TensorDataType get type => dataType;

  /// The number of elements in the tensor
  int get size => data.length;

  /// The location of the tensor data
  /// TODO: What does that mean?
  String get location => '';

  /// Create an onnx tensor with the given type
  static Future<Tensor<T>> create<T>({
    required TensorDataType dataType,
    required List<T> data,
    List<int>? shape,
  }) async {
    // Assume 1-D tensor if shape omitted
    shape ??= [1, data.length];

    final OrtDataType ortDataType = _tensorDataTypeToOrtDataType(dataType);
    final dynamic ortListData = _getOrtListData(dataType, data);

    // First create the tensor
    final ortValue = await OrtValue.fromList(ortListData, shape);
    // Now convert the tensor to the correct type
    await ortValue.to(ortDataType);

    return Tensor._(dataType, data, shape, ortValue);
  }

  static Future<Tensor<T>> createFromOrtValue<T>(OrtValue ortValue) async {
    final TensorDataType dataType = switch (ortValue.dataType) {
      OrtDataType.float32 => TensorDataType.float32,
      OrtDataType.float16 => TensorDataType.float16,
      OrtDataType.bfloat16 => TensorDataType.bfloat16,

      OrtDataType.int64 => TensorDataType.int64,
      OrtDataType.uint64 => TensorDataType.uint64,
      OrtDataType.int32 => TensorDataType.int32,
      OrtDataType.uint32 => TensorDataType.uint32,
      OrtDataType.int16 => TensorDataType.int16,
      OrtDataType.uint16 => TensorDataType.uint16,
      OrtDataType.int8 => TensorDataType.int8,
      OrtDataType.uint8 => TensorDataType.uint8,

      OrtDataType.bool => TensorDataType.bool,
      OrtDataType.string => TensorDataType.string,

      OrtDataType.complex64 => throw UnimplementedError(
          'OrtDataType.complex64 is not currently supported'),
      OrtDataType.complex128 => throw UnimplementedError(
          'OrtDataType.complex128 is not currently supported'),
    };
    final List<T> data = List<T>.from(await ortValue.asFlattenedList());
    final List<int> shape = ortValue.shape;

    return Tensor._(dataType, data, shape, ortValue);
  }

  static OrtDataType _tensorDataTypeToOrtDataType(TensorDataType dataType) => switch (dataType) {
    TensorDataType.float64 => throw UnimplementedError(
        'float64 is not implemented for the flutter_onnxruntime package'),

    TensorDataType.float32 => OrtDataType.float32,
    TensorDataType.float16 => OrtDataType.float16,
    TensorDataType.bfloat16 => OrtDataType.bfloat16,

    TensorDataType.int64 => OrtDataType.int64,
    TensorDataType.uint64 => OrtDataType.uint64,
    TensorDataType.int32 => OrtDataType.int32,
    TensorDataType.uint32 => OrtDataType.uint32,
    TensorDataType.int16 => OrtDataType.int16,
    TensorDataType.uint16 => OrtDataType.uint16,
    TensorDataType.int8 => OrtDataType.int8,
    TensorDataType.uint8 => OrtDataType.uint8,
    TensorDataType.uint4 => OrtDataType.uint8,
    TensorDataType.int4 => OrtDataType.int8,

    TensorDataType.string => OrtDataType.string,
    TensorDataType.bool => OrtDataType.bool,
  };

  static dynamic _getOrtListData<T>(TensorDataType dataType, List<T> data) {
    // TODO: Might be nice to coerce the type?
    final dynamic ortListData;

    switch (dataType) {
      case TensorDataType.float64:
        throw UnimplementedError(
          'float64 is not implemented for the flutter_onnxruntime package',
        );

      case TensorDataType.float32:
      case TensorDataType.float16:
      case TensorDataType.bfloat16:
        if (T != double && T != dynamic) {
          throw ArgumentError(
            'The data for a float32, float16, or bfloat16 tensor must be of '
                'type List<double> not List<$T>',
          );
        }

        ortListData = Float32List.fromList(List<double>.from(data));
        break;

      case TensorDataType.int64:
      case TensorDataType.uint64:
        if (T != int && T != dynamic) {
          throw ArgumentError(
            'The data for an int64 or uint64 tensor must be of type List<int> '
                'not List<$T>',
          );
        }

        ortListData = Int64List.fromList(List<int>.from(data));
        break;
      case TensorDataType.int32:
      case TensorDataType.uint32:
      case TensorDataType.int16:
      case TensorDataType.uint16:
      case TensorDataType.int8:
        if (T != int && T != dynamic) {
          throw ArgumentError(
            'The data for an int32, uint32, int16, uint16, or int8 tensor must '
                'be of type List<int> not List<$T>',
          );
        }

        ortListData = Int32List.fromList(List<int>.from(data));
        break;
      case TensorDataType.uint8:
      case TensorDataType.uint4:
      case TensorDataType.int4:
        if (T != int && T != dynamic) {
          throw ArgumentError(
            'The data for an uint8, uint4, or int4 tensor must be of type '
                'List<int> not List<$T>',
          );
        }

        ortListData = Uint8List.fromList(List<int>.from(data));
        break;

      case TensorDataType.string:
        if (T != String && T != dynamic) {
          throw ArgumentError(
            'The data for a String tensor must be of type List<String> not '
                'List<$T>',
          );
        }

        ortListData = List<String>.from(data);
        break;
      case TensorDataType.bool:
        if (T != bool && T != dynamic) {
          throw ArgumentError(
            'The data for a bool tensor must be of type List<bool> not '
                'List<$T>',
          );
        }

        ortListData = List<bool>.from(data);
        break;
    }

    return ortListData;
  }

  /// Casts the generic [T] of this class to a new type. Ideal for updating a
  /// [Tensor<dynamic>] to it's known type.
  ///
  /// Note: This does not convert the data to a new type - only updating dart's
  /// generic type for this class.
  ///
  /// Important: The returned [Tensor] points to the same underlying data and is
  /// not copied into a new Tensor.
  Tensor<U> cast<U>() => Tensor._(dataType, data.cast<U>(), shape, ortValue);

  Future<void> updateData(List<T> data) async {
    await this.ortValue.dispose();

    final dynamic ortListData = _getOrtListData(dataType, data);
    final ortValue = await OrtValue.fromList(ortListData, shape);
    await ortValue.to(_tensorDataTypeToOrtDataType(dataType));
    this.data = data;
    this.ortValue = ortValue;
  }

  Future<void> dispose() async {
    await ortValue.dispose();
  }
}

/// Map a device to the execution providers to use for the given device.
/// @param {import("../utils/devices.js").DeviceType|"auto"|null} [device=null] (Optional) The device to run the inference on.
/// @returns {ONNXExecutionProviders[]} The execution providers to use for the given device.
Future<List<OrtProvider>> deviceToExecutionProviders([DeviceType? device]) async {
  // Use the default execution providers if the user hasn't specified anything
  if (device == null) return defaultDevices;

  final List<OrtProvider> supportedDevices = await getSupportedDevices();

  // Handle overloaded cases
  switch (device) {
    case DeviceType.auto:
      return supportedDevices;
    case DeviceType.gpu:
      return supportedDevices.where((x) => [
        OrtProvider.WEB_GPU,
        OrtProvider.CUDA,
        OrtProvider.DIRECT_ML,
        OrtProvider.WEB_NN, // webnn-gpu TODO
      ].contains(x)).toList();
    default:
      // Continue on :)
  }

  final OrtProvider? ortDevice = DEVICE_TO_EXECUTION_PROVIDER_MAPPING[device];
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
Future<OrtSession> createInferenceSession(
  dynamic buffer_or_path,
  OrtSessionOptions session_options,
  Map<String, dynamic> session_config,
) async {
  // TODO: session_config is not currently used. Should it be? How would it be used?
  if (buffer_or_path is String) {
    return await ort.createSession(buffer_or_path, options: session_options);
  }

  return await createInferenceSessionBytes(buffer_or_path, session_options);
}

/// Create sn ONNX Runtime session with the given bytes as the model
Future<OrtSession> createInferenceSessionBytes(List<int> buffer, OrtSessionOptions sessionOptions) async {
  // The flutter_onnxruntime package doesn't have a way to load a model from
  // memory so we will first save it to a path and load it from there
  final Directory tempDir = await getTemporaryDirectory();

  // Create filename as a hash from the bytes so it can be reused if it exists
  final String filename = sha256.convert(buffer).toString();
  final File modelFile = File(path.join(tempDir.path, filename));

  if (!await modelFile.exists()) {
    await modelFile.writeAsBytes(buffer);
  }

  return await ort.createSession(modelFile.path, options: sessionOptions);
}

/// TODO: This is not implemented and will always be false
/// Check if ONNX's WASM backend is being proxied.
/// @returns {boolean} Whether ONNX's WASM backend is being proxied.
bool isONNXProxy() => false;
