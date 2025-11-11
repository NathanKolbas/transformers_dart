import 'package:transformers/src/utils/devices.dart';

/// TODO
/// Checks if WebGPU fp16 support is available in the current environment.
Future<bool> isWebGpuFp16Supported() async => false;

enum DataType {
  /// Auto-detect based on environment
  auto('auto'),
  fp32('fp32'),
  fp16('fp16'),
  q8('q8'),
  int8('int8'),
  uint8('uint8'),
  q4('q4'),
  bnb4('bnb4'),

  /// fp16 model with int4 block weight quantization
  q4f16('q4f16');

  const DataType(this.value);

  final String value;

  /// Get the [DataType] from it's [String] value.
  static DataType fromString(String dataType) => DataType
      .values
      .firstWhere((e) => e.value == dataType);

  String toJson() => value;

  @override
  String toString() => value;
}

const Map<DeviceType, DataType> DEFAULT_DEVICE_DTYPE_MAPPING = {
  // NOTE: If not specified, will default to fp32
  DeviceType.wasm: DataType.q8,
};

/// @type {Record<Exclude<DataType, "auto">, string>}
const Map<DataType, String> DEFAULT_DTYPE_SUFFIX_MAPPING = {
  DataType.fp32: '',
  DataType.fp16: '_fp16',
  DataType.int8: '_int8',
  DataType.uint8: '_uint8',
  DataType.q8: '_quantized',
  DataType.q4: '_q4',
  DataType.q4f16: '_q4f16',
  DataType.bnb4: '_bnb4',
};
