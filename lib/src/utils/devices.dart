// ignore_for_file: non_constant_identifier_names

/// The mapping of devices supported by Transformers.dart
enum DeviceType {
  /// Auto-detect based on device and environment
  auto('auto'),
  /// Auto-detect GPU
  gpu('gpu'),
  /// CPU
  cpu('cpu'),
  /// WebAssembly
  wasm('wasm'),
  /// WebGPU
  webgpu('webgpu'),
  /// CUDA
  cuda('cuda'),
  /// DirectML
  dml('dml'),

  /// WebNN (default)
  webnn('webnn'),

  /// WebNN NPU
  webnnNpu('webnn-npu'),

  /// WebNN GPU
  webnnGpu('webnn-gpu'),

  /// WebNN CPU
  webnnCpu('webnn-cpu');

  const DeviceType(this.value);

  final String value;

  /// Get the [DeviceType] from it's [String] value.
  static DeviceType fromString(String deviceType) => DeviceType
      .values
      .firstWhere((e) => e.value == deviceType);
}

/// @typedef {keyof typeof DEVICE_TYPES} DeviceType
/// The list of devices supported by Transformers.dart
// sealed class DeviceType {}
//
// class DeviceTypeAuto extends DeviceType {}
// class DeviceTypeGpu extends DeviceType {}
// class DeviceTypeCpu extends DeviceType {}
// class DeviceTypeWasm extends DeviceType {}
// class DeviceTypeWebgpu extends DeviceType {}
// class DeviceTypeCuda extends DeviceType {}
// class DeviceTypeDml extends DeviceType {}
// class DeviceTypeWebnn extends DeviceType {}
// class DeviceTypeWebnnNpu extends DeviceType {}
// class DeviceTypeWebnnGpu extends DeviceType {}
// class DeviceTypeWebnnCpu extends DeviceType {}
