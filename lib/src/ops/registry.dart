import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:transformers/src/backends/onnx.dart' show createInferenceSessionBytes;
import 'package:transformers/src/utils/tensor.dart';

typedef WrapReturnFunctionType = Future<List<Tensor>> Function(Map<String, Tensor> inputs);

/// Asynchronously creates a wrapper function for running an ONNX inference session.
///
/// @param {number[]} session_bytes The session data in bytes.
/// @param {import('onnxruntime-common').InferenceSession.SessionOptions} session_options The options for the ONNX session.
/// @template {string | [string] | string[]} T
/// @param {T} names The name(s) of the output tensor(s).
///
/// @returns {Promise<function(Record<string, Tensor>): Promise<T extends string ? Tensor : T extends string[] ? { [K in keyof T]: Tensor } : never>>}
/// The wrapper function for running the ONNX inference session.
Future<WrapReturnFunctionType> wrap(List<int> session_bytes, OrtSessionOptions session_options, List<String> names) async {
  final session = await createInferenceSessionBytes(session_bytes, session_options);

  Future<List<Tensor>> op(Map<String, Tensor> inputs) async {
    final Map<String, OrtValue> ortFeed = Map.fromEntries(
        inputs.entries.map((e) => MapEntry(e.key, e.value.ort_tensor.ortValue)));

    final outputs = await session.run(ortFeed);

    if (names.length > 1) {
      return await Future.wait(names.map((n) => Tensor.createFromOrtValue(outputs[n]!)));
    } else {
      return [await Tensor.createFromOrtValue(outputs[names.first]!)];
    }
  }

  return op;
}

// In-memory registry of initialized ONNX operators
class TensorOpRegistry {
  static final OrtSessionOptions session_options = OrtSessionOptions();
  // {
  //   // TODO: Allow for multiple execution providers
  //   // executionProviders: ['webgpu'],
  // };

  static Future<WrapReturnFunctionType>? _nearest_interpolate_4d;
  static Future<WrapReturnFunctionType>? _bilinear_interpolate_4d;
  static Future<WrapReturnFunctionType>? _bicubic_interpolate_4d;
  static Future<WrapReturnFunctionType>? _matmul;
  static Future<WrapReturnFunctionType>? _stft;
  static Future<WrapReturnFunctionType>? _rfft;
  static Future<WrapReturnFunctionType>? _top_k;
  static Future<WrapReturnFunctionType>? _slice;

  static Future<WrapReturnFunctionType> get nearest_interpolate_4d {
    _nearest_interpolate_4d ??= wrap(
      [8, 10, 18, 0, 58, 129, 1, 10, 41, 10, 1, 120, 10, 0, 10, 0, 10, 1, 115, 18, 1, 121, 34, 6, 82, 101, 115, 105, 122, 101, 42, 18, 10, 4, 109, 111, 100, 101, 34, 7, 110, 101, 97, 114, 101, 115, 116, 160, 1, 3, 18, 1, 114, 90, 31, 10, 1, 120, 18, 26, 10, 24, 8, 1, 18, 20, 10, 3, 18, 1, 98, 10, 3, 18, 1, 99, 10, 3, 18, 1, 104, 10, 3, 18, 1, 119, 90, 15, 10, 1, 115, 18, 10, 10, 8, 8, 7, 18, 4, 10, 2, 8, 4, 98, 31, 10, 1, 121, 18, 26, 10, 24, 8, 1, 18, 20, 10, 3, 18, 1, 98, 10, 3, 18, 1, 99, 10, 3, 18, 1, 104, 10, 3, 18, 1, 119, 66, 2, 16, 21],
      session_options,
      ['y'],
    );
    return _nearest_interpolate_4d!;
  }

  static Future<WrapReturnFunctionType> get bilinear_interpolate_4d {
    _bilinear_interpolate_4d ??= wrap(
      [8, 9, 18, 0, 58, 128, 1, 10, 40, 10, 1, 120, 10, 0, 10, 0, 10, 1, 115, 18, 1, 121, 34, 6, 82, 101, 115, 105, 122, 101, 42, 17, 10, 4, 109, 111, 100, 101, 34, 6, 108, 105, 110, 101, 97, 114, 160, 1, 3, 18, 1, 114, 90, 31, 10, 1, 120, 18, 26, 10, 24, 8, 1, 18, 20, 10, 3, 18, 1, 98, 10, 3, 18, 1, 99, 10, 3, 18, 1, 104, 10, 3, 18, 1, 119, 90, 15, 10, 1, 115, 18, 10, 10, 8, 8, 7, 18, 4, 10, 2, 8, 4, 98, 31, 10, 1, 121, 18, 26, 10, 24, 8, 1, 18, 20, 10, 3, 18, 1, 98, 10, 3, 18, 1, 99, 10, 3, 18, 1, 104, 10, 3, 18, 1, 119, 66, 2, 16, 20],
      session_options,
      ['y'],
    );
    return _bilinear_interpolate_4d!;
  }

  static Future<WrapReturnFunctionType> get bicubic_interpolate_4d {
    _bicubic_interpolate_4d ??= wrap(
      [8, 9, 18, 0, 58, 127, 10, 39, 10, 1, 120, 10, 0, 10, 0, 10, 1, 115, 18, 1, 121, 34, 6, 82, 101, 115, 105, 122, 101, 42, 16, 10, 4, 109, 111, 100, 101, 34, 5, 99, 117, 98, 105, 99, 160, 1, 3, 18, 1, 114, 90, 31, 10, 1, 120, 18, 26, 10, 24, 8, 1, 18, 20, 10, 3, 18, 1, 98, 10, 3, 18, 1, 99, 10, 3, 18, 1, 104, 10, 3, 18, 1, 119, 90, 15, 10, 1, 115, 18, 10, 10, 8, 8, 7, 18, 4, 10, 2, 8, 4, 98, 31, 10, 1, 121, 18, 26, 10, 24, 8, 1, 18, 20, 10, 3, 18, 1, 98, 10, 3, 18, 1, 99, 10, 3, 18, 1, 104, 10, 3, 18, 1, 119, 66, 2, 16, 20],
      session_options,
      ['y'],
    );
    return _bicubic_interpolate_4d!;
  }

  static Future<WrapReturnFunctionType> get matmul {
    _matmul ??= wrap(
      [8, 9, 18, 0, 58, 55, 10, 17, 10, 1, 97, 10, 1, 98, 18, 1, 99, 34, 6, 77, 97, 116, 77, 117, 108, 18, 1, 114, 90, 9, 10, 1, 97, 18, 4, 10, 2, 8, 1, 90, 9, 10, 1, 98, 18, 4, 10, 2, 8, 1, 98, 9, 10, 1, 99, 18, 4, 10, 2, 8, 1, 66, 2, 16, 20],
      session_options,
      ['c'],
    );
    return _matmul!;
  }

  static Future<WrapReturnFunctionType> get stft {
    _stft ??= wrap(
      [8, 7, 18, 0, 58, 148, 1, 10, 38, 10, 1, 115, 10, 1, 106, 10, 1, 119, 10, 1, 108, 18, 1, 111, 34, 4, 83, 84, 70, 84, 42, 15, 10, 8, 111, 110, 101, 115, 105, 100, 101, 100, 24, 1, 160, 1, 2, 18, 1, 115, 90, 26, 10, 1, 115, 18, 21, 10, 19, 8, 1, 18, 15, 10, 3, 18, 1, 98, 10, 3, 18, 1, 115, 10, 3, 18, 1, 99, 90, 11, 10, 1, 106, 18, 6, 10, 4, 8, 7, 18, 0, 90, 16, 10, 1, 119, 18, 11, 10, 9, 8, 1, 18, 5, 10, 3, 18, 1, 119, 90, 11, 10, 1, 108, 18, 6, 10, 4, 8, 7, 18, 0, 98, 31, 10, 1, 111, 18, 26, 10, 24, 8, 1, 18, 20, 10, 3, 18, 1, 98, 10, 3, 18, 1, 102, 10, 3, 18, 1, 100, 10, 3, 18, 1, 99, 66, 2, 16, 17],
      session_options,
      ['o'],
    );
    return _stft!;
  }

  static Future<WrapReturnFunctionType> get rfft {
    _rfft ??= wrap(
      [8, 9, 18, 0, 58, 97, 10, 33, 10, 1, 120, 10, 0, 10, 1, 97, 18, 1, 121, 34, 3, 68, 70, 84, 42, 15, 10, 8, 111, 110, 101, 115, 105, 100, 101, 100, 24, 1, 160, 1, 2, 18, 1, 100, 90, 21, 10, 1, 120, 18, 16, 10, 14, 8, 1, 18, 10, 10, 3, 18, 1, 115, 10, 3, 18, 1, 99, 90, 11, 10, 1, 97, 18, 6, 10, 4, 8, 7, 18, 0, 98, 21, 10, 1, 121, 18, 16, 10, 14, 8, 1, 18, 10, 10, 3, 18, 1, 115, 10, 3, 18, 1, 99, 66, 2, 16, 20],
      session_options,
      ['y'],
    );
    return _rfft!;
  }

  static Future<WrapReturnFunctionType> get top_k {
    _top_k ??= wrap(
        [8, 10, 18, 0, 58, 73, 10, 18, 10, 1, 120, 10, 1, 107, 18, 1, 118, 18, 1, 105, 34, 4, 84, 111, 112, 75, 18, 1, 116, 90, 9, 10, 1, 120, 18, 4, 10, 2, 8, 1, 90, 15, 10, 1, 107, 18, 10, 10, 8, 8, 7, 18, 4, 10, 2, 8, 1, 98, 9, 10, 1, 118, 18, 4, 10, 2, 8, 1, 98, 9, 10, 1, 105, 18, 4, 10, 2, 8, 7, 66, 2, 16, 21],
        session_options,
        [ /* Values */ 'v', /* Indices */ 'i']
    );
    return _top_k!;
  }

  static Future<WrapReturnFunctionType> get slice {
    _slice ??= wrap(
      [8, 7, 18, 0, 58, 96, 10, 25, 10, 1, 120, 10, 1, 115, 10, 1, 101, 10, 1, 97, 10, 1, 116, 18, 1, 121, 34, 5, 83, 108, 105, 99, 101, 18, 1, 114, 90, 9, 10, 1, 120, 18, 4, 10, 2, 8, 1, 90, 9, 10, 1, 115, 18, 4, 10, 2, 8, 7, 90, 9, 10, 1, 101, 18, 4, 10, 2, 8, 7, 90, 9, 10, 1, 97, 18, 4, 10, 2, 8, 7, 90, 9, 10, 1, 116, 18, 4, 10, 2, 8, 7, 98, 9, 10, 1, 121, 18, 4, 10, 2, 8, 1, 66, 2, 16, 13],
      session_options,
      ['y'],
    );
    return _slice!;
  }
}
