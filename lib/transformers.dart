// This is the main entry point for the library.
// It exports the public-facing functions and classes.

export 'src/pipelines.dart';
export 'src/tokenizers.dart';
export 'src/generation/streamers.dart';
export 'src/utils/devices.dart';
export 'src/utils/dtypes.dart';
export 'src/utils/hub.dart';
export 'src/utils/image.dart';
export 'src/utils/tensor.dart';

export 'package:huggingface_hub/huggingface_hub.dart' show HuggingfaceHubThrowOnError;
export 'package:ort/ort.dart' show OrtInitializationOptions;

import 'package:huggingface_hub/huggingface_hub.dart';
import 'package:transformers/src/models.dart';
import 'package:ort/ort.dart';

/// Throw on error for only specific [Transformers] initialization errors
class TransformersThrowOnError {
  /// Throw if setting up HuggingfaceHub fails
  final bool huggingfaceHub;

  /// Throw if setting up ORT (ONNX Runtime) fails. Recommended to be true (default).
  final bool onnxRuntime;

  const TransformersThrowOnError({
    this.huggingfaceHub = false,
    this.onnxRuntime = true,
  });
}

class Transformers {
  static final Transformers _instance = Transformers._internal();

  factory Transformers() => _instance;

  Transformers._internal();

  bool _initialized = false;

  /// If transformers is initialized. Typically you don't need this and can
  /// just call [ensureInitialized] directly without checking if initialized
  /// prior.
  static bool get initialized => Transformers._instance._initialized;

  /// Make sure transformers is initialized.
  ///
  /// If [throwOnFail] is set to true then an exception will be thrown if
  /// initialization fails. By default this is false.
  ///
  /// [throwOnSpecificErrors] - Let's you only throw when certain
  /// initializations fail. If [throwOnFail] is true then this is irrelevant.
  ///
  /// Returns [bool] whether or not initialization was successful. If
  /// [throwOnFail] is true then you must catch the error.
  static Future<bool> ensureInitialized({
    bool throwOnFail = false,
    TransformersThrowOnError throwOnSpecificErrors = const TransformersThrowOnError(),
    HuggingfaceHubThrowOnError huggingfaceHubThrowOnError = const HuggingfaceHubThrowOnError(),
    OrtInitializationOptions ortOptions = const OrtInitializationOptions(),
  }) async {
    if (Transformers._instance._initialized) return true;

    try {
      setupModelMappings();

      List<Object?> errors = await Future.wait([
        _setupHfHub(
          throwOnFail || throwOnSpecificErrors.huggingfaceHub,
          huggingfaceHubThrowOnError,
        ),
        _setupOrt(
          throwOnFail || throwOnSpecificErrors.onnxRuntime,
          ortOptions,
        ),
      ]);
      errors = errors.nonNulls.toList(growable: false);
      if (throwOnFail && errors.isNotEmpty) {
        throw StateError('An error occurred during Transformers initialization: $errors');
      }

      Transformers._instance._initialized = true;
      return Transformers._instance._initialized;
    } catch (_) {
      if (throwOnFail) {
        rethrow;
      }
    }

    return Transformers._instance._initialized;
  }

  static Future<Object?> _setupHfHub(
    bool throwOnFail,
    HuggingfaceHubThrowOnError huggingfaceHubThrowOnError,
  ) async {
    try {
      await HuggingfaceHub.ensureInitialized(
        throwOnFail: throwOnFail,
        throwOnSpecificErrors: huggingfaceHubThrowOnError,
      );
    } catch (e) {
      if (throwOnFail) return e;
    }

    return null;
  }

  static Future<Object?> _setupOrt(
    bool throwOnFail,
    OrtInitializationOptions ortInitializationOptions,
  ) async {
    try {
      await Ort.ensureInitialized(
        throwOnFail: throwOnFail,
        options: ortInitializationOptions,
      );
    } catch (e) {
      if (throwOnFail) return e;
    }

    return null;
  }
}
