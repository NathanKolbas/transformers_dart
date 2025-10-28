// This is the main entry point for the library.
// It exports the public-facing functions and classes.

export 'src/tokenizers.dart';
export 'src/generation/streamers.dart';
export 'src/utils/dtypes.dart';
export 'src/utils/image.dart';
export 'src/utils/tensor.dart';

import 'package:transformers/src/models.dart';

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
  }) async {
    if (Transformers._instance._initialized) return true;

    try {
      setupModelMappings();

      Transformers._instance._initialized = true;
      return Transformers._instance._initialized;
    } catch (_) {
      if (throwOnFail) {
        rethrow;
      }
    }

    return Transformers._instance._initialized;
  }
}
