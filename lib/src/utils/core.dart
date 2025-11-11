import 'package:transformers/extensions/list_extensions.dart';

sealed class ProgressInfo {
  String get status => throw UnimplementedError('Implement status');
}

final class InitiateProgressInfo extends ProgressInfo {
  @override
  String get status => 'initiate';

  /// The model id or directory path.
  String name;

  /// The name of the file.
  String file;

  InitiateProgressInfo(this.name, this.file);
}

final class DownloadProgressInfo extends ProgressInfo {
  @override
  String get status => 'download';

  /// The model id or directory path.
  String name;

  /// The name of the file.
  String file;

  DownloadProgressInfo(this.name, this.file);
}

final class ProgressStatusInfo extends ProgressInfo {
  @override
  String get status => 'progress';

  /// The model id or directory path.
  String name;

  /// The name of the file.
  String file;

  /// A number between 0 and 100.
  num progress;

  /// The number of bytes loaded.
  num loaded;

  /// The total number of bytes to be loaded.
  num total;

  ProgressStatusInfo(this.name, this.file, this.progress, this.loaded, this.total);
}

final class DoneProgressInfo extends ProgressInfo {
  @override
  String get status => 'done';

  /// The model id or directory path.
  String name;

  /// The name of the file.
  String file;

  DoneProgressInfo(this.name, this.file);
}

final class ReadyProgressInfo extends ProgressInfo {
  @override
  String get status => 'ready';

  /// The loaded task.
  String task;

  /// The loaded model.
  String model;

  ReadyProgressInfo(this.task, this.model);
}

/**
 * @typedef {InitiateProgressInfo | DownloadProgressInfo | ProgressStatusInfo | DoneProgressInfo | ReadyProgressInfo} ProgressInfo
 */

/// A callback function that is called with progress information.
typedef ProgressCallback = void Function(ProgressInfo progressInfo);

/// Helper function to dispatch progress callbacks.
///
/// @param {ProgressCallback | null | undefined} progress_callback The progress callback function to dispatch.
/// @param {ProgressInfo} data The data to pass to the progress callback function.
/// @returns {void}
/// @private
void dispatchCallback(ProgressCallback? progress_callback, ProgressInfo data) {
  if (progress_callback != null) progress_callback(data);
}

/// Reverses the keys and values of an object.
///
/// @param {Object} data The object to reverse.
/// @returns {Object} The reversed object.
/// @see https://ultimatecourses.com/blog/reverse-object-keys-and-values-in-javascript
Map<V, K> reverseDictionary<K, V>(Map<K, V> data) {
  return data.map((k, v) => MapEntry(v, k));
}

/// Escapes regular expression special characters from a string by replacing them with their escaped counterparts.
///
/// @param {string} string The string to escape.
/// @returns {string} The escaped string.
String escapeRegExp(String string) {
  return string.replaceAllMapped(
    r'[.*+?^${}()|[\]\\]',
        (m) => '\\${m.group(0)!}',
  );
}

/// Check if a value is an integer.
/// @param {*} x The value to check.
/// @returns {boolean} True if the value is an integer, false otherwise.
bool isIntegralNumber(dynamic x) => x is int || x is BigInt;

/// Determine if a provided width or height is nullish.
/// @param {*} x The value to check.
/// @returns {boolean} True if the value is `null`, `undefined` or `-1`, false otherwise.
bool isNullishDimension(dynamic x) => x == null || x == -1;

/// Calculates the index offset for a given index and window size.
/// @param {number} i The index.
/// @param {number} w The window size.
/// @returns {number} The index offset.
int calculateReflectOffset(int i, int w) => ((i + w) % (2 * w) - w).abs();

/// Efficiently merge arrays, creating a new copy.
/// Adapted from https://stackoverflow.com/a/6768642/13989043
/// @param  {Array[]} arrs Arrays to merge.
/// @returns {Array} The merged array.
List<T> mergeArrays<T>(List<List<T>> arrs) {
  return List<T>.from(arrs.flat());
}

///
/// @param {Object} o
/// @param {string[]} props
/// @returns {Object}
Map<String, T> pick<T>(Map<String, dynamic> o, List<String> props) {
  final Map<String, T> result = {};
  for (final prop in props) {
    if (o.containsKey(prop)) {
      result[prop] = o[prop];
    }
  }
  return result;
}

/// Calculate the length of a string, taking multi-byte characters into account.
/// This mimics the behavior of Python's `len` function.
/// @param {string} s The string to calculate the length of.
/// @returns {number} The length of the string.
int len(String s) => s.runes.length;

/// Count the occurrences of a value in an array or string.
/// This mimics the behavior of Python's `count` method.
/// @param {any[]|string} arr The array or string to search.
/// @param {any} value The value to count.
int count<V, T extends Iterable<V>>(T arr, V value) {
  int count = 0;
  for (final v in arr) {
    if (v == value) ++count;
  }
  return count;
}

/// Count the occurrences of a value in an array or string.
/// This mimics the behavior of Python's `count` method.
/// @param {any[]|string} arr The array or string to search.
/// @param {any} value The value to count.
int countString(String s, String value) {
  int count = 0;
  for (final v in s.split('')) {
    if (v == value) ++count;
  }
  return count;
}
