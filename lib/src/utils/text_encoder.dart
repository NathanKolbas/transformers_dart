import 'dart:convert';
import 'dart:typed_data';

/// A Dart equivalent to JavaScripts TextEncoder mimicking it's class
/// implementation. According to the Mozilla docs:
/// The TextEncoder interface enables you to encode a JavaScript string using
/// UTF-8.
/// Ref: https://developer.mozilla.org/en-US/docs/Web/API/TextEncoder
class TextEncoder {
  /// Takes a string as input, and returns a Uint8Array containing the string
  /// encoded using UTF-8.
  Uint8List encode(String text) => utf8.encode(text);
}
