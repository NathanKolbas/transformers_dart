import 'dart:convert';
import 'dart:typed_data';

/// A Dart equivalent to JavaScripts TextDecoder mimicking it's class
/// implementation. According to the Mozilla docs:
/// The TextDecoder interface represents a decoder for a specific text encoding,
/// such as UTF-8, ISO-8859-2, or GBK. A decoder takes an array of bytes as
/// input and returns a JavaScript string.
/// Ref: https://developer.mozilla.org/en-US/docs/Web/API/TextDecoder
class TextDecoder {
  /// A string identifying the character encoding that this decoder will use.
  /// This may be any valid label.
  final String label;

  /// A boolean value indicating if the TextDecoder.decode() method must throw a
  /// TypeError when decoding invalid data. It defaults to false, which means
  /// that the decoder will substitute malformed data with a replacement
  /// character.
  final bool fatal;

  /// A boolean value indicating whether the byte order mark will be included in
  /// the output or skipped over. It defaults to false, which means that the
  /// byte order mark will be skipped over when decoding and will not be
  /// included in the decoded text.
  final bool ignoreBOM;

  TextDecoder({
    this.label = 'utf-8',
    this.fatal = false,
    this.ignoreBOM = false,
  }) {
    // if (!ignoreBOM) {
    //   print(
    //     'It is not possible to set `ignoreBOM` to false.\n'
    //     'The Utf8Decoder in dart:convert automatically handles BOMs in UTF-8 '
    //         'encoded data. It will recognize and skip the UTF-8 BOM if present '
    //         'at the beginning of the byte stream, effectively behaving as if '
    //         'ignoreBOM were true for UTF-8.'
    //         '\n\n'
    //         'Because of this, ignoreBOM will be true and will continue on.'
    //   );
    // }
  }

  /// Decodes the given bytes into a JavaScript string and returns it.
  String decode(Uint8List buffer) => utf8.decode(buffer);
}
