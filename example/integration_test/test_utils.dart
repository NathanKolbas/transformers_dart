import 'package:flutter_test/flutter_test.dart';

/// When running tests `stdout.write` does not work. This is a workaround.
void Function(Object?) getTestStdout() {
  final StringBuffer buffer = StringBuffer();

  addTearDown(() {
    final text = buffer.toString();
    if (text.isNotEmpty) {
      print(buffer.toString());
    }
    buffer.clear();
  });

  return (Object? x) {
    buffer.write(x);
    final text = buffer.toString();
    if (!text.contains('\n')) return;

    if (text.endsWith('\n')) {
      print(text);
      buffer.clear();
    } else {
      final texts = text.split('\n');
      buffer.clear();
      buffer.write(texts.removeAt(texts.length - 1));
      texts.forEach(print);
    }
  };
}
