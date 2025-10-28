import 'package:flutter_test/flutter_test.dart';
import 'package:transformers/src/models/auto/processing_auto.dart';

void main() {
  group('AutoProcessor', () {
    test('can load from pretrained', () async {
      final autoProcessor = await AutoProcessor.from_pretrained('ibm-granite/granite-docling-258M');
      expect(autoProcessor, isNotNull);
    });
  });
}
