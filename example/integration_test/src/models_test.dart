import 'package:flutter_test/flutter_test.dart';
import 'package:transformers/src/models.dart';
import 'package:transformers/src/utils/hub.dart';
import 'package:integration_test/integration_test.dart';
import 'package:transformers/transformers.dart';

Future<void> main() async {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  setUpAll(() async {
    await Transformers.ensureInitialized(throwOnFail: true);
  });

  group('AutoModelForVision2Seq', () {
    test('can load from pretrained', () async {
      final model = await AutoModelForVision2Seq.from_pretrained(
        "onnx-community/granite-docling-258M-ONNX",
        PretrainedModelOptions(dtype: DataType.fp32),
      );
      expect(model, isNotNull);
    });
  });
}
