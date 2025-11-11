import 'package:flutter_test/flutter_test.dart';
import 'package:transformers/src/models.dart';
import 'package:transformers/src/models/auto/processing_auto.dart';
import 'package:transformers/src/utils/hub.dart';
import 'package:integration_test/integration_test.dart';
import 'package:transformers/transformers.dart';

Future<void> main() async {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  await Transformers.ensureInitialized(throwOnFail: true);

  group('AutoModelForVision2Seq', () {
    test('can load from pretrained', () async {
      final model = await AutoModelForVision2Seq.from_pretrained(
        "ibm-granite/granite-docling-258M",
        // PretrainedModelOptions(
        //   torch_dtype=torch.bfloat16,
        //   _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "sdpa",
        // )
      );
      expect(model, isNotNull);
    });
  });
}
