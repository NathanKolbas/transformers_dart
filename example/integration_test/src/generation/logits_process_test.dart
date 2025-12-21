import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:transformers/transformers.dart';
import 'package:transformers/src/generation/logits_process.dart';

Future<void> main() async {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  setUpAll(() async {
    await Transformers.ensureInitialized(throwOnFail: true);
  });

  group('ForcedBOSTokenLogitsProcessor', () {
    test('applies the BOS token', () async {
      // 1. Define parameters
      final bos_token_id = 0;
      final vocabSize = 5;

      final processor = ForcedBOSTokenLogitsProcessor(bos_token_id);

      final input_ids = [[1]];

      final logits = Tensor.create<double>(
        TensorDataType.float32,
        [10, 20, 30, 40, 50],
        [1, vocabSize],
      );

      print('Logits before processing: ${logits.data}');

      final processedLogits = await processor(input_ids, logits);

      print('Logits after processing: ${logits.data}');
      print('Logits output after processing: ${processedLogits.data}');

      expect(processedLogits.data, logits.data);
      expect(logits.data, [0, -double.infinity, -double.infinity, -double.infinity, -double.infinity]);
    });
  });
}
