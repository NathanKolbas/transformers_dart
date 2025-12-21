import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:transformers/transformers.dart';

Future<void> main() async {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  setUpAll(() async {
    await Transformers.ensureInitialized(throwOnFail: true);
  });

  group('Tensor', () {
    test('can create Float32', () {
      final tensor = Tensor.create<double>(
        TensorDataType.float32,
        [10, 20, 30, 40, 50],
      );

      expect(tensor.data, [10, 20, 30, 40, 50]);
    });

    test('can modify tensor data', () {
      final tensor = Tensor.create<double>(
        TensorDataType.float32,
        [10, 20, 30, 40, 50],
      );

      tensor.data[0] = 42;

      const expected = [42, 20, 30, 40, 50];
      expect(tensor.data, expected);
      expect(tensor.ort_tensor.ortTensor.data, expected);
    });
  });
}
