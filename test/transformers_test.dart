import 'package:flutter_test/flutter_test.dart';

import 'package:transformers/transformers.dart';

void main() {
  test('check that it works with xlm-roberta-base', () async {
    Stopwatch stopwatch = Stopwatch()..start();
    final tokenizer = await AutoTokenizer.from_pretrained("facebookAI/xlm-roberta-base");
    stopwatch.stop();

    print('Loaded AutoTokenizer in ${stopwatch.elapsed}');

    stopwatch = Stopwatch()..start();
    final tokenized = await tokenizer('test', return_tensor: false);
    stopwatch.stop();

    print('Tokenized text in ${stopwatch.elapsed}');
    print('Tokenization: $tokenized');

    stopwatch = Stopwatch()..start();
    final decoded = tokenizer.decode(tokenized.input_ids, skip_special_tokens: true);
    stopwatch.stop();

    print('Decoded in ${stopwatch.elapsed}');
    print('Decoded: $decoded');
  });
}
