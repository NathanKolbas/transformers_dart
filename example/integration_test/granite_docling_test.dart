import 'package:flutter_test/flutter_test.dart';
import 'package:transformers/src/models.dart';
import 'package:transformers/src/models/auto/processing_auto.dart';
import 'package:transformers/src/utils/hub.dart';
import 'package:integration_test/integration_test.dart';
import 'package:transformers/transformers.dart';

Future<void> main() async {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  setUpAll(() async {
    await Transformers.ensureInitialized(throwOnFail: true);
  });

  test('can run inference with granite-docling-258M-ONNX', () async {
    const model_id = "onnx-community/granite-docling-258M-ONNX";
    Stopwatch stopwatch = Stopwatch()..start();
    final processor = await AutoProcessor.from_pretrained(model_id);
    stopwatch.stop();

    print('Loaded processor in ${stopwatch.elapsed}');

    stopwatch = Stopwatch()..start();
    final model = await AutoModelForVision2Seq.from_pretrained(
      model_id,
      PretrainedModelOptions(dtype: DataType.fp32),
    );
    stopwatch.stop();

    print('Loaded model in ${stopwatch.elapsed}');

    stopwatch = Stopwatch()..start();
    final image1 = await load_image("https://huggingface.co/ibm-granite/granite-docling-258M/resolve/main/assets/new_arxiv.png");
    stopwatch.stop();

    print('Loaded image in ${stopwatch.elapsed}');

    const messages = [
      Message(
        role: 'user',
        content: [
          MessageContent(type: 'image'),
          MessageContent(type: 'text', text: 'Convert this page to docling.'),
        ],
      ),
    ];

    stopwatch = Stopwatch()..start();
    final text = await processor.apply_chat_template(messages, ApplyChatTemplateOptions(add_generation_prompt: true));
    stopwatch.stop();
    print('Applied chat template in ${stopwatch.elapsed}');
    stopwatch = Stopwatch()..start();
    final inputs = await processor.call(text, [[image1], {
      // Set `do_image_splitting: true` to split images into multiple patches.
      // NOTE: This uses more memory, but can provide more accurate results.
      'do_image_splitting': true,
    }]);
    stopwatch.stop();

    print('Extracted features in ${stopwatch.elapsed}');

    stopwatch = Stopwatch()..start();
    final Tensor generated_ids = await model.generate({
      ...inputs,
      'max_new_tokens': 4096,
      'streamer': TextStreamer(processor.tokenizer!, {
        'skip_prompt': true,
        'skip_special_tokens': false,
      }),
    });
    stopwatch.stop();
    print('Generated ids in ${stopwatch.elapsed}');
    stopwatch = Stopwatch()..start();
    final generated_texts = processor.batch_decode(
      (await generated_ids.slice([null, [inputs.input_ids.dims.at(-1), null]])).data as List<List<int>>,
      skip_special_tokens: true,
    );
    stopwatch.stop();
    print('Generated text in ${stopwatch.elapsed}');
    print(generated_texts[0]);

    // expect(model, isNotNull);
  });
}
