import 'package:flutter_test/flutter_test.dart';
import 'package:transformers/src/models.dart';
import 'package:transformers/src/models/auto/processing_auto.dart';
import 'package:transformers/src/utils/hub.dart';
import 'package:integration_test/integration_test.dart';
import 'package:transformers/transformers.dart';

import 'test_utils.dart';

Future<void> main() async {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  setUpAll(() async {
    await Transformers.ensureInitialized(throwOnFail: true);
  });

  test('can run inference with granite-docling-258M-ONNX', () async {
    final writeStdout = getTestStdout();

    const model_id = "onnx-community/granite-docling-258M-ONNX";
    final processor = await AutoProcessor.from_pretrained(model_id);

    final model = await AutoModelForVision2Seq.from_pretrained(
      model_id,
      PretrainedModelOptions(dtype: DataType.fp32),
    );

    final image1 = await load_image("https://huggingface.co/ibm-granite/granite-docling-258M/resolve/main/assets/new_arxiv.png");

    const messages = [
      Message(
        role: 'user',
        content: [
          MessageContent(type: 'image'),
          MessageContent(type: 'text', text: 'Convert this page to docling.'),
        ],
      ),
    ];

    final text = await processor.apply_chat_template(messages, ApplyChatTemplateOptions(add_generation_prompt: true));
    final inputs = await processor.call(text, [[image1], {
      // Set `do_image_splitting: true` to split images into multiple patches.
      // NOTE: This uses more memory, but can provide more accurate results.
      'do_image_splitting': true,
    }]);

    final Tensor generated_ids = await model.generate({
      ...inputs,
      'max_new_tokens': 4096,
      'streamer': TextStreamer(processor.tokenizer!, {
        'skip_prompt': true,
        'skip_special_tokens': false,
        'callback_function': writeStdout,
      }),
    });
    final generated_texts = processor.batch_decode(
      await generated_ids.slice([null, [(inputs['input_ids'] as Tensor).dims.last, null]]),
      skip_special_tokens: true,
    );
    print(generated_texts[0]);

    // expect(model, isNotNull);
  });
}
