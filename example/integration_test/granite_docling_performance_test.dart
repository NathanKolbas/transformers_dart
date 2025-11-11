import 'dart:developer';

import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:transformers/src/models.dart';
import 'package:transformers/src/models/auto/processing_auto.dart';
import 'package:transformers/src/utils/hub.dart';
import 'package:transformers/transformers.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Transformers.ensureInitialized(throwOnFail: true);

  const model_id = "onnx-community/granite-docling-258M-ONNX";
  Stopwatch stopwatch = Stopwatch()..start();
  Timeline.startSync('Loading processor');
  final processor = await AutoProcessor.from_pretrained(model_id);
  Timeline.finishSync();
  stopwatch.stop();

  print('Loaded processor in ${stopwatch.elapsed}');

  stopwatch = Stopwatch()..start();
  Timeline.startSync('Loading model');
  final model = await AutoModelForVision2Seq.from_pretrained(
    model_id,
    PretrainedModelOptions(dtype: DataType.fp32),
  );
  Timeline.finishSync();
  stopwatch.stop();

  print('Loaded model in ${stopwatch.elapsed}');

  stopwatch = Stopwatch()..start();
  Timeline.startSync('Loading image');
  final image1 = await load_image("https://huggingface.co/ibm-granite/granite-docling-258M/resolve/main/assets/new_arxiv.png");
  Timeline.finishSync();
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
  Timeline.startSync('Applying chat template');
  final text = await processor.apply_chat_template(messages, ApplyChatTemplateOptions(add_generation_prompt: true));
  Timeline.finishSync();
  stopwatch.stop();
  print('Applied chat template in ${stopwatch.elapsed}');
  stopwatch = Stopwatch()..start();
  Timeline.startSync('Extracting features');
  final inputs = await processor.call(text, [[image1], {
    // Set `do_image_splitting: true` to split images into multiple patches.
    // NOTE: This uses more memory, but can provide more accurate results.
    'do_image_splitting': true,
  }]);
  Timeline.finishSync();
  stopwatch.stop();

  print('Extracted features in ${stopwatch.elapsed}');

  stopwatch = Stopwatch()..start();
  Timeline.startSync('Generating ids');
  final Tensor generated_ids = await model.generate({
    ...inputs,
    'max_new_tokens': 4096,
    'streamer': TextStreamer(processor.tokenizer!, {
      'skip_prompt': true,
      'skip_special_tokens': false,
    }),
  });
  Timeline.finishSync();
  stopwatch.stop();
  print('Generated ids in ${stopwatch.elapsed}');
  stopwatch = Stopwatch()..start();
  Timeline.startSync('Decoding ids');
  final generated_texts = processor.batch_decode(
    await generated_ids.slice([null, [(inputs['input_ids'] as Tensor).dims.last, null]]),
    skip_special_tokens: true,
  );
  Timeline.finishSync();
  stopwatch.stop();
  print('Generated text in ${stopwatch.elapsed}');
  print(generated_texts[0]);
}
