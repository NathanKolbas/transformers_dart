import 'dart:developer';

import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:transformers/transformers.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Transformers.ensureInitialized(
    throwOnFail: true,
    ortOptions: OrtInitializationOptions(
      showDebugMessages: true,
    ),
  );

  // const waitFor = Duration(seconds: 15);
  // print('Starting test in ${waitFor.inSeconds} seconds...');
  // await Future.delayed(waitFor);
  debugger();

  final task = PipelineType.textGeneration;
  final modelId = 'onnx-community/granite-4.0-350m-ONNX';

  Stopwatch stopwatch = Stopwatch()..start();
  Timeline.startSync('Loading generator');
  final generator = await pipeline(task, modelId, PretrainedModelOptions(
    // NOTE: PlatformException(CONVERSION_ERROR, float16 is not supported on Windows, false, null)
    // dtype: DataType.q4f16,
    // dtype: DataType.uint8,
  ));
  Timeline.finishSync();
  stopwatch.stop();

  print('Loaded generator in ${stopwatch.elapsed}');

  const messages = [
    Message(role: 'system', content: 'You are a helpful assistant.'),
    Message(role: 'user', content: 'What is the capital of France?'),
  ];

  stopwatch = Stopwatch()..start();
  Timeline.startSync('Generating output');
  final output = await generator(messages, TextGenerationConfig(
    max_new_tokens: 512,
    do_sample: false,
    streamer: TextStreamer(generator.tokenizer!, {
      'skip_prompt': true,
      'skip_special_tokens': true,
    }),
  ));
  Timeline.finishSync();
  stopwatch.stop();

  print('Output in ${stopwatch.elapsed}');

  final generated_text = List<Message>.from(output[0]['generated_text']);
  final String outputText = generated_text.last.content;

  assert(outputText == 'The capital of France is Paris.');
}
