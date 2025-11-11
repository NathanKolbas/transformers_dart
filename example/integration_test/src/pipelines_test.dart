import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:transformers/transformers.dart';

import '../test_utils.dart';

Future<void> main() async {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  setUpAll(() async {
    await Transformers.ensureInitialized(throwOnFail: true);
  });

  group('pipelines', () {
    /// https://huggingface.co/Xenova/bert-base-uncased
    group('can fill-mask with Xenova/bert-base-uncased', () {
      final task = PipelineType.fillMask;
      final modelId = 'Xenova/bert-base-uncased';

      test('default', () async {
        final unmasker = await pipeline(task, modelId);
        final output = await unmasker('The goal of life is [MASK].');
        // [
        //   {score: 0.10933241993188858, token: 2166, token_str: life, sequence: the goal of life is life.},
        //   {score: 0.03941909596323967, token: 7691, token_str: survival, sequence: the goal of life is survival.},
        //   {score: 0.03293018788099289, token: 2293, token_str: love, sequence: the goal of life is love.},
        //   {score: 0.030096078291535378, token: 4071, token_str: freedom, sequence: the goal of life is freedom.},
        //   {score: 0.024967413395643234, token: 17839, token_str: simplicity, sequence: the goal of life is simplicity.},
        // ]

        expect(output.first, {
          'score': 0.10933241993188858,
          'token': 2166,
          'token_str': 'life',
          'sequence': 'the goal of life is life.',
        });
      });

      test('int8', () async {
        final unmasker = await pipeline(task, modelId, PretrainedModelOptions(
          dtype: DataType.int8,
        ));
        final output = await unmasker('The goal of life is [MASK].');
        // [
        //   {score: 0.029873134568333626, token: 7691, token_str: survival, sequence: the goal of life is survival.},
        //   {score: 0.027043689042329788, token: 3722, token_str: simple, sequence: the goal of life is simple.},
        //   {score: 0.01596716232597828, token: 2166, token_str: life, sequence: the goal of life is life.},
        //   {score: 0.015874158591032028, token: 17839, token_str: simplicity, sequence: the goal of life is simplicity.},
        //   {score: 0.012147907167673111, token: 4071, token_str: freedom, sequence: the goal of life is freedom.},
        // ]

        expect(output.first, {
          'score': 0.029873134568333626,
          'token': 7691,
          'token_str': 'survival',
          'sequence': 'the goal of life is survival.',
        });
      });
    });

    /// https://huggingface.co/Xenova/bert-base-cased
    group('can fill-mask with Xenova/bert-base-cased', () {
      final task = PipelineType.fillMask;
      final modelId = 'Xenova/bert-base-cased';

      test('default', () async {
        final unmasker = await pipeline(task, modelId);
        final output = await unmasker('The goal of life is [MASK].');
        // [
        //   {score: 0.11368464678525925, token: 8115, token_str: survival, sequence: The goal of life is survival.},
        //   {score: 0.053510650992393494, token: 1567, token_str: love, sequence: The goal of life is love.},
        //   {score: 0.050412051379680634, token: 9266, token_str: happiness, sequence: The goal of life is happiness.},
        //   {score: 0.033218201249837875, token: 4438, token_str: freedom, sequence: The goal of life is freedom.},
        //   {score: 0.033011458814144135, token: 2244, token_str: success, sequence: The goal of life is success.},
        // ]

        expect(output.first, {
          'score': 0.11368464678525925,
          'token': 8115,
          'token_str': 'survival',
          'sequence': 'The goal of life is survival.',
        });
      });

      test('int8', () async {
        final unmasker = await pipeline(task, modelId, PretrainedModelOptions(
          dtype: DataType.int8,
        ));
        final output = await unmasker('The goal of life is [MASK].');
        // [
        //   {score: 0.11542430520057678, token: 8115, token_str: survival, sequence: The goal of life is survival.},
        //   {score: 0.030219027772545815, token: 9266, token_str: happiness, sequence: The goal of life is happiness.},
        //   {score: 0.02652108296751976, token: 2244, token_str: success, sequence: The goal of life is success.},
        //   {score: 0.026136169210076332, token: 1567, token_str: love, sequence: The goal of life is love.},
        //   {score: 0.016573654487729073, token: 4438, token_str: freedom, sequence: The goal of life is freedom.},
        // ]

        expect(output.first, {
          'score': 0.11542430520057678,
          'token': 8115,
          'token_str': 'survival',
          'sequence': 'The goal of life is survival.',
        });
      });
    });

    /// https://huggingface.co/FacebookAI/xlm-roberta-base
    /// https://huggingface.co/Xenova/xlm-roberta-base
    group('can fill-mask with Xenova/xlm-roberta-base', () {
      final task = PipelineType.fillMask;
      final modelId = 'Xenova/xlm-roberta-base';

      test('default', () async {
        final unmasker = await pipeline(task, modelId);
        final output = await unmasker("Hello I'm a <mask> model.");
        // [
        //   {score: 0.10563567280769348, token: 54543, token_str: fashion, sequence: Hello I'm a fashion model.},
        //   {score: 0.08014999330043793, token: 3525, token_str: new, sequence: Hello I'm a new model.},
        //   {score: 0.03341289237141609, token: 3299, token_str: model, sequence: Hello I'm a model model.},
        //   {score: 0.03021736815571785, token: 92265, token_str: French, sequence: Hello I'm a French model.},
        //   {score: 0.026435906067490578, token: 17473, token_str: sexy, sequence: Hello I'm a sexy model.},
        // ]

        expect(output.first, {
          'score': 0.10563567280769348,
          'token': 54543,
          'token_str': 'fashion',
          'sequence': "Hello I'm a fashion model.",
        });
      });
    });

    /// https://huggingface.co/Xenova/gpt2
    group('can text-generation with Xenova/gpt2', () {
      final task = PipelineType.textGeneration;
      final modelId = 'Xenova/gpt2';

      test('default', () async {
        const text = "Once upon a time,";

        final generator = await pipeline(task, modelId);
        // Generate text (default parameters)
        final output = await generator(text);
        expect(
          output.first['generated_text'],
        'Once upon a time, the world was a place of great beauty and great danger. The world was',
        );

        // Generate text (custom parameters)
        final output2 = await generator(text, TextGenerationConfig(
          max_new_tokens: 20,
          do_sample: true,
          top_k: 5,
        ));
        expect(
          output2.first['generated_text'],
          isNot(equals(output.first['generated_text'])),
        );
      });
    });

    /// https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX
    group('can text-generation with onnx-community/Qwen3-0.6B-ONNX', () {
      final task = PipelineType.textGeneration;
      final modelId = 'onnx-community/Qwen3-0.6B-ONNX';

      test('default', () async {
        final writeStdout = getTestStdout();

        final generator = await pipeline(task, modelId, PretrainedModelOptions(
          // NOTE: PlatformException(CONVERSION_ERROR, float16 is not supported on Windows, false, null)
          // dtype: DataType.q4f16,
          dtype: DataType.uint8,
        ));

        const messages = [
          Message(role: 'system', content: 'You are a helpful assistant.'),
          Message(role: 'user', content: 'Write me a poem about Machine Learning.'),
        ];

        final output = await generator(messages, TextGenerationConfig(
          max_new_tokens: 512,
          do_sample: false,
          streamer: TextStreamer(generator.tokenizer!, {
            'skip_prompt': true,
            'skip_special_tokens': true,
            'callback_function': writeStdout,
          }),
        ));

        final generated_text = List<Message>.from(output[0]['generated_text']);
        final String outputText = generated_text.last.content;

        expect(outputText, '');
      });
    });

    /// https://huggingface.co/onnx-community/Qwen3-Embedding-0.6B-ONNX
    group('can feature-extraction with onnx-community/Qwen3-Embedding-0.6B-ONNX', () {
      final task = PipelineType.featureExtraction;
      final modelId = 'onnx-community/Qwen3-Embedding-0.6B-ONNX';

      test('default', () async {
        // Create a feature extraction pipeline
        final extractor = await pipeline(task, modelId, PretrainedModelOptions(
          dtype: DataType.fp32,
        ));

        String get_detailed_instruct(task_description, query) {
          return 'Instruct: $task_description\nQuery:$query';
        }

        // Each query must come with a one-sentence instruction that describes the task
        const instruct = "Given a web search query, retrieve relevant passages that answer the query";
        final queries = [
          get_detailed_instruct(instruct, "What is the capital of China?"),
          get_detailed_instruct(instruct, "Explain gravity"),
        ];

        // No need to add instruction for retrieval documents
        const documents = [
          "The capital of China is Beijing.",
          "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
        ];
        final input_texts = [...queries, ...documents];

        // Extract embeddings for queries and documents
        final Tensor output = await extractor(input_texts, FeatureExtractionPipelineOptions(
          pooling: "last_token",
          normalize: true,
        ));
        final scores = await matmul(
          await output.slice([[0, queries.length]]), // Query embeddings
          await (await output.slice([[queries.length, null]])).transpose([1, 0]), // Document embeddings
        );

        expect(
          scores.tolist(),
          [[0.7645584344863892, 0.14142566919326782], [0.13549798727035522, 0.599955141544342]],
        );
      });
    });

    /// https://huggingface.co/onnx-community/granite-4.0-350m-ONNX-web
    group('can text-generation with onnx-community/granite-4.0-350m-ONNX', () {
      final task = PipelineType.textGeneration;
      final modelId = 'onnx-community/granite-4.0-350m-ONNX';

      test('default', () async {
        final writeStdout = getTestStdout();

        final generator = await pipeline(task, modelId, PretrainedModelOptions(
          // NOTE: PlatformException(CONVERSION_ERROR, float16 is not supported on Windows, false, null)
          // dtype: DataType.q4f16,
          // dtype: DataType.uint8,
        ));

        const messages = [
          Message(role: 'system', content: 'You are a helpful assistant.'),
          Message(role: 'user', content: 'What is the capital of France?'),
        ];

        final output = await generator(messages, TextGenerationConfig(
          max_new_tokens: 512,
          do_sample: false,
          streamer: TextStreamer(generator.tokenizer!, {
            'skip_prompt': true,
            'skip_special_tokens': true,
            'callback_function': writeStdout,
          }),
        ));

        final generated_text = List<Message>.from(output[0]['generated_text']);
        final String outputText = generated_text.last.content;

        expect(outputText, 'The capital of France is Paris.');
      });
    });
  });
}
