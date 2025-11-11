# transformers

🚧 **THIS IS CURRENTLY A WORK IN PROGRESS** 🚧

State-of-the-art Machine Learning for Dart. Run 🤗 Transformers cross-platform on your device, with no need for a server!

This repo is based off of [transformers.js](https://github.com/huggingface/transformers.js).

Currently I have only tested this on Windows and Android. More manually testing should be done for the other platforms.

Web is not currently supported (also due to [huggingface_hub](https://github.com/NathanKolbas/huggingface_hub_dart) needing to be updated to support web) but the plan for this is to use [transformers.js](https://github.com/huggingface/transformers.js) for web and the dart implementation for for all other platforms. This is due to the differences in filesystem.

There is still a lot of work to be done here and any help is appreciated! If a model is not yet added that you would like to use [make an issue](https://github.com/NathanKolbas/transformers_dart/issues/new/choose) and better yet make a PR!

> 🗒️ Inference
> 
> Inference is currently under heavy active development.
> 
> It is highly recommended to hold off on using this for inference until the underlying issues are resolved.
> 
> The ONNX backend that is currently being used was working fine for Windows but fails when used on Android due to slight variances. The inference backend will be replaced to better handle this.
> 
> Proper checks need to be made to confirm that all memory is properly released for Tensors.
> 
> Inference is quite slow when it shouldn't be. This is due to overly/inefficient type "casting" and manipulation of Tensor's underlying data needing to create a new Tensor each time.

## Version

This library is based off of commit [a5847c9fb6ca410df6fc35ee584140f867840150](https://github.com/huggingface/transformers.js/tree/a5847c9fb6ca410df6fc35ee584140f867840150) from the official [transformers.js](https://github.com/huggingface/transformers.js) library.

## Supported Devices

In theory, this library should work across all platforms except for the Web do to no file storage. Please see each section to know which platform has been tested.

### Windows

✔️ Tested and works.

### MacOS

❌ Issues compiling with ONNX library.

### Linux

❓ Not tested yet.

### Android

⭕ Inference has problems but tokenizer works.

### iOS

❓ Not tested yet.

### Web

❌ Not yet implemented.

## Setup

To make sure everything is set up call `Transformers.ensureInitialized` from the initialization of your application. Here is an example for a flutter application:

```dart
import 'package:flutter/widgets.dart';
import 'package:transformers/transformers.dart';

void main() async {  
  WidgetsFlutterBinding.ensureInitialized();
  await Transformers.ensureInitialized();
  
  // Rest of your main function...
}
```

## Examples

### Tokenizers

#### `xlm-roberta-base`

Here is how to run the tokenizer for `xlm-roberta-base`:

```dart
import 'package:transformers/transformers.dart';

Future<void> main() async {
  final tokenizer = await AutoTokenizer.from_pretrained("facebookAI/xlm-roberta-base");
  final tokenized = await tokenizer('test', return_tensor: false);
  print('Tokenization: $tokenized');

  final decoded = tokenizer.decode(tokenized.input_ids, skip_special_tokens: true);
  print('Decoded: $decoded');
}
```

### Pipelines

#### Text Generation with GPT2

Here is how to run a pipeline for `GPT2` text generation:

```dart
import 'package:transformers/transformers.dart';

Future<void> main() async {
  final generator = await pipeline(PipelineType.textGeneration, 'Xenova/gpt2');
  final output = await generator('Once upon a time,');
  // 'Once upon a time, the world was a place of great beauty and great danger. The world was',
}
```

#### Text Generation with Qwen3

Here is how to run a pipeline for `Qwen3` text generation:

```dart
import 'package:transformers/transformers.dart';

Future<void> main() async {
  final generator = await pipeline(
    PipelineType.textGeneration,
    'onnx-community/Qwen3-0.6B-ONNX',
    PretrainedModelOptions(
      // If your device supports float16:
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
    }),
  ));

  final generated_text = List<Message>.from(output[0]['generated_text']);
  final String outputText = generated_text.last.content;
}
```

#### Text Generation with Granite 4.0

Here is how to run a pipeline for `granite-4.0-350m` text generation:

```dart
import 'package:transformers/transformers.dart';

Future<void> main() async {
  final task = PipelineType.textGeneration;
  final modelId = 'onnx-community/granite-4.0-350m-ONNX';


  final generator = await pipeline(
    PipelineType.textGeneration,
    'onnx-community/granite-4.0-350m-ONNX',
    PretrainedModelOptions(
      // If your device supports float16:
      // dtype: DataType.q4f16,
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
    }),
  ));

  final generated_text = List<Message>.from(output[0]['generated_text']);
  final String outputText = generated_text.last.content;
  // The capital of France is Paris.
}
```

#### Fill Mask with bert-base-uncased

Here is how to run a pipeline for `bert-base-uncased` fill mask:

```dart
import 'package:transformers/transformers.dart';

Future<void> main() async {
  final unmasker = await pipeline(PipelineType.fillMask, 'Xenova/bert-base-uncased');
  final output = await unmasker('The goal of life is [MASK].');
  // [
  //   {score: 0.10933241993188858, token: 2166, token_str: life, sequence: the goal of life is life.},
  //   {score: 0.03941909596323967, token: 7691, token_str: survival, sequence: the goal of life is survival.},
  //   {score: 0.03293018788099289, token: 2293, token_str: love, sequence: the goal of life is love.},
  //   {score: 0.030096078291535378, token: 4071, token_str: freedom, sequence: the goal of life is freedom.},
  //   {score: 0.024967413395643234, token: 17839, token_str: simplicity, sequence: the goal of life is simplicity.},
  // ]
}
```

#### Feature Extraction with Qwen3

Here is how to run a pipeline for `Qwen3` feature extraction:

```dart
import 'package:transformers/transformers.dart';

Future<void> main() async {
  // Create a feature extraction pipeline
  final extractor = await pipeline(
    PipelineType.featureExtraction,
    'onnx-community/Qwen3-Embedding-0.6B-ONNX',
    PretrainedModelOptions(
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
  // scores.tolist()
  // [
  //   [0.7645584344863892, 0.14142566919326782],
  //   [0.13549798727035522, 0.599955141544342]
  // ]
}
```
