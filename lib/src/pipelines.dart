import 'package:transformers/src/base/processing_utils.dart';
import 'package:transformers/src/generation/configuration_utils.dart';
import 'package:transformers/src/models.dart';
import 'package:transformers/src/utils/core.dart';
import 'package:transformers/src/utils/hub.dart';
import 'package:transformers/src/tokenizers.dart';
import 'package:transformers/src/utils/maths.dart';
import 'package:transformers/src/utils/tensor.dart';

class PipelineOptions {}

/// The Pipeline class is the class from which all pipelines inherit.
/// Refer to this class for methods shared across different pipelines.
class Pipeline {
  String task;

  PreTrainedModel model;

  PreTrainedTokenizer? tokenizer;

  Processor? processor;

  /// Create a new Pipeline.
  /// @param {Object} options An object containing the following properties:
  /// @param {string} [options.task] The task of the pipeline. Useful for specifying subtasks.
  /// @param {PreTrainedModel} [options.model] The model used by the pipeline.
  /// @param {PreTrainedTokenizer} [options.tokenizer=null] The tokenizer used by the pipeline (if any).
  /// @param {Processor} [options.processor=null] The processor used by the pipeline (if any).
  Pipeline({
    required this.task,
    required this.model,
    this.tokenizer,
    this.processor,
  });

  factory Pipeline.fromJson(Map<String, dynamic> json) => Pipeline(
    task: json['task'],
    model: json['model'],
    tokenizer: json['tokenizer'],
    processor: json['processor'],
  );

  Future<dynamic> call(dynamic texts, [dynamic options]) async {
    throw ArgumentError('Method should be overridden in subclass.');
  }

  Future<void> dispose() async {
    await model.dispose();
  }
}

class FillMaskPipelineOptions extends PipelineOptions {
  /// The number of top predictions to return.
  int? top_k;

  FillMaskPipelineOptions({int? top_k}) : top_k = top_k ?? 5;
}

/// Masked language modeling prediction pipeline using any `ModelWithLMHead`.
///
/// **Example:** Perform masked language modelling (a.k.a. "fill-mask") with `Xenova/bert-base-uncased`.
/// ```javascript
/// const unmasker = await pipeline('fill-mask', 'Xenova/bert-base-cased');
/// const output = await unmasker('The goal of life is [MASK].');
/// // [
/// //   { token_str: 'survival', score: 0.06137419492006302, token: 8115, sequence: 'The goal of life is survival.' },
/// //   { token_str: 'love', score: 0.03902450203895569, token: 1567, sequence: 'The goal of life is love.' },
/// //   { token_str: 'happiness', score: 0.03253183513879776, token: 9266, sequence: 'The goal of life is happiness.' },
/// //   { token_str: 'freedom', score: 0.018736306577920914, token: 4438, sequence: 'The goal of life is freedom.' },
/// //   { token_str: 'life', score: 0.01859794743359089, token: 1297, sequence: 'The goal of life is life.' }
/// // ]
/// ```
///
/// **Example:** Perform masked language modelling (a.k.a. "fill-mask") with `Xenova/bert-base-cased` (and return top result).
/// ```javascript
/// const unmasker = await pipeline('fill-mask', 'Xenova/bert-base-cased');
/// const output = await unmasker('The Milky Way is a [MASK] galaxy.', { top_k: 1 });
/// // [{ token_str: 'spiral', score: 0.6299987435340881, token: 14061, sequence: 'The Milky Way is a spiral galaxy.' }]
/// ```
class FillMaskPipeline extends Pipeline {
  FillMaskPipeline({required super.task, required super.model, super.tokenizer, super.processor});

  factory FillMaskPipeline.fromJson(Map<String, dynamic> json) => FillMaskPipeline(
    task: json['task'],
    model: json['model'],
    tokenizer: json['tokenizer'],
    processor: json['processor'],
  );

  /// [texts] - [String] or [List<String>]
  @override
  Future<dynamic> call(dynamic texts, [dynamic options]) async {
    if (!(texts is String || texts is List<String>)) {
      throw ArgumentError('Argument `text` must be a String or a List<String>. Provided: "$texts"');
    }
    if (options is! FillMaskPipelineOptions?) {
      throw ArgumentError('Argument `options` must be FillMaskPipelineOptions');
    }

    final List<String> textsAsList = texts is List<String> ? texts : [texts];
    options ??= FillMaskPipelineOptions();
    final top_k = options.top_k;

    final tokenizer = this.tokenizer;
    if (tokenizer == null) {
      throw Exception('Tokenizer is not set');
    }

    // Run tokenization
    final model_inputs = await tokenizer(
      textsAsList,
      padding: true,
      truncation: true,
    );

    // Run model
    final modelOutput = await model(model_inputs.toJson());
    final Tensor logits = modelOutput['logits']!;

    final List toReturn = [];
    final List<List<int>> input_ids = (model_inputs.input_ids as Tensor)
        .tolist()
        .map((x) => List<int>.from(x))
        .toList();
    for (int i = 0; i < input_ids.length; ++i) {
      final ids = input_ids[i];
      final mask_token_index = ids.indexWhere((x) => x == tokenizer.mask_token_id);
      if (mask_token_index == -1) {
        throw ArgumentError('Mask token (${tokenizer.mask_token}) not found in text.');
      }
      final itemLogits = await (await logits[i])[mask_token_index];

      final scores = await topk(await Tensor.create(
        TensorDataType.float32,
        softmax(List<num>.from(itemLogits.data)),
        itemLogits.dims,
      ), top_k);
      final values = scores.$1.tolist();
      final indices = List<int>.from(scores.$2.tolist());

      toReturn.add(indices.indexed.map((e) {
        final (i, x) = e;
        final sequence = ids.toList();
        sequence[mask_token_index] = x;

        return {
          'score': values[i],
          'token': x,
          'token_str': tokenizer.decode([x]),
          'sequence': tokenizer.decode(sequence, skip_special_tokens: true),
        };
      }));
    }

    return texts is List ? toReturn : toReturn.first;
  }
}

class TextGenerationConfig extends GenerationConfig {
  /// Whether or not to add special tokens when tokenizing the sequences.
  bool? add_special_tokens;

  /// If set to `false` only added text is returned, otherwise the full text is returned.
  bool return_full_text;

  TextGenerationConfig({
    super.max_length,
    super.max_new_tokens,
    super.min_length,
    super.min_new_tokens,
    super.early_stopping,
    super.max_time,
    super.do_sample,
    super.num_beams,
    super.num_beam_groups,
    super.penalty_alpha,
    super.use_cache,
    super.temperature,
    super.top_k,
    super.top_p,
    super.typical_p,
    super.epsilon_cutoff,
    super.eta_cutoff,
    super.diversity_penalty,
    super.repetition_penalty,
    super.encoder_repetition_penalty,
    super.length_penalty,
    super.no_repeat_ngram_size,
    super.bad_words_ids,
    super.force_words_ids,
    super.renormalize_logits,
    super.constraints,
    super.forced_bos_token_id,
    super.forced_eos_token_id,
    super.remove_invalid_values,
    super.exponential_decay_length_penalty,
    super.suppress_tokens,
    super.streamer,
    super.begin_suppress_tokens,
    super.forced_decoder_ids,
    super.guidance_scale,
    super.num_return_sequences,
    super.output_attentions,
    super.output_hidden_states,
    super.output_scores,
    super.return_dict_in_generate,
    super.pad_token_id,
    super.bos_token_id,
    super.eos_token_id,
    super.encoder_no_repeat_ngram_size,
    super.decoder_start_token_id,
    super.generation_kwargs,

    this.add_special_tokens,
    bool? return_full_text,
  }) : return_full_text = return_full_text ?? true;

  factory TextGenerationConfig.fromJson(Map<String, dynamic> json) => TextGenerationConfig(
    max_length: json['max_length'],
    max_new_tokens: json['max_new_tokens'],
    min_length: json['min_length'],
    min_new_tokens: json['min_new_tokens'],
    early_stopping: json['early_stopping'],
    max_time: json['max_time'],
    do_sample: json['do_sample'],
    num_beams: json['num_beams'],
    num_beam_groups: json['num_beam_groups'],
    penalty_alpha: json['penalty_alpha'],
    use_cache: json['use_cache'],
    temperature: json['temperature'],
    top_k: json['top_k'],
    top_p: json['top_p'],
    typical_p: json['typical_p'],
    epsilon_cutoff: json['epsilon_cutoff'],
    eta_cutoff: json['eta_cutoff'],
    diversity_penalty: json['diversity_penalty'],
    repetition_penalty: json['repetition_penalty'],
    encoder_repetition_penalty: json['encoder_repetition_penalty'],
    length_penalty: json['length_penalty'],
    no_repeat_ngram_size: json['no_repeat_ngram_size'],
    bad_words_ids: json['bad_words_ids'],
    force_words_ids: json['force_words_ids'],
    renormalize_logits: json['renormalize_logits'],
    constraints: json['constraints'],
    forced_bos_token_id: json['forced_bos_token_id'],
    forced_eos_token_id: json['forced_eos_token_id'],
    remove_invalid_values: json['remove_invalid_values'],
    exponential_decay_length_penalty: json['exponential_decay_length_penalty'],
    suppress_tokens: json['suppress_tokens'],
    streamer: json['streamer'],
    begin_suppress_tokens: json['begin_suppress_tokens'],
    forced_decoder_ids: json['forced_decoder_ids'],
    guidance_scale: json['guidance_scale'],
    num_return_sequences: json['num_return_sequences'],
    output_attentions: json['output_attentions'],
    output_hidden_states: json['output_hidden_states'],
    output_scores: json['output_scores'],
    return_dict_in_generate: json['return_dict_in_generate'],
    pad_token_id: json['pad_token_id'],
    bos_token_id: json['bos_token_id'],
    eos_token_id: json['eos_token_id'],
    encoder_no_repeat_ngram_size: json['encoder_no_repeat_ngram_size'],
    decoder_start_token_id: json['decoder_start_token_id'],
    generation_kwargs: json['generation_kwargs'],
    add_special_tokens: json['add_special_tokens'],
    return_full_text: json['return_full_text'],
  );

  @override
  Map<String, dynamic> toJson() => {
    ...super.toJson(),
    'add_special_tokens': add_special_tokens,
    'return_full_text': return_full_text,
  };
}

bool isChat(dynamic x) {
  return x is List && x.every((x) => x is Message);
}

/// Language generation pipeline using any `ModelWithLMHead` or `ModelForCausalLM`.
/// This pipeline predicts the words that will follow a specified text prompt.
/// NOTE: For the full list of generation parameters, see [`GenerationConfig`](./utils/generation#module_utils/generation.GenerationConfig).
///
/// **Example:** Text generation with `Xenova/distilgpt2` (default settings).
/// ```javascript
/// const generator = await pipeline('text-generation', 'Xenova/distilgpt2');
/// const text = 'I enjoy walking with my cute dog,';
/// const output = await generator(text);
/// // [{ generated_text: "I enjoy walking with my cute dog, and I love to play with the other dogs." }]
/// ```
///
/// **Example:** Text generation with `Xenova/distilgpt2` (custom settings).
/// ```javascript
/// const generator = await pipeline('text-generation', 'Xenova/distilgpt2');
/// const text = 'Once upon a time, there was';
/// const output = await generator(text, {
///   temperature: 2,
///   max_new_tokens: 10,
///   repetition_penalty: 1.5,
///   no_repeat_ngram_size: 2,
///   num_beams: 2,
///   num_return_sequences: 2,
/// });
/// // [{
/// //   "generated_text": "Once upon a time, there was an abundance of information about the history and activities that"
/// // }, {
/// //   "generated_text": "Once upon a time, there was an abundance of information about the most important and influential"
/// // }]
/// ```
///
/// **Example:** Run code generation with `Xenova/codegen-350M-mono`.
/// ```javascript
/// const generator = await pipeline('text-generation', 'Xenova/codegen-350M-mono');
/// const text = 'def fib(n):';
/// const output = await generator(text, {
///   max_new_tokens: 44,
/// });
/// // [{
/// //   generated_text: 'def fib(n):\n' +
/// //     '    if n == 0:\n' +
/// //     '        return 0\n' +
/// //     '    elif n == 1:\n' +
/// //     '        return 1\n' +
/// //     '    else:\n' +
/// //     '        return fib(n-1) + fib(n-2)\n'
/// // }]
/// ```
class TextGenerationPipeline extends Pipeline {
  TextGenerationPipeline({required super.task, required super.model, super.tokenizer, super.processor});

  factory TextGenerationPipeline.fromJson(Map<String, dynamic> json) => TextGenerationPipeline(
    task: json['task'],
    model: json['model'],
    tokenizer: json['tokenizer'],
    processor: json['processor'],
  );

  @override
  Future<dynamic> call(dynamic texts, [dynamic options]) async {
    if (options is! TextGenerationConfig?) {
      throw ArgumentError('Argument `options` must be TextGenerationConfig');
    }

    options ??= TextGenerationConfig();

    final tokenizer = this.tokenizer;
    if (tokenizer == null) {
      throw Exception('Tokenizer is not set');
    }

    bool isBatched = false;
    bool isChatInput = false;

    // Normalize inputs
    List<String> inputs;
    if (texts is String) {
      inputs = texts = [texts];
    } else if (texts is List<String>) {
      isBatched = true;
      inputs = texts;
    } else {
      if (isChat(texts)) {
        texts = [texts];
      } else if (texts is List && texts.every(isChat)) {
        isBatched = true;
      } else {
        throw ArgumentError('Input must be a string, a list of strings, a Message, or a list of Message');
      }
      isChatInput = true;

      // If the input is a chat, we need to apply the chat template
      inputs = await Future.wait((texts as List)
          .map((x) async => await tokenizer.apply_chat_template(x, ApplyChatTemplateOptions(
        tokenize: false,
        add_generation_prompt: true,
      ))));
    }

    // By default, do not add special tokens
    final add_special_tokens = options.add_special_tokens ?? false;

    // By default, return full text
    final return_full_text = isChatInput ? false : options.return_full_text;

    tokenizer.padding_side = 'left';
    final text_inputs = await tokenizer(
      inputs,
      add_special_tokens: add_special_tokens,
      padding: true,
      truncation: true,
    );

    final Tensor outputTokenIds = await model.generate({
      ...text_inputs.toJson(),
      ...options.toJson(),
    });

    final decoded = tokenizer.batch_decode(
      outputTokenIds,
      skip_special_tokens: true,
    );

    List<int>? promptLengths;
    if (!return_full_text && (text_inputs.input_ids as Tensor).dims.last > 0) {
      promptLengths = tokenizer.batch_decode(
        text_inputs.input_ids,
        skip_special_tokens: true,
      ).map((x) => x.length).toList();
    }

    /** @type {TextGenerationOutput[]} */
    final List<List<dynamic>> toReturn = List.generate(texts.length, (_) => []);
    for (int i = 0; i < decoded.length; ++i) {
      final textIndex = (i / outputTokenIds.dims.first * texts.length).floor();

      if (promptLengths != null) {
        // Trim the decoded text to only include the generated part
        decoded[i] = decoded[i].substring(promptLengths[textIndex]);
      }
      toReturn[textIndex].add({
        'generated_text': isChatInput ? [
          ...texts[textIndex],
          Message(role: 'assistant', content: decoded[i]),
        ] : decoded[i]
      });
    }
    return (!isBatched && toReturn.length == 1) ? toReturn.first : toReturn;
  }
}

class FeatureExtractionPipelineOptions extends PipelineOptions {
  final String pooling;
  final bool normalize;
  final bool quantize;
  final QuantizeEmbeddingsPrecision precision;

  FeatureExtractionPipelineOptions({
    String? pooling,
    bool? normalize,
    bool? quantize,
    QuantizeEmbeddingsPrecision? precision,
  }) : pooling = pooling ?? 'none',
        normalize = normalize ?? false,
        quantize = quantize ?? false,
        precision = precision ?? QuantizeEmbeddingsPrecision.binary;
}

/// Feature extraction pipeline using no model head. This pipeline extracts the hidden
/// states from the base transformer, which can be used as features in downstream tasks.
///
/// **Example:** Run feature extraction with `bert-base-uncased` (without pooling/normalization).
/// ```javascript
/// const extractor = await pipeline('feature-extraction', 'Xenova/bert-base-uncased', { revision: 'default' });
/// const output = await extractor('This is a simple test.');
/// // Tensor {
/// //   type: 'float32',
/// //   data: Float32Array [0.05939924716949463, 0.021655935794115067, ...],
/// //   dims: [1, 8, 768]
/// // }
/// ```
///
/// **Example:** Run feature extraction with `bert-base-uncased` (with pooling/normalization).
/// ```javascript
/// const extractor = await pipeline('feature-extraction', 'Xenova/bert-base-uncased', { revision: 'default' });
/// const output = await extractor('This is a simple test.', { pooling: 'mean', normalize: true });
/// // Tensor {
/// //   type: 'float32',
/// //   data: Float32Array [0.03373778983950615, -0.010106077417731285, ...],
/// //   dims: [1, 768]
/// // }
/// ```
///
/// **Example:** Calculating embeddings with `sentence-transformers` models.
/// ```javascript
/// const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
/// const output = await extractor('This is a simple test.', { pooling: 'mean', normalize: true });
/// // Tensor {
/// //   type: 'float32',
/// //   data: Float32Array [0.09094982594251633, -0.014774246141314507, ...],
/// //   dims: [1, 384]
/// // }
/// ```
/// **Example:** Calculating binary embeddings with `sentence-transformers` models.
/// ```javascript
/// const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
/// const output = await extractor('This is a simple test.', { pooling: 'mean', quantize: true, precision: 'binary' });
/// // Tensor {
/// //   type: 'int8',
/// //   data: Int8Array [49, 108, 24, ...],
/// //   dims: [1, 48]
/// // }
/// ```
class FeatureExtractionPipeline extends Pipeline {
  FeatureExtractionPipeline({required super.task, required super.model, super.tokenizer, super.processor});

  factory FeatureExtractionPipeline.fromJson(Map<String, dynamic> json) => FeatureExtractionPipeline(
    task: json['task'],
    model: json['model'],
    tokenizer: json['tokenizer'],
    processor: json['processor'],
  );

  /// @type {FeatureExtractionPipelineCallback}
  @override
  Future<dynamic> call(dynamic texts, [dynamic options]) async {
    if (options is! FeatureExtractionPipelineOptions?) {
      throw ArgumentError('Argument `options` must be TextGenerationConfig');
    }

    options ??= FeatureExtractionPipelineOptions();

    final tokenizer = this.tokenizer;
    if (tokenizer == null) {
      throw Exception('Tokenizer is not set');
    }

    // Run tokenization
    final model_inputs = await tokenizer(
      texts,
      padding: true,
      truncation: true,
    );

    // Run model
    final outputs = await model(model_inputs.toJson());

    // TODO: Provide warning to the user that they might be using model which was not exported
    // specifically for feature extraction
    // console.log(this.model.config)
    // console.log(outputs)

    /** @type {Tensor} */
    Tensor? result = outputs['last_hidden_state'] ?? outputs['logits'] ?? outputs['token_embeddings'];
    if (result == null) throw Exception('No hidden state found');

    switch (options.pooling) {
      case 'none':
        // Skip pooling
        break;
      case 'mean':
        result = await mean_pooling(result, model_inputs.attention_mask);
        break;
      case 'first_token':
      case 'cls':
        result = await result.slice([null, 0]);
        break;
      case 'last_token':
      case 'eos':
        result = await result.slice([null, -1]);
        break;
      default:
        throw ArgumentError("Pooling method '${options.pooling}' not supported.");
    }

    if (options.normalize) {
      result = await result.normalize(2, -1);
    }

    if (options.quantize) {
      result = await quantize_embeddings(result, options.precision);
    }

    return result;
  }
}

enum PipelineType {
  // --- START TaskType ---
  textClassification('text-classification'),
  tokenClassification('token-classification'),
  questionAnswering('question-answering'),
  fillMask('fill-mask'),
  summarization('summarization'),
  translation('translation'),
  text2textGeneration('text2text-generation'),
  textGeneration('text-generation'),
  zeroShotClassification('zero-shot-classification'),
  audioClassification('audio-classification'),
  zeroShotAudioClassification('zero-shot-audio-classification'),
  automaticSpeechRecognition('automatic-speech-recognition'),
  textToAudio('text-to-audio'),
  imageToText('image-to-text'),
  imageClassification('image-classification'),
  imageSegmentation('image-segmentation'),
  backgroundRemoval('background-removal'),
  zeroShotImageClassification('zero-shot-image-classification'),
  objectDetection('object-detection'),
  zeroShotObjectDetection('zero-shot-object-detection'),
  documentQuestionAnswering('document-question-answering'),
  imageToImage('image-to-image'),
  depthEstimation('depth-estimation'),
  featureExtraction('feature-extraction'),
  imageFeatureExtraction('image-feature-extraction'),
  // --- END TaskType ---

  // --- START AliasType ---
  sentimentAnalysis('sentiment-analysis'),
  ner('ner'),
  // "vqa": "visual-question-answering", // TODO: Add
  asr('asr'),
  textToSpeech('text-to-speech'),

  // Add for backwards compatibility
  embeddings('embeddings');
  // --- END AliasType ---

  final String value;

  const PipelineType(this.value);

  /// Get the [PipelineType] from it's [String] value.
  static PipelineType fromString(String pipelineType) => PipelineType
      .values
      .firstWhere((e) => e.value == pipelineType);

  String toJson() => value;

  @override
  String toString() => value;
}

const Map<String, Map<String, dynamic>> SUPPORTED_TASKS = {
  // "text-classification": {
  //   "tokenizer": AutoTokenizer,
  //   "pipeline": TextClassificationPipeline,
  //   "model": AutoModelForSequenceClassification,
  //   "default": {
  //     // TODO: replace with original
  //     // "model": "distilbert-base-uncased-finetuned-sst-2-english",
  //     "model": "Xenova/distilbert-base-uncased-finetuned-sst-2-english",
  //   },
  //   "type": "text",
  // },
  // "token-classification": {
  //   "tokenizer": AutoTokenizer,
  //   "pipeline": TokenClassificationPipeline,
  //   "model": AutoModelForTokenClassification,
  //   "default": {
  //     // TODO: replace with original
  //     // "model": "Davlan/bert-base-multilingual-cased-ner-hrl",
  //     "model": "Xenova/bert-base-multilingual-cased-ner-hrl",
  //   },
  //   "type": "text",
  // },
  // "question-answering": {
  //   "tokenizer": AutoTokenizer,
  //   "pipeline": QuestionAnsweringPipeline,
  //   "model": AutoModelForQuestionAnswering,
  //   "default": {
  //     // TODO: replace with original
  //     // "model": "distilbert-base-cased-distilled-squad",
  //     "model": "Xenova/distilbert-base-cased-distilled-squad",
  //   },
  //   "type": "text",
  // },

  "fill-mask": {
    "tokenizer": AutoTokenizer.from_pretrained,
    "pipeline": FillMaskPipeline.fromJson,
    "model": AutoModelForMaskedLM.from_pretrained,
    "default": {
      // TODO: replace with original
      // "model": "bert-base-uncased",
      "model": "Xenova/bert-base-uncased",
    },
    "type": "text",
  },
  // "summarization": {
  //   "tokenizer": AutoTokenizer,
  //   "pipeline": SummarizationPipeline,
  //   "model": AutoModelForSeq2SeqLM,
  //   "default": {
  //     // TODO: replace with original
  //     // "model": "sshleifer/distilbart-cnn-6-6",
  //     "model": "Xenova/distilbart-cnn-6-6",
  //   },
  //   "type": "text",
  // },
  // "translation": {
  //   "tokenizer": AutoTokenizer,
  //   "pipeline": TranslationPipeline,
  //   "model": AutoModelForSeq2SeqLM,
  //   "default": {
  //     // TODO: replace with original
  //     // "model": "t5-small",
  //     "model": "Xenova/t5-small",
  //   },
  //   "type": "text",
  // },
  // "text2text-generation": {
  //   "tokenizer": AutoTokenizer,
  //   "pipeline": Text2TextGenerationPipeline,
  //   "model": AutoModelForSeq2SeqLM,
  //   "default": {
  //     // TODO: replace with original
  //     // "model": "google/flan-t5-small",
  //     "model": "Xenova/flan-t5-small",
  //   },
  //   "type": "text",
  // },
  "text-generation": {
    "tokenizer": AutoTokenizer.from_pretrained,
    "pipeline": TextGenerationPipeline.fromJson,
    "model": AutoModelForCausalLM.from_pretrained,
    "default": {
      // TODO: replace with original
      // "model": "gpt2",
      "model": "Xenova/gpt2",
    },
    "type": "text",
  },
  // "zero-shot-classification": {
  //   "tokenizer": AutoTokenizer,
  //   "pipeline": ZeroShotClassificationPipeline,
  //   "model": AutoModelForSequenceClassification,
  //   "default": {
  //     // TODO: replace with original
  //     // "model": "typeform/distilbert-base-uncased-mnli",
  //     "model": "Xenova/distilbert-base-uncased-mnli",
  //   },
  //   "type": "text",
  // },
  // "audio-classification": {
  //   "pipeline": AudioClassificationPipeline,
  //   "model": AutoModelForAudioClassification,
  //   "processor": AutoProcessor,
  //   "default": {
  //     // TODO: replace with original
  //     // "model": "superb/wav2vec2-base-superb-ks",
  //     "model": "Xenova/wav2vec2-base-superb-ks",
  //   },
  //   "type": "audio",
  // },
  // "zero-shot-audio-classification": {
  //   "tokenizer": AutoTokenizer,
  //   "pipeline": ZeroShotAudioClassificationPipeline,
  //   "model": AutoModel,
  //   "processor": AutoProcessor,
  //   "default": {
  //     // TODO: replace with original
  //     // "model": "laion/clap-htsat-fused",
  //     "model": "Xenova/clap-htsat-unfused",
  //   },
  //   "type": "multimodal",
  // },
  // "automatic-speech-recognition": {
  //   "tokenizer": AutoTokenizer,
  //   "pipeline": AutomaticSpeechRecognitionPipeline,
  //   "model": [AutoModelForSpeechSeq2Seq, AutoModelForCTC],
  //   "processor": AutoProcessor,
  //   "default": {
  //     // TODO: replace with original
  //     // "model": "openai/whisper-tiny.en",
  //     "model": "Xenova/whisper-tiny.en",
  //   },
  //   "type": "multimodal",
  // },
  // "text-to-audio": {
  //   "tokenizer": AutoTokenizer,
  //   "pipeline": TextToAudioPipeline,
  //   "model": [AutoModelForTextToWaveform, AutoModelForTextToSpectrogram],
  //   "processor": [AutoProcessor, /* Some don't use a processor */ null],
  //   "default": {
  //     // TODO: replace with original
  //     // "model": "microsoft/speecht5_tts",
  //     "model": "Xenova/speecht5_tts",
  //   },
  //   "type": "text",
  // },
  // "image-to-text": {
  //   "tokenizer": AutoTokenizer,
  //   "pipeline": ImageToTextPipeline,
  //   "model": AutoModelForVision2Seq,
  //   "processor": AutoProcessor,
  //   "default": {
  //     // TODO: replace with original
  //     // "model": "nlpconnect/vit-gpt2-image-captioning",
  //     "model": "Xenova/vit-gpt2-image-captioning",
  //   },
  //   "type": "multimodal",
  // },
  //
  // "image-classification": {
  //   // no tokenizer
  //   "pipeline": ImageClassificationPipeline,
  //   "model": AutoModelForImageClassification,
  //   "processor": AutoProcessor,
  //   "default": {
  //     // TODO: replace with original
  //     // "model": "google/vit-base-patch16-224",
  //     "model": "Xenova/vit-base-patch16-224",
  //   },
  //   "type": "multimodal",
  // },
  //
  // "image-segmentation": {
  //   // no tokenizer
  //   "pipeline": ImageSegmentationPipeline,
  //   "model": [AutoModelForImageSegmentation, AutoModelForSemanticSegmentation, AutoModelForUniversalSegmentation],
  //   "processor": AutoProcessor,
  //   "default": {
  //     // TODO: replace with original
  //     // "model": "facebook/detr-resnet-50-panoptic",
  //     "model": "Xenova/detr-resnet-50-panoptic",
  //   },
  //   "type": "multimodal",
  // },
  // "background-removal": {
  //   // no tokenizer
  //   "pipeline": BackgroundRemovalPipeline,
  //   "model": [AutoModelForImageSegmentation, AutoModelForSemanticSegmentation, AutoModelForUniversalSegmentation],
  //   "processor": AutoProcessor,
  //   "default": {
  //     "model": "Xenova/modnet",
  //   },
  //   "type": "image",
  // },
  //
  // "zero-shot-image-classification": {
  //   "tokenizer": AutoTokenizer,
  //   "pipeline": ZeroShotImageClassificationPipeline,
  //   "model": AutoModel,
  //   "processor": AutoProcessor,
  //   "default": {
  //     // TODO: replace with original
  //     // "model": "openai/clip-vit-base-patch32",
  //     "model": "Xenova/clip-vit-base-patch32",
  //   },
  //   "type": "multimodal",
  // },
  //
  // "object-detection": {
  //   // no tokenizer
  //   "pipeline": ObjectDetectionPipeline,
  //   "model": AutoModelForObjectDetection,
  //   "processor": AutoProcessor,
  //   "default": {
  //     // TODO: replace with original
  //     // "model": "facebook/detr-resnet-50",
  //     "model": "Xenova/detr-resnet-50",
  //   },
  //   "type": "multimodal",
  // },
  // "zero-shot-object-detection": {
  //   "tokenizer": AutoTokenizer,
  //   "pipeline": ZeroShotObjectDetectionPipeline,
  //   "model": AutoModelForZeroShotObjectDetection,
  //   "processor": AutoProcessor,
  //   "default": {
  //     // TODO: replace with original
  //     // "model": "google/owlvit-base-patch32",
  //     "model": "Xenova/owlvit-base-patch32",
  //   },
  //   "type": "multimodal",
  // },
  // "document-question-answering": {
  //   "tokenizer": AutoTokenizer,
  //   "pipeline": DocumentQuestionAnsweringPipeline,
  //   "model": AutoModelForDocumentQuestionAnswering,
  //   "processor": AutoProcessor,
  //   "default": {
  //     // TODO: replace with original
  //     // "model": "naver-clova-ix/donut-base-finetuned-docvqa",
  //     "model": "Xenova/donut-base-finetuned-docvqa",
  //   },
  //   "type": "multimodal",
  // },
  // "image-to-image": {
  //   // no tokenizer
  //   "pipeline": ImageToImagePipeline,
  //   "model": AutoModelForImageToImage,
  //   "processor": AutoProcessor,
  //   "default": {
  //     // TODO: replace with original
  //     // "model": "caidas/swin2SR-classical-sr-x2-64",
  //     "model": "Xenova/swin2SR-classical-sr-x2-64",
  //   },
  //   "type": "image",
  // },
  // "depth-estimation": {
  //   // no tokenizer
  //   "pipeline": DepthEstimationPipeline,
  //   "model": AutoModelForDepthEstimation,
  //   "processor": AutoProcessor,
  //   "default": {
  //     // TODO: replace with original
  //     // "model": "Intel/dpt-large",
  //     "model": "Xenova/dpt-large",
  //   },
  //   "type": "image",
  // },

  // This task serves as a useful interface for dealing with sentence-transformers (https://huggingface.co/sentence-transformers).
  "feature-extraction": {
    "tokenizer": AutoTokenizer.from_pretrained,
    "pipeline": FeatureExtractionPipeline.fromJson,
    "model": AutoModel.from_pretrained,
    "default": {
      // TODO: replace with original
      // "model": "sentence-transformers/all-MiniLM-L6-v2",
      "model": "Xenova/all-MiniLM-L6-v2",
    },
    "type": "text",
  },
  // "image-feature-extraction": {
  //   "processor": AutoProcessor,
  //   "pipeline": ImageFeatureExtractionPipeline,
  //   "model": [AutoModelForImageFeatureExtraction, AutoModel],
  //   "default": {
  //     // TODO: replace with original
  //     // "model": "google/vit-base-patch16-224",
  //     "model": "Xenova/vit-base-patch16-224-in21k",
  //   },
  //   "type": "image",
  // },
};

const Map<PipelineType, PipelineType> TASK_ALIASES = {
  PipelineType.sentimentAnalysis: PipelineType.textClassification,
  PipelineType.ner: PipelineType.tokenClassification,
  // PipelineType.vqa: PipelineType.visualQuestionAnswering, // TODO: Add
  PipelineType.asr: PipelineType.automaticSpeechRecognition,
  PipelineType.textToSpeech: PipelineType.textToAudio,
  PipelineType.embeddings: PipelineType.featureExtraction,
};

/// Utility factory method to build a `Pipeline` object.
///
/// @template {PipelineType} T The type of pipeline to return.
/// @param {T} task The task defining which pipeline will be returned. Currently accepted tasks are:
///  - `"audio-classification"`: will return a `AudioClassificationPipeline`.
///  - `"automatic-speech-recognition"`: will return a `AutomaticSpeechRecognitionPipeline`.
///  - `"depth-estimation"`: will return a `DepthEstimationPipeline`.
///  - `"document-question-answering"`: will return a `DocumentQuestionAnsweringPipeline`.
///  - `"feature-extraction"`: will return a `FeatureExtractionPipeline`.
///  - `"fill-mask"`: will return a `FillMaskPipeline`.
///  - `"image-classification"`: will return a `ImageClassificationPipeline`.
///  - `"image-segmentation"`: will return a `ImageSegmentationPipeline`.
///  - `"image-to-text"`: will return a `ImageToTextPipeline`.
///  - `"object-detection"`: will return a `ObjectDetectionPipeline`.
///  - `"question-answering"`: will return a `QuestionAnsweringPipeline`.
///  - `"summarization"`: will return a `SummarizationPipeline`.
///  - `"text2text-generation"`: will return a `Text2TextGenerationPipeline`.
///  - `"text-classification"` (alias "sentiment-analysis" available): will return a `TextClassificationPipeline`.
///  - `"text-generation"`: will return a `TextGenerationPipeline`.
///  - `"token-classification"` (alias "ner" available): will return a `TokenClassificationPipeline`.
///  - `"translation"`: will return a `TranslationPipeline`.
///  - `"translation_xx_to_yy"`: will return a `TranslationPipeline`.
///  - `"zero-shot-classification"`: will return a `ZeroShotClassificationPipeline`.
///  - `"zero-shot-audio-classification"`: will return a `ZeroShotAudioClassificationPipeline`.
///  - `"zero-shot-image-classification"`: will return a `ZeroShotImageClassificationPipeline`.
///  - `"zero-shot-object-detection"`: will return a `ZeroShotObjectDetectionPipeline`.
/// @param {string} [model=null] The name of the pre-trained model to use. If not specified, the default model for the task will be used.
/// @param {import('./utils/hub.js').PretrainedModelOptions} [options] Optional parameters for the pipeline.
/// @returns {Promise<AllTasks[T]>} A Pipeline object for the specified task.
/// @throws {Error} If an unsupported pipeline is requested.
Future<Pipeline> pipeline(PipelineType task, [String? model, PretrainedModelOptions? options]) async {
  options ??= PretrainedModelOptions();

  // Apply aliases
  task = TASK_ALIASES[task] ?? task;

  // Get pipeline info
  final pipelineInfo = SUPPORTED_TASKS[task.value.split('_').first];
  if (pipelineInfo == null) {
    throw ArgumentError('Unsupported pipeline: $task. Must be one of [${SUPPORTED_TASKS.keys}]');
  }

  // Use model if specified, otherwise, use default
  if (model == null) {
    model = pipelineInfo['default']['model'];
    // console.log
    print('No model specified. Using default model: "$model".');
  }

  final classes = {
    'tokenizer': pipelineInfo['tokenizer'],
    'model': pipelineInfo['model'],
    'processor': pipelineInfo['processor'],
  };

  // Load model, tokenizer, and processor (if they exist)
  final results = await loadItems(classes, model!, options);
  results['task'] = task.value;

  dispatchCallback(options.progress_callback, ReadyProgressInfo(task.value, model));

  final pipelineClass = pipelineInfo['pipeline'];
  return pipelineClass(results);
}

/// Helper function to get applicable model, tokenizer, or processor classes for a given model.
/// @param {Map<string, any>} mapping The mapping of names to classes, arrays of classes, or null.
/// @param {string} model The name of the model to load.
/// @param {import('./utils/hub.js').PretrainedOptions} pretrainedOptions The options to pass to the `from_pretrained` method.
/// @private
Future<Map<String, dynamic>> loadItems(
  Map<String, dynamic> mapping,
  String model,
  PretrainedOptions pretrainedOptions,
) async {
  final Map<String, dynamic> result = {};

  final List<Future<void>> futures = [];
  for (final e in mapping.entries) {
    final name = e.key;
    final cls = e.value;
    if (cls == null) continue;

    final Future<void> future;
    if (cls is List) {
      future = (() async {
        // Error e;
        for (final c in cls) {
          if (c == null) {
            // If null, we resolve it immediately, meaning the relevant
            // class was not found, but it is optional.
            return null;
          }
          try {
            return await c.from_pretrained(model, pretrainedOptions);
          } catch (err) {
            // TODO
            // if (err.message?.includes('Unsupported model type')) {
            //   // If the error is due to an unsupported model type, we
            //   // save the error and try the next class.
            //   e = err;
            // } else if (err.message?.includes('Could not locate file')) {
            //   e = err;
            // } else {
            //   reject(err);
            //   return;
            // }
            rethrow;
          }
        }
        // throw e;
      })();
    } else {
      final PretrainedOptions options = switch (name) {
        'tokenizer' => PretrainedTokenizerOptions.fromJson(pretrainedOptions.toJson()),
        _ => pretrainedOptions,
      };

      future = cls(model, options);
    }

    result[name] = future;
    futures.add(future);
  }

  await Future.wait(futures);

  for (final e in result.entries) {
    result[e.key] = await e.value;
  }

  return result;
}