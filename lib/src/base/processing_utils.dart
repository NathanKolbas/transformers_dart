import 'package:transformers/src/base/feature_extraction_utils.dart';
import 'package:transformers/src/base/image_processing_utils.dart';
import 'package:transformers/src/utils/hub.dart';

import '../tokenizers.dart';
import '../utils/constants.dart';

class ImageProcessorClassReflection {
  final Future<ImageProcessor> Function(
    String pretrained_model_name_or_path,
    [PretrainedOptions? options]
  ) from_pretrained;

  ImageProcessorClassReflection({required this.from_pretrained});
}

class TokenizerClassReflection {
  final Future<PreTrainedTokenizer> Function(
    String pretrained_model_name_or_path,
    [PretrainedTokenizerOptions? options]
  ) from_pretrained;

  TokenizerClassReflection({required this.from_pretrained});
}

class FeatureExtractorClassReflection {
  final Future<FeatureExtractor> Function(
    String pretrained_model_name_or_path,
    [PretrainedOptions? options]
  ) from_pretrained;

  FeatureExtractorClassReflection({required this.from_pretrained});
}

class ProcessorReflection {
  final Processor Function(
    Map<String, dynamic> config,
    Map<String, dynamic> components,
  ) constructor;

  final List<String> classes;

  final ImageProcessorClassReflection? image_processor_class;

  final TokenizerClassReflection? tokenizer_class;

  final FeatureExtractorClassReflection? feature_extractor_class;

  final bool uses_processor_config;

  const ProcessorReflection({
    required this.constructor,
    this.classes = const [
      'image_processor_class',
      'tokenizer_class',
      'feature_extractor_class',
    ],
    this.image_processor_class,
    this.tokenizer_class,
    this.feature_extractor_class,
    this.uses_processor_config = false,
  });
}

class ProcessorPretrainedOptions extends PretrainedTokenizerOptions {}

/// Represents a Processor that extracts features from an input.
class Processor {
  static final ProcessorReflection reflection = ProcessorReflection(
    constructor: Processor.new,
  );

  final Map<String, dynamic> config;
  final Map<String, dynamic> components;

  /// Creates a new Processor with the given components
  /// @param {Object} config
  /// @param {Record<string, Object>} components
  Processor(this.config, this.components);

  /// The image processor of the processor, if it exists.
  ImageProcessor? get image_processor => components['image_processor'];

  /// The tokenizer of the processor, if it exists.
  PreTrainedTokenizer? get tokenizer => components['tokenizer'];

  /// The feature extractor of the processor, if it exists.
  FeatureExtractor? get feature_extractor => components['feature_extractor'];

  /// @param {Parameters<PreTrainedTokenizer['apply_chat_template']>[0]} messages
  /// @param {Parameters<PreTrainedTokenizer['apply_chat_template']>[1]} options
  /// @returns {ReturnType<PreTrainedTokenizer['apply_chat_template']>}
  /// @returns {string | Tensor | number[]| number[][]|BatchEncoding} The tokenized output.
  Future<dynamic> apply_chat_template(
    List<Message> messages,
    [ApplyChatTemplateOptions? options]
  ) async {
    options ??= ApplyChatTemplateOptions.empty();

    final tokenizer = this.tokenizer;
    if (tokenizer == null) {
      throw StateError('Unable to apply chat template without a tokenizer.');
    }

    return await tokenizer.apply_chat_template(messages, options.copyWith(
      tokenize: false, // default to false
    ));
  }

  /// @param {Parameters<PreTrainedTokenizer['batch_decode']>} args
  /// @returns {ReturnType<PreTrainedTokenizer['batch_decode']>}
  List<String> batch_decode(List<List<int>> batch, {
    bool skip_special_tokens = false,
    bool clean_up_tokenization_spaces = true,
  }) {
    final tokenizer = this.tokenizer;
    if (tokenizer == null) {
      throw StateError('Unable to decode without a tokenizer.');
    }
    return tokenizer.batch_decode(
      batch,
      clean_up_tokenization_spaces: clean_up_tokenization_spaces,
      skip_special_tokens: skip_special_tokens,
    );
  }

  /// @param {Parameters<PreTrainedTokenizer['decode']>} args
  /// @returns {ReturnType<PreTrainedTokenizer['decode']>}
  String decode(List<int> token_ids, {
    bool skip_special_tokens = false,
    bool clean_up_tokenization_spaces = true,
  }) {
    final tokenizer = this.tokenizer;
    if (tokenizer == null) {
      throw StateError('Unable to decode without a tokenizer.');
    }
    return tokenizer.decode(
      token_ids,
      clean_up_tokenization_spaces: clean_up_tokenization_spaces,
      skip_special_tokens: skip_special_tokens,
    );
  }

  /// Calls the feature_extractor function with the given input.
  /// @param {any} input The input to extract features from.
  /// @param {...any} args Additional arguments.
  /// @returns {Promise<any>} A Promise that resolves with the extracted features.
  Future<dynamic> call(dynamic input, [List<dynamic> args = const []]) async {
    for (final item in [image_processor, feature_extractor, tokenizer]) {
      if (item != null) {
        return switch (item) {
          ImageProcessor() => item.call(input, args),
          PreTrainedTokenizer() => item.call(
            input,
            // TODO: Maybe make this args a class, pass along the Map, or handle this better?
            truncation: config['truncation'],
            return_token_type_ids: config['return_token_type_ids'],
            return_tensor: config['return_tensor'],
            padding: config['padding'],
            max_length: config['max_length'],
            add_special_tokens: config['add_special_tokens'],
            text_pair: config['text_pair'],
          ),
          FeatureExtractor() => item.call(),
          _ => throw ArgumentError('Unknown processor type.'),
        };
      }
    }
    throw StateError('No image processor, feature extractor, or tokenizer found.');
  }

  static Future<Processor> Function(
      String pretrained_model_name_or_path,
      [ProcessorPretrainedOptions? options]
  ) setup_from_pretrained(ProcessorReflection reflection) => (
    String pretrained_model_name_or_path,
    [ProcessorPretrainedOptions? options]
  ) => Processor.from_pretrained(reflection, pretrained_model_name_or_path, options);

  static Future<Processor> default_from_pretrained(
    String pretrained_model_name_or_path,
    [ProcessorPretrainedOptions? options]
  ) => Processor.from_pretrained(reflection, pretrained_model_name_or_path, options);

  /// Instantiate one of the processor classes of the library from a pretrained model.
  ///
  /// The processor class to instantiate is selected based on the `image_processor_type` (or `feature_extractor_type`; legacy)
  /// property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible)
  ///
  /// @param {string} pretrained_model_name_or_path The name or path of the pretrained model. Can be either:
  /// - A string, the *model id* of a pretrained processor hosted inside a model repo on huggingface.co.
  ///   Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
  ///   user or organization name, like `dbmdz/bert-base-german-cased`.
  /// - A path to a *directory* containing processor files, e.g., `./my_model_directory/`.
  /// @param {PretrainedProcessorOptions} options Additional options for loading the processor.
  ///
  /// @returns {Promise<Processor>} A new instance of the Processor class.
  static Future<Processor> from_pretrained(
    ProcessorReflection reflection,
    String pretrained_model_name_or_path,
    [ProcessorPretrainedOptions? options]
  ) async {
    final [config, components] = await Future.wait([
      // Not sure why this TODO was in the transformers.js code?
      // TODO:
      reflection.uses_processor_config
          ? getModelJSON(pretrained_model_name_or_path, PROCESSOR_NAME, true, options)
          : Future.value(<String, dynamic>{}),
      Future.wait(
          reflection.classes.map((cls) async {
            final component = switch (cls) {
              'image_processor_class' => await reflection.image_processor_class?.from_pretrained(pretrained_model_name_or_path, options),
              'tokenizer_class' => await reflection.tokenizer_class?.from_pretrained(pretrained_model_name_or_path, options),
              'feature_extractor_class' => await reflection.feature_extractor_class?.from_pretrained(pretrained_model_name_or_path, options),
              _ => null,
            };
            if (component == null) return null;

            return MapEntry(cls.replaceAll(RegExp(r'_class$'), ''), component);
          })
      ).then((e) => Map.fromEntries(e.nonNulls)),
    ]);

    return reflection.constructor(config, components);
  }
}
